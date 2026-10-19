"""Finite-residue and unit/valuation codebooks with real QAT decoding.

Code indices have 2--4 bits. Matrix products and optimization remain real.
The same decoded codebook has an identical ordinary lookup implementation.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

GROUP = 64

def alphabet(kind, bits):
    if bits not in (2, 3, 4, 8):
        raise ValueError("supported code widths are 2,3,4,8")
    if kind == "int":
        q = 2**(bits-1)-1
        return np.arange(-q, q+1, dtype=np.float64)
    if kind == "int_full":
        return np.arange(-2**(bits-1), 2**(bits-1), dtype=np.float64)
    if kind == "fp4":
        if bits != 4: raise ValueError("E2M1 needs four bits")
        return np.array([-6.,-4.,-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.,4.,6.])
    if bits == 8: return alphabet("int", 8)
    role, prime = kind.rsplit("_", 1); p = int(prime)
    if p not in (2,3,5): raise ValueError("prime must be 2,3,5")
    if role == "residue":
        m = 1
        while m*p <= 2**bits: m *= p
        if m == 1: raise ValueError("budget cannot encode one radix digit")
        return np.arange(-(m//2), (m+1)//2, dtype=np.float64)
    if role != "unit": raise ValueError("unknown codebook")
    k = 2 if p == 2 else 1
    units = [u for u in range(-(p**k//2), (p**k+1)//2) if u % p]
    exponents = (2**bits-1)//len(units)
    if not exponents: raise ValueError("budget cannot encode the unit alphabet")
    values = np.array(sorted({0, *(u*p**e for e in range(exponents) for u in units)}), dtype=np.float64)
    assert len(values) <= 2**bits
    return values

def nearest(x, values):
    mids = (values[1:]+values[:-1])/2
    return values[torch.bucketize(x.contiguous(), mids, right=False)]

def fake_quant(x, scale, values):
    z = x/scale
    hard = nearest(z, values)
    return (z+(hard-z).detach())*scale

def initial_scale(groups, values):
    base = (groups.abs().amax(-1, keepdim=True)/values.abs().max()).clamp_min(1e-8)
    best, error = base, torch.full_like(base, float("inf"))
    for factor in (.25,.35,.5,.65,.8,1.,1.25,1.5,2.):
        scale = base*factor
        err = (nearest(groups/scale, values)*scale-groups).square().mean(-1,keepdim=True)
        choose = err < error
        best, error = torch.where(choose, scale, best), torch.minimum(err,error)
    return best

class QuantLinear(nn.Module):
    def __init__(self, original, family, wbits, abits, activation_family=None):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        if self.in_features % GROUP: raise ValueError("input width must be divisible by64")
        self.wbits, self.abits, self.family = wbits, abits, family
        weight = original.weight.detach().float()
        self.weight = nn.Parameter(weight.clone())
        self.register_buffer("bias", None if original.bias is None else original.bias.detach().float().clone())
        self.register_buffer("levels", torch.tensor(alphabet(family,wbits),dtype=torch.float32,device=weight.device))
        akind = "int" if abits == 8 else (activation_family or family)
        self.register_buffer("activation_levels", torch.tensor(alphabet(akind,abits),dtype=torch.float32,device=weight.device))
        self.log_scale = nn.Parameter(initial_scale(weight.reshape(self.out_features,-1,GROUP),self.levels).log())
        self.log_clip = nn.Parameter(torch.zeros((),device=weight.device))
        self.log_scale.register_hook(lambda g:g/math.sqrt(GROUP))
        self.log_clip.register_hook(lambda g:g/math.sqrt(GROUP))

    def forward(self, x):
        dtype = x.dtype
        wg = self.weight.reshape(self.out_features,-1,GROUP)
        scale = self.log_scale.clamp(-25,5).exp()
        w = fake_quant(wg,scale,self.levels).reshape_as(self.weight)
        xx = x.float().reshape(-1,self.in_features//GROUP,GROUP)
        base = (xx.detach().abs().amax(-1,keepdim=True)/self.activation_levels.abs().max()).clamp_min(1e-8)
        sx = base*self.log_clip.clamp(-2.3,2.3).exp()
        xx = fake_quant(xx,sx,self.activation_levels).reshape_as(x)
        # TF32 is disabled by the runner. Accumulation is FP32 with no modular wrap.
        return F.linear(xx,w,self.bias).to(dtype)

def spec_for(regime, family, layer):
    if regime.startswith("mix"):
        w = 2 if layer % 2 == 0 else 4
        a = 4 if regime == "mix24_a4" else 8
        return w,a,"int"
    w,a = [int(v) for v in regime.removeprefix("w").split("_a")]
    return w,a,("int" if a == 8 or regime == "w3_a4" else family)

def signature(regime,family,layers=30):
    return tuple((tuple(alphabet(family,spec_for(regime,family,l)[0])),
                  tuple(alphabet(spec_for(regime,family,l)[2],spec_for(regime,family,l)[1]))) for l in range(layers))

def replace_projections(model,regime,family):
    for p in model.parameters(): p.requires_grad_(False)
    modules, packed, scales, parameter_count = [],0,0,0
    for index,layer in enumerate(model.model.layers):
        w,a,af = spec_for(regime,family,index)
        for owner in (layer.self_attn,layer.mlp):
            for name,old in list(owner.named_children()):
                if not isinstance(old,nn.Linear): continue
                new = QuantLinear(old,family,w,a,af)
                setattr(owner,name,new);modules.append(new)
                n = new.weight.numel();parameter_count += n
                packed += math.ceil(n*math.ceil(math.log2(len(new.levels)))/8)
                scales += 4*(new.log_scale.numel()+1)
    total = sum(p.numel() for p in model.parameters() if not any(p is v for m in modules for v in (m.weight,m.log_scale,m.log_clip)))
    return modules,dict(quantized_projection_parameters=parameter_count,projection_code_bytes=packed,
                       scale_metadata_bytes=scales,remaining_bf16_parameter_bytes=2*total,
                       ideal_inference_parameter_bytes=packed+scales+2*total,
                       group_size=GROUP,packed_inference_implemented=False)

def cases():
    regimes = ["w2_a2","w3_a3","w4_a4","w2_a8","w3_a8","w4_a8","w3_a4","mix24_a4"]
    families = ["int","residue_3","unit_2","unit_3"]
    requested = [(r,f) for r in regimes for f in families]
    requested += [(r,f) for r in ["w3_a8","w4_a8"] for f in ["residue_5","unit_5"]]
    requested += [(f"w{b}_a8","int_full") for b in (2,3,4)]
    requested += [(r,"fp4") for r in ["w4_a4","w4_a8"]]
    unique,aliases={},{}
    for regime,family in requested:
        sig = signature(regime,family)
        tag=regime+"_"+family
        if sig not in unique:unique[sig]=tag
        aliases[tag]=unique[sig]
    return requested,aliases

def pack_indices(indices,bits):
    a=np.asarray(indices)
    if bits not in (2,3,4) or a.dtype.kind not in 'iu' or np.any(a>=2**bits) or np.any(a<0):
        raise ValueError('valid unsigned2/3/4-bit indices required')
    a=a.reshape(-1).astype(np.uint8)
    flags=((a[:,None]>>np.arange(bits,dtype=np.uint8))&1).reshape(-1)
    return np.packbits(flags,bitorder='little').tobytes()

def unpack_indices(payload,count,bits):
    if bits not in (2,3,4) or count<0 or len(payload)!=(count*bits+7)//8:
        raise ValueError('packed length differs')
    flags=np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder='little')
    if np.any(flags[count*bits:]):raise ValueError('nonzero padding')
    return (flags[:count*bits].reshape(count,bits)*(1<<np.arange(bits))).sum(1).astype(np.uint8)
