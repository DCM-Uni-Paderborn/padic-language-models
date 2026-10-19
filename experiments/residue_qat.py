"""Exact low-digit projection versus clipped decoding under the same QAT budget."""
import argparse,json,math,sys,gc,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import low_precision_study as study
import padic_lm.low_precision as core
_original_alphabet=core.alphabet
_OriginalLinear=core.QuantLinear
def cyclic_alphabet(kind,bits):
    return _original_alphabet("residue_"+kind.split("_")[1],bits) if kind.startswith("wrap_") else _original_alphabet(kind,bits)

def project(x,scale,values):
    # FP64 grid conversion preserves the low integer digits before reduction.
    m=len(values);half=m//2
    with torch.no_grad():
        n=torch.round(x.detach().double()/scale.detach().double())
        hard=(torch.remainder(n+half,m)-half).float()
    z=x/scale
    return (z+(hard-z).detach())*scale

class ProjectionLinear(_OriginalLinear):
    def __init__(self,original,family,wbits,abits,activation_family=None):
        super().__init__(original,family,wbits,abits,activation_family)
        self.cyclic=family.startswith("wrap_")
        self.cyclic_activation=abits!=8 and (activation_family or family).startswith("wrap_")
        if not self.cyclic:return
        g=self.weight.detach().reshape(self.out_features,-1,core.GROUP)
        base=(g.abs().amax(-1,keepdim=True)/self.levels.abs().max()).clamp_min(1e-8)
        best=base;error=torch.full_like(base,float("inf"))
        for factor in (.25,.35,.5,.65,.8,1.,1.25,1.5,2.):
            scale=base*factor
            err=(project(g,scale,self.levels)-g).square().mean(-1,keepdim=True)
            choose=err<error;best=torch.where(choose,scale,best);error=torch.minimum(err,error)
        with torch.no_grad():self.log_scale.copy_(best.log())
    def forward(self,x):
        if not self.cyclic:return super().forward(x)
        dtype=x.dtype
        wg=self.weight.reshape(self.out_features,-1,core.GROUP);s=self.log_scale.clamp(-25,5).exp()
        w=project(wg,s,self.levels).reshape_as(self.weight)
        xx=x.float().reshape(-1,self.in_features//core.GROUP,core.GROUP)
        base=(xx.detach().abs().amax(-1,keepdim=True)/self.activation_levels.abs().max()).clamp_min(1e-8)
        sx=base*self.log_clip.clamp(-2.3,2.3).exp()
        quant=project if self.cyclic_activation else core.fake_quant
        xx=quant(xx,sx,self.activation_levels).reshape_as(x)
        return F.linear(xx,w,self.bias).to(dtype)

def physical_signature(regime,family,layers=30):
    values=[]
    for layer in range(layers):
        w,a,af=core.spec_for(regime,family,layer)
        values.append((family.startswith("wrap_"),af.startswith("wrap_") and a!=8,
                       tuple(cyclic_alphabet(family,w)),tuple(cyclic_alphabet(af,a))))
    return tuple(values)

def run(inputs,output,size):
    core.alphabet=cyclic_alphabet;core.QuantLinear=ProjectionLinear
    output.mkdir(parents=True,exist_ok=False)
    freeze=study.freeze(output)
    freeze["additional_sources"]={"experiments/residue_qat.py":study.sha(__file__),
        "research/residue-qat-protocol-20261019.md":study.sha(ROOT/"research/residue-qat-protocol-20261019.md")}
    study.write(output/"freeze.json",freeze)
    meta=json.loads((inputs/"manifest.json").read_text());assert study.sha(inputs/"tokens.npz")==meta["tokens_sha256"]
    with np.load(inputs/"tokens.npz",allow_pickle=False) as z:tokens={k:torch.from_numpy(z[k].copy()) for k in z.files}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    teacher=study.load_model(size).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    baseline=study.evaluate(teacher,tokens,["development","wikitext","pg19"])
    study.write(output/(size+"-reference.json"),{"metrics":baseline,"state_sha256":study.state_dict_hash(teacher)})
    cache=[]
    with torch.inference_mode():
        for i in range(0,128,2):cache.append(teacher(tokens["train"][i:i+2].cuda(),use_cache=False).logits[:,:-1].cpu())
    cache=torch.cat(cache);del teacher;gc.collect();torch.cuda.empty_cache()
    requested=[(r,f) for r in ("w2_a8","w3_a8","w4_a8","w4_a4","mix24_a4") for f in ("wrap_2","wrap_3")]
    requested+=[("w4_a4","int_full"),("mix24_a4","int_full")]
    if size=="360M":requested=[("w4_a8","wrap_2"),("w4_a8","wrap_3"),("w4_a8","int_full"),("w4_a8","residue_3")]
    aliases={};unique={}
    for r,f in requested:
        sig=physical_signature(r,f,layers=30 if size=="135M" else 32);tag=r+"_"+f
        if sig not in unique:unique[sig]=tag
        aliases[tag]=unique[sig]
    study.write(output/"cases.json",{"requested":[{"regime":r,"family":f,"canonical":aliases[r+"_"+f]} for r,f in requested]})
    for r,f in requested:
        tag=r+"_"+f
        if aliases[tag]!=tag:continue
        for seed in study.SEEDS:
            try:study.train_one(cache,tokens,size,r,f,seed,output/size/tag/f"seed-{seed}")
            except Exception as e:print(json.dumps({"run_closed":tag,"seed":seed,"error_type":type(e).__name__,"message":str(e)}),flush=True)
    study.write(output/"completed.json",{"status":"cyclic_projection_matrix_finished",
        "complete_fits":len(list(output.glob("*/**/completed.json"))),"closed_failed_fits":len(list(output.glob("*/**/failed.json")))})
    print("CYCLIC_PROJECTION_COMPLETE",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--size",choices=["135M","360M"],required=True)
    p.add_argument("--after",nargs="*",type=Path,default=[])
    a=p.parse_args()
    while any(not (q/"completed.json").exists() for q in a.after):time.sleep(15)
    run(a.inputs,a.output,a.size)
