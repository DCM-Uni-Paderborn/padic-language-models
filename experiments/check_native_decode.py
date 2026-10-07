"""Frozen complete-fixture code/mask identity and selected FP64 readout gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from padic_lm import native_decode
from padic_lm.native_decode import FiniteBridge, depth_lookup, selected_keys, fixed_keys, attention_component

DEV = 'ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8'
AUDIT = '28b682bdf864333c6161606cfc4e6c67260551cab60fe5eb6f162676a6439a36'
TRAIN = 'f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1'
CONTEXT = {}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write(path, value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as r: return {k:r[k] for k in r.files}
def decode(bits): return (bits.astype(np.uint32)<<16).view(np.float32)
def tensor(bits): return torch.tensor(bits.astype(np.uint16),device='cuda').view(torch.bfloat16)
def bits(t): return t.contiguous().view(torch.uint16).cpu().numpy()

def error(actual, reference, limit):
    a = actual.float().cpu().numpy().astype(np.float64)
    delta = a-reference
    nrmse = float(np.sqrt(np.sum(delta**2)/max(np.sum(reference**2),1e-30)))
    maximum = float(np.max(np.abs(delta)))
    allowed = float(.03*np.max(np.abs(reference))+1e-4)
    return {'nrmse':nrmse,'max_abs':maximum,'max_abs_allowed':allowed,
            'nrmse_limit':limit,'passed':nrmse<=limit and maximum<=allowed}

def fp64(query, keys, values, weight, chosen):
    head=[]
    for h in range(15):
        ids=chosen[h//3]
        score=keys[h//3,ids].astype(np.float64)@query[h,0].astype(np.float64)/8
        exp=np.exp(score-score.max()); probability=exp/exp.sum()
        head.append(probability@values[h//3,ids].astype(np.float64))
    head=np.array(head)[None,:,None]
    projected=head.transpose(0,2,1,3).reshape(1,1,960)@weight.astype(np.float64).T
    return projected,head

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('development','development-audit','training','protocol','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args(); started=time.monotonic()
    for n in ('development','development_audit','training','protocol','output'):
        setattr(args,n,getattr(args,n).resolve())
    if args.output.exists():raise FileExistsError('fresh identity output required')
    if digest(args.development/'completed.json')!=DEV or digest(args.development_audit)!=AUDIT or digest(args.training/'completed.json')!=TRAIN:
        raise ValueError('requires unchanged audited study parents')
    assert json.loads(args.development_audit.read_text())['status']=='shared_budget_audit_passed'
    args.output.mkdir(parents=True); CONTEXT.update(output=args.output,started=started)
    def guard():
        if time.monotonic()-started>=3600:raise TimeoutError('native correctness3600s guard reached')
    protocol=args.protocol.read_bytes(); sources={Path(__file__).name:Path(__file__).read_bytes(),
            'native_decode.py':Path(native_decode.__file__).read_bytes()}
    (args.output/'sources').mkdir()
    for n,b in sources.items():(args.output/'sources'/n).write_bytes(b)
    (args.output/'prospective_protocol.txt').write_bytes(protocol)
    input_hashes={}
    def verified(directory,parent,name):
        path=directory/name
        if digest(path)!=parent['artifact_sha256'][name] or path.stat().st_size!=parent['artifact_bytes'][name]:
            raise ValueError('input differs: '+name)
        input_hashes[str(path.relative_to(ROOT))]=parent['artifact_sha256'][name]
        return path
    dev=json.loads((args.development/'completed.json').read_text())
    train=json.loads((args.training/'completed.json').read_text())
    weight_bits=arrays(verified(args.development,dev,'output_projection.npz'))['weight_bits']
    weights=tensor(weight_bits); weight=decode(weight_bits)
    frozen={}
    for seed in (17,29,43):
        encoder=arrays(verified(args.training,train,f'seed-{seed}/final_encoder.npz'))
        state=arrays(verified(args.training,train,f'seed-{seed}/routing_state.npz'))
        frozen[seed]=(encoder,state)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)
    environment={'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,
                 'python':sys.version,'device':str(torch.cuda.get_device_properties(0)),
                 'tf32':False,'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    write(args.output/'config.json',{'protocol_sha256':hashlib.sha256(protocol).hexdigest(),
          'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
          'parents':{'development':DEV,'audit':AUDIT,'training':TRAIN},'environment':environment})
    code_records=[];mask_records=[];readouts=[];raw_outputs={};backend_errors=[];backend=SDPBackend.FLASH_ATTENTION
    fixtures={length:verified(args.development,dev,f'windows/length-{length}-article-00.npz') for length in (512,2048)}
    # Probe every declared shape before any readout; a failure selects MATH
    # consistently for both contexts rather than changing backend mid-study.
    for length,path in fixtures.items():
        with np.load(path,allow_pickle=False) as r:
            pq,pk,pv=(tensor(r['reference_'+n+'_bits']) for n in ('queries','keys','values'))
        for size in (length,128):
            try:
                with warnings.catch_warnings(record=True) as ws, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention_component(pq[:,-1:],pk[:,:size],pv[:,:size],weights)
                backend_errors.extend({'length':length,'keys':size,'warning':str(w.message)} for w in ws)
            except RuntimeError as exc:
                backend_errors.append({'length':length,'keys':size,'error':str(exc)});backend=SDPBackend.MATH
        del pq,pk,pv
    for length in (512,2048):
        guard()
        with np.load(fixtures[length],allow_pickle=False) as r:
            names=[n for n in r.files if n in ('reference_queries_bits','reference_keys_bits','reference_values_bits')
                   or n.endswith(('_query_latents','_key_latents','_query_codes','_key_codes'))
                   or n.endswith('_group_selection_bits')]
            raw={n:r[n] for n in names}
        q,k,v=(tensor(raw['reference_'+n+'_bits']) for n in ('queries','keys','values'))
        qnp,knp,vnp=(decode(raw['reference_'+n+'_bits']) for n in ('queries','keys','values'))
        methods={'stock':None,'recency':None,'uniform':None}
        for seed in (17,29,43):
            encoder,state=frozen[seed]
            for family,prime,digits in (('p2',2,(4,4)),('p3',3,(3,2))):
                book={n:state[family+'_'+n] for n in ('center','projection','thresholds')}
                bridge=FiniteBridge(encoder,book,prime,digits,'cuda')
                codes={}
                for role,values in (('query',q),('key',k.repeat_interleave(3,dim=0))):
                    actual=[];latent=[]
                    for pos in range(length):
                        guard(); c,z=bridge.encode(values[:,pos:pos+1],role);actual.append(c);latent.append(z)
                    actual=torch.cat(actual,1);latent=torch.cat(latent,1).cpu().numpy()
                    expected=raw[f's{seed}_{family}_{role}_codes'][...,0]
                    differences=np.argwhere(actual.cpu().numpy()!=expected)
                    record={'length':length,'seed':seed,'family':family,'role':role,'elements':int(expected.size),
                            'mismatch_elements':int(len(differences)),'examples':differences[:20].tolist(),
                            'latent_max_abs_difference':float(np.max(np.abs(latent-raw[f's{seed}_{role}_latents'])))}
                    code_records.append(record);codes[role]=actual
                    raw_outputs[f'L{length}_s{seed}_{family}_{role}_codes']=actual.cpu().numpy()
                kinds=[(family,1)]+([('coarsened_p2',2)] if family=='p2' else [])
                for kind,coarse in kinds:
                    lut=depth_lookup(prime,digits,coarse,'cuda');mismatched_rows=0;examples=[]
                    expected=np.unpackbits(raw[f's{seed}_{kind}_group_selection_bits'],axis=-1,bitorder='little')[...,:length].astype(bool)
                    for pos in range(length):
                        guard();chosen=selected_keys(codes['query'][:,pos],codes['key'][:,:pos+1],lut).cpu().numpy()
                        mask=np.zeros((5,length),bool);np.put_along_axis(mask,chosen,True,axis=1)
                        bad=np.flatnonzero(np.any(mask!=expected[:,pos],axis=1));mismatched_rows+=len(bad)
                        if bad.size and len(examples)<20:examples.append({'position':pos,'groups':bad.tolist()})
                    mask_records.append({'length':length,'seed':seed,'kind':kind,'rows':5*length,
                                         'mismatch_rows':int(mismatched_rows),'examples':examples})
                    methods[f's{seed}_{kind}']=(codes['query'],codes['key'],lut)
                print(json.dumps({'length':length,'seed':seed,'family':family,'codes_mismatched':sum(r['mismatch_elements'] for r in code_records[-2:])}),flush=True)
        for prefix in (length//2,3*length//4,length):
            pos=prefix-1
            for method,data in methods.items():
                guard()
                if method=='stock':chosen=None;ids=np.broadcast_to(np.arange(prefix),(5,prefix))
                elif method in ('recency','uniform'):chosen=fixed_keys(prefix,method,'cuda');ids=chosen.cpu().numpy()
                else:chosen=selected_keys(data[0][:,pos],data[1][:,:prefix],data[2]);ids=chosen.cpu().numpy()
                with sdpa_kernel(backend):projected,head=attention_component(q[:,pos:pos+1],k[:,:prefix],v[:,:prefix],weights,chosen)
                ref_projected,ref_head=fp64(qnp[:,pos:pos+1],knp[:,:prefix],vnp[:,:prefix],weight,ids)
                rec={'length':length,'prefix':prefix,'method':method,'backend':str(backend),
                     'head':error(head,ref_head,.01),'projected':error(projected,ref_projected,.015)}
                readouts.append(rec);label=f'L{length}_P{prefix}_{method}'
                raw_outputs[label+'_head_bits']=bits(head);raw_outputs[label+'_projected_bits']=bits(projected)
                raw_outputs[label+'_head_fp64']=ref_head;raw_outputs[label+'_projected_fp64']=ref_projected
        del raw,q,k,v
    guard()
    assert len(code_records)==24 and sum(r['elements'] for r in code_records)==460800
    assert len(mask_records)==18 and sum(r['rows'] for r in mask_records)==115200 and len(readouts)==72
    same_sources=Path(__file__).read_bytes()==sources[Path(__file__).name] and Path(native_decode.__file__).read_bytes()==sources['native_decode.py']
    if not same_sources or args.protocol.read_bytes()!=protocol:raise RuntimeError('executed source/protocol changed')
    passed=all(r['mismatch_elements']==0 for r in code_records) and all(r['mismatch_rows']==0 for r in mask_records) and all(r['head']['passed'] and r['projected']['passed'] for r in readouts)
    np.savez_compressed(args.output/'native_arrays.npz',**raw_outputs)
    report={'status':'native_decode_correctness_passed' if passed else 'native_decode_correctness_failed',
            'created_utc':datetime.now(timezone.utc).isoformat(),'selected_input_sha256':input_hashes,
            'code_records':code_records,'mask_records':mask_records,'readouts':readouts,'backend_probe_messages':backend_errors,
            'process_wall_seconds':time.monotonic()-started,'scope':'Complete first-development-article native code/mask identity and one-query FP64 component agreement; no native timing or whole-model NLL claim.'}
    write(args.output/'results.json',report)
    guard()
    if not passed:raise AssertionError('complete native identity/readout gate failed; all observed failures retained')
    manifest={str(x.relative_to(args.output)):digest(x) for x in sorted(args.output.rglob('*')) if x.is_file()}
    write(args.output/'completed.json',{'status':report['status'],'artifact_sha256':manifest,
            'artifact_bytes':{n:(args.output/n).stat().st_size for n in manifest},'process_wall_seconds':time.monotonic()-started})
    (args.output/'completed.sha256').write_text(digest(args.output/'completed.json')+'\n');guard()
    print(json.dumps({'status':report['status'],'code_elements':460800,'mask_rows':115200,'readouts':72}),flush=True)

if __name__=='__main__':
    try:
        with torch.inference_mode():main()
    except Exception as exc:
        if CONTEXT:
            out=CONTEXT['output'];(out/'completed.json').unlink(missing_ok=True);(out/'completed.sha256').unlink(missing_ok=True)
            write(out/'failed.json',{'status':'native_decode_correctness_failed','type':type(exc).__name__,
                    'message':str(exc),'process_wall_seconds':time.monotonic()-CONTEXT['started']})
        raise
