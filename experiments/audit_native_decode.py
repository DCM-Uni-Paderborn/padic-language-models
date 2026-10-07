"""Read-only independent congruence/rank and FP64 audit of failed native gate.

No Torch, native implementation, production selector or gate-driver imports.
Reconstructs the mask differences implied by every archived native code and
independently checks all saved arithmetic outputs. It does not claim a second
complete GPU mask replay because those masks were not saved in full.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import time
import numpy as np

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def bf16(a):return (a.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
def chosen(q,k,p,d,coarsen):
    length=k.shape[1]
    if length<=128:return np.broadcast_to(np.arange(length),(5,length))
    stop=length-8; modulus=p**d[0]
    qc=np.stack((q.astype(int)%modulus,q.astype(int)//modulus),-1)
    kc=np.stack((k[:,:stop].astype(int)%modulus,k[:,:stop].astype(int)//modulus),-1)
    score=np.zeros((15,stop),np.int16)
    for level in range(1,max(d)+1):score+=np.all((qc[:,None]-kc)%(p**level)==0,axis=-1)
    score//=coarsen
    # Sorting/searchsorted forms ranks independently of the GPU histogram.
    ordered=np.sort(score,axis=1)
    ranks=np.array([np.searchsorted(row,s,side='left')+np.searchsorted(row,s,side='right')-1
                    for row,s in zip(ordered,score)])
    group=ranks.reshape(5,3,stop).sum(1)
    return np.array([np.sort(np.r_[np.lexsort((np.arange(stop),r))[-120:],np.arange(stop,length)]) for r in group])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('gate','development','output'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args();start=time.monotonic()
    if a.output.exists():raise FileExistsError('fresh diagnostic audit required')
    a.output.mkdir(parents=True)
    report=read(a.gate/'results.json');assert report['status']=='native_decode_correctness_failed'
    assert (a.gate/'failed.json').exists() and not (a.gate/'completed.json').exists()
    config=read(a.gate/'config.json')
    for n,h in config['source_sha256'].items():assert digest(a.gate/'sources'/n)==h
    assert digest(a.gate/'prospective_protocol.txt')==config['protocol_sha256']
    saved=np.load(a.gate/'native_arrays.npz',allow_pickle=False)
    parent=read(a.development/'completed.json')
    with np.load(a.development/'output_projection.npz',allow_pickle=False) as r:weight=bf16(r['weight_bits'])
    code_count=mask_count=mask_mismatches=code_mismatches=0
    records=[];readouts=[];max_metric_difference=0.;max_reference_difference=0.
    gate_code={(r['length'],r['seed'],r['family'],r['role']):r for r in report['code_records']}
    gate_mask={(r['length'],r['seed'],r['kind']):r for r in report['mask_records']}
    gate_readout={(r['length'],r['prefix'],r['method']):r for r in report['readouts']}
    for length in (512,2048):
        name=f'windows/length-{length}-article-00.npz';path=a.development/name
        assert digest(path)==parent['artifact_sha256'][name]
        with np.load(path,allow_pickle=False) as raw:
            q,k,v=(bf16(raw['reference_'+n+'_bits']) for n in ('queries','keys','values'))
            for seed in (17,29,43):
                for family,prime,digits in (('p2',2,(4,4)),('p3',3,(3,2))):
                    for role in ('query','key'):
                        new=saved[f'L{length}_s{seed}_{family}_{role}_codes']
                        old=raw[f's{seed}_{family}_{role}_codes'][...,0]
                        diff=np.argwhere(new!=old);claimed=gate_code[length,seed,family,role]
                        assert len(diff)==claimed['mismatch_elements'] and diff[:20].tolist()==claimed['examples']
                        code_count+=old.size;code_mismatches+=len(diff)
                    qc=saved[f'L{length}_s{seed}_{family}_query_codes'];kc=saved[f'L{length}_s{seed}_{family}_key_codes']
                    for kind,coarse in [(family,1)]+([('coarsened_p2',2)] if family=='p2' else []):
                        expected=np.unpackbits(raw[f's{seed}_{kind}_group_selection_bits'],axis=-1,bitorder='little')[...,:length].astype(bool)
                        mismatches=0;examples=[]
                        for pos in range(length):
                            ids=chosen(qc[:,pos],kc[:,:pos+1],prime,digits,coarse)
                            assert np.all(ids>=0) and np.all(ids<=pos) and ids.shape==(5,min(128,pos+1))
                            assert np.all(np.diff(ids,axis=1)>0)
                            mask=np.zeros((5,length),bool);np.put_along_axis(mask,ids,True,axis=1)
                            bad=np.flatnonzero(np.any(mask!=expected[:,pos],axis=1));mismatches+=bad.size
                            if bad.size and len(examples)<20:examples.append({'position':pos,'groups':bad.tolist()})
                            mask_count+=5
                        claimed=gate_mask[length,seed,kind]
                        assert int(mismatches)==claimed['mismatch_rows'] and examples==claimed['examples']
                        mask_mismatches+=mismatches;records.append({'length':length,'seed':seed,'kind':kind,'implied_mismatch_rows':int(mismatches)})
            for prefix in (length//2,3*length//4,length):
                for method in ['stock','recency','uniform']+[f's{s}_{f}' for s in (17,29,43) for f in ('p2','coarsened_p2','p3')]:
                    if method=='stock':ids=np.broadcast_to(np.arange(prefix),(5,prefix))
                    elif method=='recency':ids=np.broadcast_to(np.arange(prefix-128,prefix),(5,128))
                    elif method=='uniform':ids=np.broadcast_to(np.r_[((2*np.arange(120)+1)*(prefix-8))//240,np.arange(prefix-8,prefix)],(5,128))
                    else:
                        seed=int(method.split('_')[0][1:]);family='p3' if method.endswith('p3') else 'p2'
                        prime,digits=(3,(3,2)) if family=='p3' else (2,(4,4))
                        qc=saved[f'L{length}_s{seed}_{family}_query_codes'];kc=saved[f'L{length}_s{seed}_{family}_key_codes']
                        ids=chosen(qc[:,prefix-1],kc[:,:prefix],prime,digits,2 if 'coarsened' in method else 1)
                    # Explicit coordinate sums and scalar math.fsum probability
                    # denominator differ from the driver's matrix contraction.
                    heads=[]
                    for h in range(15):
                        selected=ids[h//3]
                        scores=np.sum(k[h//3,selected]*q[h,prefix-1],axis=1)/8.
                        exp=np.exp(scores-scores.max());prob=exp/math.fsum(exp.tolist())
                        heads.append(np.sum(prob[:,None]*v[h//3,selected],axis=0))
                    head=np.array(heads)[None,:,None];projected=head.reshape(960)@weight.T
                    projected=projected.reshape(1,1,960);label=f'L{length}_P{prefix}_{method}'
                    native_head=bf16(saved[label+'_head_bits']);native_projected=bf16(saved[label+'_projected_bits'])
                    record={'length':length,'prefix':prefix,'method':method}
                    for field,native,ref,limit in (('head',native_head,head,.01),('projected',native_projected,projected,.015)):
                        max_reference_difference=max(max_reference_difference,float(np.max(np.abs(ref-saved[label+'_'+field+'_fp64']))))
                        delta=native-ref;nrmse=float(np.sqrt(np.sum(delta**2)/max(np.sum(ref**2),1e-30)))
                        maximum=float(np.max(np.abs(delta)));allowed=float(.03*np.max(np.abs(ref))+1e-4)
                        assert nrmse<=limit and maximum<=allowed
                        claimed=gate_readout[length,prefix,method][field]
                        for x,y in ((nrmse,claimed['nrmse']),(maximum,claimed['max_abs']),(allowed,claimed['max_abs_allowed'])):
                            max_metric_difference=max(max_metric_difference,abs(x-y));assert abs(x-y)<1e-11
                        record[field]={'nrmse':nrmse,'max_abs':maximum,'max_abs_allowed':allowed,'passed':True}
                    readouts.append(record)
    assert code_count==460800 and code_mismatches==10 and mask_count==115200 and mask_mismatches==378 and len(readouts)==72
    result={'status':'native_failure_diagnostic_audit_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'gate_result_sha256':digest(a.gate/'results.json'),'source_sha256':digest(Path(__file__)),
        'code_elements':code_count,'code_mismatches':code_mismatches,'implied_mask_rows':mask_count,
        'implied_mask_mismatches':int(mask_mismatches),'readouts_verified':72,'mask_records':records,'readouts':readouts,
        'max_reference_difference':max_reference_difference,'max_metric_difference':max_metric_difference,
        'process_wall_seconds':time.monotonic()-start,
        'scope':'No production imports. All saved code differences, implied independent masks and72 saved selected-set arithmetic readouts independently reconstructed. Implied mask counts/examples match the native gate, but this does not independently replay every GPU mask or convert the failed strict identity into a pass.'}
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('mask_records','readouts')},indent=2))

if __name__=='__main__':main()
