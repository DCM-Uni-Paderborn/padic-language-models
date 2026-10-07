"""Post hoc recency/coarse-ball diagnosis of frozen masks; no new forwards."""
import argparse
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def coarse_empty(packed_query,packed_key,prime,first_digits):
    q,k=packed_query[...,0].astype(np.int64),packed_key[...,0].astype(np.int64)
    signature=lambda x:x%prime+prime*((x//prime**first_digits)%prime)
    q,k=signature(q),signature(k)
    counts=np.cumsum(k[...,None]==np.arange(prime**2),axis=1)
    h=np.arange(len(q))[:,None];position=np.arange(128,512)[None,:]
    # Old candidates end at t-8, immediately before the mandatory latest eight.
    old_match=counts[h,position-8,q[:,128:]]
    return old_match==0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('input','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();p=args.input.resolve();out=args.output.resolve();started=time.monotonic()
    if out.exists():raise FileExistsError('fresh mechanism-diagnostic output required')
    completion=json.loads((p/'completed.json').read_text());r=json.loads((p/'results.json').read_text())
    assert digest(p/'completed.json')==(p/'completed.sha256').read_text().strip()=='2c39f405bfd0ded6e86850d3bd46320877a22e2ed98a2f02321cbb75196f4660'
    names=[n for n in r['config']['methods'] if n!='full_control']
    counts={n:{'exact_recency_rows':np.zeros(15,dtype=np.int64),'overlap_keys':np.zeros(15,dtype=np.int64),
        'empty_coarse_rows':np.zeros(15,dtype=np.int64),'coarse_available':False} for n in names}
    for index in range(64):
        path=p/'windows'/f'window-{index:03d}.npz'
        assert digest(path)==completion['artifact_sha256'][str(path.relative_to(p))]
        with np.load(path,allow_pickle=False) as w:
            rec=np.unpackbits(w['recency_selection_bits'],axis=-1,bitorder='little').astype(bool)[:,128:]
            for name in names:
                mask=np.unpackbits(w[name+'_selection_bits'],axis=-1,bitorder='little').astype(bool)[:,128:]
                identical=(mask==rec).all(-1)
                counts[name]['exact_recency_rows']+=identical.sum(-1)
                counts[name]['overlap_keys']+=(mask&rec).sum(axis=(-1,-2))
                if name.endswith(('_p2','_p3')):
                    prime=3 if name.endswith('_p3') else 2
                    empty=coarse_empty(w[name+'_packed_query_codes'],w[name+'_packed_key_codes'],prime,3 if prime==3 else 4)
                    assert np.all(identical[empty]),'empty old coarse ball must reduce exactly to recency'
                    counts[name]['coarse_available']=True;counts[name]['empty_coarse_rows']+=empty.sum(-1)
    reports={}
    for name,c in counts.items():
        report={'affected_query_head_rows':64*15*384,'exact_recency_fraction':float(c['exact_recency_rows'].sum()/(64*15*384)),
            'mean_fraction_selected_keys_shared_with_recency':float(c['overlap_keys'].sum()/(64*15*384*128)),
            'mean_selected_keys_outside_recency':float(128-c['overlap_keys'].sum()/(64*15*384)),
            'per_head_exact_recency_fraction':(c['exact_recency_rows']/(64*384)).tolist()}
        if c['coarse_available']:
            report['empty_nonmandatory_coarse_ball_fraction']=float(c['empty_coarse_rows'].sum()/(64*15*384))
            report['per_head_empty_nonmandatory_coarse_ball_fraction']=(c['empty_coarse_rows']/(64*384)).tolist()
            report['empty_coarse_implies_exact_recency_verified']=True
        reports[name]=report
    result={'status':'qk_routing_overlap_complete','created_utc':datetime.now(timezone.utc).isoformat(),
        'input_completion_sha256':digest(p/'completed.json'),'source_sha256':digest(__file__),'methods':reports,
        'process_wall_seconds':time.monotonic()-started,'new_model_forwards':0,'new_training_or_tuning':False,
        'scope':'Post hoc mechanism diagnosis on saved development masks; descriptive correlated rows, not preregistered quality or confirmation'}
    out.mkdir(parents=True);(out/'qk_routing_overlap.py').write_bytes(Path(__file__).read_bytes())
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({n:{k:v for k,v in row.items() if not k.startswith('per_head')} for n,row in reports.items()},indent=2))


if __name__=='__main__':main()
