"""Check post hoc coarse-empty helper by signed residue differences on fixed rows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from qk_routing_overlap import coarse_empty


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();p=args.input;out=args.output
    if out.exists():raise FileExistsError('fresh verification output required')
    with np.load(p/'diagnostic_indices.npz',allow_pickle=False) as a:rows=a['rows']
    checked=0
    for index in range(64):
        sample=rows[rows[:,0]==index]
        with np.load(p/'windows'/f'window-{index:03d}.npz',allow_pickle=False) as w:
            for seed in (17,29,43):
                for kind in ('initial_p2','p2','p3'):
                    name=f's{seed}_{kind}';prime=3 if kind=='p3' else 2;digits=3 if prime==3 else 4
                    q,k=w[name+'_packed_query_codes'],w[name+'_packed_key_codes']
                    observed=coarse_empty(q,k,prime,digits)
                    for _,head,pos in sample:
                        jointq=int(q[head,pos,0]);jointk=k[head,:pos-7,0].astype(np.int64)
                        qq=np.array([jointq%prime**digits,jointq//prime**digits],dtype=np.int64)
                        kk=np.stack((jointk%prime**digits,jointk//prime**digits),axis=-1)
                        expected=not bool(np.any(np.all((kk-qq)%prime==0,axis=-1)))
                        assert bool(observed[head,pos-128])==expected
                        checked+=1
    assert checked==4096*9
    helper=Path(__file__).with_name('qk_routing_overlap.py')
    report={'status':'sampled_coarse_ball_verification_passed','diagnostic_rows':4096,'prime_seed_state_rows_checked':checked,
        'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Independent signed-congruence formula checked against production helper on fixed4096 rows; imports that helper; no model forwarding or full-corpus second coarse-count reconstruction'}
    out.mkdir(parents=True);(out/'verify_qk_coarse_sample.py').write_bytes(Path(__file__).read_bytes())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
