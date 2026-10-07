"""Separate verification of frozen training states, targets and surrogate errors.

No production module is imported. This audits fitting, not held-out quality.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.input.resolve();out=args.output.resolve();started=time.monotonic()
    if out.exists():raise FileExistsError('fresh audit output required')
    c=json.loads((p/'completed.json').read_text());r=json.loads((p/'results.json').read_text())
    assert digest(p/'completed.json')==(p/'completed.sha256').read_text().strip()
    assert c['status']=='learned_encoders_frozen_before_development'
    for n,h in c['artifact_sha256'].items():assert digest(p/n)==h and (p/n).stat().st_size==c['artifact_bytes'][n]
    assert r['development_model_outputs_read'] is False and r['test_split_read'] is False
    assert r['full_budget_512_fixture_stock_exact'] and r['original_model_state_unchanged']
    assert r['model_loaded_state_sha256']=='b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583'
    assert digest(p/'prospective_protocol.txt')==r['config']['protocol_sha256']
    with np.load(p/'training_trace.npz',allow_pickle=False) as t:trace={n:t[n] for n in t.files}
    for n in ('queries','keys','values','native_values'):assert trace[n].shape==(16,15,512,64) and trace[n].dtype==np.float32
    torch.backends.cuda.matmul.allow_tf32=False;native_score_checks=native_value_checks=0
    with torch.inference_mode():
        for b in range(16):
            q,k,v=[torch.tensor(trace[n][b],device='cuda',dtype=torch.bfloat16) for n in ('queries','keys','values')]
            scores=torch.matmul(q,k.transpose(-1,-2))/8
            future=torch.ones(512,512,device='cuda',dtype=torch.bool).triu(1)
            scores=scores+torch.zeros_like(scores).masked_fill(future,torch.finfo(torch.bfloat16).min)
            assert np.array_equal(scores.float().cpu().numpy(),trace['native_scores'][b])
            weights=torch.softmax(scores,dim=-1,dtype=torch.float32).to(torch.bfloat16)
            value=weights@v
            assert np.array_equal(value.float().cpu().numpy(),trace['native_values'][b])
            native_score_checks+=15*512*512;native_value_checks+=15*512*64
    for h in range(15):
        base=(h//3)*3
        for n in ('keys','values'):assert trace[n][:,h].tobytes()==trace[n][:,base].tobytes()
    seed_reports=[];fold_max=0.;loss_difference_max=0.
    for sr in r['seed_results']:
        seed=sr['seed'];directory=p/f'seed-{seed}'
        with np.load(directory/'initial_training_state.npz') as t:initial={n:t[n] for n in t.files}
        with np.load(directory/'final_training_state.npz') as t:final={n:t[n] for n in t.files}
        with np.load(directory/'training_history.npz') as t:history={n:t[n] for n in t.files}
        assert history['loss'].shape==(200,) and np.all(np.isfinite(history['loss']))
        rng=np.random.default_rng(np.random.SeedSequence([seed,19001]))
        for step in range(200):
            blocks=rng.choice(16,2,replace=False)
            positions=np.stack([rng.choice(np.arange(128,512),32,replace=False) for _ in blocks])
            assert np.array_equal(history['windows'][step],blocks) and np.array_equal(history['queries'][step],positions)
        for h in range(15):
            rng=np.random.default_rng(np.random.SeedSequence([seed,h,7603]))
            q,rr=np.linalg.qr(rng.normal(size=(64,2)));expected=(q*np.where(np.diag(rr)<0,-1,1)).astype(np.float32)
            assert np.array_equal(initial['query_weight'][h],expected) and np.array_equal(initial['key_weight'][h],expected)
        for role,name in (('query','queries'),('key','keys')):
            source=trace[name].astype(np.float64);mean=source.mean(axis=(0,2));std=source.std(axis=(0,2)).clip(1e-4)
            assert np.allclose(initial[role+'_mean'],mean,rtol=2e-6,atol=2e-6)
            assert np.allclose(initial[role+'_std'],std,rtol=2e-6,atol=2e-6)
            assert np.array_equal(initial[role+'_mean'],final[role+'_mean']) and np.array_equal(initial[role+'_std'],final[role+'_std'])
            assert np.count_nonzero(initial[role+'_bias'])==0
        errors={}
        for label,state in (('initial',initial),('final',final)):
            with np.load(directory/f'{label}_encoder.npz') as t:deployment={n:t[n] for n in t.files}
            assert sum(a.nbytes for a in deployment.values())==15608
            for role in ('query','key'):
                w=state[role+'_weight'].astype(np.float64)/state[role+'_std'][...,None]
                bias=state[role+'_bias'].astype(np.float64)-np.einsum('hd,hdr->hr',state[role+'_mean'].astype(np.float64),w)
                assert np.allclose(w,deployment[role+'_weight'],rtol=2e-6,atol=2e-6)
                assert np.allclose(bias,deployment[role+'_bias'],rtol=2e-6,atol=3e-6)
            numerator=denominator=0.
            with torch.inference_mode():
                for b in range(16):
                    latents={}
                    for role,name in (('query','queries'),('key','keys')):
                        block=torch.tensor(trace[name][b],device='cuda')
                        w=torch.tensor(deployment[role+'_weight'],device='cuda');bias=torch.tensor(deployment[role+'_bias'],device='cuda')
                        latents[role]=torch.stack([F.linear(block[h],w[h].T,bias[h]) for h in range(15)])
                    # Independent quadratic-distance expression, not direct differences.
                    query=latents['query'][:,128:];key=latents['key']
                    distances=query.square().sum(-1)[...,None]+key.square().sum(-1)[:,None,:]-2*(query@key.transpose(-1,-2))
                    logits=-distances/2
                    allowed=torch.arange(512,device='cuda')[None,None]<=torch.arange(128,512,device='cuda')[None,:,None]
                    weights=torch.softmax(logits.masked_fill(~allowed,-torch.inf),dim=-1,dtype=torch.float32)
                    values=torch.tensor(trace['values'][b],device='cuda');truth=torch.tensor(trace['native_values'][b,:,128:],device='cuda')
                    predicted=weights@values
                    numerator+=float((predicted.double()-truth.double()).square().sum());denominator+=float(truth.double().square().sum())
            error=float(np.sqrt(numerator/denominator));expected_error=sr[label+'_training_error']['value_output_nrmse']
            loss_difference_max=max(loss_difference_max,abs(error-expected_error));assert abs(error-expected_error)<2e-6
            errors[label]=error
        assert errors['final']<errors['initial']
        with np.load(directory/'routing_state.npz') as t:states={n:t[n] for n in t.files}
        assert np.array_equal(states['p2_center'],states['p3_center']) and np.array_equal(states['p2_projection'],states['p3_projection'])
        assert np.array_equal(states['flat_center'],states['p2_center']) and np.array_equal(states['flat_projection'],states['p2_projection'])
        for prefix,key in (('p2','p2'),('p3','p3'),('flat','flat')):
            count=15608+sum(a.nbytes for n,a in states.items() if n.startswith(prefix+'_'))
            assert count==sr['standalone_numeric_state_bytes'][key]
        assert sr['standalone_numeric_state_bytes']['flat']<=sr['standalone_numeric_state_bytes']['p2']
        with np.load(directory/'training_latents.npz') as t:latq=t['queries'];latk=t['keys']
        for h in range(15):
            pooled=np.concatenate([latq[:,h].reshape(-1,2),latk[:,h].reshape(-1,2)]).astype(np.float64)
            center=states['p2_center'][h];projection=states['p2_projection'][h]
            assert np.allclose(center,pooled.mean(0),rtol=0,atol=1e-13)
            assert np.allclose(projection.T@projection,np.eye(2),rtol=0,atol=1e-13)
            projected=(pooled-center)@projection
            for prime,prefix in ((2,'p2'),(3,'p3')):
                for coordinate,digits in enumerate(states[prefix+'_digits']):
                    alphabet=prime**int(digits);cuts=np.quantile(projected[:,coordinate],np.arange(1,alphabet)/alphabet)
                    assert np.allclose(cuts,states[prefix+'_thresholds'][h,coordinate,:alphabet-1],rtol=0,atol=1e-13)
        seed_reports.append({'seed':seed,'independent_training_nrmse':errors,'schedule_verified_steps':200,
            'numeric_state_bytes':sr['standalone_numeric_state_bytes']})
    for n,h in c['artifact_sha256'].items():assert digest(p/n)==h
    report={'status':'qk_training_audit_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'training_completion_sha256':digest(p/'completed.json'),'source_sha256':digest(__file__),
        'native_score_elements_exact':native_score_checks,'native_value_elements_exact':native_value_checks,
        'maximum_independent_nrmse_absolute_difference':loss_difference_max,'seed_reports':seed_reports,
        'gpu_enabled_audit_process_wall_seconds':time.monotonic()-started,
        'scope':'Separate-algorithm verification by same operator; all losses are training-only; no development model outputs or confirmation'}
    out.mkdir(parents=True);(out/'audit_qk_training.py').write_bytes(Path(__file__).read_bytes())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
