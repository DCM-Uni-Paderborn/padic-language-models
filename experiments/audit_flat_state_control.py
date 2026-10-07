"""Separate NumPy/Python verification of the immutable state-capped control.

No production modules are imported. The fit, hard selectors, teacher readouts,
energy aggregation and GQA unions are reconstructed from declared inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_completed(p):
    c=json.loads((p/'completed.json').read_text())
    assert digest(p/'completed.json')==(p/'completed.sha256').read_text().strip()
    for f,h in c['artifact_sha256'].items():
        assert digest(p/f)==h and (p/f).stat().st_size==c['artifact_bytes'][f],f
    return c


def independent_fit(points, head):
    points=points.copy();points[points==0]=0
    unique,counts=np.unique(points,axis=0,return_counts=True)
    rng=np.random.default_rng(np.random.SeedSequence([17,head,4409]))
    centers=unique[np.sort(rng.choice(len(unique),13,replace=False))].copy()
    for iteration in range(1,21):
        difference=unique[:,None,:]-centers[None,:,:]
        labels=np.argmin(np.einsum('nkd,nkd->nk',difference,difference),axis=1)
        means=[]
        for index in range(len(centers)):
            group=labels==index
            if np.any(group):
                means.append(np.average(unique[group],axis=0,weights=counts[group]))
        updated=np.unique(np.asarray(means),axis=0)
        if np.array_equal(updated,centers):
            return updated,iteration
        centers=updated
    return centers,20


def run(args):
    root=Path(args.input).resolve();parent=Path(args.parent).resolve();trace_path=Path(args.trace).resolve()
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('fresh audit output required')
    c=load_completed(root);pc=load_completed(parent)
    assert c['status']=='state_capped_flat_completed'
    assert c['parent_completion_sha256']==digest(parent/'completed.json')
    assert c['trace_sha256']==digest(trace_path)
    result=json.loads((root/'results.json').read_text());config=result['config']
    assert digest(root/'prospective_plan.txt')==config['plan_sha256']
    for n,h in config['source_sha256'].items():assert digest(root/'sources'/n)==h
    with np.load(trace_path,allow_pickle=False) as t:
        q,k,v=(t[n].astype(np.float64) for n in ('queries','keys','values'))
    with np.load(root/'encoder_state.npz',allow_pickle=False) as t:state={n:t[n] for n in t.files}
    with np.load(root/'key_codes.npz',allow_pickle=False) as t:codes=t['key_codes'];lengths=t['valid_lengths']
    with np.load(root/'rawstats.npz',allow_pickle=False) as t:raw={n:t[n] for n in t.files}
    with np.load(parent/'pca-17/rawstats.npz',allow_pickle=False) as t:old={n:t[n] for n in t.files}
    with np.load(parent/'pca-17/encoder_state.npz',allow_pickle=False) as t:
        assert state['center'].tobytes()==t['p2_center'].tobytes()
        assert state['projection'].tobytes()==t['p2_projection'].tobytes()
        binary_bytes=sum(t[n].nbytes for n in t.files if n.startswith('p2_'))
    assert config['max_centroids']==13 and config['seed']==17 and config['max_lloyd_iterations']==20
    assert config['calibration_chunks']==[0,1] and config['evaluation_chunks']==list(range(2,32))
    assert sum(a.nbytes for a in state.values())==result['memory']['numeric_state_bytes']<=binary_bytes==16004
    assert codes.dtype==np.uint8 and codes.nbytes==36864
    for name,a in state.items():assert a.nbytes==result['memory']['state_array_bytes'][name]
    assert np.all(state['centroid_counts']<=13) and np.all(state['iterations']<=20)
    fitted_max=0.;encoded_mismatches=0
    projected=np.empty((*q.shape[:3],2))
    for head in range(9):
        projected[:,head]=(q[:,head]-state['center'][head])@state['projection'][head]
        projected_k=(k[:,head]-state['center'][head])@state['projection'][head]
        calibration=np.concatenate([projected_k[chunk,:int(lengths[chunk])] for chunk in (0,1)])
        fitted,iteration=independent_fit(calibration,head)
        active=state['centroids'][head,:int(state['centroid_counts'][head])]
        assert fitted.shape==active.shape and iteration==int(state['iterations'][head])
        fitted_max=max(fitted_max,float(np.max(np.abs(fitted-active))))
        assert np.allclose(fitted,active,rtol=0,atol=2e-13)
        difference=projected_k[:,:,None,:]-active[None,None,:,:]
        expected=np.argmin(np.einsum('csnd,csnd->csn',difference,difference),axis=-1)
        encoded_mismatches+=int(np.count_nonzero(expected!=codes[:,head]))
    assert encoded_mismatches==0
    methods=raw['methods'].tolist();assert methods==['recency','padic_2','trie_2','flat_euclidean','flat_dot_product']
    ids=raw['identifiers'];selection=raw['selected_indices']
    expected_ids={(chunk,h,pos) for chunk in range(2,32) for h in range(9) for pos in range(int(lengths[chunk]))}
    idmap={tuple(map(int,row)):i for i,row in enumerate(ids)}
    assert len(idmap)==len(ids) and set(idmap)==expected_ids
    assert np.array_equal(ids,old['identifiers'])
    for i,method in enumerate(methods[:3]):
        oi=old['methods'].tolist().index(method)
        for name in ('selected_indices','kept_attention_mass','output_error_squared'):
            assert np.array_equal(raw[name][i],old[name][oi])
    assert np.array_equal(raw['output_reference_squared'],old['output_reference_squared'])
    assert np.array_equal(selection[1],selection[2])
    max_mass=max_error=max_reference=0.;mask_checks=0
    for chunk in range(2,32):
        for head in range(9):
            length=int(lengths[chunk]);scores=q[chunk,head,:length]@k[chunk,head,:length].T/8
            for pos in range(length):
                row=idmap[(chunk,head,pos)];count=min(pos+1,32);r=min(8,count)
                probabilities=np.exp(scores[pos,:pos+1]-max(scores[pos,:pos+1]));probabilities/=sum(probabilities)
                reference=np.einsum('n,nd->d',probabilities,v[chunk,head,:pos+1])
                max_reference=max(max_reference,abs(float(reference@reference)-raw['output_reference_squared'][row]))
                for method in range(5):
                    padded=selection[method,row];selected=padded[padded!=-1].astype(int).tolist()
                    assert len(selected)==count and sorted(set(selected))==selected and min(selected)>=0 and max(selected)<=pos
                    assert selected[-r:]==list(range(pos+1-r,pos+1))
                    assert len(padded)-len(selected)==32-count and np.all(padded[count:]==-1)
                    if method>=3:
                        active=state['centroids'][head,:int(state['centroid_counts'][head])]
                        query=projected[chunk,head,pos]
                        if method==3:
                            distance=active-query
                            centroid_scores=-np.einsum('nd,nd->n',distance,distance)
                        else:
                            # Direct uncentered projected query; original-origin centroids.
                            query=q[chunk,head,pos]@state['projection'][head]
                            offset=state['center'][head]@state['projection'][head]
                            centroid_scores=(active+offset)@query
                        mandatory=list(range(pos+1-r,pos+1))
                        optional=sorted(range(pos+1-r),key=lambda x:(float(centroid_scores[int(codes[chunk,head,x])]),x),reverse=True)
                        expected=sorted(mandatory+optional[:count-r]);assert selected==expected,(chunk,head,pos,method)
                    mass=sum(probabilities[selected]);weight=probabilities[selected]/mass
                    value=np.einsum('n,nd->d',weight,v[chunk,head,selected]);delta=value-reference
                    max_mass=max(max_mass,abs(float(mass)-raw['kept_attention_mass'][method,row]))
                    max_error=max(max_error,abs(float(delta@delta)-raw['output_error_squared'][method,row]))
                    mask_checks+=1
    assert max_mass<2e-13 and max_error<2e-13 and max_reference<2e-13
    summary_checks=0
    for s in result['summaries']:
        method=methods.index(s['method']);mask=np.ones(len(ids),dtype=bool) if s['stratum']=='all_queries' else ids[:,2]>=32
        for g in [s['all_heads'],*s['per_head']]:
            selected=mask if g['head'] is None else mask&(ids[:,1]==g['head'])
            numerator=float(raw['output_error_squared'][method,selected].sum());denominator=float(raw['output_reference_squared'][selected].sum())
            assert g['queries']==int(selected.sum())
            assert g['error_squared_sum']==numerator and g['reference_squared_sum']==denominator
            assert g['value_output_nrmse']==float(np.sqrt(numerator/denominator))
            assert g['kept_mass_mean']==float(raw['kept_attention_mass'][method,selected].mean())
            expected_agreement=np.all(selection[method,selected]==selection[0,selected],axis=1)
            assert g['recency_exact_agreement_fraction']==float(expected_agreement.mean())
            summary_checks+=1
    union_rows=raw['union_identifiers'];unions=raw['union_counts'];union_checks=0
    for row,(chunk,kv,pos) in enumerate(union_rows):
        bounds=min(int(pos)+1,8+3*(32-8)) if pos>=32 else int(pos)+1
        assert raw['shared_recent_union_bounds'][row]==bounds
        for method in range(5):
            group=set()
            for head in range(int(kv)*3,int(kv)*3+3):
                group.update(x for x in selection[method,idmap[(int(chunk),head,int(pos))]].tolist() if x!=-1)
            assert unions[method,row]==len(group)<=bounds
            union_checks+=1
    for s in result['logical_access_summaries']:
        index=methods.index(s['method']);mask=np.ones(len(union_rows),dtype=bool) if s['stratum']=='all_queries' else union_rows[:,2]>=32
        values=unions[index,mask]
        assert s['selected_position_union_mean']==float(values.mean())
        assert s['selected_position_union_median']==float(np.median(values))
        assert s['selected_position_union_max']==int(values.max())
    assert load_completed(root)==c and load_completed(parent)==pc
    report={'status':'state_capped_flat_audit_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'input_completion_sha256':digest(root/'completed.json'),'audit_source_sha256':digest(__file__),
        'python':sys.version,'numpy':np.__version__,'mask_checks':mask_checks,'union_checks':union_checks,
        'summary_head_stratum_checks':summary_checks,'key_code_mismatches':encoded_mismatches,
        'maximum_centroid_fit_absolute_difference':fitted_max,'maximum_mass_absolute_difference':max_mass,
        'maximum_squared_error_absolute_difference':max_error,'maximum_reference_energy_absolute_difference':max_reference,
        'scope':'Separate algorithm verification by the same operator; not scientific confirmation or external peer review'}
    output.mkdir(parents=True);(output/'audit_flat_state_control.py').write_bytes(Path(__file__).read_bytes())
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','parent','trace','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
