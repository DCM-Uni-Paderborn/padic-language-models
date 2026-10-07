"""Separate mask/native-readout/loss accounting audit; no production imports."""
import argparse
from datetime import datetime,timezone
import hashlib,json,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

AUDIT_CONTEXT={}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as a:return {n:a[n] for n in a.files}
def float_bits(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
def native(bits):return torch.tensor(bits,device='cuda',dtype=torch.uint16).view(torch.bfloat16)
def record_bits(t):return t.contiguous().view(torch.uint16).cpu().numpy()


def rank_mask(scores,full=False):
    h,q,k=scores.shape
    position=np.arange(q)
    causal=position[None,:]<=position[:,None]
    if full:return np.broadcast_to(causal,(h,q,k)).copy()
    work=scores.astype(np.float64).copy()
    work[:,~causal]=-np.inf
    mandatory=causal&(position[None,:]>position[:,None]-8)
    work[:,mandatory]=np.inf
    keys=np.broadcast_to(-position,(h,q,k))
    # Two independent lexicographic sort keys, in descending-score/newer-ID order.
    order=np.lexsort((keys,-work),axis=-1)[...,:128]
    mask=np.zeros((h,q,k),dtype=bool)
    np.put_along_axis(mask,order,True,axis=-1)
    return mask&causal


def quantized(latents,state,prefix,prime):
    digits=state[prefix+'digits'];result=np.empty((15,512,2),dtype=np.uint16)
    for head in range(15):
        projected=(latents[head].astype(np.float64)-state[prefix+'center'][head])@state[prefix+'projection'][head]
        for coordinate,d in enumerate(digits):
            rank=np.searchsorted(state[prefix+'thresholds'][head,coordinate,:prime**int(d)-1],projected[:,coordinate],side='right')
            reversed_rank=np.zeros_like(rank)
            for _ in range(int(d)):
                reversed_rank=reversed_rank*prime+rank%prime;rank=rank//prime
            result[head,:,coordinate]=reversed_rank
    packed=(result[...,0]+result[...,1]*prime**int(digits[0])).astype(np.uint8)[...,None]
    return result,packed


def finite_scores(q,k,digits,prime):
    scores=np.zeros((15,512,512),dtype=np.uint8)
    for level in range(1,max(int(d) for d in digits)+1):
        common=np.ones(scores.shape,dtype=bool)
        for coordinate,d in enumerate(digits):
            modulus=prime**min(level,int(d))
            common&=(q[...,coordinate,None]%modulus)==(k[...,None,:,coordinate]%modulus)
        scores+=common
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('input','training','data','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();p=args.input.resolve();training=args.training.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('fresh audit directory required')
    started=time.monotonic();out.mkdir(parents=True)
    source=Path(__file__).read_bytes();(out/'audit_qk_development.py').write_bytes(source)
    AUDIT_CONTEXT.update(output=out,started=started,prior=0.)
    completion=json.loads((p/'completed.json').read_text());r=json.loads((p/'results.json').read_text());config=r['config']
    assert digest(p/'completed.json')==(p/'completed.sha256').read_text().strip()
    assert completion['status']=='learned_bridge_development_complete'
    for n,h in completion['artifact_sha256'].items():assert digest(p/n)==h and (p/n).stat().st_size==completion['artifact_bytes'][n]
    assert config['data_completion_sha256']==digest(args.data/'completed.json')=='033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454'
    assert config['training_completion_sha256']==digest(training/'completed.json')=='f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1'
    assert config['seeds']==[17,29,43] and config['budget']==128 and config['recent_window']==8
    expected=['full_control','recency','sink_recency','angular']
    for seed in (17,29,43):expected.extend(f's{seed}_{kind}' for kind in ('initial_p2','p2','p3','flat_euclidean','flat_dot','gaussian'))
    assert config['methods']==expected and not r['test_split_read'] and not r['new_training_or_tuning']
    assert digest(p/'development_supplement.txt')==config['supplement_sha256']
    assert digest(p/'prospective_protocol.txt')==config['main_protocol_sha256']
    for n,h in config['source_sha256'].items():assert digest(p/'sources'/n)==h
    prior=completion['cumulative_experiment_stage_seconds']
    AUDIT_CONTEXT['prior']=prior
    def check_time():
        if prior+time.monotonic()-started>=3600:raise TimeoutError('cumulative development/audit ceiling reached')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    indices=load(p/'diagnostic_indices.npz')['rows'];metadata=json.loads((args.data/'manifest.json').read_text());windows=metadata['windows']['development']
    ids=load(args.data/'tokens.npz')['development'];assert ids.shape==(64,512)
    assert config['windows']==windows and indices.shape==(4096,3) and len(np.unique(indices,axis=0))==4096
    article_ids=list(dict.fromkeys(w['article_id'] for w in windows));assert len(article_ids)==58
    # Independent structural check of the saved frozen diagnostic, without using
    # the production sampling function: every cell covers all four quarters.
    cell_counts=[]
    for article in article_ids:
        blocks=[i for i,w in enumerate(windows) if w['article_id']==article]
        for head in range(15):
            cell=indices[np.isin(indices[:,0],blocks)&(indices[:,1]==head)]
            assert len(cell) in (4,5) and set((cell[:,2]-128)//96)==set(range(4));cell_counts.append(len(cell))
    assert cell_counts.count(5)==616 and set(indices[:,0])==set(range(64))
    states={seed:load(training/f'seed-{seed}'/'routing_state.npz') for seed in (17,29,43)}
    initial_states={seed:load(training/f'seed-{seed}'/'initial_routing_state.npz') for seed in (17,29,43)}
    deployments={(seed,stage):load(training/f'seed-{seed}'/f'{stage}_encoder.npz') for seed in (17,29,43) for stage in ('initial','final')}
    planes=load(p/'angular_state.npz')['planes'];assert planes.shape==(15,64,8)
    weight=native(load(p/'output_projection.npz')['weight_bits'])
    totals={n:{'losses':[],'pe':[],'ve':[],'energy':[],'value_energy':[],'mass':[],'unions':[],
        'diagnostic_error':0.,'diagnostic_energy':0.,'diagnostic_mass':0.} for n in expected}
    ref_losses=[];mask_rows=0;native_elements=0;projected_elements=0;latent_elements=0;max_metric_error=0.;first_window=None
    causal=torch.ones(512,512,device='cuda',dtype=torch.bool).tril()
    with torch.inference_mode():
        for index in range(64):
            check_time();w=load(p/'windows'/f'window-{index:03d}.npz')
            assert np.array_equal(w['tokens'],ids[index]);ref_losses.append(w['reference_target_nll'])
            q,k,v=(native(w['reference_'+n+'_bits'])[None] for n in ('queries','keys','values'))
            for head in range(15):
                for name in ('keys','values'):
                    assert np.array_equal(w['reference_'+name+'_bits'][head],w['reference_'+name+'_bits'][head//3*3])
            scores=(q@k.transpose(-1,-2))/8
            scores=scores+torch.zeros_like(scores).masked_fill(~causal,torch.finfo(torch.bfloat16).min)
            assert np.array_equal(record_bits(scores)[0],w['reference_scores_bits'])
            probability=torch.softmax(scores,dim=-1,dtype=torch.float32)
            dense_value=probability.to(torch.bfloat16)@v
            assert np.array_equal(record_bits(dense_value)[0],w['reference_native_values_bits'])
            dense_projected=F.linear(dense_value.transpose(1,2).contiguous().reshape(1,512,960),weight)
            assert np.array_equal(record_bits(dense_projected)[0],w['reference_projected_bits'])
            native_elements+=15*512*64;projected_elements+=512*960
            # Restore the recorded input strides for exact FP32 GEMM arithmetic.
            real={}
            for role,name in (('q','queries'),('k','keys')):
                t=torch.empty_strided((1,15,512,64),tuple(int(a) for a in w[f'reference_{role}_fp32_stride']),device='cuda',dtype=torch.float32)
                t.copy_(native(w['reference_'+name+'_bits'])[None].float());real[role]=t
            for seed in (17,29,43):
                for stage in ('initial','final'):
                    d=deployments[(seed,stage)]
                    for role,label in (('q','query'),('k','key')):
                        mapped=torch.stack([F.linear(real[role][0,h],torch.tensor(d[label+'_weight'][h].T,device='cuda'),
                            torch.tensor(d[label+'_bias'][h],device='cuda')) for h in range(15)])
                        assert np.array_equal(mapped.cpu().numpy(),w[f's{seed}_{stage}_{role}_latents'])
                        latent_elements+=15*512*2
            for name in expected:
                check_time();mask=np.unpackbits(w[name+'_selection_bits'],axis=-1,bitorder='little').astype(bool)
                assert mask.shape==(15,512,512) and not np.any(np.triu(mask,1))
                count=512 if name=='full_control' else 128
                assert np.array_equal(mask.sum(-1),np.broadcast_to(np.minimum(np.arange(1,513),count),(15,512)))
                position=np.arange(512);recent=(position[None,:]<=position[:,None])&(position[None,:]>position[:,None]-8)
                assert np.all(mask[:,recent])
                if name in ('full_control','recency','sink_recency'):
                    ranking=np.broadcast_to(position,(15,512,512)).astype(np.float64).copy()
                    if name=='sink_recency':ranking[...,:4]+=512
                    truth=rank_mask(ranking,full=name=='full_control')
                elif name=='angular':
                    qq=np.einsum('hld,hdr->hlr',float_bits(w['reference_queries_bits']).astype(np.float64),planes)>=0
                    kk=np.einsum('hld,hdr->hlr',float_bits(w['reference_keys_bits']).astype(np.float64),planes)>=0
                    assert np.array_equal(np.packbits(qq,axis=-1,bitorder='little'),w[name+'_packed_query_codes'])
                    assert np.array_equal(np.packbits(kk,axis=-1,bitorder='little'),w[name+'_packed_key_codes'])
                    ranking=(qq[:,:,None,:]==kk[:,None,:,:]).sum(-1)
                    truth=rank_mask(ranking)
                else:
                    seed=int(name.split('_')[0][1:]);kind=name.split('_',1)[1];stage='initial' if kind=='initial_p2' else 'final'
                    lq,lk=(w[f's{seed}_{stage}_{role}_latents'] for role in ('q','k'))
                    if kind in ('initial_p2','p2','p3'):
                        prime=3 if kind=='p3' else 2;state=initial_states[seed] if stage=='initial' else states[seed]
                        prefix='' if stage=='initial' else f'p{prime}_'
                        qc,qpacked=quantized(lq,state,prefix,prime);kc,kpacked=quantized(lk,state,prefix,prime)
                        assert np.array_equal(qpacked,w[name+'_packed_query_codes']) and np.array_equal(kpacked,w[name+'_packed_key_codes'])
                        truth=rank_mask(finite_scores(qc,kc,state[prefix+'digits'],prime))
                    elif kind.startswith('flat'):
                        state=states[seed];ranking=np.empty((15,512,512),dtype=np.float64);keycodes=[]
                        for head in range(15):
                            qproj=(lq[head].astype(np.float64)-state['flat_center'][head])@state['flat_projection'][head]
                            kproj=(lk[head].astype(np.float64)-state['flat_center'][head])@state['flat_projection'][head]
                            centers=state['flat_centroids'][head,:int(state['flat_centroid_counts'][head])]
                            codes=np.argmin(((kproj[:,None]-centers[None])**2).sum(-1),axis=-1);keycodes.append(codes)
                            if kind=='flat_euclidean':s=-((qproj[:,None]-centers[None])**2).sum(-1)
                            else:
                                offset=state['flat_center'][head]@state['flat_projection'][head]
                                s=(qproj+offset)@(centers+offset).T
                            ranking[head]=s[:,codes]
                        assert np.array_equal(np.stack(keycodes)[...,None].astype(np.uint8),w[name+'_packed_key_codes'])
                        truth=rank_mask(ranking)
                    else:
                        assert np.array_equal(lq,w[name+'_real_query_latents']) and np.array_equal(lk,w[name+'_real_key_latents'])
                        qlatent,klatent=torch.tensor(lq,device='cuda'),torch.tensor(lk,device='cuda')
                        ranking=-(qlatent[:,:,None]-klatent[:,None]).square().sum(-1)/2
                        truth=rank_mask(ranking.cpu().numpy())
                assert np.array_equal(mask,truth),f'hard mask mismatch {index} {name}'
                mask_rows+=15*512
                selected=torch.tensor(mask,device='cuda')[None]
                weights=torch.softmax(scores.masked_fill(~selected,torch.finfo(torch.bfloat16).min),dim=-1,dtype=torch.float32).to(torch.bfloat16)
                value=weights@v
                projected=F.linear(value.transpose(1,2).contiguous().reshape(1,512,960),weight)
                assert np.array_equal(record_bits(projected)[0],w[name+'_projected_bits'])
                native_elements+=15*512*64;projected_elements+=512*960
                pe=(projected.double()-dense_projected.double()).square().sum(-1)[0].cpu().numpy()
                energy=dense_projected.double().square().sum(-1)[0].cpu().numpy()
                ve=(value.double()-dense_value.double()).square().sum(-1)[0].cpu().numpy()
                value_energy=dense_value.double().square().sum(-1)[0].cpu().numpy()
                mass=(probability*selected).sum(-1)[0].cpu().numpy()
                for actual,recorded in ((pe,w[name+'_projected_error']),(ve,w[name+'_native_value_error']),
                    (energy,w['projected_reference_energy']),(value_energy,w['native_value_reference_energy']),
                    (mass,w[name+'_retained_mass'])):
                    discrepancy=float(np.max(np.abs(actual.astype(np.float64)-recorded.astype(np.float64))))
                    max_metric_error=max(max_metric_error,discrepancy);assert discrepancy<1e-10
                union=np.array([np.logical_or.reduce(mask[h:h+3]).sum(-1) for h in range(0,15,3)],dtype=np.uint16)
                assert np.array_equal(union,w[name+'_gqa_unions'])
                diag=indices[indices[:,0]==index];dh,dp=diag[:,1],diag[:,2]
                assert np.array_equal(record_bits(value)[0,dh,dp],w[name+'_diagnostic_native_values_bits'])
                if name=='full_control':assert np.array_equal(w[name+'_target_nll'],w['reference_target_nll'])
                t=totals[name]
                for key,a in (('losses',w[name+'_target_nll']),('pe',pe),('ve',ve),('energy',energy),('value_energy',value_energy),('mass',mass),('unions',union)):t[key].append(a)
                t['diagnostic_error']+=float(ve[dh,dp].sum());t['diagnostic_energy']+=float(value_energy[dh,dp].sum());t['diagnostic_mass']+=float(mass[dh,dp].astype(np.float64).sum())
            if index==0:first_window=w
            print(json.dumps({'verified_windows':index+1,'cumulative_seconds':round(prior+time.monotonic()-started,3)}),flush=True)
    reference=np.stack(ref_losses).astype(np.float64);summary_checks=0
    for name,t in totals.items():
        losses=np.stack(t['losses']).astype(np.float64);s=r['methods'][name]
        assert abs(losses.mean()-s['mean_nll'])<1e-13 and abs((losses-reference).mean()-s['paired_delta_nll'])<1e-13
        assert s['targets']==32704 and s['affected_targets']==24512
        assert np.allclose(losses.mean(1),s['window_mean_nll'],rtol=0,atol=1e-13)
        for record in s['article_records']:
            blocks=[i for i,w in enumerate(windows) if w['article_id']==record['article_id']]
            assert record['windows']==blocks and record['targets']==len(blocks)*511
            assert abs(losses[blocks].sum()-record['nll_sum'])<1e-10
            assert abs((losses[blocks]-reference[blocks]).mean()-record['paired_delta_nll'])<1e-13
        pe,en,ve,ven,mass,unions=(np.stack(t[key]) for key in ('pe','energy','ve','value_energy','mass','unions'))
        pairs={'projected_nrmse':np.sqrt(pe.sum()/en.sum()),'affected_projected_nrmse':np.sqrt(pe[:,128:].sum()/en[:,128:].sum()),
            'native_value_nrmse':np.sqrt(ve.sum()/ven.sum()),'affected_native_value_nrmse':np.sqrt(ve[:,:,128:].sum()/ven[:,:,128:].sum()),
            'affected_mean_retained_mass':mass[:,:,128:].astype(np.float64).mean(),'affected_mean_logical_gqa_union':unions[:,:,128:].mean(),
            'diagnostic_native_value_nrmse':np.sqrt(t['diagnostic_error']/t['diagnostic_energy']),
            'diagnostic_mean_retained_mass':t['diagnostic_mass']/4096}
        for key,value in pairs.items():assert abs(float(value)-s[key])<1e-12;summary_checks+=1
        assert int(unions[:,:,128:].max())==s['affected_maximum_logical_gqa_union']
    # Independently replay all declared masks through one complete model window.
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
    check_time();model=AutoModelForCausalLM.from_pretrained(config['model'],revision=config['model_revision'],
        torch_dtype=torch.bfloat16,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
    attention=model.model.layers[0].self_attn;old_forward=attention.forward;active={}
    def independent_forward(module,hidden_states,position_embeddings,attention_mask,**kwargs):
        shape=(*hidden_states.shape[:-1],-1,64)
        qq=module.q_proj(hidden_states).view(shape).transpose(1,2)
        kk=module.k_proj(hidden_states).view(shape).transpose(1,2);vv=module.v_proj(hidden_states).view(shape).transpose(1,2)
        qq,kk=apply_rotary_pos_emb(qq,kk,*position_embeddings);kk,vv=repeat_kv(kk,3),repeat_kv(vv,3)
        logits=(qq@kk.transpose(-1,-2))/8+attention_mask
        logits=logits.masked_fill(~active['selected'],torch.finfo(torch.bfloat16).min)
        probability=torch.softmax(logits,-1,dtype=torch.float32).to(torch.bfloat16)
        values=(probability@vv).transpose(1,2).contiguous().reshape(1,512,960)
        return F.linear(values,module.o_proj.weight,module.o_proj.bias),probability
    batch=torch.tensor(ids[0:1],device='cuda');model_checks=0
    with torch.inference_mode():
        logits=model(batch,use_cache=False).logits
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1),reduction='none').cpu().numpy()
        assert np.array_equal(loss,first_window['reference_target_nll']);model_checks+=1
        attention.forward=types.MethodType(independent_forward,attention)
        try:
            for name in expected:
                check_time();active['selected']=torch.tensor(np.unpackbits(first_window[name+'_selection_bits'],axis=-1,bitorder='little').astype(bool),device='cuda')[None]
                logits=model(batch,use_cache=False).logits
                loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1),reduction='none').cpu().numpy()
                assert np.array_equal(loss,first_window[name+'_target_nll']),name;model_checks+=1
        finally:attention.forward=old_forward
    for n,h in completion['artifact_sha256'].items():assert digest(p/n)==h
    report={'status':'qk_development_audit_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'development_completion_sha256':digest(p/'completed.json'),'audit_source_sha256':digest(__file__),
        'windows_verified':64,'hard_mask_query_head_rows_exact':mask_rows,'native_weighted_value_elements_reconstructed':native_elements,
        'projected_output_elements_exact':projected_elements,'folded_latent_elements_exact':latent_elements,
        'maximum_raw_metric_absolute_difference':max_metric_error,'summary_endpoint_checks':summary_checks,
        'complete_model_fixture_windows':1,'complete_model_fixture_methods_including_reference':model_checks,
        'full_budget_loss_identity_windows':64,'gpu_enabled_audit_stage_seconds':time.monotonic()-started,
        'cumulative_experiment_stage_seconds':prior+time.monotonic()-started,
        'scope':'Same-operator separate algorithms; all masks/readouts/loss aggregations checked; complete model replay only on first512-token window; development, not confirmation'}
    assert Path(__file__).read_bytes()==source
    check_time()
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if AUDIT_CONTEXT:
            report={'status':'development_audit_failed','type':type(error).__name__,'message':str(error),
                'gpu_enabled_audit_stage_seconds':time.monotonic()-AUDIT_CONTEXT['started'],
                'cumulative_experiment_stage_seconds':AUDIT_CONTEXT['prior']+time.monotonic()-AUDIT_CONTEXT['started']}
            (AUDIT_CONTEXT['output']/'failed.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
