"""Fixed learned-bridge hard-selection development; dense BF16 emulation only."""
import argparse
from datetime import datetime,timezone
import hashlib,json,importlib.metadata,time
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from padic_lm import development,dense_masks,learned,routing,flat_routing
from padic_lm.development import diagnostic_indices,baseline_mask,angular_mask,group_unions,paired_quality
from padic_lm.dense_masks import padic_mask,flat_mask,ranked_mask
from padic_lm.learned import folded_latents
from padic_lm.routing import QuantileCodebook,PrefixTree,pack_codes,angular_hyperplanes
from padic_lm.flat_routing import FlatKeyCodebook
import llm_routing
from llm_routing import stock_eager_logits,routed_eager_attention,state_dict_hash,validate_selection_mask

DATA_HASH='033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454'
TRAIN_HASH='f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1'
MODEL_HASH='b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583'
SEEDS=(17,29,43)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as a:return {n:a[n].copy() for n in a.files}
def bf16_bits(tensor):
    if tensor.dtype!=torch.bfloat16:raise ValueError('lossless BF16 recording required')
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


def verify_manifest(directory,expected):
    completion=json.loads((directory/'completed.json').read_text())
    if digest(directory/'completed.json')!=expected or (directory/'completed.sha256').read_text().strip()!=expected:
        raise ValueError('frozen completion differs')
    for n,h in completion['artifact_sha256'].items():
        if digest(directory/n)!=h or (directory/n).stat().st_size!=completion['artifact_bytes'][n]:
            raise ValueError('frozen artifact mismatch: '+n)
    return completion


def book_from_state(state,prefix,seed,prime):
    return QuantileCodebook(state[prefix+'center'],state[prefix+'projection'],state[prefix+'thresholds'],
        tuple(int(d) for d in state[prefix+'digits']),'pca',seed,8192,prime)


def verify_trie(selected,qcodes,kcodes,book,check_time):
    count=0
    for b in range(len(qcodes)):
        for head in range(qcodes.shape[1]):
            check_time();tree=PrefixTree(2,book.digits,prime=book.prime)
            for pos in range(qcodes.shape[2]):
                tree.append(kcodes[b,head,pos])
                truth=tree.select(qcodes[b,head,pos],128,8)
                if not np.array_equal(np.flatnonzero(selected[b,head,pos]),truth):
                    raise AssertionError('same-code finite/trie mismatch')
                count+=1
    return count


class FixedMaskAttention(torch.nn.Module):
    """Replay a causal mask; require unchanged original first-layer states."""
    def __init__(self,original,selected,reference):
        super().__init__();self.original=original;self.selected=selected;self.reference=reference;self.last=None
        if original.config._attn_implementation!='eager':raise ValueError('native eager interface required')
        validate_selection_mask(selected,min(selected.shape[-1],int(selected[0,0,-1].sum())))

    def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,**kwargs):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        if self.training or self.original.training or past_key_values is not None or kwargs.get('past_key_value') is not None:
            raise ValueError('frozen uncached evaluation required')
        shape=(*hidden_states.shape[:-1],-1,self.original.head_dim)
        q=self.original.q_proj(hidden_states).view(shape).transpose(1,2)
        k=self.original.k_proj(hidden_states).view(shape).transpose(1,2)
        v=self.original.v_proj(hidden_states).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        for n,t in (('q',q),('k',k),('v',v)):
            if not torch.equal(t,self.reference[n]):raise AssertionError('mask replay input differs: '+n)
        logits=stock_eager_logits(q,k,attention_mask,scaling=self.original.scaling,groups=self.original.num_key_value_groups)
        if not torch.equal(logits,self.reference['scores']):raise AssertionError('native scores changed during replay')
        selected=torch.from_numpy(self.selected).to(q.device)
        value,probability,mass=routed_eager_attention(q,k,v,selected,attention_mask,scaling=self.original.scaling,
            groups=self.original.num_key_value_groups,precomputed_logits=logits)
        projected=self.original.o_proj(value.reshape(*hidden_states.shape[:-1],-1).contiguous())
        self.last={'projected':projected,'values':value.transpose(1,2),'mass':mass}
        return projected,probability


def baseline_forward(model,batch):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    attention=model.model.layers[0].self_attn;record={}
    def before(module,args,kwargs):
        hidden=kwargs['hidden_states'];shape=(*hidden.shape[:-1],-1,module.head_dim)
        q=module.q_proj(hidden).view(shape).transpose(1,2)
        k=module.k_proj(hidden).view(shape).transpose(1,2)
        v=module.v_proj(hidden).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
        record.update(q=q,k=k,v=v,scores=stock_eager_logits(q,k,kwargs['attention_mask'],
            scaling=module.scaling,groups=module.num_key_value_groups))
    def pre_projection(module,args):record['values']=args[0].view(1,batch.shape[1],15,64).transpose(1,2)
    def after(module,args,result):record['projected']=result[0]
    hooks=[attention.register_forward_pre_hook(before,with_kwargs=True),
           attention.o_proj.register_forward_pre_hook(pre_projection),attention.register_forward_hook(after)]
    try:
        logits=model(batch,use_cache=False).logits
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1),reduction='none')
        prediction=logits[:,:-1].argmax(-1)[0]
    finally:
        for hook in hooks:hook.remove()
    return record,loss.cpu().numpy(),prediction.cpu().numpy()


def replay(model,original,batch,selected,reference):
    wrapper=FixedMaskAttention(original,selected,reference).eval()
    model.model.layers[0].self_attn=wrapper
    try:
        logits=model(batch,use_cache=False).logits
        losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1),reduction='none')
        predictions=logits[:,:-1].argmax(-1)[0]
        record=wrapper.last
    finally:model.model.layers[0].self_attn=original
    return record,losses.cpu().numpy(),predictions.cpu().numpy()


def native_metrics(record,reference):
    pe=(record['projected'].double()-reference['projected'].double()).square().sum(-1)[0].cpu().numpy()
    energy=reference['projected'].double().square().sum(-1)[0].cpu().numpy()
    ve=(record['values'].double()-reference['values'].double()).square().sum(-1)[0].cpu().numpy()
    value_energy=reference['values'].double().square().sum(-1)[0].cpu().numpy()
    return pe,energy,ve,value_energy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('data','training','training-audit','supplement','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();data=args.data.resolve();training=args.training.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('fresh development output required')
    started=time.monotonic();cpu_started=time.process_time()
    train_completion=verify_manifest(training,TRAIN_HASH);verify_manifest(data,DATA_HASH)
    metadata=json.loads((data/'manifest.json').read_text());audit=json.loads(args.training_audit.read_text())
    if audit['status']!='qk_training_audit_passed' or audit['training_completion_sha256']!=TRAIN_HASH:
        raise ValueError('requires completed training verification')
    prior_seconds=train_completion['gpu_enabled_process_wall_seconds']+audit['gpu_enabled_audit_process_wall_seconds']
    def check_time():
        if prior_seconds+time.monotonic()-started>=3600:raise TimeoutError('cumulative capture/train/evaluation ceiling reached')
    windows=metadata['windows']['development'];diagnostic=diagnostic_indices(windows)
    if metadata['development_scored_targets']!=32704 or metadata['test_split_read']:raise ValueError('fixed data accounting differs')
    tokens=torch.tensor(arrays(data/'tokens.npz')['development'],dtype=torch.int64)
    if tokens.shape!=(64,512):raise ValueError('fixed development windows required')
    frozen={}
    for seed in SEEDS:
        directory=training/f'seed-{seed}';state=arrays(directory/'routing_state.npz')
        fields={n[len('flat_'):]:a for n,a in state.items() if n.startswith('flat_')}
        frozen[seed]={'final':arrays(directory/'final_encoder.npz'),'initial':arrays(directory/'initial_encoder.npz'),
            'p2':book_from_state(state,'p2_',seed,2),'p3':book_from_state(state,'p3_',seed,3),
            'initial_book':book_from_state(arrays(directory/'initial_routing_state.npz'),'',seed,2),
            'flat':FlatKeyCodebook(**fields),'numeric_state':json.loads((directory/'results.json').read_text())['standalone_numeric_state_bytes']}
    source_paths=[Path(__file__),Path(development.__file__),Path(dense_masks.__file__),Path(learned.__file__),
        Path(routing.__file__),Path(flat_routing.__file__),Path(llm_routing.__file__)]
    sources={p.name:p.read_bytes() for p in source_paths}
    supplement=args.supplement.read_bytes();main_protocol=(training/'prospective_protocol.txt').read_bytes()
    output.mkdir(parents=True);(output/'sources').mkdir();(output/'windows').mkdir()
    for n,b in sources.items():(output/'sources'/n).write_bytes(b)
    (output/'prospective_protocol.txt').write_bytes(main_protocol);(output/'development_supplement.txt').write_bytes(supplement)
    np.savez_compressed(output/'diagnostic_indices.npz',rows=diagnostic)
    planes=angular_hyperplanes(15,64,8,17)
    np.savez_compressed(output/'angular_state.npz',planes=planes,seed_bits=np.array([17,8],dtype=np.uint64))
    np.savez_compressed(output/'gaussian_metadata.npz',temperature=np.array([2],dtype=np.float32))
    names=['full_control','recency','sink_recency','angular']
    for seed in SEEDS:names.extend(f's{seed}_{kind}' for kind in ('initial_p2','p2','p3','flat_euclidean','flat_dot','gaussian'))
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'hard_selection_development',
        'data_completion_sha256':DATA_HASH,'training_completion_sha256':TRAIN_HASH,
        'training_audit_sha256':digest(args.training_audit),'main_protocol_sha256':hashlib.sha256(main_protocol).hexdigest(),
        'supplement_sha256':hashlib.sha256(supplement).hexdigest(),'prior_experiment_stage_seconds':prior_seconds,
        'cumulative_experiment_wall_limit_seconds':3600,'model':metadata['model'],'model_revision':metadata['model_revision'],
        'methods':names,'seeds':list(SEEDS),'layer':0,'length':512,'budget':128,'recent_window':8,
        'query_heads':15,'physical_kv_heads':5,'gqa_group_size':3,'development_articles':58,'windows':windows,
        'diagnostic_indices_sha256':digest(output/'diagnostic_indices.npz'),'target_positions':'0..510; affected128..510',
        'layer_error_positions':'all0..511; affected128..511','bf16_storage':'uint16 bit patterns, lossless',
        'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
        'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')},'command':list(sys.argv),
        'scope':'All real KV retained; CPU masks plus dense GPU readout; descriptive development, no quality/native-speed decision'}
    json_write(output/'config.json',config)
    totals={n:{'losses':[],'projected_error':0.,'projected_energy':0.,'affected_projected_error':0.,'affected_projected_energy':0.,
        'value_error':0.,'value_energy':0.,'affected_value_error':0.,'affected_value_energy':0.,'mass_sum':0.,'affected_mass_sum':0.,
        'unions':[],'diagnostic_value_error':0.,'diagnostic_value_energy':0.,'diagnostic_mass_sum':0.,
        'routing_wall_seconds':0.,'routing_cpu_seconds':0.,'trie_verification_seconds':0.,'trie_rows_verified':0,
        'model_forward_wall_seconds_with_diagnostics':0.} for n in names}
    baseline_losses=[];fixture_reports=[];encoder_wall_seconds=0.
    try:
        check_time();torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA Spark evaluation required')
        model=AutoModelForCausalLM.from_pretrained(metadata['model'],revision=metadata['model_revision'],
            torch_dtype=torch.bfloat16,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
        original=model.model.layers[0].self_attn
        if (model.config.num_attention_heads,model.config.num_key_value_heads,model.config.num_hidden_layers)!=(15,5,32):
            raise ValueError('pinned model configuration differs')
        if state_dict_hash(model)!=MODEL_HASH:raise ValueError('pinned model state differs')
        np.savez_compressed(output/'output_projection.npz',weight_bits=bf16_bits(original.o_proj.weight))
        with torch.inference_mode():
            for index,token_window in enumerate(tokens):
                check_time();batch=token_window[None].to('cuda');reference,ref_loss,ref_prediction=baseline_forward(model,batch)
                from transformers.models.llama.modeling_llama import repeat_kv
                q=reference['q'].float();k=repeat_kv(reference['k'],3).float();v=repeat_kv(reference['v'],3)
                for h in range(15):
                    base=(h//3)*3
                    if not torch.equal(k[:,h],k[:,base]) or not torch.equal(v[:,h],v[:,base]):raise AssertionError('GQA repetition differs')
                qnp,knp=q.cpu().numpy(),k.cpu().numpy();baseline_losses.append(ref_loss)
                saved={'tokens':token_window.numpy(),'reference_target_nll':ref_loss,'reference_predictions':ref_prediction,
                    'reference_queries_bits':bf16_bits(reference['q'])[0],'reference_keys_bits':bf16_bits(repeat_kv(reference['k'],3))[0],
                    'reference_values_bits':bf16_bits(v)[0],'reference_scores_bits':bf16_bits(reference['scores'])[0],
                    'reference_native_values_bits':bf16_bits(reference['values'])[0],
                    'reference_projected_bits':bf16_bits(reference['projected'])[0],
                    'reference_q_fp32_stride':np.array(q.stride(),dtype=np.int64),
                    'reference_k_fp32_stride':np.array(k.stride(),dtype=np.int64)}
                latent={};torch.cuda.synchronize();encoder_start=time.monotonic()
                for seed,state in frozen.items():
                    for stage in ('initial','final'):
                        latent[(seed,stage,'q')]=folded_latents(q,state[stage],'query')
                        latent[(seed,stage,'k')]=folded_latents(k,state[stage],'key')
                        for role in ('q','k'):saved[f's{seed}_{stage}_{role}_latents']=latent[(seed,stage,role)].cpu().numpy()[0]
                torch.cuda.synchronize();encoder_wall_seconds+=time.monotonic()-encoder_start
                diag=diagnostic[diagnostic[:,0]==index];dh,dp=diag[:,1],diag[:,2]
                for name in names:
                    check_time();rw=time.monotonic();rc=time.process_time();codes={};book=None
                    if name in ('full_control','recency','sink_recency'):
                        selected=baseline_mask(512,15,kind='full' if name=='full_control' else name)
                    elif name=='angular':selected,codes=angular_mask(qnp,knp,planes)
                    else:
                        seed=int(name.split('_')[0][1:]);kind=name.split('_',1)[1];state=frozen[seed]
                        stage='initial' if kind=='initial_p2' else 'final'
                        lq,lk=(latent[(seed,stage,role)] for role in ('q','k'));lqnp,lknp=lq.cpu().numpy(),lk.cpu().numpy()
                        if kind in ('initial_p2','p2','p3'):
                            book=state['initial_book' if kind=='initial_p2' else kind]
                            qc,kc=book.encode(lqnp),book.encode(lknp)
                            selected=padic_mask(qc,kc,digits=book.digits,prime=book.prime)
                            codes={'packed_query_codes':pack_codes(qc,book.digits,book.prime),
                                   'packed_key_codes':pack_codes(kc,book.digits,book.prime)}
                        elif kind in ('flat_euclidean','flat_dot'):
                            flat=state['flat'];kc=flat.encode_keys(lknp);pq=flat.project_queries(lqnp)
                            offset=np.einsum('hd,hdr->hr',flat.center,flat.projection)
                            selected=flat_mask(pq,kc,flat.centroids,flat.centroid_counts,
                                score_kind='euclidean' if kind=='flat_euclidean' else 'dot_product',offset=offset)
                            codes={'packed_key_codes':kc[...,None]}
                        else:
                            delta=lq[:,:,:,None]-lk[:,:,None]
                            score=-delta.square().sum(-1)/2
                            selected=ranked_mask(score.cpu().numpy(),128,8)
                            codes={'real_query_latents':lqnp,'real_key_latents':lknp}
                    validate_selection_mask(selected,512 if name=='full_control' else 128)
                    recent=np.arange(512)[None,:]>np.arange(512)[:,None]-8
                    recent&=np.tri(512,dtype=bool)
                    if not np.all(selected[...,recent]):raise AssertionError('mandatory recent keys omitted')
                    record=totals[name];record['routing_wall_seconds']+=time.monotonic()-rw;record['routing_cpu_seconds']+=time.process_time()-rc
                    if name.endswith(('_p2','_p3')) and 'initial' not in name:
                        ts=time.monotonic();record['trie_rows_verified']+=verify_trie(selected,qc,kc,book,check_time)
                        record['trie_verification_seconds']+=time.monotonic()-ts
                    check_time();torch.cuda.synchronize();forward_start=time.monotonic()
                    actual,loss,prediction=replay(model,original,batch,selected,reference)
                    torch.cuda.synchronize();record['model_forward_wall_seconds_with_diagnostics']+=time.monotonic()-forward_start
                    if name=='full_control':
                        if not torch.equal(actual['projected'],reference['projected']) or not torch.equal(actual['values'],reference['values']) or not np.array_equal(loss,ref_loss):
                            raise AssertionError('full-budget model/attention fixture differs from stock')
                    if index==0 and name.endswith(('_p2','_p3')) and 'initial' not in name:
                        # Independently traversed tree rows were checked above; replay the
                        # resulting identical mask through a complete model fixture.
                        tree_mask=np.zeros_like(selected)
                        for head in range(15):
                            tree=PrefixTree(2,book.digits,prime=book.prime)
                            for pos in range(512):
                                tree.append(kc[0,head,pos]);tree_mask[0,head,pos,tree.select(qc[0,head,pos],128,8)]=True
                        alternate,alt_loss,_=replay(model,original,batch,tree_mask,reference)
                        if not torch.equal(alternate['projected'],actual['projected']) or not np.array_equal(alt_loss,loss):
                            raise AssertionError('complete finite/trie model fixture differs')
                        saved[name+'_trie_fixture_target_nll']=alt_loss
                        fixture_reports.append({'method':name,'window':0,'length':512,'projected_and_target_losses_exact':True})
                    pe,energy,ve,value_energy=native_metrics(actual,reference)
                    mass=actual['mass'].cpu().numpy()[0]
                    record['losses'].append(loss);record['projected_error']+=float(pe.sum());record['projected_energy']+=float(energy.sum())
                    record['affected_projected_error']+=float(pe[128:].sum());record['affected_projected_energy']+=float(energy[128:].sum())
                    record['value_error']+=float(ve.sum());record['value_energy']+=float(value_energy.sum())
                    record['affected_value_error']+=float(ve[:,128:].sum());record['affected_value_energy']+=float(value_energy[:,128:].sum())
                    record['mass_sum']+=float(mass.astype(np.float64).sum());record['affected_mass_sum']+=float(mass[:,128:].astype(np.float64).sum())
                    record['diagnostic_value_error']+=float(ve[dh,dp].sum());record['diagnostic_value_energy']+=float(value_energy[dh,dp].sum())
                    record['diagnostic_mass_sum']+=float(mass[dh,dp].astype(np.float64).sum())
                    unions=(np.broadcast_to(np.arange(1,513),(1,5,512)).astype(np.uint16) if name=='full_control' else group_unions(selected))
                    record['unions'].append(unions[0])
                    saved.update({name+'_target_nll':loss,name+'_predictions':prediction,
                        name+'_selection_bits':np.packbits(selected[0],axis=-1,bitorder='little'),
                        name+'_projected_bits':bf16_bits(actual['projected'])[0],name+'_projected_error':pe,
                        name+'_native_value_error':ve,name+'_retained_mass':mass,name+'_gqa_unions':unions[0],
                        name+'_diagnostic_native_values_bits':bf16_bits(actual['values'])[0,dh,dp]})
                    for n,a in codes.items():saved[name+'_'+n]=a[0]
                saved['projected_reference_energy']=energy;saved['native_value_reference_energy']=value_energy
                check_time();np.savez_compressed(output/'windows'/f'window-{index:03d}.npz',**saved)
                json_write(output/'progress.json',{'completed_windows':index+1,'total_windows':64,
                    'cumulative_experiment_stage_seconds':prior_seconds+time.monotonic()-started})
                print(json.dumps({'completed_windows':index+1,'cumulative_seconds':round(prior_seconds+time.monotonic()-started,3)}),flush=True)
        if state_dict_hash(model)!=MODEL_HASH:raise AssertionError('original model state changed')
        baseline_losses=np.stack(baseline_losses);summaries={}
        for name,record in totals.items():
            losses=np.stack(record.pop('losses'));unions=np.stack(record.pop('unions'))
            primary=paired_quality(losses,baseline_losses,windows)
            pe,pa,ve,va=(float(np.sqrt(record[n]/max(record[d],1e-300))) for n,d in
                (('projected_error','projected_energy'),('affected_projected_error','affected_projected_energy'),
                 ('value_error','value_energy'),('affected_value_error','affected_value_energy')))
            if name in ('full_control','recency','sink_recency'):state_bytes=0;key_bytes=0
            elif name=='angular':state_bytes=planes.nbytes+16;key_bytes=15*512
            else:
                seed=int(name.split('_')[0][1:]);kind=name.split('_',1)[1]
                state_bytes=(15612 if kind=='gaussian' else frozen[seed]['numeric_state']['flat' if kind.startswith('flat') else 'p2' if kind=='initial_p2' else kind])
                key_bytes=15*512*(8 if kind=='gaussian' else 1)
            summaries[name]={**primary,**record,'projected_nrmse':pe,'affected_projected_nrmse':pa,
                'native_value_nrmse':ve,'affected_native_value_nrmse':va,
                'mean_retained_mass':record['mass_sum']/(64*15*512),'affected_mean_retained_mass':record['affected_mass_sum']/(64*15*384),
                'diagnostic_native_value_nrmse':float(np.sqrt(record['diagnostic_value_error']/max(record['diagnostic_value_energy'],1e-300))),
                'diagnostic_mean_retained_mass':record['diagnostic_mass_sum']/4096,
                'affected_mean_logical_gqa_union':float(unions[:,:,128:].mean()),'affected_maximum_logical_gqa_union':int(unions[:,:,128:].max()),
                'standalone_numeric_state_bytes':state_bytes,'key_metadata_bytes_per_window':key_bytes,
                'key_metadata_bytes_whole_development':64*key_bytes}
        verify_manifest(training,TRAIN_HASH);verify_manifest(data,DATA_HASH)
        if args.supplement.read_bytes()!=supplement:raise RuntimeError('supplement changed during evaluation')
        for p in source_paths:
            if p.read_bytes()!=sources[p.name]:raise RuntimeError('executed source changed during evaluation')
        reference_quality=paired_quality(baseline_losses,baseline_losses,windows)
        result={'status':'learned_bridge_development_complete','config':config,'reference':reference_quality,'methods':summaries,
            'trie_model_fixtures':fixture_reports,'full_control_exact_windows':64,'original_model_state_unchanged':True,
            'gqa_repeated_keys_values_exact':True,'test_split_read':False,'new_training_or_tuning':False,
            'six_encoder_states_and_latent_transfer_wall_seconds':encoder_wall_seconds,
            'folded_block_head_gemm_calls':64*6*2*15,
            'development_stage_wall_seconds':time.monotonic()-started,'process_cpu_seconds':time.process_time()-cpu_started,
            'cumulative_experiment_stage_seconds':prior_seconds+time.monotonic()-started,
            'dense_emulation_buffer_payload_bytes':{'cpu_selected_mask':15*512*512,'gpu_selected_mask':15*512*512,
                'native_bf16_scores':15*512*512*2,'one_fp32_probability_matrix':15*512*512*4,
                'cpu_float64_ranking_work':15*512*512*8,'cpu_int64_sort_indices':15*512*512*8,
                'original_physical_bf16_kv':5*512*64*2*2},
            'scope':'Descriptive development; all real KV and dense operations retained; no confirmation or native efficiency claim'}
        json_write(output/'results.json',result);check_time()
        artifacts=sorted(p for p in output.rglob('*') if p.is_file())
        completion={'status':result['status'],'training_completion_sha256':TRAIN_HASH,'data_completion_sha256':DATA_HASH,
            'cumulative_experiment_stage_seconds':prior_seconds+time.monotonic()-started,
            'artifact_sha256':{str(p.relative_to(output)):digest(p) for p in artifacts},
            'artifact_bytes':{str(p.relative_to(output)):p.stat().st_size for p in artifacts}}
        check_time();json_write(output/'completed.json',completion);(output/'completed.sha256').write_text(digest(output/'completed.json')+'\n')
        print(json.dumps({'status':result['status'],'reference_ppl':reference_quality['perplexity'],
            'cumulative_seconds':completion['cumulative_experiment_stage_seconds'],
            'paired_nll':{n:r['paired_delta_nll'] for n,r in summaries.items()}},indent=2),flush=True)
    except Exception as error:
        (output/'completed.json').unlink(missing_ok=True);(output/'completed.sha256').unlink(missing_ok=True)
        json_write(output/'failed.json',{'status':'learned_development_failed','type':type(error).__name__,'message':str(error),
            'completed_windows':len(baseline_losses),'cumulative_experiment_stage_seconds':prior_seconds+time.monotonic()-started})
        raise


if __name__=='__main__':main()
