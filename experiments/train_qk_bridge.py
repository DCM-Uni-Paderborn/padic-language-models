"""Capture only new training states and fit the three declared real bridges.

Development model outputs are not read in this stage. Its completed artifact
freezes all initial/final encoders and calibration-only routing states first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"));sys.path.insert(0, str(ROOT / "experiments"))
from padic_lm import learned, routing, flat_routing
from padic_lm.learned import DualAffine, fit_encoder, folded_latents, gaussian_values
from padic_lm.routing import QuantileCodebook
from padic_lm.flat_routing import FlatKeyCodebook
import llm_routing
from llm_routing import state_dict_hash, stock_eager_logits, routed_eager_attention

DATA_HASH = "033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454"
MODEL_STATE_HASH = "b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583"
SEEDS = (17, 29, 43)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_write(path, value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def verify_data(directory):
    completion=json.loads((directory/'completed.json').read_text())
    actual=digest(directory/'completed.json')
    if actual!=DATA_HASH or (directory/'completed.sha256').read_text().strip()!=actual:
        raise ValueError('requires the corrected article-based frozen dataset')
    for name,value in completion['artifact_sha256'].items():
        if digest(directory/name)!=value or (directory/name).stat().st_size!=completion['artifact_bytes'][name]:
            raise ValueError(f'data artifact mismatch: {name}')
    return json.loads((directory/'manifest.json').read_text())


def capture_training(model, tokens, check_time):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
    attention=model.model.layers[0].self_attn
    records={name:[] for name in ('queries','keys','values','native_scores','native_values')}
    current={};losses=[]
    def before(module,args,kwargs):
        hidden=kwargs['hidden_states'];shape=(*hidden.shape[:-1],-1,module.head_dim)
        q=module.q_proj(hidden).view(shape).transpose(1,2)
        k=module.k_proj(hidden).view(shape).transpose(1,2)
        v=module.v_proj(hidden).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
        logits=stock_eager_logits(q,k,kwargs['attention_mask'],scaling=module.scaling,groups=module.num_key_value_groups)
        current.update(q=q,k=k,v=v,logits=logits,bias=kwargs['attention_mask'])
        for name,value in (('queries',q),('keys',repeat_kv(k,module.num_key_value_groups)),
                           ('values',repeat_kv(v,module.num_key_value_groups)),('native_scores',logits)):
            records[name].append(value[0].float().cpu().numpy())
    def before_projection(module,args):
        tensor=args[0].view(1,512,15,64).transpose(1,2)
        records['native_values'].append(tensor[0].float().cpu().numpy())
    def after(module,args,result):current['projected']=result[0].detach()
    hooks=[attention.register_forward_pre_hook(before,with_kwargs=True),
           attention.o_proj.register_forward_pre_hook(before_projection),attention.register_forward_hook(after)]
    exact=False
    try:
        with torch.inference_mode():
            for index,window in enumerate(tokens):
                check_time();batch=window[None].to('cuda')
                logits=model(batch,use_cache=False).logits
                losses.append(float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1))))
                if index==0:
                    # Calling o_proj would trigger the capture hook again: use F.linear.
                    selected=torch.ones((1,15,512,512),device='cuda',dtype=torch.bool).tril()
                    value,_,_=routed_eager_attention(current['q'],current['k'],current['v'],selected,current['bias'],
                        scaling=attention.scaling,groups=attention.num_key_value_groups,precomputed_logits=current['logits'])
                    before_o=value.reshape(1,512,960)
                    projected=F.linear(before_o,attention.o_proj.weight,attention.o_proj.bias)
                    if not torch.equal(projected,current['projected']):
                        raise AssertionError('complete512-token full-budget fixture is not exactly stock')
                    exact=True
                current.clear()
    finally:
        for hook in hooks:hook.remove()
    arrays={name:np.stack(values) for name,values in records.items()}
    if any(len(values)!=16 for values in records.values()):raise AssertionError('capture/window count differs')
    for kv in range(5):
        for h in (kv*3+1,kv*3+2):
            for name in ('keys','values'):
                if arrays[name][:,h].tobytes()!=arrays[name][:,kv*3].tobytes():
                    raise AssertionError('GQA repetition differs inside a physical group')
    return arrays,losses,exact


def training_error(queries,keys,values,target,arrays,check_time):
    numerator=denominator=0.
    with torch.no_grad():
        for window in range(len(queries)):
            check_time()
            q=folded_latents(queries[window:window+1],arrays,'query')
            k=folded_latents(keys[window:window+1],arrays,'key')
            positions=torch.arange(128,512,device='cuda')[None]
            predicted=gaussian_values(q[:,:,128:],k,values[window:window+1],positions)
            truth=target[window:window+1,:,128:]
            numerator+=float((predicted.double()-truth.double()).square().sum())
            denominator+=float(truth.double().square().sum())
    return {'value_output_nrmse':float(np.sqrt(numerator/denominator)),
            'error_squared_sum':numerator,'reference_squared_sum':denominator,'query_head_rows':16*15*384}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();data=args.data.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('fresh training output required')
    wall=time.monotonic();cpu=time.process_time()
    metadata=verify_data(data);protocol_bytes=args.protocol.read_bytes()
    source_paths=(Path(__file__),Path(learned.__file__),Path(routing.__file__),Path(flat_routing.__file__),Path(llm_routing.__file__))
    sources={p.name:p.read_bytes() for p in source_paths}
    if metadata['training_tokens']!=8192 or metadata['development_articles']!=58 or metadata['test_split_read']:
        raise ValueError('frozen article/data accounting differs')
    with np.load(data/'tokens.npz',allow_pickle=False) as archive:
        tokens=torch.as_tensor(archive['train'].copy(),dtype=torch.int64)
    if tokens.shape!=(16,512):raise ValueError('fixed training tokens required')
    output.mkdir(parents=True);(output/'sources').mkdir()
    for name,content in sources.items():(output/'sources'/name).write_bytes(content)
    (output/'prospective_protocol.txt').write_bytes(protocol_bytes)
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':'training_only_before_development_outputs',
        'data_directory':str(data),'data_completion_sha256':DATA_HASH,'protocol_sha256':hashlib.sha256(protocol_bytes).hexdigest(),
        'model':metadata['model'],'model_revision':metadata['model_revision'],'layer':0,'sequence_length':512,
        'seeds':list(SEEDS),'steps':200,'optimizer':'Adam','learning_rate':0.001,'temperature':2,
        'batch_windows':2,'queries_per_window':32,'affected_training_query_start':128,
        'primary_loss':'batch output squared-error sum / native pre-o_proj target squared-norm sum',
        'gpu_enabled_process_wall_limit_seconds':3600,'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
        'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')},'command':list(sys.argv)}
    json_write(output/'config.json',config)
    def check_time():
        if time.monotonic()-wall>3600:raise TimeoutError('cumulative GPU-enabled work reached one-hour ceiling')
    try:
        torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required on Spark')
        model=AutoModelForCausalLM.from_pretrained(metadata['model'],revision=metadata['model_revision'],
            torch_dtype=torch.bfloat16,attn_implementation='eager',use_safetensors=True).to('cuda').eval()
        if (model.config.num_attention_heads,model.config.num_key_value_heads,model.config.num_hidden_layers)!=(15,5,32):
            raise ValueError('pinned configuration differs')
        state_hash=state_dict_hash(model)
        if state_hash!=MODEL_STATE_HASH:raise ValueError('loaded original model-state fingerprint differs')
        arrays,losses,exact=capture_training(model,tokens,check_time)
        np.savez_compressed(output/'training_trace.npz',**arrays)
        # No development IDs/states/NLL are consumed by this fitting stage.
        q,k,v,target=(torch.as_tensor(arrays[name],device='cuda') for name in ('queries','keys','values','native_values'))
        records=[]
        for seed in SEEDS:
            check_time();directory=output/f'seed-{seed}';directory.mkdir()
            encoder,initial,history=fit_encoder(q,k,v,target,seed,check_time)
            initial_model=DualAffine(q,k,seed)
            initial_model.load_state_dict({n:torch.as_tensor(a,device='cuda') for n,a in initial.items()})
            initial_folded=initial_model.folded_arrays();final_folded=encoder.folded_arrays()
            np.savez_compressed(directory/'initial_training_state.npz',**initial)
            np.savez_compressed(directory/'final_training_state.npz',**{n:a.detach().cpu().numpy() for n,a in encoder.state_dict().items()})
            np.savez_compressed(directory/'initial_encoder.npz',**initial_folded,seed=np.asarray([seed],dtype=np.uint64))
            np.savez_compressed(directory/'final_encoder.npz',**final_folded,seed=np.asarray([seed],dtype=np.uint64))
            np.savez_compressed(directory/'training_history.npz',**history)
            with torch.no_grad():
                initial_q=folded_latents(q,initial_folded,'query').cpu().numpy()
                initial_k=folded_latents(k,initial_folded,'key').cpu().numpy()
            initial_book=QuantileCodebook.fit(initial_q,initial_k,coordinates=2,prime=2,code_bit_budget=8,
                                              projection_kind='pca',seed=seed)
            np.savez_compressed(directory/'initial_routing_state.npz',center=initial_book.center,
                projection=initial_book.projection,thresholds=initial_book.thresholds,
                digits=np.asarray(initial_book.digits,dtype=np.uint16),prime_seed=np.asarray([2,seed],dtype=np.uint64))
            with torch.no_grad():
                qlat=folded_latents(q,final_folded,'query');klat=folded_latents(k,final_folded,'key')
                folding={role:float((encoder.encode(values,role)-latents).abs().max())
                    for role,values,latents in (('query',q,qlat),('key',k,klat))}
                qlatent,klatent=qlat.cpu().numpy(),klat.cpu().numpy()
            books={p:QuantileCodebook.fit(qlatent,klatent,coordinates=2,prime=p,code_bit_budget=8,
                    projection_kind='pca',seed=seed) for p in (2,3)}
            binary=books[2];np.testing.assert_array_equal(binary.center,books[3].center)
            np.testing.assert_array_equal(binary.projection,books[3].projection)
            flat=FlatKeyCodebook.fit(klatent,center=binary.center,projection=binary.projection,
                max_centroids=13,seed=17,max_iterations=20)
            bridge_bytes=sum(a.nbytes for a in final_folded.values())+8
            states={};memory={}
            for p,book in books.items():
                fields={'center':book.center,'projection':book.projection,'thresholds':book.thresholds,
                    'digits':np.asarray(book.digits,dtype=np.uint16),'prime_seed':np.asarray([p,seed],dtype=np.uint64)}
                states.update({f'p{p}_{n}':a for n,a in fields.items()})
                memory[f'p{p}']=bridge_bytes+sum(a.nbytes for a in fields.values())
            states.update({f'flat_{n}':a for n,a in flat.state_arrays.items()})
            memory['flat']=bridge_bytes+flat.state_bytes
            if memory['flat']>memory['p2']:raise AssertionError('flat numeric-state cap fails before development')
            np.savez_compressed(directory/'routing_state.npz',**states)
            np.savez_compressed(directory/'training_latents.npz',queries=qlatent,keys=klatent)
            record={'seed':seed,'steps_completed':len(history['loss']),'initial_training_error':training_error(q,k,v,target,initial_folded,check_time),
                'final_training_error':training_error(q,k,v,target,final_folded,check_time),
                'folding_maximum_latent_absolute_difference':folding,'standalone_numeric_state_bytes':memory,
                'flat_centroid_counts':flat.centroid_counts.tolist(),'flat_iterations':flat.iterations.tolist(),
                'shared_latent_pca_exact':True,'first_batch_loss':float(history['loss'][0]),'last_batch_loss':float(history['loss'][-1])}
            json_write(directory/'results.json',record);records.append(record)
        if state_dict_hash(model)!=state_hash:raise AssertionError('original model parameters/buffers changed')
        if verify_data(data)!=metadata or args.protocol.read_bytes()!=protocol_bytes:
            raise RuntimeError('data/protocol changed during fit')
        for path in source_paths:
            if path.read_bytes()!=sources[path.name] or (output/'sources'/path.name).read_bytes()!=sources[path.name]:
                raise RuntimeError('executed source or snapshot changed')
        result={'status':'learned_encoders_frozen_before_development','config':config,'model_loaded_state_sha256':state_hash,
            'training_baseline_window_nll':losses,'full_budget_512_fixture_stock_exact':exact,'seed_results':records,
            'development_model_outputs_read':False,'test_split_read':False,
            'original_model_state_unchanged':True,'gpu_enabled_process_wall_seconds':time.monotonic()-wall,
            'process_cpu_seconds':time.process_time()-cpu,'scope':'Training-only fit evidence; no development/confirmation quality or native performance claim'}
        json_write(output/'results.json',result);check_time()
        artifacts=sorted(p for p in output.rglob('*') if p.is_file())
        completion={'status':result['status'],'data_completion_sha256':DATA_HASH,'gpu_enabled_process_wall_seconds':time.monotonic()-wall,
            'artifact_sha256':{str(p.relative_to(output)):digest(p) for p in artifacts},
            'artifact_bytes':{str(p.relative_to(output)):p.stat().st_size for p in artifacts}}
        check_time();json_write(output/'completed.json',completion)
        (output/'completed.sha256').write_text(digest(output/'completed.json')+'\n')
        print(json.dumps({'status':result['status'],'seconds':completion['gpu_enabled_process_wall_seconds'],'seed_results':records},indent=2))
    except Exception as error:
        (output/'completed.json').unlink(missing_ok=True);(output/'completed.sha256').unlink(missing_ok=True)
        json_write(output/'failed.json',{'status':'learned_training_failed','type':type(error).__name__,'message':str(error),
            'gpu_enabled_process_wall_seconds':time.monotonic()-wall})
        raise


if __name__=='__main__':main()
