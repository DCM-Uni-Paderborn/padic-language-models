"""Frozen full-Transformer quantization and mixed-precision continuation."""
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os,sys,time,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from padic_lm.low_precision import cases,replace_projections,alphabet
from padic_lm.corpus import articles
from llm_routing import state_dict_hash

REVISIONS={"135M":"93efa2f097d58c2a74874c7e644dbc9b0cee75a2","360M":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49"}
DATASET="b08601e04326c79dfdd32d625aee71d232d685c3"
SEEDS=(17,29,43)
STEPS=128
BATCH=2
LENGTH=256

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".writing")
    temp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temp.replace(p)

def freeze(output):
    names=["src/padic_lm/low_precision.py","src/padic_lm/corpus.py","experiments/low_precision_study.py",
           "experiments/llm_routing.py","tests/test_low_precision.py","research/low-precision-protocol-20261019.md"]
    value={"source_sha256":{n:sha(ROOT/n) for n in names},"models":REVISIONS,"dataset_revision":DATASET,
           "seeds":list(SEEDS),"steps":STEPS,"batch":BATCH,"length":LENGTH,
           "created_utc":datetime.now(timezone.utc).isoformat(),"training_loss":"0.5*CE+0.5*teacher_KL",
           "primary_quality_log_margin":float(np.log(1.10))}
    write(output/"freeze.json",value)
    return value

def prepare(output):
    output.mkdir(parents=True,exist_ok=False);freeze(output)
    tokenizer=AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M",revision=REVISIONS["135M"],local_files_only=True)
    other=AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M",revision=REVISIONS["360M"],local_files_only=True)
    prior=json.loads((ROOT/"results/learned-data-articles-360m-spark-20261004/manifest.json").read_text())
    excluded={x["article_text_sha256"] for x in prior["windows"]["train"]}
    rows=[r["text"] for r in load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",revision=DATASET,split="train",streaming=True)]
    eligible=[]
    for doc in articles(rows):
        if doc.text_sha256 in excluded:continue
        ids=tokenizer(doc.text,add_special_tokens=False)["input_ids"]
        if len(ids)<LENGTH:continue
        key=hashlib.sha256(("low-precision-20261019|"+doc.text_sha256).encode()).hexdigest()
        eligible.append((key,doc,ids[:LENGTH]))
    eligible.sort(key=lambda x:x[0])
    chosen=[];texts=set();token_hashes=set()
    for _,doc,ids in eligible:
        h=hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()
        if h in token_hashes or doc.text_sha256 in texts:continue
        if other(doc.text,add_special_tokens=False)["input_ids"][:LENGTH]!=ids:
            raise ValueError("tokenizers differ on selected input")
        chosen.append((doc,ids,h));texts.add(doc.text_sha256);token_hashes.add(h)
        if len(chosen)==144:break
    if len(chosen)!=144:raise ValueError("requires128 train+16 disjoint development articles")
    tokens={"train":np.array([x[1] for x in chosen[:128]],dtype="<i8"),
            "development":np.array([x[1] for x in chosen[128:]],dtype="<i8")}
    records={role:[{"title":x[0].title,"text_sha256":x[0].text_sha256,"start_row":x[0].start_row,
                     "stop_row":x[0].stop_row,"token_sha256":x[2]} for x in group]
             for role,group in [("train",chosen[:128]),("development",chosen[128:])]}
    for name,directory,field in [
        ("wikitext","heldout-context-data-spark-20261004","tokens"),
        ("pg19","pg19-context-data-spark-20261004","tokens4096")]:
        d=ROOT/"results"/directory
        completion=json.loads((d/"completed.json").read_text())
        if sha(d/"tokens.npz")!=completion["artifact_sha256"]["tokens.npz"]:
            raise ValueError("frozen test tokens differ")
        with np.load(d/"tokens.npz",allow_pickle=False) as z:
            key=field if field in z.files else ("tokens2048" if "tokens2048" in z.files else z.files[0])
            tokens[name]=z[key][:,:LENGTH].copy()
        if tokens[name].shape!=(40,LENGTH):raise ValueError("forty test prefixes required")
        meta=json.loads((d/"manifest.json").read_text())
        selected=meta.get("books",meta.get("selected",meta.get("articles",[])))
        if isinstance(selected,dict):selected=selected.get("test",[])
        records[name]={"original_manifest_sha256":sha(d/"manifest.json"),"documents":selected}
    np.savez_compressed(output/"tokens.npz",**tokens)
    write(output/"manifest.json",{"models":REVISIONS,"dataset_revision":DATASET,"length":LENGTH,"records":records,
        "training_documents":128,"development_documents":16,"test_documents_per_corpus":40,
        "prior_adaptation_article_hashes_excluded":sorted(excluded),"tokenizers_match_selected_training":True,
        "tokens_sha256":sha(output/"tokens.npz"),"scope":"Held out from this continuation; test corpora were used in earlier component studies"})
    print(json.dumps({"inputs_ready":str(output),"tokens_sha256":sha(output/"tokens.npz")}),flush=True)

def load_model(size):
    model=AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-"+size,revision=REVISIONS[size],
        torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation="sdpa").to("cuda")
    model.config.use_cache=False
    return model

def evaluate(model,tokens,labels):
    model.eval();result={}
    with torch.inference_mode():
        for name in labels:
            rows=[]
            for i in range(0,len(tokens[name]),2):
                ids=tokens[name][i:i+2].to("cuda")
                logits=model(ids,use_cache=False).logits[:,:-1].float()
                losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction="none").reshape(len(ids),-1)
                rows.extend([{"document":i+j,"targets":LENGTH-1,"loss_sum":float(row.double().sum())} for j,row in enumerate(losses)])
            mean=sum(r["loss_sum"] for r in rows)/sum(r["targets"] for r in rows)
            result[name]={"nll":mean,"ppl":math.exp(mean),"documents":rows}
    return result

def train_one(teacher_cache,tokens,size,regime,family,seed,output):
    if output.exists():
        if (output/"completed.json").exists():return
        raise FileExistsError("incomplete output is not automatically restarted")
    output.mkdir(parents=True);start=time.monotonic()
    torch.manual_seed(seed)
    model=load_model(size)
    modules,storage=replace_projections(model,regime,family)
    groups=[{"params":[m.weight for m in modules],"lr":1e-5},
            {"params":[p for m in modules for p in (m.log_scale,m.log_clip)],"lr":1e-3}]
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    before=evaluate(model,tokens,["development","wikitext","pg19"])
    initial_codes=[torch.bucketize((m.weight.reshape(m.out_features,-1,64)/m.log_scale.exp()).detach(),
                      (m.levels[1:]+m.levels[:-1])/2).to(torch.uint8) for m in modules]
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261019]))
    schedule=np.concatenate([rng.permutation(128) for _ in range(2)]).reshape(STEPS,BATCH)
    history=[];model.train()
    try:
        for step,indices in enumerate(schedule):
            if time.monotonic()-start>1800:raise TimeoutError("fixed per-fit1800-second cap")
            ids=tokens["train"][indices].to("cuda")
            targets=ids[:,1:].reshape(-1)
            logits=model(ids,use_cache=False).logits[:,:-1].float()
            student=logits.reshape(-1,logits.shape[-1])
            with torch.no_grad():
                reference=teacher_cache[indices].to("cuda").float().reshape_as(student)
                probs=torch.softmax(reference,dim=-1);teacher_logp=torch.log_softmax(reference,dim=-1)
            logp=torch.log_softmax(student,dim=-1)
            ce=F.nll_loss(logp,targets)
            kl=(probs*(teacher_logp-logp)).sum(-1).mean()
            loss=.5*(ce+kl)
            if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]],1.,error_if_nonfinite=True)
            optimizer.step()
            history.append({"step":step,"ce":float(ce.detach()),"kl":float(kl.detach()),"loss":float(loss.detach()),"gradient_norm":float(norm)})
            if step%16==15:print(json.dumps({"model":size,"case":regime+"_"+family,"seed":seed,"step":step+1,"ce":history[-1]["ce"],"elapsed":time.monotonic()-start}),flush=True)
        after=evaluate(model,tokens,["development","wikitext","pg19"])
        changed=total=0
        for initial,m in zip(initial_codes,modules):
            codes=torch.bucketize((m.weight.reshape(m.out_features,-1,64)/m.log_scale.exp()).detach(),
                                  (m.levels[1:]+m.levels[:-1])/2).to(torch.uint8)
            changed+=int((codes!=initial).sum());total+=codes.numel()
        write(output/"results.json",{"model":size,"regime":regime,"family":family,"seed":seed,"steps":STEPS,
              "training_input_tokens":STEPS*BATCH*LENGTH,"ptq":before,"qat":after,"storage":storage,
              "changed_weight_codes":changed,"total_weight_codes":total,"history":history,
              "elapsed_seconds":time.monotonic()-start,"minibatch_schedule":schedule.tolist()})
        write(output/"completed.json",{"results_sha256":sha(output/"results.json"),"status":"complete"})
        print(json.dumps({"complete":str(output),"wikitext_ppl":after["wikitext"]["ppl"],"pg19_ppl":after["pg19"]["ppl"],"changed_codes":changed,"seconds":time.monotonic()-start}),flush=True)
    except Exception as e:
        write(output/"failed.json",{"type":type(e).__name__,"message":str(e),"history":history,"elapsed_seconds":time.monotonic()-start})
        raise
    finally:
        del model,optimizer,modules,groups,initial_codes
        gc.collect();torch.cuda.empty_cache()

def run(inputs,output,sizes,only):
    meta=json.loads((inputs/"manifest.json").read_text())
    if sha(inputs/"tokens.npz")!=meta["tokens_sha256"]:raise ValueError("input checksum differs")
    output.mkdir(parents=True,exist_ok=False);freeze(output)
    with np.load(inputs/"tokens.npz",allow_pickle=False) as z:
        tokens={k:torch.from_numpy(z[k].copy()) for k in z.files}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    req,aliases=cases();by_tag={r+"_"+f:(r,f) for r,f in req}
    write(output/"cases.json",{"requested":[{"regime":r,"family":f,"canonical":aliases[r+"_"+f]} for r,f in req]})
    for size in sizes:
        teacher=load_model(size).eval()
        for p in teacher.parameters():p.requires_grad_(False)
        baseline=evaluate(teacher,tokens,["development","wikitext","pg19"])
        write(output/(size+"-reference.json"),{"metrics":baseline,"state_sha256":state_dict_hash(teacher)})
        cache=[]
        with torch.inference_mode():
            for i in range(0,128,2):
                cache.append(teacher(tokens["train"][i:i+2].to("cuda"),use_cache=False).logits[:,:-1].cpu())
        teacher_cache=torch.cat(cache);del teacher,cache
        gc.collect();torch.cuda.empty_cache()
        canonical=list(dict.fromkeys(aliases.values()))
        if size=="360M":
            canonical=[r+"_"+f for r in ["w4_a8"] for f in ["int","fp4","unit_2","unit_3"]]
        if only:canonical=[k for k in canonical if k in only]
        for tag in canonical:
            regime,family=by_tag[tag]
            for seed in SEEDS:
                try:
                    train_one(teacher_cache,tokens,size,regime,family,seed,output/size/tag/f"seed-{seed}")
                except Exception as error:
                    print(json.dumps({"run_closed":tag,"seed":seed,"error_type":type(error).__name__,"message":str(error)}),flush=True)
        del teacher_cache
    write(output/"completed.json",{"status":"requested_matrix_finished","complete_fits":len(list(output.glob("*/**/completed.json"))),"closed_failed_fits":len(list(output.glob("*/**/failed.json")))})
    print("STUDY_COMPLETE",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["prepare","run"])
    p.add_argument("--inputs",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sizes",nargs="+",default=["135M","360M"])
    p.add_argument("--only",nargs="*")
    a=p.parse_args()
    if a.mode=="prepare":prepare(a.output)
    else:run(a.inputs,a.output,a.sizes,a.only)

if __name__=="__main__":main()
