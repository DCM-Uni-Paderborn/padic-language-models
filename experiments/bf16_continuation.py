"""Matched BF16 continuation with FP32 shadow weights and wide accumulation."""
import argparse,gc,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import low_precision_study as study
class BF16Linear(nn.Module):
 def __init__(self,original):
  super().__init__();assert original.bias is None
  self.weight=nn.Parameter(original.weight.detach().float().clone())
 def forward(self,x):
  return F.linear(x.float(),self.weight.to(torch.bfloat16).float()).to(x.dtype)
def replace(model):
 for p in model.parameters():p.requires_grad_(False)
 modules=[]
 for layer in model.model.layers:
  for owner in (layer.self_attn,layer.mlp):
   for name,old in list(owner.named_children()):
    if isinstance(old,nn.Linear):
     new=BF16Linear(old);setattr(owner,name,new);modules.append(new)
 count=sum(m.weight.numel() for m in modules)
 rest=sum(p.numel() for p in model.parameters())-count
 storage={"quantized_projection_parameters":count,"projection_code_bytes":2*count,
  "scale_metadata_bytes":0,"remaining_bf16_parameter_bytes":2*rest,
  "ideal_inference_parameter_bytes":2*(count+rest),"group_size":64,"packed_inference_implemented":False}
 return modules,storage
def train(cache,tokens,size,seed,output):
 output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.manual_seed(seed)
 model=study.load_model(size);modules,storage=replace(model)
 optimizer=torch.optim.AdamW([m.weight for m in modules],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 before=study.evaluate(model,tokens,["development","wikitext","pg19"])
 initial=[m.weight.detach().to(torch.bfloat16).view(torch.int16).clone() for m in modules]
 rng=np.random.default_rng(np.random.SeedSequence([seed,20261019]))
 schedule=np.concatenate([rng.permutation(128) for _ in range(2)]).reshape(128,2)
 history=[];model.train()
 try:
  for step,indices in enumerate(schedule):
   if time.monotonic()-start>1800:raise TimeoutError("fixed per-fit1800-second cap")
   ids=tokens["train"][indices].cuda();logits=model(ids,use_cache=False).logits[:,:-1].float()
   student=logits.reshape(-1,logits.shape[-1])
   with torch.no_grad():
    ref=cache[indices].cuda().float().reshape_as(student)
    probs=torch.softmax(ref,-1);teacher_logp=torch.log_softmax(ref,-1)
   logp=torch.log_softmax(student,-1);ce=F.nll_loss(logp,ids[:,1:].reshape(-1))
   kl=(probs*(teacher_logp-logp)).sum(-1).mean();loss=.5*(ce+kl)
   if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite training loss")
   optimizer.zero_grad(set_to_none=True);loss.backward()
   norm=torch.nn.utils.clip_grad_norm_([m.weight for m in modules],1.,error_if_nonfinite=True);optimizer.step()
   history.append({"step":step,"ce":float(ce.detach()),"kl":float(kl.detach()),"loss":float(loss.detach()),"gradient_norm":float(norm)})
   if step%16==15:print(json.dumps({"model":size,"case":"w16_a16_bf16","seed":seed,"step":step+1,"ce":float(ce.detach()),"elapsed":time.monotonic()-start}),flush=True)
  after=study.evaluate(model,tokens,["development","wikitext","pg19"])
  changed=sum(int((a!=m.weight.detach().to(torch.bfloat16).view(torch.int16)).sum()) for a,m in zip(initial,modules))
  study.write(output/"results.json",{"model":size,"regime":"w16_a16","family":"bf16","seed":seed,"steps":128,
   "training_input_tokens":65536,"ptq":before,"qat":after,"storage":storage,"changed_weight_codes":changed,
   "total_weight_codes":sum(a.numel() for a in initial),"history":history,"elapsed_seconds":time.monotonic()-start,
   "minibatch_schedule":schedule.tolist()})
  study.write(output/"completed.json",{"results_sha256":study.sha(output/"results.json"),"status":"complete"})
  print(json.dumps({"complete":str(output),"wikitext_ppl":after["wikitext"]["ppl"],"pg19_ppl":after["pg19"]["ppl"]}),flush=True)
 except Exception as e:
  study.write(output/"failed.json",{"type":type(e).__name__,"message":str(e),"history":history});raise
 finally:
  del model,modules,optimizer,initial;gc.collect();torch.cuda.empty_cache()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--size",choices=["135M","360M"],required=True);p.add_argument("--after",nargs="*",type=Path,default=[]);a=p.parse_args()
 while any(not (r/"completed.json").exists() for r in a.after):time.sleep(15)
 a.output.mkdir(parents=True,exist_ok=False);freeze=study.freeze(a.output)
 freeze["additional_sources"]={"experiments/bf16_continuation.py":study.sha(__file__),"research/bf16-continuation-20261019.md":study.sha(ROOT/"research/bf16-continuation-20261019.md")}
 study.write(a.output/"freeze.json",freeze)
 meta=json.loads((a.inputs/"manifest.json").read_text());assert study.sha(a.inputs/"tokens.npz")==meta["tokens_sha256"]
 with np.load(a.inputs/"tokens.npz",allow_pickle=False) as z:tokens={k:torch.from_numpy(z[k].copy()) for k in z.files}
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
 teacher=study.load_model(a.size).eval()
 for q in teacher.parameters():q.requires_grad_(False)
 study.write(a.output/(a.size+"-reference.json"),{"metrics":study.evaluate(teacher,tokens,["development","wikitext","pg19"]),"state_sha256":study.state_dict_hash(teacher)})
 cache=[]
 with torch.inference_mode():
  for i in range(0,128,2):cache.append(teacher(tokens["train"][i:i+2].cuda(),use_cache=False).logits[:,:-1].cpu())
 cache=torch.cat(cache);del teacher;gc.collect();torch.cuda.empty_cache()
 study.write(a.output/"cases.json",{"requested":[{"regime":"w16_a16","family":"bf16","canonical":"w16_a16_bf16"}]})
 for seed in study.SEEDS:
  try:train(cache,tokens,a.size,seed,a.output/a.size/"w16_a16_bf16"/f"seed-{seed}")
  except Exception as e:print(json.dumps({"run_closed":"w16_a16_bf16","seed":seed,"error_type":type(e).__name__,"message":str(e)}),flush=True)
 study.write(a.output/"completed.json",{"status":"matched_bf16_continuation_finished"})
if __name__=="__main__":main()
