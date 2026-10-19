"""Independent parameter-payload audit from pinned Transformer configurations."""
import argparse,json,math
from pathlib import Path
def cardinality(family,bits):
 if family=="fp4":assert bits==4;return 15
 if family=="int":return 2**bits-1
 if family=="int_full":return 2**bits
 if family=="unit_wide_3":return 3 if bits==2 else 1+6*((2**bits-1)//6)
 role,p=family.rsplit("_",1);p=int(p)
 if role in ("wrap","residue"):
  m=p
  while m*p<=2**bits:m*=p
  assert m<=2**bits
  return m
 assert role=="unit"
 k=2 if p==2 else 1
 units=p**k-p**(k-1)
 return 1+units*((2**bits-1)//units)
def predicted(config,regime,family):
 c=config;h=c["hidden_size"];kv=c["num_key_value_heads"]*(h//c["num_attention_heads"])
 n=c["num_hidden_layers"];inter=c["intermediate_size"]
 assert c["tie_word_embeddings"] and not c.get("attention_bias",False) and not c.get("mlp_bias",False)
 shapes=[h*h,kv*h,kv*h,h*h,inter*h,inter*h,h*inter]
 assert all(v%64==0 for v in shapes)
 count=n*sum(shapes);codes=0
 for layer in range(n):
  bits=(2 if layer%2==0 else 4) if regime.startswith("mix") else int(regime.split("_")[0][1:])
  width=16 if family=="bf16" else (cardinality(family,bits)-1).bit_length()
  codes+=sum((v*width+7)//8 for v in shapes)
 scales=0 if family=="bf16" else 4*(count//64+7*n)
 remaining=c["vocab_size"]*h+(2*n+1)*h
 return {"quantized_projection_parameters":count,"projection_code_bytes":codes,"scale_metadata_bytes":scales,
  "remaining_bf16_parameter_bytes":2*remaining,"ideal_inference_parameter_bytes":codes+scales+2*remaining}
def audit(data,configs):
 checked=0;payloads={}
 for x in data["records"]:
  expected=predicted(configs[x["model"]]["config"],x["regime"],x["family"])
  for k,v in expected.items():assert x["storage"][k]==v,(x["model"],x["regime"],x["family"],k,x["storage"][k],v)
  assert not x["storage"]["packed_inference_implemented"]
  checked+=1
  payloads["|".join((x["model"],x["regime"],x["family"]))]=expected
 return {"status":"passed","fits_checked":checked,"parameter_payloads":payloads,
  "scope":"Ideal parameter index, scale and remaining BF16 payloads. Excludes activations, optimizer states, temporary dequantization and runtime/serialization overhead."}
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fits",type=Path,required=True);p.add_argument("--configs",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();x=audit(json.loads(a.fits.read_text()),json.loads(a.configs.read_text()))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("x") as f:json.dump(x,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({"status":x["status"],"fits_checked":x["fits_checked"]}))
