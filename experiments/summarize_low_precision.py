"""Build compact paper tables from the complete fixed continuation matrix."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from analyze_low_precision import study_group
LABELS={"int":"Integer","int_full":"Full integer","residue_3":"Ternary residues","residue_5":"Quinary residues",
 "unit_2":"Binary units","unit_3":"Ternary units","unit_5":"Quinary units","unit_wide_3":"Extended ternary units",
 "fp4":"Real FP4","bf16":"Continued BF16","wrap_2":"Binary low digits","wrap_3":"Ternary low digits"}
REGIMES={"mix24_a4":"W2/W4, A4"}
def regime_label(r):
 if r in REGIMES:return REGIMES[r]
 w,a=r.removeprefix("w").split("_a")
 return "W"+w+"/A"+a
def interval(values,precision=3):
 lo,hi=min(values),max(values)
 return f"{lo:.{precision}f}" if round(lo,precision)==round(hi,precision) else f"{lo:.{precision}f}--{hi:.{precision}f}"
def ppl_interval(values):
 def number(x):
  if x>=10000:
   e=int(math.floor(math.log10(x)));return rf"{x/10**e:.2f}\times10^{{{e}}}"
  return f"{x:.2f}"
 lo,hi=min(values),max(values)
 return r"\("+number(lo)+(r"\text{--}"+number(hi) if round(lo,2)!=round(hi,2) else "")+r"\)"
def run(directory,output):
 summary=json.loads((directory/"summary.json").read_text());assert summary["matrix_complete"] and summary["complete_fits"]==168
 data=json.loads((directory/"compact-fits.json").read_text());output.mkdir(parents=True,exist_ok=False)
 records=data["records"];groups={}
 for x in records:groups.setdefault((x["model"],x["regime"],x["family"]),[]).append(x)
 assert len(groups)==56
 for rows in groups.values():assert sorted(x["seed"] for x in rows)==[17,29,43]
 decisions={(x["model"],x["regime"],x["family"]):x for x in summary["family_decisions"]}
 numeric=[]
 for key,rows in sorted(groups.items()):
  for phase in ("ptq","qat"):
   for corpus in ("development","wikitext","pg19"):
    values=[x[phase][corpus]["nll"] for x in rows];ppls=[x[phase][corpus]["ppl"] for x in rows]
    numeric.append(dict(model=key[0],regime=key[1],family=key[2],phase=phase,corpus=corpus,
     minimum_nll=min(values),maximum_nll=max(values),minimum_ppl=min(ppls),maximum_ppl=max(ppls),
     final_quality_retained=decisions[key]["retains_quality"]))
 with (output/"seed-ranges.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,list(numeric[0]));w.writeheader();w.writerows(numeric)
 def table(keys,caption,label):
  lines=[r"\begin{table}[p]",r"\centering\scriptsize",r"\setlength{\tabcolsep}{3.5pt}",
   r"\caption{"+caption+"}",r"\label{"+label+"}",r"\begin{tabular}{llrrrrl}",r"\toprule",
   r"Schedule & Alphabet & \multicolumn{2}{c}{WikiText NLL} & \multicolumn{2}{c}{PG-19 NLL} & Quality\\",
   r"& & Before & After & Before & After & \\",r"\midrule"]
  for key in keys:
   rows=groups[key];fields=[regime_label(key[1]),LABELS[key[2]]]
   for corpus in ("wikitext","pg19"):
    for phase in ("ptq","qat"):fields.append(interval([x[phase][corpus]["nll"] for x in rows]))
   fields.append("Pass" if decisions[key]["retains_quality"] else "Fail")
   lines.append(" & ".join(fields)+r" \\")
  lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}",""])
  return "\n".join(lines)
 si=[]
 primary=sorted(k for k in groups if k[0]=="135M" and study_group(*k)=="primary")
 extra=sorted(k for k in groups if k[0]=="135M" and study_group(*k)!="primary")
 confirmation=sorted(k for k in groups if k[0]=="360M")
 si.append(table(primary,"All primary SmolLM2-135M settings. Before and after refer to the fixed 128-update continuation. Entries give ranges over the three fixed seeds, collapsing to one number when they agree at the shown precision. Quality requires all six upper bounds against the original BF16 checkpoint to meet the declared allowance. Exact codebook aliases are evaluated once.","tab:low-precision-primary"))
 si.append(table(extra,"Additional SmolLM2-135M unit-precision allocations, low-digit projections, full-integer controls and matched BF16 continuation. Values are NLL ranges over the same three seeds before and after continuation. Binary low-digit projection has the same decoded alphabet as the full integer control but a different encoder. The ternary W3/A8 low-digit setting is an exact alias of its W2/A8 setting.","tab:low-precision-projection"))
 si.append(table(confirmation,"All SmolLM2-360M W4/A8 settings and the matched BF16 continuation on the second size/device pair. Entries give three-seed NLL ranges. The two nearest-real residue controls match the alphabets of the cyclic projections. This comparison does not isolate a model-size effect.","tab:low-precision-confirmation"))
 (output/"supporting-tables.tex").write_text("\n".join(si))
 main=[r"\begin{table}[htbp]",r"\centering\small",
  r"\caption{W4/A8 perplexity after the fixed continuation. Ranges include all three seeds. Original BF16 gives the unadapted checkpoints. Continued BF16 applies the same weight updates with BF16 forward precision. The extended unit alphabet has 13 states, while the primary unit and real FP4 alphabets have 15. The low-bit rows use the same four-bit index width and scale recipe. These are numerical emulations with real accumulation.}",
  r"\label{tab:low-precision-central}",r"\begin{tabular}{llrr}",r"\toprule",
  r"Model & Weight alphabet & WikiText PPL & PG-19 PPL\\",r"\midrule"]
 for size in ("135M","360M"):
  ref=data["references"][size]["metrics"]
  main.append(size+" & Original BF16 & "+f"{ref['wikitext']['ppl']:.2f} & {ref['pg19']['ppl']:.2f}"+r" \\")
  continued=groups[(size,"w16_a16","bf16")]
  main.append(" & ".join([size,"Continued BF16",ppl_interval([x["qat"]["wikitext"]["ppl"] for x in continued]),ppl_interval([x["qat"]["pg19"]["ppl"] for x in continued])])+r" \\")
  for family in ("int","fp4","unit_2","unit_3","unit_wide_3"):
   rows=groups[(size,"w4_a8",family)]
   main.append(" & ".join([size,LABELS[family],ppl_interval([x["qat"]["wikitext"]["ppl"] for x in rows]),ppl_interval([x["qat"]["pg19"]["ppl"] for x in rows])])+r" \\")
  if size=="135M":main.append(r"\midrule")
 main.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}",""])
 (output/"main-table.tex").write_text("\n".join(main))
 storage_keys=[("135M","w16_a16","bf16"),("135M","w2_a8","int"),("135M","w3_a8","int"),("135M","w4_a8","int"),("360M","w16_a16","bf16"),("360M","w4_a8","int")]
 storage=[r"\begin{table}[htbp]",r"\centering\small",
  r"\caption{Ideal parameter payloads in millions of bytes. Projection values use packed code indices for the low-bit methods and sixteen-bit words for the continued BF16 controls. Scales include the learned clipping factors. Remaining parameters include tied embeddings once. These counts exclude activations, optimizer state, dequantization and runtime overhead.}",
  r"\label{tab:low-precision-storage}",r"\begin{tabular}{llrrrr}",r"\toprule",
  r"Model & Schedule & Projections & Scales & Remaining & Total\\",r"\midrule"]
 for key in storage_keys:
  q=groups[key][0]["storage"]
  assert all(x["storage"]==q for x in groups[key])
  values=[q[n]/1e6 for n in ("projection_code_bytes","scale_metadata_bytes","remaining_bf16_parameter_bytes","ideal_inference_parameter_bytes")]
  assert q["projection_code_bytes"]+q["scale_metadata_bytes"]+q["remaining_bf16_parameter_bytes"]==q["ideal_inference_parameter_bytes"]
  storage.append(" & ".join([key[0],regime_label(key[1]),*[f"{v:.3f}" for v in values]])+r" \\")
 storage.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}",""])
 (output/"storage-table.tex").write_text("\n".join(storage))
 decision_counts={}
 for size in ("135M","360M"):
  ds=[v for k,v in decisions.items() if k[0]==size]
  decision_counts[size]={"canonical_settings":len(ds),"retains_quality":sum(x["retains_quality"] for x in ds),
   "beats_trained_integer":sum(x["beats_integer"] for x in ds),"beats_matched_clipped":sum(x["beats_clipped"] for x in ds),
   "beats_uniform_three_bit":sum(x["beats_uniform_w3_a4"] for x in ds),
   "supplementary_retains_continued_bf16":sum(x["supplementary_retains_continued_bf16"] for x in ds)}
 (output/"decisions.json").write_text(json.dumps(decision_counts,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"settings":len(groups),"fits":len(records),"decisions":decision_counts}))
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--statistics",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();run(a.statistics,a.output)
