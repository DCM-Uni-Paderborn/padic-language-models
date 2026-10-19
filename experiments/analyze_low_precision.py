"""Audit raw fits or rebuild compact low-precision publication statistics."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def schedule_hash(x):return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
def study_group(size,regime,family):
    if family=="bf16":return "continuation"
    if family=="unit_wide_3":return "mantissa"
    if family.startswith("wrap_") or (family=="int_full" and regime in ("w4_a4","mix24_a4")) or (size=="360M" and family in ("int_full","residue_3") and regime=="w4_a8"):return "projection"
    return "primary"
def verify_metrics(metrics):
    for corpus,q in metrics.items():
        rows=q["documents"];n=16 if corpus=="development" else 40
        assert len(rows)==n
        assert [v["document"] for v in rows]==list(range(n))
        assert all(v["targets"]==255 and math.isfinite(v["loss_sum"]) for v in rows)
        nll=sum(v["loss_sum"] for v in rows)/(255*n)
        assert abs(nll-q["nll"])<1e-12
        assert abs(math.log(q["ppl"])-nll)<1e-12

def raw_inputs(runs):
    records={};reference={};requested={};failures=[];schedules={}
    for root in runs:
        if not root.exists():continue
        sizes=[]
        for p in root.glob("*-reference.json"):
            size=p.name.removesuffix("-reference.json");sizes.append(size);x=read(p)
            verify_metrics(x["metrics"])
            if size in reference:assert x==reference[size],size
            reference[size]=x
        if (root/"cases.json").exists():
            for entry in read(root/"cases.json")["requested"]:
                entry=dict(entry)
                # Correct administrative aliases only. Numerical cyclic results stay cyclic.
                if entry["family"]=="int_full" and "wrap_" in entry["canonical"]:
                    entry["canonical"]=entry["regime"]+"_int_full"
                for size in sizes:
                    if size=="360M" and not ((entry["regime"]=="w16_a16" and entry["family"]=="bf16") or (entry["regime"]=="w4_a8" and entry["family"] in ("int","fp4","unit_2","unit_3","unit_wide_3","wrap_2","wrap_3","int_full","residue_3"))):continue
                    key=(size,entry["regime"],entry["family"])
                    if key in requested:assert requested[key]==entry
                    requested[key]=entry
        for p in root.glob("*/*/seed-*/results.json"):
            if not (p.parent/"completed.json").exists():continue
            digest=sha(p);assert digest==read(p.parent/"completed.json")["results_sha256"]
            x=read(p);key=(x["model"],x["regime"],x["family"],x["seed"])
            assert key not in records,key
            for phase in ("ptq","qat"):verify_metrics(x[phase])
            assert x["steps"]==128 and x["training_input_tokens"]==65536 and len(x["history"])==128
            assert all(math.isfinite(v["loss"]) and math.isfinite(v["gradient_norm"]) for v in x["history"])
            schedule=x["minibatch_schedule"]
            assert len(schedule)==128 and all(len(row)==2 for row in schedule)
            flat=[i for row in schedule for i in row]
            assert sorted(flat[:128])==list(range(128)) and sorted(flat[128:])==list(range(128))
            digest_schedule=schedule_hash(schedule)
            if x["seed"] in schedules:assert schedules[x["seed"]]["sha256"]==digest_schedule
            else:schedules[x["seed"]]={"sha256":digest_schedule,"minibatches":schedule}
            assert 0<=x["changed_weight_codes"]<=x["total_weight_codes"]
            assert not x["storage"]["packed_inference_implemented"]
            fields=("model","regime","family","seed","steps","training_input_tokens","ptq","qat","storage","changed_weight_codes","total_weight_codes")
            compact={k:x[k] for k in fields}
            compact["code_change_measure"]="not_recorded_for_cyclic_projection" if x["family"].startswith("wrap_") else "decoded_index_changes"
            if x["family"].startswith("wrap_"):compact["changed_weight_codes"]=None
            compact["schedule_sha256"]=digest_schedule
            compact["source_results_sha256"]=digest
            compact["training_audit"]={"steps_verified":128,"all_losses_and_gradient_norms_finite":True,
                "first_loss":x["history"][0]["loss"],"last_loss":x["history"][-1]["loss"]}
            records[key]=compact
        for p in root.glob("*/*/seed-*/failed.json"):
            failures.append({"model":p.parents[2].name,"case":p.parent.parent.name,"seed":p.parent.name,
                "status":"closed_without_completed_fit"})
    aliases=[{"model":k[0],**v} for k,v in sorted(requested.items())]
    return {"schema":1,"records":[records[k] for k in sorted(records)],"references":reference,
        "requested_aliases":aliases,"schedules":schedules,"closed_fits":failures}

def pack_metrics(metrics):
    return {c:{"nll":q["nll"],"ppl":q["ppl"],
        "document_loss_sums":[row["loss_sum"] for row in q["documents"]]} for c,q in metrics.items()}
def pack_data(data):
    result={**data,"schema":2,"targets_per_document":255}
    result["records"]=[{**x,"ptq":pack_metrics(x["ptq"]),"qat":pack_metrics(x["qat"])} for x in data["records"]]
    result["references"]={size:{**x,"metrics":pack_metrics(x["metrics"])} for size,x in data["references"].items()}
    return result
def expand_data(data):
    if data["schema"]==1:return data
    assert data["schema"]==2 and data["targets_per_document"]==255
    def expand(metrics):
        return {c:{"nll":q["nll"],"ppl":q["ppl"],
            "documents":[{"document":i,"targets":255,"loss_sum":v} for i,v in enumerate(q["document_loss_sums"])]} for c,q in metrics.items()}
    return {**data,"records":[{**x,"ptq":expand(x["ptq"]),"qat":expand(x["qat"])} for x in data["records"]],
        "references":{size:{**x,"metrics":expand(x["metrics"])} for size,x in data["references"].items()}}

def summarize(data,output):
    output.mkdir(parents=True,exist_ok=False)
    records={(x["model"],x["regime"],x["family"],x["seed"]):x for x in data["records"]}
    assert len(records)==len(data["records"])
    reference=data["references"];requested=data["requested_aliases"];failures=data["closed_fits"]
    for schedule in data["schedules"].values():
        assert schedule_hash(schedule["minibatches"])==schedule["sha256"]
        flat=[i for row in schedule["minibatches"] for i in row]
        assert len(flat)==256 and sorted(flat[:128])==list(range(128)) and sorted(flat[128:])==list(range(128))
    for x in records.values():
        for phase in ("ptq","qat"):verify_metrics(x[phase])
        assert x["steps"]==128 and x["training_input_tokens"]==65536
        assert x["schedule_sha256"]==data["schedules"][str(x["seed"])]["sha256"] if str(x["seed"]) in data["schedules"] else x["schedule_sha256"]==data["schedules"][x["seed"]]["sha256"]
    aliases={(x["model"],x["regime"],x["family"]):x["canonical"] for x in requested}
    def canonical(size,regime,family,seed):
        tag=aliases.get((size,regime,family),regime+"_"+family)
        found=[x for (m,r,f,s),x in records.items() if m==size and s==seed and r+"_"+f==tag]
        assert len(found)<=1
        return found[0] if found else None
    draws=np.random.Generator(np.random.PCG64(514203)).integers(0,40,size=(20000,40))
    frequencies=np.stack([np.bincount(row,minlength=40) for row in draws])
    compact=[];comparisons=[];max_audit_error=0.
    for key,x in sorted(records.items()):
        size,regime,family,seed=key
        for phase in ("ptq","qat"):
            for corpus,q in x[phase].items():
                compact.append({"model":size,"regime":regime,"family":family,"seed":seed,"phase":phase,"corpus":corpus,
                    "nll":q["nll"],"ppl":q["ppl"],"code_bytes":x["storage"]["projection_code_bytes"],
                    "scale_bytes":x["storage"]["scale_metadata_bytes"],"bf16_bytes":x["storage"]["remaining_bf16_parameter_bytes"],
                    "changed_codes":x["changed_weight_codes"],"code_count":x["total_weight_codes"]})
        for corpus in ("wikitext","pg19"):
            candidate=np.array([r["loss_sum"] for r in x["qat"][corpus]["documents"]])
            baselines={"bf16":reference[size]["metrics"][corpus]}
            continued=canonical(size,"w16_a16","bf16",seed)
            if continued is not None:baselines["matched_continued_bf16"]=continued["qat"][corpus]
            integer=canonical(size,regime,"int",seed)
            if integer is not None:baselines["matched_int"]=integer["qat"][corpus]
            if family.startswith("wrap_"):
                clipped=canonical(size,regime,"int_full" if family=="wrap_2" else "residue_3",seed)
                if clipped is not None:baselines["matched_clipped"]=clipped["qat"][corpus]
            if regime=="mix24_a4":
                uniform=canonical(size,"w3_a4","unit_2" if family=="unit_wide_3" else family,seed)
                if uniform is not None:baselines["uniform_w3_a4"]=uniform["qat"][corpus]
            for label,q in baselines.items():
                base=np.array([r["loss_sum"] for r in q["documents"]])
                difference=candidate-base
                direct=difference[draws].sum(1)/(255*40)
                independent=frequencies@difference/(255*40)
                error=float(np.max(np.abs(direct-independent)));assert error<1e-10,error
                max_audit_error=max(max_audit_error,error)
                lo,hi=np.quantile(direct,[.025,.975],method="linear")
                comparisons.append({"model":size,"regime":regime,"family":family,"seed":seed,"corpus":corpus,"reference":label,
                    "delta_nll":float(difference.sum()/(255*40)),"lower_025":float(lo),"upper_975":float(hi),
                    "relative_ppl_percent":float(100*np.expm1(difference.sum()/(255*40))),
                    "quality_within_ten_percent":bool(hi<np.log(1.10)),"superior":bool(hi<0)})
    def csv_write(name,rows):
        if not rows:return
        with (output/name).open("w",newline="") as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write("all-endpoints.csv",compact)
    csv_write("paired-comparisons.csv",comparisons)
    decisions=[]
    for size,regime,kind in sorted({k[:3] for k in records}):
        q=[r for r in comparisons if (r["model"],r["regime"],r["family"])==(size,regime,kind)]
        relative=[r for r in q if r["reference"]=="bf16"]
        matched=[r for r in q if r["reference"]=="matched_int"]
        clipped=[r for r in q if r["reference"]=="matched_clipped"]
        uniform=[r for r in q if r["reference"]=="uniform_w3_a4"]
        continued=[r for r in q if r["reference"]=="matched_continued_bf16"]
        ready=len(relative)==6 and len({r["seed"] for r in relative})==3
        decisions.append({"model":size,"regime":regime,"family":kind,"complete_three_seeds_two_corpora":ready,
            "retains_quality":ready and all(r["quality_within_ten_percent"] for r in relative),
            "beats_integer":len(matched)==6 and all(r["superior"] for r in matched),
            "beats_clipped":len(clipped)==6 and all(r["superior"] for r in clipped),
            "beats_uniform_w3_a4":len(uniform)==6 and all(r["superior"] for r in uniform),
            "supplementary_retains_continued_bf16":len(continued)==6 and all(r["quality_within_ten_percent"] for r in continued),
            "largest_upper_bf16":max((r["upper_975"] for r in relative),default=None)})
    counts={}
    expected={("135M","primary"):93,("360M","primary"):12,("135M","mantissa"):9,
        ("360M","mantissa"):3,("135M","projection"):33,("360M","projection"):12,("135M","continuation"):3,("360M","continuation"):3}
    for (size,group),n in expected.items():
        counts[size+"_"+group]={"complete":sum(k[0]==size and study_group(k[0],k[1],k[2])==group for k in records),"expected":n}
    all_aliases_resolved=all(canonical(x["model"],x["regime"],x["family"],seed) is not None for x in requested for seed in (17,29,43))
    report={"complete_fits":len(records),"expected_fits":sum(expected.values()),"counts":counts,
        "matrix_complete":all(v["complete"]==v["expected"] for v in counts.values()) and not failures and all_aliases_resolved,
        "all_requested_aliases_resolved":all_aliases_resolved,"closed_failed_fits":len(failures),"family_decisions":decisions,
        "bootstrap_seed":514203,"bootstrap_draws":20000,"max_independent_bootstrap_difference":max_audit_error,
        "per_document_aggregations_verified":len(records)*6,"same_seed_schedules_identical":True,
        "scope":"Fixed short QAT continuation; real-decoded codebooks and cyclic projection, not native low-bit kernels or a pure p-adic-valued LLM."}
    write(output/"summary.json",report);write(output/"compact-fits.json",pack_data(data))
    write(output/"baseline-documents.json",pack_data(data)["references"])
    print(json.dumps({k:v for k,v in report.items() if k!="family_decisions"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument("--runs",nargs="+",type=Path);g.add_argument("--published",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();summarize(raw_inputs(a.runs) if a.runs else expand_data(read(a.published)),a.output)
