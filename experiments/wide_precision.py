"""Additional two-unit-digit allocation under the same low-bit QAT recipe."""
import argparse,json,math,os,sys,time,gc
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
import low_precision_study as study
import padic_lm.low_precision as core

original_alphabet=core.alphabet
def extended_alphabet(kind,bits):
    if kind!="unit_wide_3":return original_alphabet(kind,bits)
    if bits==8:return original_alphabet("int",8)
    if bits==2:return original_alphabet("unit_3",2)
    units=[-4,-2,-1,1,2,4]
    exponents=(2**bits-1)//6
    return np.array(sorted({0,*(u*3**e for e in range(exponents) for u in units)}),dtype=np.float64)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--size",choices=["135M","360M"],required=True)
    p.add_argument("--after",nargs="+",type=Path,required=True);a=p.parse_args()
    while any(not (d/"completed.json").exists() for d in a.after):time.sleep(15)
    core.alphabet=extended_alphabet
    assert np.array_equal(extended_alphabet("unit_wide_3",3),original_alphabet("unit_2",3))
    # Binary unit precision3 with exponents0..2 has the same decoded4-bit set.
    binary=np.array(sorted({0,*(u*2**e for e in range(3) for u in [-3,-1,1,3])}),dtype=np.float64)
    assert np.array_equal(extended_alphabet("unit_wide_3",4),binary)
    a.output.mkdir(parents=True,exist_ok=False)
    freeze=study.freeze(a.output);freeze["additional_sources"]={str(Path(__file__).relative_to(ROOT)):study.sha(__file__),
        "research/low-precision-mantissa-20261019.md":study.sha(ROOT/"research/low-precision-mantissa-20261019.md")}
    study.write(a.output/"freeze.json",freeze)
    meta=study.read if hasattr(study,"read") else None
    record=json.loads((a.inputs/"manifest.json").read_text())
    assert study.sha(a.inputs/"tokens.npz")==record["tokens_sha256"]
    with np.load(a.inputs/"tokens.npz",allow_pickle=False) as z:
        tokens={k:torch.from_numpy(z[k].copy()) for k in z.files}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    teacher=study.load_model(a.size).eval()
    for q in teacher.parameters():q.requires_grad_(False)
    reference=study.evaluate(teacher,tokens,["development","wikitext","pg19"])
    study.write(a.output/(a.size+"-reference.json"),{"metrics":reference,"state_sha256":study.state_dict_hash(teacher)})
    cache=[]
    with torch.inference_mode():
        for i in range(0,128,2):
            cache.append(teacher(tokens["train"][i:i+2].to("cuda"),use_cache=False).logits[:,:-1].cpu())
    cache=torch.cat(cache);del teacher;gc.collect();torch.cuda.empty_cache()
    regimes=["w4_a4","w4_a8","mix24_a4"] if a.size=="135M" else ["w4_a8"]
    study.write(a.output/"cases.json",{"requested":[{"regime":r,"family":"unit_wide_3","canonical":r+"_unit_wide_3"} for r in regimes]})
    for regime in regimes:
        for seed in study.SEEDS:
            try:study.train_one(cache,tokens,a.size,regime,"unit_wide_3",seed,a.output/a.size/(regime+"_unit_wide_3")/f"seed-{seed}")
            except Exception as e:print(json.dumps({"run_closed":regime,"seed":seed,"error_type":type(e).__name__,"message":str(e)}),flush=True)
    study.write(a.output/"completed.json",{"status":"additional_allocation_finished",
        "complete_fits":len(list(a.output.glob("*/**/completed.json"))),"closed_failed_fits":len(list(a.output.glob("*/**/failed.json")))})
    print("ADDITIONAL_ALLOCATION_COMPLETE",flush=True)

if __name__=="__main__":main()
