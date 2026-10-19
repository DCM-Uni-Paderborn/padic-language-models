"""Reproduce the complete low-precision study from its supplied frozen inputs."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--download-models",action="store_true")
 a=p.parse_args()
 import torch
 if not torch.cuda.is_available():raise SystemExit("A supported CUDA-enabled PyTorch environment is required for the model fits.")
 data=ROOT/"results/low-precision-inputs-20261019"
 if a.download_models:
  from huggingface_hub import snapshot_download
  for size,revision in json.loads((data/"manifest.json").read_text())["models"].items():
   snapshot_download("HuggingFaceTB/SmolLM2-"+size,revision=revision)
 a.output.mkdir(parents=True,exist_ok=False)
 def call(file,*args):subprocess.run([sys.executable,str(ROOT/file),*map(str,args)],cwd=ROOT,check=True)
 call("experiments/audit_low_precision_inputs.py","--data",data,"--output",a.output/"input-audit.json")
 core=a.output/"core"
 call("experiments/low_precision_study.py","run","--inputs",data,"--output",core,"--sizes","135M","360M")
 runs=[core]
 for size in ("135M","360M"):
  wide=a.output/("unit-precision-"+size);cyclic=a.output/("low-digits-"+size)
  call("experiments/wide_precision.py","--inputs",data,"--output",wide,"--size",size,"--after",core)
  call("experiments/residue_qat.py","--inputs",data,"--output",cyclic,"--size",size)
  continued=a.output/("continued-bf16-"+size)
  call("experiments/bf16_continuation.py","--inputs",data,"--output",continued,"--size",size)
  runs.extend([wide,cyclic,continued])
 stats=a.output/"statistics"
 call("experiments/analyze_low_precision.py","--runs",*runs,"--output",stats)
 if not json.loads((stats/"summary.json").read_text())["matrix_complete"]:raise SystemExit("The full fixed matrix did not complete.")
 call("experiments/audit_low_precision_storage.py","--fits",stats/"compact-fits.json","--configs",ROOT/"inputs/low-precision-model-configs.json","--output",a.output/"storage-audit.json")
 call("experiments/summarize_low_precision.py","--statistics",stats,"--output",a.output/"tables")
if __name__=="__main__":main()
