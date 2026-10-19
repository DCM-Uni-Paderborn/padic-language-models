"""Independently verify frozen continuation inputs and their test provenance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(data,root):
 manifest=json.loads((data/"manifest.json").read_text());freeze_path=root/"inputs/low-precision-training-freeze.json"
 if not freeze_path.exists():freeze_path=root/"results/low-precision-primary-0-20261019/freeze.json"
 freeze=json.loads(freeze_path.read_text())
 assert digest(data/"tokens.npz")==manifest["tokens_sha256"]
 with np.load(data/"tokens.npz",allow_pickle=False) as z:tokens={k:z[k].copy() for k in z.files}
 assert set(tokens)=={"train","development","wikitext","pg19"}
 expected={"train":128,"development":16,"wikitext":40,"pg19":40}
 seen=set();split_hashes={}
 for name,n in expected.items():
  a=tokens[name];assert a.shape==(n,256) and a.dtype.kind=="i" and np.all(a>=0)
  hashes=[hashlib.sha256(row.astype("<i8").tobytes()).hexdigest() for row in a]
  assert len(set(hashes))==n and not seen.intersection(hashes)
  seen.update(hashes);split_hashes[name]=hashes
 for name in ("train","development"):
  records=manifest["records"][name];assert len(records)==expected[name]
  assert [q["token_sha256"] for q in records]==split_hashes[name]
 texts={q["text_sha256"] for q in manifest["records"]["train"]}
 devtexts={q["text_sha256"] for q in manifest["records"]["development"]}
 assert len(texts)==128 and len(devtexts)==16 and not texts.intersection(devtexts)
 assert not (texts|devtexts).intersection(manifest["prior_adaptation_article_hashes_excluded"])
 provenance={}
 for name,old,field in [("wikitext","heldout-context-data-spark-20261004","tokens"),("pg19","pg19-context-data-spark-20261004","tokens4096")]:
  d=root/"results"/old
  m=json.loads((d/"manifest.json").read_text())
  assert m["split"]=="test"
  assert digest(d/"manifest.json")==manifest["records"][name]["original_manifest_sha256"]
  with np.load(d/"tokens.npz",allow_pickle=False) as z:
   key=field if field in z.files else "tokens2048"
   assert np.array_equal(tokens[name],z[key][:,:256])
  records=m["books" if name=="pg19" else "articles"]
  assert len(records)==40
  provenance[name]={"original_manifest":str((d/"manifest.json").relative_to(root)),
   "original_manifest_sha256":digest(d/"manifest.json"),"dataset_revision":m.get("dataset_revision",m.get("hf_dataset_revision")),
   "tokens_match_original_prefixes":True,"documents":records}
 source_matches={name:digest(root/name)==sha for name,sha in freeze["source_sha256"].items()}
 assert all(source_matches.values()),source_matches
 return {"status":"passed","tokens_sha256":digest(data/"tokens.npz"),"counts":expected,
  "all_224_prefixes_distinct":True,"training_and_development_article_hashes_disjoint":True,
  "earlier_adaptation_articles_excluded":True,"source_freeze_matches":source_matches,
  "test_provenance":provenance,"scope":"Exact input and provenance checks do not establish semantic independence or absence from model pretraining."}
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,required=True)
 p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();x=check(a.data,a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("x") as f:json.dump(x,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps({k:v for k,v in x.items() if k!="test_provenance"}))
