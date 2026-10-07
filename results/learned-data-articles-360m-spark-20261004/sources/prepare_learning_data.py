"""Freeze article-preserving train/development windows; never read test text."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import padic_lm.corpus as corpus

MODEL = "HuggingFaceTB/SmolLM2-360M"
REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError('fresh output required')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    if not tokenizer.is_fast:raise ValueError('offset mapping requires pinned fast tokenizer')
    rows={split:[entry['text'] for entry in load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',
               revision=DATASET_REVISION,split=split,streaming=True)] for split in ('train','validation')}
    prior_ids=json.loads((args.prior/'input-token-ids.json').read_text())
    prior_manifest=json.loads((args.prior/'manifest.json').read_text())
    if (len(prior_ids)!=4096 or prior_manifest['model_revision']!=REVISION
            or prior_manifest['dataset_revision']!=DATASET_REVISION):
        raise ValueError('prior workload differs from fixed pinned source')
    pieces=[];characters=0
    for row in rows['validation']:
        pieces.append(row);characters+=len(row)
        if characters>=4096*16:break
    encoded=tokenizer('\n\n'.join(pieces),add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids'][:4096]!=prior_ids:raise ValueError('cannot reproduce prior discovery prefix')
    character_end=max(end for _,end in encoded['offset_mapping'][:4096])
    excluded_through=corpus.last_consumed_row(pieces,int(character_end))
    tokenized={}
    inventory={}
    for split in ('train','validation'):
        docs=corpus.articles(rows[split])
        tokenized[split]=[(doc,tokenizer(doc.text,add_special_tokens=False)['input_ids']) for doc in docs]
        inventory[split]=[{'id':f'{split}-{doc.start_row:06d}-{doc.text_sha256[:16]}',
            'start_row':doc.start_row,'stop_row':doc.stop_row,'title':doc.title,'text_sha256':doc.text_sha256,
            'tokens':len(ids),'prior_overlap_excluded':split=='validation' and doc.start_row<=excluded_through}
            for doc,ids in tokenized[split]]
    # Exclude exact article duplicates across official train/validation roles.
    training_hashes={doc.text_sha256 for doc,_ in tokenized['train']}
    selections={
        'train':corpus.windows(tokenized['train'],count=16,length=512,max_per_article=1),
        'development':corpus.windows(tokenized['validation'],count=64,length=512,max_per_article=4,
                                   exclude_through_row=excluded_through,excluded_hashes=training_hashes)}
    ids={};records={}
    for role,selected in selections.items():
        split='train' if role=='train' else 'validation'
        ids[role]=np.asarray([tokens for _,_,_,tokens in selected],dtype=np.int64)
        records[role]=[{'window':i,'article_id':f'{split}-{doc.start_row:06d}-{doc.text_sha256[:16]}',
            'article_title':doc.title,'article_text_sha256':doc.text_sha256,'start_row':doc.start_row,
            'stop_row':doc.stop_row,'token_start':start,'token_stop':stop,
            'token_sha256':hashlib.sha256(np.asarray(tokens,dtype=np.int64).tobytes()).hexdigest()}
            for i,(doc,start,stop,tokens) in enumerate(selected)]
    sources={p.name:p.read_bytes() for p in (Path(__file__),Path(corpus.__file__))}
    output.mkdir(parents=True);(output/'sources').mkdir()
    for name,content in sources.items():(output/'sources'/name).write_bytes(content)
    np.savez_compressed(output/'tokens.npz',**ids)
    manifest={'status':'learning_data_frozen','created_utc':datetime.now(timezone.utc).isoformat(),
        'model':MODEL,'model_revision':REVISION,'tokenizer_revision':REVISION,
        'dataset':'Salesforce/wikitext','dataset_config':'wikitext-2-raw-v1','dataset_revision':DATASET_REVISION,
        'sequence_length':512,'training_tokens':8192,'development_input_tokens':32768,
        'development_scored_targets':64*511,'train_articles':len({r['article_id'] for r in records['train']}),
        'development_articles':len({r['article_id'] for r in records['development']}),
        'prior_input_sha256':digest(args.prior/'input-token-ids.json'),
        'prior_manifest_sha256':digest(args.prior/'manifest.json'),
        'prior_prefix_reproduced_exactly':True,'prior_last_consumed_row':excluded_through,
        'excluded_article_rule':'exclude each validation article starting at or before the last consumed prior row',
        'selection_rule':'source-order round robin; 1 train window/article, at most4 dev; nonoverlapping512-token windows; no cross-article concatenation',
        'test_split_read':False,'exact_train_validation_article_hash_duplicates_excluded':True,
        'inventory':inventory,'windows':records,'tokens_file_sha256':digest(output/'tokens.npz'),
        'source_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in sources.items()},
        'scope':'Frozen training/development data; reused official validation split, new article units after prior prefix; no quality outputs or confirmation'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    artifacts=sorted(p for p in output.rglob('*') if p.is_file())
    completed={'status':manifest['status'],'artifact_sha256':{str(p.relative_to(output)):digest(p) for p in artifacts},
               'artifact_bytes':{str(p.relative_to(output)):p.stat().st_size for p in artifacts}}
    (output/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    (output/'completed.sha256').write_text(digest(output/'completed.json')+'\n')
    print(json.dumps({key:manifest[key] for key in ('status','training_tokens','development_input_tokens',
        'development_scored_targets','train_articles','development_articles','prior_last_consumed_row')},indent=2))


if __name__=='__main__':main()
