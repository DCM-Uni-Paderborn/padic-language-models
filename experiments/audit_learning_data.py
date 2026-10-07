"""Separate boundary/token verification of the corrected frozen learning data."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('input','prior','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();p=args.input.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('fresh audit directory required')
    c=json.loads((p/'completed.json').read_text());m=json.loads((p/'manifest.json').read_text())
    assert digest(p/'completed.json')=='033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454'
    assert digest(p/'completed.json')==(p/'completed.sha256').read_text().strip()
    for n,h in c['artifact_sha256'].items():assert digest(p/n)==h and (p/n).stat().st_size==c['artifact_bytes'][n]
    with np.load(p/'tokens.npz',allow_pickle=False) as t:tokens={n:t[n] for n in t.files}
    assert tokens['train'].shape==(16,512) and tokens['development'].shape==(64,512)
    tokenizer=AutoTokenizer.from_pretrained(m['model'],revision=m['tokenizer_revision'])
    rows={split:[r['text'] for r in load_dataset(m['dataset'],m['dataset_config'],revision=m['dataset_revision'],
                                               split=split,streaming=True)] for split in ('train','validation')}
    boundary_counts={};text_hashes={}
    for split in rows:
        boundaries=[]
        for i,row in enumerate(rows[split]):
            parts=row.strip().split()
            if len(parts)>=3 and parts[0]=='=' and parts[-1]=='=' and parts[1]!='=' and parts[-2]!='=':
                assert '\n' not in row.strip()
                boundaries.append(i)
        boundary_counts[split]=len(boundaries)
        inventory=m['inventory'][split]
        assert boundaries==[d['start_row'] for d in inventory]
        text_hashes[split]=set()
        for i,doc in enumerate(inventory):
            stop=boundaries[i+1] if i+1<len(boundaries) else len(rows[split])
            assert stop==doc['stop_row']
            text='\n\n'.join(rows[split][doc['start_row']:stop])
            assert hashlib.sha256(text.encode()).hexdigest()==doc['text_sha256']
            text_hashes[split].add(doc['text_sha256'])
    prior_ids=json.loads((args.prior/'input-token-ids.json').read_text())
    pieces=[];chars=0
    for row in rows['validation']:
        pieces.append(row);chars+=len(row)
        if chars>=4096*16:break
    encoded=tokenizer('\n\n'.join(pieces),add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids'][:4096]==prior_ids
    end=max(b for _,b in encoded['offset_mapping'][:4096])
    starts=[];cursor=0
    for i,row in enumerate(pieces):starts.append(cursor);cursor+=len(row)+2
    # Conservative separator assignment agrees with the recorded last row.
    last=next(i for i,row in enumerate(pieces) if end<=starts[i]+len(row)+2)
    if end>starts[last]+len(pieces[last]):last+=1
    assert last==m['prior_last_consumed_row']==67
    windows_checked=0;article_counts={}
    for role,split in (('train','train'),('development','validation')):
        ranges={};cache={}
        for index,w in enumerate(m['windows'][role]):
            startrow=w['start_row'];stoprow=w['stop_row'];sha=w['article_text_sha256']
            text='\n\n'.join(rows[split][startrow:stoprow])
            assert hashlib.sha256(text.encode()).hexdigest()==sha
            if role=='development':assert startrow>last and sha not in text_hashes['train']
            if w['article_id'] not in cache:cache[w['article_id']]=tokenizer(text,add_special_tokens=False)['input_ids']
            ids=cache[w['article_id']][w['token_start']:w['token_stop']]
            assert len(ids)==512 and np.array_equal(np.asarray(ids),tokens[role][index])
            assert hashlib.sha256(tokens[role][index].tobytes()).hexdigest()==w['token_sha256']
            selected=set(range(w['token_start'],w['token_stop']))
            assert not ranges.setdefault(w['article_id'],set()).intersection(selected)
            ranges[w['article_id']].update(selected);windows_checked+=1
        article_counts[role]=len(ranges)
    assert article_counts=={'train':16,'development':58}
    assert m['test_split_read'] is False
    for n,h in c['artifact_sha256'].items():assert digest(p/n)==h
    report={'status':'learning_data_audit_passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'data_completion_sha256':digest(p/'completed.json'),'audit_source_sha256':digest(__file__),
        'top_level_article_counts':boundary_counts,'selected_article_counts':article_counts,
        'windows_verified':windows_checked,'prior_consumed_row':last,'prior_prefix_reproduced_exactly':True,
        'article_boundaries_text_tokens_and_window_hashes_verified':True,'test_split_read':False,
        'scope':'Separate parsing/token algorithm by the same operator, before interpreting model development outputs'}
    out.mkdir(parents=True);(out/'audit_learning_data.py').write_bytes(Path(__file__).read_bytes())
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
