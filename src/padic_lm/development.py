"""Frozen development sampling and descriptive accounting; no model fitting."""
import numpy as np
from .dense_masks import ranked_mask
from .routing import angular_codes


def diagnostic_indices(windows):
    articles=list(dict.fromkeys(w['article_id'] for w in windows))
    if len(windows)!=64 or len(articles)!=58:
        raise ValueError('requires the frozen 64-window/58-article workload')
    cells=len(articles)*15
    extra=set(np.random.default_rng(7079).permutation(cells)[:4096-4*cells].tolist())
    result=[]
    for article_index,article in enumerate(articles):
        blocks=np.array([i for i,w in enumerate(windows) if w['article_id']==article])
        for head in range(15):
            rng=np.random.default_rng(np.random.SeedSequence([9917,article_index,head]))
            chosen=set()
            for quarter in range(4):
                pairs=[(int(b),p) for b in blocks for p in range(128+quarter*96,224+quarter*96)]
                chosen.add(pairs[int(rng.integers(len(pairs)))])
            if article_index*15+head in extra:
                quarter=int(rng.integers(4))
                pairs=[(int(b),p) for b in blocks for p in range(128+quarter*96,224+quarter*96) if (int(b),p) not in chosen]
                chosen.add(pairs[int(rng.integers(len(pairs)))])
            result.extend((b,head,p) for b,p in chosen)
    result=np.array(sorted(result),dtype=np.int64)
    if result.shape!=(4096,3) or len(np.unique(result,axis=0))!=4096 or len(np.unique(result[:,0]))!=64:
        raise AssertionError('fixed stratified diagnostic does not cover all windows')
    return result


def baseline_mask(length,heads,*,kind,budget=128,recent_window=8):
    if kind not in ('full','recency','sink_recency'):
        raise ValueError('unknown fixed baseline')
    if kind=='full':return np.broadcast_to(np.tri(length,dtype=bool),(1,heads,length,length)).copy()
    scores=np.broadcast_to(np.arange(length,dtype=np.float64),(1,heads,length,length)).copy()
    if kind=='sink_recency':scores[...,:min(4,length)]+=length
    return ranked_mask(scores,budget,recent_window)


def angular_mask(queries,keys,planes,*,budget=128,recent_window=8):
    q=np.packbits(angular_codes(queries,planes),axis=-1,bitorder='little')
    k=np.packbits(angular_codes(keys,planes),axis=-1,bitorder='little')
    if q.shape[-1]!=1 or planes.shape[-1]!=8:raise ValueError('fixed one-byte angular code required')
    count=np.array([int(x).bit_count() for x in range(256)],dtype=np.uint8)
    scores=8-count[q[...,0,None]^k[...,None,:,0]]
    return ranked_mask(scores,budget,recent_window),{'packed_query_codes':q,'packed_key_codes':k}


def group_unions(selected,*,budget=128,recent_window=8,groups=3):
    if selected.ndim!=4 or selected.dtype!=bool or selected.shape[1]%groups:
        raise ValueError('boolean head masks must form complete GQA groups')
    b,h,q,k=selected.shape
    union=selected.reshape(b,h//groups,groups,q,k).any(axis=2).sum(axis=-1)
    prefix=np.arange(1,q+1)
    lower=np.minimum(prefix,budget)
    upper=np.minimum(prefix,min(recent_window,budget)+groups*(budget-min(recent_window,budget)))
    if np.any(union<lower) or np.any(union>upper):raise AssertionError('GQA logical union bound violated')
    return union.astype(np.uint16)


def paired_quality(losses,reference,windows):
    losses,reference=np.asarray(losses,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if losses.shape!=reference.shape or losses.ndim!=2 or len(losses)!=len(windows) or losses.shape[1]<129:
        raise ValueError('aligned finite per-target window losses required')
    if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(reference)):raise ValueError('finite losses required')
    articles=list(dict.fromkeys(w['article_id'] for w in windows));records=[]
    for article in articles:
        indices=[i for i,w in enumerate(windows) if w['article_id']==article]
        a,b=losses[indices],reference[indices]
        records.append({'article_id':article,'windows':indices,'targets':int(a.size),
            'nll_sum':float(a.sum()),'reference_nll_sum':float(b.sum()),
            'mean_nll':float(a.mean()),'paired_delta_nll':float((a-b).mean())})
    return {'targets':int(losses.size),'mean_nll':float(losses.mean()),'perplexity':float(np.exp(losses.mean())),
        'paired_delta_nll':float((losses-reference).mean()),'perplexity_ratio':float(np.exp((losses-reference).mean())),
        'affected_targets':int(losses[:,128:].size),'affected_mean_nll':float(losses[:,128:].mean()),
        'affected_paired_delta_nll':float((losses[:,128:]-reference[:,128:]).mean()),
        'window_mean_nll':losses.mean(axis=1).tolist(),'window_paired_delta_nll':(losses-reference).mean(axis=1).tolist(),
        'article_records':records,'scope':'Descriptive development losses; articles/windows and training seeds are correlated'}
