"""Vectorized CPU masks for numerical emulation; no sparse native kernel."""
import numpy as np
from .routing import digit_layout


def ranked_mask(scores, budget, recent_window=8):
    """Causal top score, mandatory recent keys, then recency for exact ties."""
    scores=np.asarray(scores)
    if scores.ndim!=4 or scores.shape[-1]!=scores.shape[-2] or not all(scores.shape):
        raise ValueError('square nonempty [batch,heads,queries,keys] scores required')
    if not isinstance(budget,int) or isinstance(budget,bool) or budget<1 or not isinstance(recent_window,int) or recent_window<0:
        raise ValueError('positive integer budget and nonnegative recent window required')
    length=scores.shape[-1];take=min(length,budget)
    position=np.arange(length)
    allowed=position[None,:]<=position[:,None]
    if not np.all(np.isfinite(scores[...,allowed])):raise ValueError('finite causal scores required')
    recent=min(recent_window,budget)
    forced=allowed&(position[None,:]>position[:,None]-recent)
    # Stable sorting preserves chronological key order for ties; take the end
    # to prioritize more recent keys. Future keys cannot enter a causal row.
    work=np.where(allowed,scores,-np.inf)
    work=np.where(forced,np.inf,work)
    rank=np.argsort(work,axis=-1,kind='stable')[...,-take:]
    selected=np.zeros(scores.shape,dtype=bool)
    np.put_along_axis(selected,rank,True,axis=-1)
    return selected&allowed


def padic_mask(query_codes,key_codes,*,digits,prime,budget=128,recent_window=8):
    """Product-ball ranks via temporary finite valuation lookup tables."""
    q,k=np.asarray(query_codes),np.asarray(key_codes)
    if q.shape!=k.shape or q.ndim!=4 or q.dtype.kind not in 'iu' or k.dtype.kind not in 'iu':
        raise ValueError('matching integer [batch,heads,length,coordinates] codes required')
    layout=digit_layout(digits,q.shape[-1],prime);depth=max(layout)
    ranks=np.full(q.shape[:-1]+(q.shape[2],),depth,dtype=np.uint8)
    for coordinate,count in enumerate(layout):
        alphabet=prime**count
        if np.any(q[...,coordinate]>=alphabet) or np.any(k[...,coordinate]>=alphabet) or np.any(q<0) or np.any(k<0):
            raise ValueError('canonical finite codes required')
        if alphabet>256:raise ValueError('bounded lookup requires at most256 bins/coordinate')
        indices=np.arange(alphabet,dtype=np.int64)
        difference=np.abs(indices[:,None]-indices[None,:])
        table=np.zeros((alphabet,alphabet),dtype=np.uint8)
        nonzero=difference!=0
        for level in range(1,count):table+=nonzero&(difference%(prime**level)==0)
        table[~nonzero]=depth
        coordinate_depth=table[q[...,coordinate,None],k[...,None,:,coordinate]]
        ranks=np.minimum(ranks,coordinate_depth)
    return ranked_mask(ranks,budget,recent_window)


def flat_mask(projected_queries,key_codes,centroids,counts,*,score_kind='euclidean',offset=None,budget=128,recent_window=8):
    q,codes=np.asarray(projected_queries),np.asarray(key_codes)
    if q.ndim!=4 or q.shape[-1]!=2 or codes.shape!=q.shape[:-1]:
        raise ValueError('matching projected query and key-code shapes required')
    scores=np.empty(q.shape[:-1]+(q.shape[2],),dtype=np.float64)
    for h in range(q.shape[1]):
        active=centroids[h,:int(counts[h])]
        if np.any(codes[:,h]<0) or np.any(codes[:,h]>=len(active)):raise ValueError('invalid centroid ID')
        for b in range(q.shape[0]):
            if score_kind=='euclidean':
                distances=q[b,h,:,None,:]-active[None,:,:]
                cluster_scores=-np.sum(distances*distances,axis=-1)
            elif score_kind=='dot_product' and offset is not None:
                cluster_scores=(q[b,h]+offset[h])@(active+offset[h]).T
            else:raise ValueError('unknown score or missing restored origin')
            scores[b,h]=cluster_scores[:,codes[b,h]]
    return ranked_mask(scores,budget,recent_window)
