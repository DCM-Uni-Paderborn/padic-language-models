import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from padic_lm.dense_masks import ranked_mask,padic_mask,flat_mask
from padic_lm.routing import select_causal_padic,PrefixTree
from padic_lm.flat_routing import FlatKeyCodebook


class DenseMaskTests(unittest.TestCase):
    def test_product_prime_masks_match_independent_tries_at_all_positions(self):
        rng=np.random.default_rng(407)
        for prime,digits in ((2,(4,4)),(3,(3,2)),(5,(2,1))):
            q=np.stack([rng.integers(prime**d,size=(2,3,57)) for d in digits],axis=-1)
            k=np.stack([rng.integers(prime**d,size=(2,3,57)) for d in digits],axis=-1)
            actual=padic_mask(q,k,digits=digits,prime=prime,budget=13,recent_window=4)
            for b in range(2):
                for h in range(3):
                    tree=PrefixTree(2,digits,prime=prime)
                    for pos in range(57):
                        tree.append(k[b,h,pos])
                        selected=np.flatnonzero(actual[b,h,pos])
                        np.testing.assert_array_equal(selected,tree.select(q[b,h,pos],13,4))
                        np.testing.assert_array_equal(selected,select_causal_padic(q[b,h,pos],k[b,h],pos,13,digits=digits,prime=prime,recent_window=4))

    def test_stable_real_ties_recent_priority_and_future_invariance(self):
        scores=np.zeros((1,1,9,9))
        actual=ranked_mask(scores,4,2)
        for pos in range(9):np.testing.assert_array_equal(np.flatnonzero(actual[0,0,pos]),np.arange(max(0,pos-3),pos+1))
        scores[0,0,8,:7]=np.arange(7)[::-1]
        np.testing.assert_array_equal(np.flatnonzero(ranked_mask(scores,4,2)[0,0,8]),[0,1,7,8])
        changed=scores.copy();changed[...,np.triu(np.ones((9,9),dtype=bool),1)]=1e9
        np.testing.assert_array_equal(ranked_mask(changed,4,2),ranked_mask(scores,4,2))

    def test_flat_masks_match_scalar_with_restored_origin(self):
        rng=np.random.default_rng(411);keys=rng.normal(size=(2,3,57,2));q=rng.normal(size=keys.shape)
        center=rng.normal(size=(3,2));projection=np.repeat(np.eye(2)[None],3,axis=0)
        book=FlatKeyCodebook.fit(keys[:1],center=center,projection=projection,max_centroids=13)
        codes=book.encode_keys(keys);projected=book.project_queries(q)
        for score in ('euclidean','dot_product'):
            masks=flat_mask(projected,codes,book.centroids,book.centroid_counts,score_kind=score,offset=center,budget=13,recent_window=4)
            for b in range(2):
                for h in range(3):
                    for pos in range(57):
                        expected=book.select_causal(projected[b,h,pos],codes[b,h],pos,13,4,head=h,score_kind=score)
                        np.testing.assert_array_equal(np.flatnonzero(masks[b,h,pos]),expected)

    def test_full_prefix_boundary_and_constant_codes(self):
        codes=np.zeros((1,2,160,2),dtype=np.uint16)
        masks=padic_mask(codes,codes,digits=(4,4),prime=2)
        for pos in (0,127,128,159):
            np.testing.assert_array_equal(masks[0,:,pos].sum(-1),min(pos+1,128))
            np.testing.assert_array_equal(np.flatnonzero(masks[0,0,pos]),np.arange(max(0,pos-127),pos+1))


if __name__=='__main__':unittest.main()
