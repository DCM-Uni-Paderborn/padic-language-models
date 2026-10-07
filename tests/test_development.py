import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from padic_lm.development import diagnostic_indices,baseline_mask,angular_mask,group_unions,paired_quality
from padic_lm.routing import angular_hyperplanes,angular_codes,select_causal_angular


class DevelopmentAccountingTests(unittest.TestCase):
    def test_diagnostic_covers_every_article_head_quarter_and_window_without_duplicates(self):
        windows=[{'article_id':f'article-{i}'} for i in range(58)]+[{'article_id':f'article-{i}'} for i in range(6)]
        rows=diagnostic_indices(windows)
        np.testing.assert_array_equal(rows,diagnostic_indices(windows))
        self.assertEqual(rows.shape,(4096,3));self.assertEqual(len(np.unique(rows,axis=0)),4096)
        self.assertEqual(set(rows[:,0]),set(range(64)))
        counts=[]
        for article in range(58):
            for head in range(15):
                cell=np.array([r for r in rows if windows[r[0]]['article_id']==f'article-{article}' and r[1]==head])
                self.assertEqual(set((cell[:,2]-128)//96),set(range(4)))
                counts.append(len(cell))
        self.assertEqual(counts.count(5),616);self.assertEqual(counts.count(4),254)

    def test_sink_and_recent_keys_share_budget_and_only_causal_positions(self):
        mask=baseline_mask(179,3,kind='sink_recency',budget=128)
        for position in range(179):
            expected=(list(range(position+1)) if position<128 else list(range(4))+list(range(position-123,position+1)))
            self.assertEqual(np.flatnonzero(mask[0,0,position]).tolist(),expected)
        self.assertFalse(np.any(np.triu(mask,1)))
        np.testing.assert_array_equal(mask[0,0],mask[0,2])

    def test_angular_vectorization_matches_origin_hyperplanes_and_scalar_rank(self):
        rng=np.random.default_rng(483)
        q,k=rng.normal(size=(2,3,37,8)),rng.normal(size=(2,3,37,8))
        planes=angular_hyperplanes(3,8,8,17)
        selected,codes=angular_mask(q,k,planes,budget=9,recent_window=4)
        qc,kc=angular_codes(q,planes),angular_codes(k,planes)
        np.testing.assert_array_equal(np.unpackbits(codes['packed_key_codes'],axis=-1,bitorder='little').astype(bool),kc)
        for b in range(2):
            for h in range(3):
                for p in range(37):
                    np.testing.assert_array_equal(np.flatnonzero(selected[b,h,p]),select_causal_angular(qc[b,h,p],kc[b,h],p,9,4))

    def test_logical_group_union_counts_shared_keys_once_and_respects_bound(self):
        selected=baseline_mask(160,3,kind='recency',budget=128)
        base=group_unions(selected)[0,0]
        np.testing.assert_array_equal(base,np.minimum(np.arange(1,161),128))
        selected[0,1,159,32]=False;selected[0,1,159,0]=True
        selected[0,2,159,33]=False;selected[0,2,159,1]=True
        self.assertEqual(group_unions(selected)[0,0,159],130)

    def test_paired_quality_weights_tokens_and_retains_article_dependence(self):
        windows=[{'article_id':'one'},{'article_id':'one'},{'article_id':'two'}]
        reference=np.ones((3,511));losses=np.array([np.full(511,2.),np.full(511,3.),np.full(511,6.)])
        result=paired_quality(losses,reference,windows)
        self.assertAlmostEqual(result['mean_nll'],11/3)
        self.assertAlmostEqual(result['paired_delta_nll'],8/3)
        self.assertEqual(result['article_records'][0]['targets'],1022)
        self.assertAlmostEqual(result['article_records'][0]['mean_nll'],2.5)
        self.assertEqual(result['affected_targets'],3*383)
        self.assertNotEqual(result['mean_nll'],np.mean([r['mean_nll'] for r in result['article_records']]))


if __name__=='__main__':unittest.main()
