"""Cached-mask readout checks without model downloads or development data."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
try:
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention,apply_rotary_pos_emb,repeat_kv
    import evaluate_qk_bridge as target
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Torch/Transformers required on experiment host')
class FixedReadoutTests(unittest.TestCase):
    def case(self,dtype):
        torch.manual_seed(743)
        config=LlamaConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=1)
        config._attn_implementation='eager'
        original=LlamaAttention(config,layer_idx=0).to(dtype=dtype).eval()
        hidden=torch.randn(1,19,32,dtype=dtype)
        position=(torch.ones(1,19,8,dtype=dtype),torch.zeros(1,19,8,dtype=dtype))
        causal=torch.tril(torch.ones(19,19,dtype=torch.bool))
        bias=torch.zeros(1,1,19,19,dtype=dtype).masked_fill(~causal,torch.finfo(dtype).min)
        shape=(1,19,-1,8)
        q=original.q_proj(hidden).view(shape).transpose(1,2);k=original.k_proj(hidden).view(shape).transpose(1,2)
        v=original.v_proj(hidden).view(shape).transpose(1,2);q,k=apply_rotary_pos_emb(q,k,*position)
        reference={'q':q,'k':k,'v':v,'scores':target.stock_eager_logits(q,k,bias,scaling=original.scaling,groups=2)}
        return original,hidden,position,bias,reference

    def test_full_budget_stock_fixture_and_sparse_native_selected_readout(self):
        for dtype in (torch.float32,torch.bfloat16):
            original,hidden,pos,bias,reference=self.case(dtype)
            with torch.inference_mode():
                stock=original(hidden_states=hidden,position_embeddings=pos,attention_mask=bias)[0]
                full=target.baseline_mask(19,4,kind='full')
                wrapper=target.FixedMaskAttention(original,full,reference).eval()
                actual=wrapper(hidden,pos,bias)[0]
                torch.testing.assert_close(actual,stock,rtol=0,atol=0)
                mask=target.baseline_mask(19,4,kind='sink_recency',budget=9,recent_window=4)
                wrapper=target.FixedMaskAttention(original,mask,reference).eval()
                actual=wrapper(hidden,pos,bias)[0]
                probability=torch.softmax(reference['scores'].masked_fill(~torch.tensor(mask),torch.finfo(dtype).min),-1,dtype=torch.float32).to(dtype)
                weighted=probability@repeat_kv(reference['v'],2)
                manual=original.o_proj(weighted.transpose(1,2).reshape(1,19,32))
                torch.testing.assert_close(actual,manual,rtol=0,atol=0)
                torch.testing.assert_close(wrapper.last['values'],weighted,rtol=0,atol=0)

    def test_replay_rejects_changed_input_native_scores_or_future_keys(self):
        original,hidden,pos,bias,reference=self.case(torch.float32)
        mask=target.baseline_mask(19,4,kind='recency',budget=9)
        wrapper=target.FixedMaskAttention(original,mask,reference).eval()
        changed=hidden.clone();changed[:,-1]+=1
        with self.assertRaisesRegex(AssertionError,'input differs'):wrapper(changed,pos,bias)
        corrupted={**reference,'scores':reference['scores']+1}
        wrapper=target.FixedMaskAttention(original,mask,corrupted).eval()
        with self.assertRaisesRegex(AssertionError,'scores changed'):wrapper(hidden,pos,bias)

    def test_replay_rejects_noncausal_mask_training_and_cache(self):
        original,hidden,pos,bias,reference=self.case(torch.float32)
        selected=target.baseline_mask(19,4,kind='full');selected[0,0,0,-1]=True
        with self.assertRaises(AssertionError):target.FixedMaskAttention(original,selected,reference)
        selected=target.baseline_mask(19,4,kind='full')
        wrapper=target.FixedMaskAttention(original,selected,reference).eval()
        with self.assertRaises(ValueError):wrapper(hidden,pos,bias,past_key_values=object())
        wrapper.train()
        with self.assertRaises(ValueError):wrapper(hidden,pos,bias)


if __name__=='__main__':unittest.main()
