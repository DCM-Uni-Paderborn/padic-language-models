"""Adversarial finite-shell and physical-GQA semantics (CPU/GPU)."""
import unittest
import numpy as np
import torch
from padic_lm.native_decode import FiniteBridge, depth_lookup, selected_keys, fixed_keys, attention_component


class NativeDecodeTests(unittest.TestCase):
    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_shell_ranks_with_exact_ties_match_pairwise_definition(self):
        rng = np.random.default_rng(2083)
        for device in self.devices():
            for prime, digits, coarse in ((2, (4,4), 1), (3, (3,2), 1), (2, (4,4), 2)):
                k = rng.integers(prime ** sum(digits), size=(15,171), dtype=np.uint8)
                q = rng.integers(prime ** sum(digits), size=15, dtype=np.uint8)
                lut = depth_lookup(prime,digits,coarse,device)
                actual = selected_keys(torch.tensor(q,device=device),torch.tensor(k,device=device),lut).cpu().numpy()
                # Independent pairwise counts, not the production histogram.
                c0,c1=prime**digits[0],prime**digits[1]
                kc=np.stack((k.astype(int)%c0,k.astype(int)//c0),-1)
                qc=np.stack((q.astype(int)%c0,q.astype(int)//c0),-1)
                score=np.zeros((15,163),int)
                for j in range(1,max(digits)+1):
                    score+=np.all((kc[:,:163]-qc[:,None])%(prime**j)==0,axis=-1)
                score//=coarse
                ranks=np.array([[2*np.count_nonzero(row<s)+np.count_nonzero(row==s)-1 for s in row] for row in score])
                aggregate=ranks.reshape(5,3,163).sum(1)
                expected=np.array([np.sort(np.r_[np.lexsort((np.arange(163),row))[-120:],np.arange(163,171)]) for row in aggregate])
                np.testing.assert_array_equal(actual,expected)

    def test_all_ties_recent_empty_and_early_prefix(self):
        for device in self.devices():
            lut=depth_lookup(2,(4,4),1,device)
            for length in (1,8,127,128,129,2048):
                q=torch.zeros(15,dtype=torch.uint8,device=device)
                k=torch.zeros((15,length),dtype=torch.uint8,device=device)
                result=selected_keys(q,k,lut).cpu().numpy()
                np.testing.assert_array_equal(result,np.broadcast_to(np.arange(max(0,length-128),length),result.shape))
            only=selected_keys(q,k,lut,budget=8,recent=8).cpu().numpy()
            np.testing.assert_array_equal(only,np.broadcast_to(np.arange(2040,2048),(5,8)))

    def test_uniform_and_single_query_causality_gqa(self):
        for device in self.devices():
            length=171
            actual=fixed_keys(length,'uniform',device).cpu().numpy()
            expected=np.r_[((2*np.arange(120)+1)*163)//240,np.arange(163,171)]
            np.testing.assert_array_equal(actual,np.broadcast_to(expected,(5,128)))
            q=torch.zeros((15,1,64),device=device)
            k=torch.zeros((5,9,64),device=device)
            v=torch.arange(9,device=device).float()[None,:,None].expand(5,9,64)
            weight=torch.eye(960,device=device)
            projected,head=attention_component(q,k,v,weight)
            # Every causal-cache key participates. Upper-left masking would
            # incorrectly return zero for this one-query example.
            torch.testing.assert_close(head,torch.full_like(head,4.))
            torch.testing.assert_close(projected,torch.full_like(projected,4.))

    def test_bridge_duplicate_cutpoints_and_digit_reversal(self):
        for device in self.devices():
            encoder={}
            for role in ('query','key'):
                encoder[role+'_weight']=np.zeros((15,64,2),np.float32)
                encoder[role+'_weight'][:,0,0]=1
                encoder[role+'_weight'][:,1,1]=1
                encoder[role+'_bias']=np.zeros((15,2),np.float32)
            book={'center':np.zeros((15,2)), 'projection':np.broadcast_to(np.eye(2),(15,2,2)).copy(),
                  'thresholds':np.zeros((15,2,15))}
            bridge=FiniteBridge(encoder,book,2,(4,4),device)
            values=torch.zeros((15,3,64),dtype=torch.bfloat16,device=device)
            values[:,0,:2]=-1; values[:,2,:2]=1
            for role in ('query','key'):
                codes,_=bridge.encode(values,role)
                # side='right': all fifteen zero cutpoints belong below zero.
                np.testing.assert_array_equal(codes.cpu(),np.broadcast_to([0,255,255],(15,3)))


if __name__ == '__main__':unittest.main()
