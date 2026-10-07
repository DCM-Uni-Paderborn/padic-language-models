import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from padic_lm.learned import DualAffine, folded_latents, gaussian_values, output_loss, fit_encoder


class LearnedBridgeTests(unittest.TestCase):
    def test_normalization_folding_and_role_specific_means(self):
        torch.manual_seed(310)
        q=torch.randn(2,3,12,5)+3;k=torch.randn_like(q)-4
        m=DualAffine(q,k,17);arrays=m.folded_arrays()
        self.assertFalse(torch.equal(m.query_mean,m.key_mean))
        for values,role in ((q,'query'),(k,'key')):
            torch.testing.assert_close(m.encode(values,role),folded_latents(values,arrays,role),rtol=2e-6,atol=2e-6)
        self.assertEqual(sum(a.nbytes for a in arrays.values()),2*3*5*2*4+2*3*2*4)
        clone=DualAffine(q,k,17)
        for name,value in m.state_dict().items():torch.testing.assert_close(value,clone.state_dict()[name],rtol=0,atol=0)

    def test_causal_gaussian_matches_manual_weights_and_future_invariance(self):
        q=torch.tensor([[[[1.,0.],[0.,1.]]]])
        k=torch.tensor([[[[0.,0.],[1.,0.],[0.,1.]]]])
        v=torch.tensor([[[[2.,3.],[4.,-1.],[99.,50.]]]])
        positions=torch.tensor([[0,1]])
        actual=gaussian_values(q,k,v,positions)
        expected=[]
        for i,p in enumerate(positions[0].tolist()):
            logits=-((q[0,0,i]-k[0,0,:p+1])**2).sum(-1)/2
            weights=torch.exp(logits);weights/=weights.sum()
            expected.append(sum(weights[j]*v[0,0,j] for j in range(p+1)))
        torch.testing.assert_close(actual[0,0],torch.stack(expected))
        k[:,:,2]=1e6;v[:,:,2]=-1e6
        torch.testing.assert_close(gaussian_values(q,k,v,positions),actual,rtol=0,atol=0)

    def test_output_energy_loss_and_gradients(self):
        q=torch.randn(2,2,4,2,requires_grad=True);k=torch.randn(2,2,7,2,requires_grad=True)
        v=torch.randn(2,2,7,3);positions=torch.tensor([[0,2,4,6],[1,3,5,6]])
        out=gaussian_values(q,k,v,positions);target=torch.randn_like(out)
        loss=output_loss(out,target)
        torch.testing.assert_close(loss,((out-target)**2).sum()/(target**2).sum())
        loss.backward()
        self.assertGreater(float(q.grad.abs().sum()),0);self.assertGreater(float(k.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(q.grad).all());self.assertTrue(torch.isfinite(k.grad).all())
        self.assertEqual(float(output_loss(torch.zeros(1),torch.zeros(1))),0)

    def test_deployment_block_shape_does_not_change_latents(self):
        q=torch.randn(3,2,12,5);k=torch.randn_like(q)
        arrays=DualAffine(q,k,29).folded_arrays()
        full=folded_latents(q,arrays,'query')
        for block in range(3):torch.testing.assert_close(folded_latents(q[block:block+1],arrays,'query')[0],full[block],rtol=0,atol=0)

    def test_invalid_causal_positions_rejected(self):
        q=torch.zeros(1,1,1,2);k=torch.zeros(1,1,2,2);v=torch.zeros(1,1,2,3)
        for positions in (torch.tensor([[-1]]),torch.tensor([[2]]),torch.tensor([[0.]])):
            with self.assertRaises(ValueError):gaussian_values(q,k,v,positions)

    def test_fixed_optimizer_schedule_learns_a_known_output_target(self):
        torch.manual_seed(315)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        old_threads=torch.get_num_threads()
        if device=='cpu':torch.set_num_threads(1)
        try:
            q=torch.randn(2,1,160,4,device=device);k=torch.randn_like(q);v=torch.randn(2,1,160,3,device=device)
            positions=torch.arange(160,device=device)[None].repeat(2,1)
            target=gaussian_values(q[...,:2],k[...,:2],v,positions).detach()
            callbacks=[]
            model,initial,history=fit_encoder(q,k,v,target,17,lambda:callbacks.append(1))
            self.assertEqual(len(callbacks),200)
            self.assertEqual(history['queries'].shape,(200,2,32))
            self.assertTrue(np.all(history['queries']>=128));self.assertTrue(np.all(history['queries']<160))
            for step in history['queries']:
                for indices in step:self.assertEqual(len(set(indices.tolist())),32)
            self.assertLess(float(history['loss'][-10:].mean()),float(history['loss'][:10].mean()))
            self.assertFalse(np.array_equal(initial['query_weight'],model.query_weight.detach().cpu().numpy()))
        finally:
            if device=='cpu':torch.set_num_threads(old_threads)


if __name__=='__main__':unittest.main()
