"""Independent code-alphabet, integer and gradient checks."""
import unittest
from fractions import Fraction
import numpy as np
import torch
from padic_lm.low_precision import alphabet,nearest,fake_quant,QuantLinear,signature,cases,pack_indices,unpack_indices

class LowPrecisionTests(unittest.TestCase):
    def test_residue_alphabets(self):
        for p in (2,3,5):
            for b in (2,3,4):
                if p>2**b:continue
                a=alphabet(f"residue_{p}",b);m=len(a)
                self.assertLessEqual(m,2**b)
                self.assertEqual(len({int(v)%m for v in a}),m)
                self.assertEqual(set(int(v) for v in a),{i-m if 2*i>=m else i for i in range(m)})

    def test_unit_valuations_are_unique(self):
        for p in (2,3,5):
            for b in (2,3,4):
                try:a=alphabet(f"unit_{p}",b)
                except ValueError:continue
                self.assertEqual(len(a),len(set(a)))
                self.assertLessEqual(len(a),2**b)
                self.assertTrue(np.array_equal(a,-a[::-1]))
                for value in a:
                    if not value:continue
                    q=int(value)
                    while q%p==0:q//=p
                    self.assertNotEqual(q%p,0)

    def test_lookup_independently(self):
        for family in ("int","int_full","residue_3","unit_2","unit_3","unit_5","fp4"):
            levels=alphabet(family,4)
            x=np.linspace(-1000,1000,1201)
            expected=np.array([min(levels,key=lambda v:abs(Fraction(float(t))-Fraction(float(v)))) for t in x])
            actual=nearest(torch.tensor(x),torch.tensor(levels)).numpy()
            np.testing.assert_array_equal(actual,expected)

    def test_packed_indices(self):
        rng=np.random.default_rng(17)
        for bits in (2,3,4):
            for count in (0,1,3,31,128,1001):
                x=rng.integers(0,2**bits,size=count,dtype=np.uint8)
                payload=pack_indices(x,bits)
                self.assertEqual(len(payload),(count*bits+7)//8)
                np.testing.assert_array_equal(unpack_indices(payload,count,bits),x)
        with self.assertRaises(ValueError):pack_indices(np.array([4]),2)

    def test_ste_gradient(self):
        x=torch.tensor([-.8,.2,.9],requires_grad=True)
        s=torch.tensor(.7,requires_grad=True)
        levels=torch.tensor([-1.,0.,1.])
        y=fake_quant(x,s,levels);y.sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        self.assertTrue(torch.isfinite(s.grad))
        self.assertGreater(abs(float(s.grad)),0.)

    def test_full_projection_matches_explicit_quantized_dot(self):
        torch.manual_seed(17)
        old=torch.nn.Linear(64,3,bias=False)
        q=QuantLinear(old,"unit_3",3,3)
        x=torch.randn(2,64)
        y=q(x).detach().numpy()
        w=old.weight.detach().numpy()
        levels=alphabet("unit_3",3)
        scales=q.log_scale.detach().exp().numpy()[:,0,0]
        act_scales=np.max(np.abs(x.numpy()),axis=1)/max(abs(levels))
        quant=lambda a:np.array([min(levels,key=lambda v:abs(v-t)) for t in a])
        expected=np.stack([[(quant(row/act_scales[i])*act_scales[i]) @
                            (quant(w[j]/scales[j])*scales[j]) for j in range(3)]
                           for i,row in enumerate(x.numpy())])
        np.testing.assert_allclose(y,expected,rtol=1e-5,atol=1e-6)

    def test_duplicate_controls_and_mixed_budget(self):
        self.assertEqual(signature("w2_a2","int"),signature("w2_a2","unit_3"))
        self.assertEqual(signature("w2_a8","int"),signature("w2_a8","residue_3"))
        req,alias=cases()
        self.assertLess(len(set(alias.values())),len(req))
        for tag in alias.values():self.assertIn(tag,alias)
        self.assertEqual(sum(2 if i%2==0 else 4 for i in range(30))/30,3)

if __name__=="__main__":unittest.main()
