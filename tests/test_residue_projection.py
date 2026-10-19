import sys,unittest
from pathlib import Path
from fractions import Fraction
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from residue_qat import project,cyclic_alphabet,physical_signature
class ResidueProjectionTests(unittest.TestCase):
 def test_exact_integer_low_digits(self):
  for p in (2,3):
   for bits in (2,3,4):
    a=cyclic_alphabet(f"wrap_{p}",bits);m=len(a)
    values=[-2**30-3,-101,-m//2,0,m//2,99,2**30+7]
    x=torch.tensor(values,dtype=torch.float64)
    actual=project(x,torch.tensor(1.,dtype=torch.float64),torch.tensor(a)).numpy()
    expected=[(n+m//2)%m-m//2 for n in values]
    np.testing.assert_array_equal(actual,expected)
 def test_encoding_role_in_aliases(self):
  self.assertNotEqual(physical_signature('w4_a4','wrap_2'),physical_signature('w4_a4','int_full'))
  self.assertEqual(physical_signature('w2_a8','wrap_3'),physical_signature('w3_a8','wrap_3'))
 def test_gradient_and_periodicity(self):
  x=torch.tensor([-.4,.8,2.2],requires_grad=True)
  levels=torch.tensor([-1.,0.,1.]);s=torch.tensor(.7,requires_grad=True)
  y=project(x,s,levels);y.sum().backward()
  torch.testing.assert_close(x.grad,torch.ones_like(x))
  self.assertTrue(torch.isfinite(s.grad))
  torch.testing.assert_close(project(x.detach()+3*.7,s.detach(),levels),y.detach(),rtol=1e-5,atol=1e-6)
if __name__=="__main__":unittest.main()
