import sys,unittest
from pathlib import Path
from fractions import Fraction
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"experiments"))
from bf16_continuation import BF16Linear
class BF16ContinuationTests(unittest.TestCase):
 def test_real_dot_of_bf16_forward_values(self):
  original=torch.nn.Linear(64,2,bias=False,dtype=torch.bfloat16)
  with torch.no_grad():original.weight.copy_(torch.arange(128).reshape(2,64).remainder(7).sub(3).to(torch.bfloat16)/64)
  module=BF16Linear(original)
  x=(torch.arange(64).remainder(5).sub(2).to(torch.bfloat16)/16).reshape(1,64)
  actual=module(x)
  expected=[]
  for row in module.weight.detach().to(torch.bfloat16):
   value=sum(Fraction(float(a))*Fraction(float(b)) for a,b in zip(row,x[0]))
   expected.append(float(value))
  torch.testing.assert_close(actual,torch.tensor([expected],dtype=torch.bfloat16),rtol=0,atol=0)
 def test_shadow_gradient_and_precision(self):
  original=torch.nn.Linear(64,2,bias=False,dtype=torch.bfloat16);module=BF16Linear(original)
  self.assertEqual(module.weight.dtype,torch.float32)
  y=module(torch.ones(2,64,dtype=torch.bfloat16));self.assertEqual(y.dtype,torch.bfloat16);y.sum().backward()
  torch.testing.assert_close(module.weight.grad,torch.full_like(module.weight,2),rtol=0,atol=0)
if __name__=="__main__":unittest.main()
