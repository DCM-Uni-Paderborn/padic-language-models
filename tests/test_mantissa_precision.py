import sys,unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"experiments")]
from wide_precision import extended_alphabet
from padic_lm.low_precision import alphabet
class MantissaPrecisionTests(unittest.TestCase):
 def test_balanced_units_and_cardinality(self):
  units=[u for u in range(-4,5) if u%3]
  self.assertEqual(units,[-4,-2,-1,1,2,4])
  for bits in (3,4):
   exponents=(2**bits-1)//6
   expected=sorted({Fraction(0),*(Fraction(u)*3**e for u in units for e in range(exponents))})
   actual=extended_alphabet("unit_wide_3",bits)
   self.assertEqual([Fraction(float(v)) for v in actual],expected)
   self.assertEqual(len(actual),1+6*exponents)
   self.assertLessEqual(len(actual),2**bits)
 def test_cross_prime_real_alphabet_equalities(self):
  np.testing.assert_array_equal(extended_alphabet("unit_wide_3",2),alphabet("int",2))
  np.testing.assert_array_equal(extended_alphabet("unit_wide_3",3),alphabet("unit_2",3))
  binary=sorted({0,*(u*2**e for u in (-3,-1,1,3) for e in range(3))})
  np.testing.assert_array_equal(extended_alphabet("unit_wide_3",4),binary)
  fp4=alphabet("fp4",4);trimmed=fp4[np.abs(fp4)!=4]
  np.testing.assert_array_equal(extended_alphabet("unit_wide_3",4)/2,trimmed)
  valuation=sorted({0,*(u*2.**e for u in (-3,-1,1,3) for e in (-1,0,1)),-4,4})
  np.testing.assert_array_equal(fp4,valuation)
if __name__=="__main__":unittest.main()
