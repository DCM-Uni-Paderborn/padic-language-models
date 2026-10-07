"""Post hoc exact learned-rule certificate; no refit or new observations."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def norm(q, p):
    q = Fraction(q)
    if not q:
        return 0.
    a, b, v = abs(q.numerator), q.denominator, 0
    while a % p == 0:
        a //= p; v += 1
    while b % p == 0:
        b //= p; v -= 1
    return float(Fraction(p)**-v)


def main():
    root = Path(sys.argv[1]); records = []; checked = 0; pins = {}
    for m, p in ((4, 2), (9, 3)):
        datafile = root/f"mod{m}-priors.npz"; data = np.load(datafile)
        pins[datafile.name] = hashlib.sha256(datafile.read_bytes()).hexdigest()
        for seed in (17, 29, 43):
            statefile = root/f"mod{m}-native-p{p}-{seed}-state.json"
            states = json.loads(statefile.read_text())["states"]
            pins[statefile.name] = hashlib.sha256(statefile.read_bytes()).hexdigest()
            units = []
            for k, state in enumerate(states):
                a, b, c = state["centers_numerator"]
                assert state["denominator"] == m and state["p"] == p
                u = a % m
                assert u % p != 0 and b % m == u and (c+u*k) % m == 0
                assert max(state["radii"]) <= 1
                units.append(u)
            predfile = root/f"mod{m}-native-p{p}-{seed}-logits.npz"
            preds = np.load(predfile); pins[predfile.name] = hashlib.sha256(predfile.read_bytes()).hexdigest()
            for part in ("train", "validation", "same_range", "large_numbers", "new_wording"):
                independent = []
                for x in data[part+"_x"]:
                    row = []
                    for s in states:
                        c = sum(int(a)*Fraction(int(b), m) for a, b in zip(x, s["centers_numerator"]))
                        r = max(norm(int(a), p)*b for a, b in zip(x, s["radii"]))
                        row.append(-math.log(max(norm(c, p), r)))
                    independent.append(row)
                z = np.array(independent); y = data[part+"_y"]
                np.testing.assert_allclose(z, preds[part]-data[part+"_base"], rtol=0., atol=1e-12)
                assert np.all(z.argmax(1) == y)
                mask = np.eye(m)[y].astype(bool)
                margins = z[np.arange(len(y)), y]-np.max(np.where(mask, -np.inf, z), axis=1)
                assert margins.min() >= math.log(p)-1e-12
                checked += z.size
                records.append(dict(modulus=m, prime=p, seed=seed, split=part, accuracy=1.,
                    minimum_correct_margin=float(margins.min()), coefficient_units_modulo_m=units))
    result = dict(passed=True, status="post_hoc_diagnostic_after_failed_primary_fusion_gate",
        independent_standalone_logit_elements=checked, records=records, inputs_sha256=pins,
        proof="For each class k, A=B=u_k mod m, C=-u_k*k mod m, p does not divide u_k, m=p^2, and all radii<=1. For integer operands and the correct residue k, the output center norm and radius are<=1; each incorrect class has center norm>=p. Hence standalone argmax is correct for every integer pair in the mathematical affine model, with margin>=ln p. This ordinary congruence identity is not a theorem of general LLM robustness. Finite int64 execution still has its stated input limits.",
        no_refit=True, primary_gate_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/"standalone-rule-certificate.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "inputs_sha256")}))


if __name__ == "__main__":
    main()
