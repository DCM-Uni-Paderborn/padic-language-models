"""Independent arithmetic, fit, data, statistical and LLM-fixture audit."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def norm(value, p):
    value = Fraction(value)
    if not value:
        return 0.
    a, b, v = abs(value.numerator), value.denominator, 0
    while a % p == 0:
        a //= p; v += 1
    while b % p == 0:
        b //= p; v -= 1
    return float(Fraction(p)**-v)


def score(x, state):
    p, den = state["p"], state["denominator"]
    centers = [Fraction(v, den) for v in state["centers_numerator"]]
    result = []
    for row in x:
        center = sum(int(v)*c for v, c in zip(row, centers))
        radius = max(norm(int(v), p)*r for v, r in zip(row, state["radii"]))
        result.append(-math.log(max(norm(center, p), radius)))
    return np.array(result)


def probabilities(logits):
    z = logits-np.max(logits, axis=1)[:, None]
    z = np.exp(z)
    return z/z.sum(axis=1)[:, None]


def measured(z, y):
    centered = z-np.max(z, axis=1, keepdims=True)
    loss = np.log(np.exp(centered).sum(axis=1))-centered[np.arange(len(y)), y]
    return np.argmax(z, axis=1) == y, loss


def real_features(x, m, kind, location, scale):
    if kind == "raw":
        return np.column_stack(((x[:, :2]-location)/scale, np.ones(len(x))))
    aa = 2*math.pi*(x[:, 0] % m)/m; bb = 2*math.pi*(x[:, 1] % m)/m
    return np.array([[math.cos(a)*math.cos(b), math.cos(a)*math.sin(b),
                       math.sin(a)*math.cos(b), math.sin(a)*math.sin(b), 1.] for a, b in zip(aa, bb)])


def replay_real(f, base, y):
    # Independent torch FP64 autograd/Adam replay of the NumPy analytic fit.
    tf = torch.tensor(f, dtype=torch.float64); tb = torch.tensor(base, dtype=torch.float64)
    ty = torch.tensor(y); w = torch.zeros((f.shape[1], base.shape[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=.03, betas=(.9, .999), eps=1e-8)
    for _ in range(300):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(tb+tf@w, ty)
        loss.backward(); opt.step()
    return w.detach().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("directory", type=Path)
    args = ap.parse_args(); out = args.directory
    start = time.monotonic(); torch.set_num_threads(4)
    def guard():
        if time.monotonic()-start > 1800:
            raise TimeoutError("fixed1800-second audit cap")
    execution = json.loads((out/"execution.json").read_text())
    for rel, pin in execution["source_sha256"].items():
        assert hashlib.sha256((out/"sources"/Path(rel).name).read_bytes()).hexdigest() == pin
    summary = json.loads((out/"summary.json").read_text())
    counts, gates, all_scores, decoded, full_vocab, fixtures = {}, [], {}, [], [], []
    env = json.loads((out/"environment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=env["model_revision"])
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
    assert all(tokenizer.decode([k]) == str(i) for i, k in enumerate(env["answer_token_ids"]))
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=env["model_revision"],
         torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    checked_elements, checked_moves = 0, 0
    for m in (4, 9):
        guard(); rows = json.loads((out/f"mod{m}-data.json").read_text())
        prior = np.load(out/f"mod{m}-priors.npz")
        counts[m] = {part: len(rr) for part, rr in rows.items()}
        seen = set()
        for part, rr in rows.items():
            n = m*m*(8 if part == "train" else 4)
            assert len(rr) == n
            pairs = [(r["a"], r["b"]) for r in rr]
            assert len(set(pairs)) == len(pairs)
            if part != "new_wording":
                assert not seen.intersection(pairs); seen.update(pairs)
            else:
                assert pairs == [(r["a"], r["b"]) for r in rows["large_numbers"]]
            lower, upper = (100000, 999999) if part in ("large_numbers", "new_wording") else (0, 999)
            assert all(lower <= a <= upper and lower <= b <= upper for a, b in pairs)
            targets = np.array([(a+b) % m for a, b in pairs])
            np.testing.assert_array_equal(targets, prior[part+"_y"])
            np.testing.assert_array_equal(np.array([[a, b, 1] for a, b in pairs]), prior[part+"_x"])
            residue_counts = np.zeros((m, m), dtype=int)
            for a, b in pairs:
                residue_counts[a % m, b % m] += 1
            assert np.all(residue_counts == (8 if part == "train" else 4))
            enc = tokenizer([r["prompt"] for r in rr], padding="max_length", max_length=80, truncation=False, return_tensors="np")
            np.testing.assert_array_equal(enc["input_ids"], prior[part+"_tokens"])
            np.testing.assert_array_equal(enc["attention_mask"], prior[part+"_masks"])
            ids = torch.tensor(prior[part+"_tokens"][:32], device="cuda")
            mask = torch.tensor(prior[part+"_masks"][:32], device="cuda")
            with torch.inference_mode():
                z = model(ids, attention_mask=mask, use_cache=False).logits.float()[torch.arange(len(ids), device="cuda"), mask.sum(1)-1]
            fresh = z[:, env["answer_token_ids"][:m]].cpu().numpy()
            np.testing.assert_allclose(fresh, prior[part+"_base"][:32], atol=0., rtol=0.)
            fixtures.append(dict(modulus=m, split=part, rows=len(ids), exact=True))
        for kind in ("raw", "fourier"):
            state = np.load(out/f"mod{m}-{kind}-state.npz")
            f = real_features(prior["train_x"], m, kind, state["location"], state["scale"])
            replay = replay_real(f, prior["train_base"], prior["train_y"])
            np.testing.assert_allclose(replay, state["w"], atol=1e-10, rtol=1e-10)
            saved = np.load(out/f"mod{m}-{kind}-logits.npz")
            for part in rows:
                xx = prior[part+"_x"]
                reference = prior[part+"_base"]+real_features(xx, m, kind, state["location"], state["scale"])@state["w"]
                np.testing.assert_allclose(reference, saved[part], atol=1e-10, rtol=1e-12)
        for p in (2, 3):
            for seed in (17, 29, 43):
                guard(); name = f"native-p{p}-{seed}"
                state = json.loads((out/f"mod{m}-{name}-state.json").read_text())
                preds = np.load(out/f"mod{m}-{name}-logits.npz")
                ordinary = np.load(out/f"mod{m}-quotient-p{p}-{seed}-logits.npz")
                current = [dict(p=p, denominator=p*p, depth=6, centers_numerator=[0, 0, 0], radii=[float(p*p)]*3) for _ in range(m)]
                native_train = np.full_like(prior["train_base"], -math.log(p*p))
                moves = 0
                for rec in state["trace"]:
                    k = rec["class_id"]; update = rec["update"]
                    if update["moved"]:
                        before = measured(prior["train_base"]+native_train, prior["train_y"])[1].mean()
                        nxt = update["state"]
                        for j, r in enumerate(nxt["radii"]):
                            expected = current[k]["radii"][j]+update["step"]*update["direction"][j]
                            assert abs(expected-r) <= 1e-10
                            assert p**-6*(1-1e-10) <= r <= p*p*(1+1e-10)
                        native_train[:, k] = score(prior["train_x"], nxt)
                        after = measured(prior["train_base"]+native_train, prior["train_y"])[1].mean()
                        assert abs(before-update["before"]) < 1e-11 and abs(after-update["after"]) < 1e-11
                        assert after <= before+1e-12
                        current[k] = nxt; moves += 1
                assert current == state["states"] and moves == state["accepted_updates"]
                checked_moves += moves
                for part in rows:
                    ref = prior[part+"_base"]+np.column_stack([score(prior[part+"_x"], s) for s in state["states"]])
                    np.testing.assert_allclose(ref, preds[part], atol=1e-12, rtol=0.)
                    np.testing.assert_allclose(ordinary[part], preds[part], atol=1e-12, rtol=0.)
                    checked_elements += ref.size
                if m == p*p:
                    for part in ("large_numbers", "new_wording"):
                        y = prior[part+"_y"]
                        acc, nll = measured(preds[part], y)
                        fz = np.load(out/f"mod{m}-fourier-logits.npz")[part]
                        fa, fn = measured(fz, y)
                        baseacc = measured(prior[part+"_base"], y)[0].mean()
                        rng = np.random.default_rng(20261004+m)
                        sample = rng.integers(0, len(y), size=(10000, len(y)))
                        da = acc.astype(float)-fa.astype(float); dn = nll-fn
                        aci = np.quantile(da[sample].mean(1), [.025, .975]).tolist()
                        nci = np.quantile(dn[sample].mean(1), [.025, .975]).tolist()
                        gate = bool(acc.mean() >= .95 and acc.mean()-baseacc >= .15 and moves > 0 and aci[0] >= -.01 and nci[1] <= .02)
                        gates.append(dict(modulus=m, seed=seed, split=part, accuracy=float(acc.mean()), stock_accuracy=float(baseacc),
                            fourier_accuracy=float(fa.mean()), delta_nll=float(dn.mean()), accuracy_difference_ci=aci, nll_difference_ci=nci, passed=gate))
        for result in [r for r in summary["results"] if r["modulus"] == m]:
            name, part = result["method"], result["split"]
            z = prior[part+"_base"] if name == "stock" else np.load(out/f"mod{m}-{name}-logits.npz")[part]
            y = prior[part+"_y"]; aa, ll = measured(z, y)
            assert abs(aa.mean()-result["accuracy"]) < 1e-12 and abs(ll.mean()-result["nll"]) < 1e-12
            pred = z.argmax(1)
            for i, k in enumerate(pred):
                decoded.append(dict(modulus=m, method=name, split=part, row=i, answer=str(int(k)), target=int(y[i])))
            zz = z-z.max(1)[:, None]
            logsum = np.log(np.exp(zz).sum(1))+z.max(1)
            full_loss = np.logaddexp(prior[part+"_logsum_other"], logsum)-z[np.arange(len(y)), y]
            valid_top = z.max(1) >= prior[part+"_max_other"]
            full_vocab.append(dict(modulus=m, method=name, split=part, full_vocabulary_answer_nll=float(full_loss.mean()),
                unrestricted_one_token_accuracy=float(np.mean(valid_top & aa)), answer_token_is_top_fraction=float(valid_top.mean())))
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name not in ("audit.json", "decoded-answers.json", "full-vocabulary-diagnostics.json")}
    result = dict(passed=True, progression_gate=all(g["passed"] for g in gates), gates=gates, counts=counts,
        independent_native_logit_elements=checked_elements, independently_verified_accepted_updates=checked_moves,
        real_fit_replays=4, fresh_llm_fixtures=fixtures, decoded_answer_count=len(decoded), sha256=pins,
        elapsed_seconds=time.monotonic()-start,
        running_auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        postprocessing_repair="Stable logsumexp loss from unchanged frozen logits; no fit rerun.")
    (out/"decoded-answers.json").write_text(json.dumps(decoded)+"\n")
    (out/"full-vocabulary-diagnostics.json").write_text(json.dumps(full_vocab, indent=2)+"\n")
    (out/"audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
