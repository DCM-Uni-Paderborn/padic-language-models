"""Frozen modular answer adapter with genuine LLM next-token priors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from padic_lm.continuous_tree import AffineDisk, metrics, softmax, quotient_logits
from padic_lm.modular_adapter import features, fit_real

REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
STATE_HASH = "b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583"


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(modulus):
    rng = np.random.default_rng(20261004+modulus)
    used = set(); rows = {}
    for part, repetitions, lower, upper in (("train", 8, 0, 999), ("validation", 4, 0, 999),
            ("same_range", 4, 0, 999), ("large_numbers", 4, 100000, 999999)):
        rr = []
        for a0 in range(modulus):
            for b0 in range(modulus):
                for _ in range(repetitions):
                    while True:
                        a = int(rng.integers((lower-a0+modulus-1)//modulus, (upper-a0)//modulus+1))*modulus+a0
                        b = int(rng.integers((lower-b0+modulus-1)//modulus, (upper-b0)//modulus+1))*modulus+b0
                        if (a, b) not in used:
                            used.add((a, b)); break
                    rr.append(dict(a=a, b=b, y=(a+b)%modulus,
                            prompt=f"Compute ({a} + {b}) mod {modulus}.\nAnswer:"))
        rows[part] = rr
    rows["new_wording"] = [dict(a=r["a"], b=r["b"], y=r["y"],
        prompt=f"What is the remainder when the sum of {r['a']} and {r['b']} is divided by {modulus}?\nAnswer:")
        for r in rows["large_numbers"]]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execution", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); execution = json.loads(args.execution.read_text())
    for rel, pin in execution["source_sha256"].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == pin, rel
    assert json.loads((ROOT/"results/continuous-tree-pilot-spark-20261004/audit.json").read_text())["passed"]
    assert not args.output.exists(); args.output.mkdir(parents=True)
    (args.output/"sources").mkdir()
    for rel in execution["source_sha256"]:
        (args.output/"sources"/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    (args.output/"execution.json").write_bytes(args.execution.read_bytes())
    start = time.monotonic()
    def guard():
        if time.monotonic()-start > 1800:
            raise TimeoutError("fixed1800-second producer cap")
    torch.set_num_threads(8); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.mem_get_info()[0] < 8*1024**3:
        raise RuntimeError("less than8GiB available")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
    answer_ids = [tokenizer.encode(str(k), add_special_tokens=False) for k in range(9)]
    assert all(len(t) == 1 for t in answer_ids)
    answer_ids = [t[0] for t in answer_ids]
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=REVISION,
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
    assert state_hash(model) == STATE_HASH
    all_splits = {}; token_pins = {}
    for modulus in (4, 9):
        rows = prepare(modulus); write(args.output/f"mod{modulus}-data.json", rows)
        splits = {}; saved = {}
        for part, rr in rows.items():
            prior, tokens, masks, max_other, logsum_other = [], [], [], [], []
            for pos in range(0, len(rr), 32):
                guard()
                encoded = tokenizer([r["prompt"] for r in rr[pos:pos+32]], padding="max_length", max_length=80,
                                    truncation=False, return_tensors="pt")
                assert encoded["input_ids"].shape[1] <= 80
                ids, mask = encoded["input_ids"].to("cuda"), encoded["attention_mask"].to("cuda")
                with torch.inference_mode():
                    z = model(ids, attention_mask=mask, use_cache=False).logits.float()
                    z = z[torch.arange(len(ids), device="cuda"), mask.sum(1)-1]
                selected = torch.tensor(answer_ids[:modulus], device="cuda")
                prior.append(z[:, selected].cpu().numpy().astype(np.float64))
                z = z.clone()  # Normal tensor for the diagnostic mask outside inference_mode.
                z[:, selected] = -torch.inf
                max_other.append(z.max(1).values.cpu().numpy())
                logsum_other.append(torch.logsumexp(z, 1).cpu().numpy())
                tokens.append(ids.cpu().numpy()); masks.append(mask.cpu().numpy())
            xx = np.array([[r["a"], r["b"], 1] for r in rr], dtype=np.int64)
            yy = np.array([r["y"] for r in rr], dtype=np.int64)
            bb = np.concatenate(prior)
            splits[part] = (xx, yy, bb)
            saved.update({part+"_x": xx, part+"_y": yy, part+"_base": bb,
                         part+"_tokens": np.concatenate(tokens), part+"_masks": np.concatenate(masks),
                         part+"_max_other": np.concatenate(max_other), part+"_logsum_other": np.concatenate(logsum_other)})
            print(json.dumps(dict(stage="llm-priors", modulus=modulus, split=part, rows=len(rr))), flush=True)
        np.savez_compressed(args.output/f"mod{modulus}-priors.npz", **saved)
        all_splits[modulus] = splits
    del model; torch.cuda.empty_cache()
    write(args.output/"environment.json", dict(torch=torch.__version__, numpy=np.__version__,
          cuda_device=torch.cuda.get_device_name(), model_revision=REVISION, model_state_sha256=STATE_HASH,
          answer_token_ids=answer_ids, maximum_input_tokens=80, reserved_gpu_exclusively=False))
    summaries = []
    for modulus, splits in all_splits.items():
        x, y, base = splits["train"]
        outputs = {"stock": {part: bb for part, (_, _, bb) in splits.items()}}
        for kind in ("raw", "fourier"):
            f, location, scale = features(x, modulus, kind)
            w, history = fit_real(f, base, y, steps=300)
            np.savez_compressed(args.output/f"mod{modulus}-{kind}-state.npz", w=w,
                                location=np.array([]) if location is None else location,
                                scale=np.array([]) if scale is None else scale,
                                train_losses=np.array(history))
            outputs[kind] = {part: bb+features(xx, modulus, kind, location, scale)[0]@w for part, (xx, yy, bb) in splits.items()}
        for p in (2, 3):
            for seed in (17, 29, 43):
                guard(); rng = np.random.default_rng(seed)
                models = [AffineDisk(p, 3, depth=6) for _ in range(modulus)]
                native = np.column_stack([-np.log(m.forward(x)[0]) for m in models])
                trace = []; moved = 0
                for sweep in range(300):
                    guard()
                    for k in rng.permutation(modulus):
                        k = int(k); model = models[k]
                        z = base+native; q = softmax(z)
                        g = q[:, k]-(y == k)
                        def objective():
                            zz = base+native.copy()
                            zz[:, k] = base[:, k]-np.log(model.forward(x)[0])
                            return metrics(zz, y)["nll"]
                        rec = model.update(x, "classification", rng, 100*p**2*(1-1/p), objective, derivative=g)
                        native[:, k] = -np.log(model.forward(x)[0])
                        moved += rec["moved"]
                        trace.append(dict(sweep=sweep, class_id=k, update=rec))
                    if sweep % 50 == 0:
                        print(json.dumps(dict(stage="native-fit", modulus=modulus, prime=p, seed=seed,
                            sweep=sweep, moved=moved, train=metrics(base+native, y))), flush=True)
                name = f"native-p{p}-{seed}"
                states = [m.state() for m in models]
                write(args.output/f"mod{modulus}-{name}-state.json", dict(states=states, trace=trace, accepted_updates=moved))
                outputs[name] = {part: bb+np.column_stack([-np.log(m.forward(xx)[0]) for m in models])
                                 for part, (xx, yy, bb) in splits.items()}
                outputs[f"quotient-p{p}-{seed}"] = {part: bb+quotient_logits(xx, states) for part, (xx, yy, bb) in splits.items()}
        for name, predictions in outputs.items():
            np.savez_compressed(args.output/f"mod{modulus}-{name}-logits.npz", **predictions)
            for part, zz in predictions.items():
                summaries.append(dict(modulus=modulus, method=name, split=part, **metrics(zz, splits[part][1])))
    write(args.output/"summary.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-start, results=summaries,
          primary_endpoint="constrained single digit conditional accuracy/NLL; not corpus token NLL"))


if __name__ == "__main__":
    main()
