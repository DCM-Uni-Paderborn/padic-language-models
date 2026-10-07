"""Frozen semantic-prefix experiment on CLINC150 with real LLM features."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from padic_lm.structural_intent import taxonomy, paths, probabilities, metrics, objective_gradient

REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
STATE_HASH = "b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583"
DATA_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
DATA_BLOB = "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
SEEDS = (17, 29, 43)
METHODS = ("semantic_prefix", "shuffled_prefix", "flat_joint", "direct_domain")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def prepare(raw, domains):
    names, labels, group, codes, held = taxonomy(domains)
    label_id = {label: i for i, label in enumerate(labels)}
    domain_id = {label: d for d, name in enumerate(names) for label in domains[name]}
    train_seen_text = {normalized(t) for t, _ in raw["train"]}
    val_seen_text = {normalized(t) for t, _ in raw["val"]}
    split_rows = {k: [] for k in ("train", "development", "seen_test", "unseen_test")}
    for label in labels:
        candidates = [(i, t) for i, (t, lab) in enumerate(raw["train"]) if lab == label]
        candidates.sort(key=lambda v: (hashlib.sha256(
            ("structural-intent-train-v1|"+label+"|"+v[1]).encode()).hexdigest(), v[0]))
        seen, chosen = set(), []
        for i, text in candidates:
            if normalized(text) not in seen:
                chosen.append((i, text)); seen.add(normalized(text))
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise ValueError("fewer than20 unique adaptation utterances for "+label)
        for i, text in chosen:
            split_rows["train"].append(dict(source_split="train", index=i, text=text,
                label=label, y=label_id[label], domain=domain_id[label]))
    exclusions = {"development": 0, "seen_test": 0, "unseen_test": 0}
    for official, target in (("val", "development"), ("test", None)):
        used = set()
        for i, (text, label) in enumerate(raw[official]):
            if label not in domain_id or (official == "val" and label in held):
                continue
            part = target or ("unseen_test" if label in held else "seen_test")
            norm = normalized(text)
            if norm in train_seen_text or norm in used or (official == "test" and norm in val_seen_text):
                exclusions[part] += 1
                continue
            used.add(norm)
            split_rows[part].append(dict(source_split=official, index=i, text=text,
                label=label, y=label_id.get(label, -1), domain=domain_id[label]))
    return names, labels, group, codes, held, split_rows, exclusions


def torch_loss(x, y, d, w, b, kind, left, right, membership):
    z = x@w.T+b
    if kind == "tree":
        logp = -torch.nn.functional.softplus(z)@left-torch.nn.functional.softplus(-z)@right
    else:
        logp = torch.log_softmax(torch.cat((z, torch.zeros((len(z), 1), device=z.device, dtype=z.dtype)), 1), 1)
    if kind == "domain":
        loss = -logp[torch.arange(len(y), device=x.device), d].mean()
    else:
        q = logp.exp()@membership
        loss = (-logp[torch.arange(len(y), device=x.device), y]-
                torch.log(q[torch.arange(len(y), device=x.device), d])).mean()
    return loss+.0005*(w*w).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists():
        raise FileExistsError("requires a fresh directory, no automatic restart")
    execution = json.loads(args.execution.read_text())
    for rel, pin in execution["source_sha256"].items():
        if digest(ROOT/rel) != pin:
            raise ValueError("frozen source differs: "+rel)
    if digest(ROOT/execution["protocol"]) != execution["protocol_sha256"]:
        raise ValueError("frozen protocol differs")
    out.mkdir(parents=True); (out/"sources").mkdir()
    for rel in execution["source_sha256"]:
        (out/"sources"/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    (out/"prospective_protocol.md").write_bytes((ROOT/execution["protocol"]).read_bytes())
    (out/"execution.json").write_bytes(args.execution.read_bytes())
    start = time.monotonic()
    def guard():
        if time.monotonic()-start > 1800:
            raise TimeoutError("prospective1800-second producer allocation exhausted")
    try:
        torch.set_num_threads(8); torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.mem_get_info()[0] < 8*1024**3:
            raise RuntimeError("less than8GiB currently available; no foreign process is changed")
        raw_bytes = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/clinc/oos-eval/{DATA_COMMIT}/data/data_full.json", timeout=60).read()
        blob = hashlib.sha1(b"blob "+str(len(raw_bytes)).encode()+b"\0"+raw_bytes).hexdigest()
        if blob != DATA_BLOB:
            raise ValueError("dataset Git blob differs")
        (out/"data_full.json").write_bytes(raw_bytes)
        domains = json.loads((ROOT/"research/clinc-domains.json").read_text())
        names, labels, group, codes, held, rows, exclusions = prepare(json.loads(raw_bytes), domains)
        assert len(labels) == 140 and len(held) == 10
        all_rows = [r for part in rows.values() for r in part]
        data_manifest = dict(names=names, labels=labels, label_domains=group.tolist(),
            semantic_codes=codes.tolist(), withheld_intents=held, rows=rows, exclusions=exclusions,
            source_commit=DATA_COMMIT, source_blob_sha1=blob, source_sha256=digest(out/"data_full.json"))
        write_json(out/"data-manifest.json", data_manifest)
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=REVISION)
        tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
        text = [r["text"] for r in all_rows]
        batch = tokenizer(text, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        original_lengths = [len(v) for v in tokenizer(text, truncation=False)["input_ids"]]
        ids = batch["input_ids"].numpy(); masks = batch["attention_mask"].numpy()
        np.savez_compressed(out/"tokens.npz", ids=ids, masks=masks)
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M", revision=REVISION,
                    torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
        if state_hash(model) != STATE_HASH:
            raise ValueError("pinned BF16 model state differs")
        hidden = []
        with torch.inference_mode():
            for i in range(0, len(ids), 32):
                guard()
                t = torch.as_tensor(ids[i:i+32], device="cuda")
                mask = torch.as_tensor(masks[i:i+32], device="cuda")
                h = model.model(t, attention_mask=mask, use_cache=False).last_hidden_state.float()
                pooled = (h*mask[:, :, None]).sum(1)/mask.sum(1)[:, None]
                hidden.append(pooled.cpu().numpy().copy())
                if i % 512 == 0:
                    print(json.dumps(dict(stage="features", rows=min(i+32, len(ids)), total=len(ids))), flush=True)
        hidden = np.concatenate(hidden)
        np.savez_compressed(out/"llm-features.npz", hidden=hidden)
        del model; torch.cuda.empty_cache(); guard()
        ntrain = len(rows["train"])
        center = hidden[:ntrain].astype(np.float64).mean(0)
        centered = hidden[:ntrain].astype(np.float64)-center
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered/ntrain)
        projection = eigenvectors[:, -64:][:, ::-1].copy()
        projection *= np.where(projection[np.argmax(np.abs(projection), axis=0), np.arange(64)] < 0, -1., 1.)
        scale = np.sqrt(np.maximum(eigenvalues[-64:][::-1], 0))+1e-6
        xall = (hidden.astype(np.float64)-center)@projection/scale
        np.savez_compressed(out/"encoder.npz", center=center, projection=projection, scale=scale,
                            eigenvalues=eigenvalues[-64:][::-1])
        splits = {}; offset = 0
        for part, rr in rows.items():
            n = len(rr)
            splits[part] = (xall[offset:offset+n], np.array([r["y"] for r in rr]),
                           np.array([r["domain"] for r in rr]))
            offset += n
        xtrain, ytrain, dtrain = splits["train"]
        tx = torch.as_tensor(xtrain, device="cuda", dtype=torch.float64)
        ty = torch.as_tensor(ytrain, device="cuda"); td = torch.as_tensor(dtrain, device="cuda")
        tm = torch.as_tensor(np.eye(10)[group], device="cuda", dtype=torch.float64)
        summaries, certificates = [], []
        for seed in SEEDS:
            initial = np.random.default_rng(seed).normal(scale=.01, size=(139, 64))
            shuffled = codes[np.random.default_rng(seed+1000).permutation(140)]
            for method in METHODS:
                guard()
                kind = "tree" if method.endswith("prefix") else "domain" if method == "direct_domain" else "flat"
                address = shuffled if method == "shuffled_prefix" else codes
                nodes, left, right = paths(address)
                heads = 9 if kind == "domain" else 139
                w0, b0 = initial[:heads].copy(), np.zeros(heads)
                w = torch.tensor(w0, device="cuda", requires_grad=True)
                b = torch.tensor(b0, device="cuda", requires_grad=True)
                tl = torch.as_tensor(left, device="cuda"); tr = torch.as_tensor(right, device="cuda")
                first = torch_loss(tx, ty, td, w, b, kind, tl, tr, tm)
                first.backward()
                checked, gw, gb = objective_gradient(xtrain, ytrain, group, w0, b0, kind, left, right)
                if abs(checked-float(first.detach())) > 1e-10 or np.max(np.abs(gw-w.grad.cpu().numpy())) > 1e-10 or np.max(np.abs(gb-b.grad.cpu().numpy())) > 1e-10:
                    raise ArithmeticError("real objective/autograd mismatch")
                optimizer = torch.optim.Adam([w, b], lr=.03, betas=(.9, .999), eps=1e-8, foreach=False)
                history = []
                for step in range(401):
                    guard(); optimizer.zero_grad()
                    loss = torch_loss(tx, ty, td, w, b, kind, tl, tr, tm)
                    if step % 25 == 0:
                        history.append(dict(step=step, objective=float(loss.detach())))
                    if step < 400:
                        loss.backward(); optimizer.step()
                wf, bf = w.detach().cpu().numpy().copy(), b.detach().cpu().numpy().copy()
                record = dict(seed=seed, method=method, kind=kind, training_history=history,
                    coefficients_changed=bool(np.any(wf != w0)), learned_parameter_count=heads*65,
                    initial_training_objective=history[0]["objective"], final_training_objective=history[-1]["objective"], splits={})
                arrays = dict(initial_w=w0, initial_b=b0, w=wf, b=bf, codes=address,
                              nodes=nodes, left=left, right=right)
                for part, (x, y, d) in splits.items():
                    p = probabilities(x, wf, bf, kind, left, right)
                    record["splits"][part] = metrics(p, y, d, group, kind)
                    arrays[part+"_probabilities"] = p
                np.savez_compressed(out/f"seed-{seed}-{method}.npz", **arrays)
                summaries.append(record)
                if method == "semantic_prefix":
                    xp, _, dp = splits["unseen_test"]
                    pp = arrays["unseen_test_probabilities"]
                    qp = pp@np.eye(10)[group]
                    z = xp@wf.T+bf
                    for i in range(min(20, len(xp))):
                        domain = int(qp[i].argmax()); leaf = int(np.flatnonzero(group == domain)[0])
                        terms = []
                        for j, (k, prefix) in enumerate(nodes):
                            if k < 4 and (left[j, leaf] or right[j, leaf]):
                                logterm = -np.logaddexp(0, -z[i, j] if right[j, leaf] else z[i, j])
                                terms.append(dict(depth=int(k), prefix=int(prefix), bit=int(right[j, leaf]), log_probability=float(logterm)))
                        delta = abs(sum(v["log_probability"] for v in terms)-np.log(qp[i, domain]))
                        if delta > 1e-10:
                            raise ArithmeticError("domain path certificate differs")
                        certificates.append(dict(seed=seed, unseen_test_row=i, predicted_domain=domain,
                            true_domain=int(dp[i]), terms=terms, max_error=float(delta)))
                print(json.dumps(dict(stage="fit", seed=seed, method=method, metrics=record["splits"]["unseen_test"])), flush=True)
        write_json(out/"results.json", summaries); write_json(out/"path-certificates.json", certificates)
        write_json(out/"config.json", dict(model="HuggingFaceTB/SmolLM2-360M", revision=REVISION,
            model_state_sha256=STATE_HASH, dtype="BF16", feature="mean final hidden state over nonpad tokens",
            hidden_dimension=960, probe_dimension=64, max_tokens=64, batch_size=32,
            truncated_utterances=int(sum(v > 64 for v in original_lengths)),
            total_real_input_tokens=int(masks.sum()), actual_rows={k:len(v) for k,v in rows.items()},
            steps=400, learning_rate=.03, l2_coefficient=.001, seeds=list(SEEDS), methods=list(METHODS),
            p=2, k=8, domain_depth=4, observed_classes=140, withheld_classes=10,
            packages={p:importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
            hardware=torch.cuda.get_device_name(), nvml_available=False,
            nvml_preflight="Driver/library version mismatch; CUDA API and small FP64 arithmetic pass; no device-cost claim",
            scope="Few-shot frozen-LLM probe, not generative LM quality, p-adic gradient or unique arithmetic advantage"))
        guard()
        if not all(v["coefficients_changed"] and v["final_training_objective"] < v["initial_training_objective"] for v in summaries):
            raise ArithmeticError("a classifier did not learn its real objective")
        files = sorted(p for p in out.rglob("*") if p.is_file())
        write_json(out/"completed.json", dict(status="structural_intent_producer_complete", process_wall_seconds=time.monotonic()-start,
            artifact_sha256={str(p.relative_to(out)):digest(p) for p in files},
            artifact_bytes={str(p.relative_to(out)):p.stat().st_size for p in files}))
    except BaseException as error:
        write_json(out/"failed.json", dict(error=repr(error), process_wall_seconds=time.monotonic()-start))
        raise


if __name__ == "__main__":
    main()
