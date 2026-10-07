"""Separate scalar-tree/readout, data, gradient and paired-outcome verification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(text):
    return " ".join(text.casefold().split())


def distribution(x, w, b, kind, codes):
    z = x@w.T+b
    if kind != "tree":
        logits = np.pad(z, ((0, 0), (0, 1)))
        logits -= logits.max(1)[:, None]
        p = np.exp(logits); p /= p.sum(1)[:, None]
        return p, None, None
    # Independently enumerate occupied prefixes and traverse one branch per leaf.
    nodes = []
    for depth in range(8):
        for prefix in sorted(set(int(c) % (1 << depth) for c in codes)):
            children = {(int(c) >> depth) & 1 for c in codes if int(c) % (1 << depth) == prefix}
            if len(children) == 2:
                nodes.append((depth, prefix))
    right = np.zeros((len(nodes), len(codes))); mask = right.copy()
    logp = np.zeros((len(x), len(codes)))
    for i, code in enumerate(codes):
        for j, (depth, prefix) in enumerate(nodes):
            if int(code) % (1 << depth) == prefix:
                bit = (int(code) >> depth) & 1
                right[j, i] = bit; mask[j, i] = 1
                logp[:, i] -= np.logaddexp(0, -z[:, j] if bit else z[:, j])
    return np.exp(logp), right, mask


def joint_gradient(x, y, groups, w, b, codes):
    p, right, mask = distribution(x, w, b, "tree", codes)
    z = x@w.T+b
    sigmoid = np.exp(-np.logaddexp(0, -z))
    domains = np.column_stack([p[:, groups == d].sum(1) for d in range(10)])
    conditional = p*(groups[None, :] == groups[y, None])/domains[np.arange(len(y)), groups[y], None]
    derivative = sigmoid*(mask[:, y].T+conditional@mask.T)-(right[:, y].T+conditional@right.T)
    loss = (-np.log(p[np.arange(len(y)), y])-np.log(domains[np.arange(len(y)), groups[y]])).mean()+.0005*(w*w).sum()
    return float(loss), derivative.T@x/len(y)+.001*w, derivative.mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); source = args.input.resolve(); out = args.output.resolve()
    if out.exists():
        raise FileExistsError("fresh independent audit directory required")
    completion = json.loads((source/"completed.json").read_text())
    for name, pin in completion["artifact_sha256"].items():
        if sha(source/name) != pin or (source/name).stat().st_size != completion["artifact_bytes"][name]:
            raise ValueError("artifact differs: "+name)
    execution = json.loads((source/"execution.json").read_text())
    for rel, pin in execution["source_sha256"].items():
        if sha(ROOT/rel) != pin:
            raise ValueError("source differs: "+rel)
    out.mkdir(parents=True)
    (out/"audit_structural_intent.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    def guard():
        if time.monotonic()-start > 1800:
            raise TimeoutError("prospective1800-second audit allocation exhausted")
    try:
        data = json.loads((source/"data-manifest.json").read_text())
        raw = json.loads((source/"data_full.json").read_text())
        raw_bytes = (source/"data_full.json").read_bytes()
        assert hashlib.sha1(b"blob "+str(len(raw_bytes)).encode()+b"\0"+raw_bytes).hexdigest() == "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
        domains = json.loads((ROOT/"research/clinc-domains.json").read_text())
        labels = data["labels"]; groups = np.array(data["label_domains"]); codes = np.array(data["semantic_codes"])
        held = [min(sorted(domains[n]), key=lambda label: hashlib.sha256(
            ("structural-intent-heldout-v1|"+n+"|"+label).encode()).hexdigest()) for n in sorted(domains)]
        assert held == data["withheld_intents"]
        for i, label in enumerate(labels):
            domain = sorted(domains).index(data["names"][groups[i]])
            fine = sorted(domains[data["names"][domain]]).index(label)
            expected_code = sum(int(bit)*(2**k) for k, bit in enumerate(f"{domain:04b}{fine:04b}"))
            assert codes[i] == expected_code
        order = ("train", "development", "seen_test", "unseen_test")
        all_rows = [r for part in order for r in data["rows"][part]]
        train_text = {clean(t) for t, _ in raw["train"]}; val_text = {clean(t) for t, _ in raw["val"]}
        for part, rows in data["rows"].items():
            if part != "train": assert len({clean(r["text"]) for r in rows}) == len(rows)
            for r in rows:
                assert raw[r["source_split"]][r["index"]] == [r["text"], r["label"]]
                assert r["label"] in domains[data["names"][r["domain"]]]
                if part == "unseen_test": assert r["label"] in held and r["y"] == -1
                else: assert labels[r["y"]] == r["label"] and r["label"] not in held
                if part != "train": assert clean(r["text"]) not in train_text
                if part.endswith("test"): assert clean(r["text"]) not in val_text
        for i, label in enumerate(labels):
            selected = [r["index"] for r in data["rows"]["train"] if r["label"] == label]
            candidates = [(j, t) for j, (t, lab) in enumerate(raw["train"]) if lab == label]
            candidates.sort(key=lambda row: (hashlib.sha256(
                ("structural-intent-train-v1|"+label+"|"+row[1]).encode()).hexdigest(), row[0]))
            seen = set(); expected = []
            for j, text in candidates:
                if clean(text) not in seen:
                    expected.append(j); seen.add(clean(text))
                if len(expected) == 20: break
            assert selected == expected
        # Check every eligible val/test row, including exclusions, not only saved rows.
        for official in ("val", "test"):
            used = set(); expected = []
            for i, (text, label) in enumerate(raw[official]):
                if label not in labels+held or (official == "val" and label in held): continue
                norm = clean(text)
                if norm in train_text or norm in used or (official == "test" and norm in val_text): continue
                used.add(norm); expected.append(i)
            actual = sorted(r["index"] for r in all_rows if r["source_split"] == official)
            assert sorted(expected) == actual
        features = np.load(source/"llm-features.npz")["hidden"]
        encoder = np.load(source/"encoder.npz")
        ntrain = len(data["rows"]["train"])
        centered = features[:ntrain].astype(np.float64)-encoder["center"]
        assert np.max(np.abs(features[:ntrain].astype(np.float64).mean(0)-encoder["center"])) == 0
        covariance = centered.T@centered/ntrain
        pca_residual = float(np.max(np.abs(covariance@encoder["projection"]-encoder["projection"]*encoder["eigenvalues"])))
        assert pca_residual < 1e-8
        assert np.max(np.abs(encoder["projection"].T@encoder["projection"]-np.eye(64))) < 1e-12
        assert np.max(np.abs(np.linalg.eigvalsh(covariance)[-64:][::-1]-encoder["eigenvalues"])) < 1e-8
        assert np.array_equal(encoder["scale"], np.sqrt(np.maximum(encoder["eigenvalues"], 0))+1e-6)
        xall = (features.astype(np.float64)-encoder["center"])@encoder["projection"]/encoder["scale"]
        parts = {}; offset = 0
        for part in order:
            rows = data["rows"][part]
            parts[part] = (xall[offset:offset+len(rows)], np.array([r["y"] for r in rows]), np.array([r["domain"] for r in rows]))
            offset += len(rows)
        records = json.loads((source/"results.json").read_text())
        max_probability_error = 0.; elements = 0; checked = []
        unseen_arrays = {}; seen_arrays = {}
        for record in records:
            guard(); seed, method, kind = record["seed"], record["method"], record["kind"]
            with np.load(source/f"seed-{seed}-{method}.npz") as model:
                expected_initial = np.random.default_rng(seed).normal(scale=.01, size=(139, 64))
                assert np.array_equal(model["initial_w"], expected_initial[:len(model["w"])])
                assert np.all(model["initial_b"] == 0)
                expected_codes = codes[np.random.default_rng(seed+1000).permutation(140)] if method == "shuffled_prefix" else codes
                assert np.array_equal(model["codes"], expected_codes)
                for part, (x, y, d) in parts.items():
                    p, _, _ = distribution(x, model["w"], model["b"], kind, model["codes"])
                    error = float(np.max(np.abs(p-model[part+"_probabilities"])))
                    max_probability_error = max(error, max_probability_error); elements += p.size
                    assert error < 1e-12
                    q = p if kind == "domain" else np.column_stack([p[:, groups == j].sum(1) for j in range(10)])
                    values = {"domain_accuracy":float(np.mean(q.argmax(1) == d)),
                        "domain_nll":float(-np.log(q[np.arange(len(d)), d]).mean())}
                    if kind != "domain" and np.all(y >= 0):
                        values.update(intent_accuracy=float(np.mean(p.argmax(1) == y)), intent_nll=float(-np.log(p[np.arange(len(y)), y]).mean()))
                    for key, value in values.items(): assert abs(value-record["splits"][part][key]) < 1e-10
                    if part == "unseen_test": unseen_arrays[seed, method] = (q, d)
                    if part == "seen_test" and kind != "domain": seen_arrays[seed, method] = float(np.mean(p.argmax(1) == y))
                assert np.any(model["initial_w"] != model["w"])
                x, y, d = parts["train"]
                for stage, key in (("initial", "initial_training_objective"), ("final", "final_training_objective")):
                    w = model["initial_w"] if stage == "initial" else model["w"]
                    b = model["initial_b"] if stage == "initial" else model["b"]
                    p, _, _ = distribution(x, w, b, kind, model["codes"])
                    q = p if kind == "domain" else np.column_stack([p[:, groups == j].sum(1) for j in range(10)])
                    value = -np.log(q[np.arange(len(d)), d]).mean()+.0005*(w*w).sum()
                    if kind != "domain": value += -np.log(p[np.arange(len(y)), y]).mean()
                    assert abs(value-record[key]) < 1e-10
                if method == "semantic_prefix":
                    certificates = [v for v in json.loads((source/"path-certificates.json").read_text()) if v["seed"] == seed]
                    assert len(certificates) == 20
                    xu, _, _ = parts["unseen_test"]
                    for cert in certificates:
                        row, domain = cert["unseen_test_row"], cert["predicted_domain"]
                        z = xu[row]@model["w"].T+model["b"]
                        pp, _, _ = distribution(xu[row:row+1], model["w"], model["b"], kind, model["codes"])
                        expected_log = np.log(pp[0, groups == domain].sum())
                        actual = 0.
                        for term in cert["terms"]:
                            j = np.flatnonzero(np.all(model["nodes"] == [term["depth"], term["prefix"]], axis=1))[0]
                            expected_term = -np.logaddexp(0, -z[j] if term["bit"] else z[j])
                            assert abs(expected_term-term["log_probability"]) < 1e-12
                            actual += expected_term
                        assert abs(actual-expected_log) < 1e-12
                if seed == 17 and method == "semantic_prefix":
                    w = model["initial_w"].copy(); b = model["initial_b"].copy()
                    mw = np.zeros_like(w); vw = mw.copy(); mb = np.zeros_like(b); vb = mb.copy()
                    x, y, _ = parts["train"]
                    for step in range(400):
                        guard(); _, gw, gb = joint_gradient(x, y, groups, w, b, model["codes"])
                        mw = .9*mw+.1*gw; vw = .999*vw+.001*gw*gw
                        mb = .9*mb+.1*gb; vb = .999*vb+.001*gb*gb
                        w -= .03*(mw/(1-.9**(step+1)))/(np.sqrt(vw/(1-.999**(step+1)))+1e-8)
                        b -= .03*(mb/(1-.9**(step+1)))/(np.sqrt(vb/(1-.999**(step+1)))+1e-8)
                        if step % 100 == 0: print(json.dumps(dict(stage="independent_real_fit_replay", step=step)), flush=True)
                    fit_error = float(max(np.max(np.abs(w-model["w"])), np.max(np.abs(b-model["b"]))))
                    assert fit_error < 1e-7
            checked.append([seed, method])
        # Fixed ten-intent paired cluster bootstrap. Fixed fits, descriptive coverage.
        unseen_rows = data["rows"]["unseen_test"]
        draws = np.random.default_rng(20261004).integers(0, 10, size=(10000, 10))
        np.savez_compressed(out/"bootstrap-draws.npz", draws=draws)
        comparisons, progression = [], []
        for seed in (17, 29, 43):
            p, d = unseen_arrays[seed, "semantic_prefix"]
            primary_loss = -np.log(p[np.arange(len(d)), d]); primary_correct = (p.argmax(1) == d).astype(float)
            counts = np.array([sum(r["label"] == label for r in unseen_rows) for label in held])
            assert np.all(counts > 0)
            den = counts[draws].sum(1)
            conditions = {}
            for method in ("flat_joint", "shuffled_prefix", "direct_domain"):
                q, _ = unseen_arrays[seed, method]
                diff = primary_loss+np.log(q[np.arange(len(d)), d])
                sums = np.array([sum(diff[i] for i, r in enumerate(unseen_rows) if r["label"] == label) for label in held])
                boot = sums[draws].sum(1)/den
                accuracy_difference = float(primary_correct.mean()-np.mean(q.argmax(1) == d))
                upper = float(np.quantile(boot, .975, method="linear"))
                comparisons.append(dict(seed=seed, comparator=method, domain_nll_difference=float(diff.mean()),
                    domain_nll_difference_95pct_upper=upper, domain_accuracy_difference=accuracy_difference))
                conditions[method] = (accuracy_difference >= .02 and upper < 0) if method != "direct_domain" else (accuracy_difference >= -.01 and diff.mean() <= .02)
            conditions["seen_intent_accuracy"] = seen_arrays[seed, "semantic_prefix"] >= seen_arrays[seed, "flat_joint"]-.01
            progression.append(dict(seed=seed, conditions={k:bool(v) for k,v in conditions.items()}, passes=bool(all(conditions.values()))))
        write = lambda p,v: p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
        write(out/"comparisons.json", comparisons); write(out/"progression.json", progression)
        # Re-tokenize all archived texts, then repeat the first original32-row GPU batch.
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
        tokenizer=AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M",revision=revision)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side="right"
        encoded=tokenizer([r["text"] for r in all_rows],padding="max_length",max_length=64,truncation=True,return_tensors="np")
        with np.load(source/"tokens.npz") as tokens:
            assert np.array_equal(encoded["input_ids"],tokens["ids"])
            assert np.array_equal(encoded["attention_mask"],tokens["masks"])
        guard()
        model=AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M",revision=revision,torch_dtype=torch.bfloat16,attn_implementation="eager",use_safetensors=True).to("cuda").eval()
        # Verify pinned state with a local independent digest, not producer imports.
        state=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            tensor=tensor.detach().contiguous()
            state.update(name.encode());state.update(str(tensor.dtype).encode());state.update(json.dumps(list(tensor.shape)).encode())
            state.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        assert state.hexdigest()=="b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583"
        with torch.inference_mode():
            token=torch.as_tensor(encoded["input_ids"][:32],device="cuda")
            mask=torch.as_tensor(encoded["attention_mask"][:32],device="cuda")
            hidden=model.model(token,attention_mask=mask,use_cache=False).last_hidden_state.float()
            repeated=((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
        feature_error=float(np.max(np.abs(repeated-features[:32])))
        assert feature_error==0
        guard()
        write(out/"audit.json",dict(status="structural_intent_independently_verified",producer_completion_sha256=sha(source/"completed.json"),
            all12_fitted_classifiers_verified=checked,probability_elements=elements,max_probability_error=max_probability_error,
            pca_eigen_residual=pca_residual,independent400_step_fit_max_error=fit_error,
            all_tokenized_rows_verified=len(all_rows),fresh_llm_feature_rows=32,max_llm_feature_error=feature_error,
            all_primary_progression_rules_pass=bool(all(v["passes"] for v in progression)),
            bootstrap_units="10 fixed unseen intents; whole-intent paired resampling, descriptive approximate coverage",
            process_wall_seconds=time.monotonic()-start,scope="Real prefix/tree equivalence and probe generalization; no generative LM or real-noise robustness result"))
    except BaseException as error:
        (out/"failed.json").write_text(json.dumps(dict(error=repr(error),process_wall_seconds=time.monotonic()-start),indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
