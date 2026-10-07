"""Independent arithmetic/readout audit and fresh LM replays for the P1 pilot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import log
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {name:a[name].copy() for name in a.files}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def independently_encode(hidden, state):
    a = (hidden.astype(np.float64)-state["center"].astype(np.float64)) @ state["projection"].astype(np.float64)
    bins = np.sum(a[:, :, None] >= state["thresholds"][None], axis=2).astype(np.uint8)
    return bins, (a-state["mean"])/state["scale"]


def independently_features(method, ordered, real, input_state, stage):
    if method == "real_linear":
        return np.concatenate([real, np.zeros((len(real), 112))], axis=1), 0
    w, b = stage["weights"], stage["bias"]
    if method == "real_threshold":
        coeff = w.astype(np.int16)
        coeff[coeff >= 8] -= 16
        intercept = b.astype(np.int16)
        intercept[intercept >= 8] -= 16
        a = (real @ (coeff.astype(float)/8).T+intercept/8)/input_state["real_scale"]
        phi = np.stack([a <= threshold for threshold in (-1., -.25, .25, 1.)], axis=-1)
        return phi.reshape(len(real), 128).astype(float), 0
    if method == "ordered_ball":
        code = ordered.astype(np.int64)
    elif method == "shuffled_ball":
        code = np.stack([input_state["shuffled"][j, ordered[:, j]] for j in range(16)], axis=1).astype(np.int64)
    else:
        # Independent digit-reversal oracle, with no producer encoding calls.
        table = np.asarray([sum(((i >> j) & 1) << (3-j) for j in range(4)) for i in range(16)])
        code = table[ordered]
    residues = np.broadcast_to(b.astype(np.int64), (len(code), 32)).copy()
    for j in range(16):
        residues = (residues+code[:, j, None]*w[None, :, j].astype(np.int64)) % 16
    if method == "reverse_signed":
        signed = residues.copy()
        signed[signed >= 8] -= 16
        a = signed.astype(float)/8
        phi = np.stack([a, a*a, a*a*a, a*a*a*a], axis=-1)
    else:
        # Congruence balls as independent low-digit prefix tree indicators.
        bits = np.stack([((residues >> digit) & 1) == 0 for digit in range(4)], axis=-1)
        phi = np.logical_and.accumulate(bits, axis=-1)
    return phi.reshape(len(code), 128).astype(float), int(residues.size)


def model_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().contiguous()
        result.update(name.encode()); result.update(str(a.dtype).encode())
        result.update(json.dumps(list(a.shape)).encode())
        result.update(a.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


class OneWindowFFN(torch.nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.hidden, self.output, self.calls = hidden, output, 0

    def forward(self, hidden):
        if self.calls or not np.array_equal(hidden[0].float().cpu().numpy(), self.hidden):
            raise AssertionError("fresh audit intervention input differs")
        self.calls += 1
        return torch.tensor(self.output[None], device=hidden.device, dtype=hidden.dtype)


def replay(model, token, hidden, output):
    layer = model.model.layers[0]
    original = layer.mlp
    module = OneWindowFFN(hidden, output)
    if output is not None:
        layer.mlp = module
    try:
        with torch.inference_mode():
            batch = torch.tensor(token[None], device="cuda")
            logits = model(batch, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   batch[:, 1:].reshape(-1), reduction="none")
            return loss.float().cpu().numpy()
    finally:
        layer.mlp = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "pilot", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    pilot, out = args.pilot.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("fresh audit required")
    out.mkdir(parents=True)
    wall = time.monotonic()
    try:
        completed = json.loads((pilot/"completed.json").read_text())
        for name, pin in completed["artifact_sha256"].items():
            if digest(pilot/name) != pin or (pilot/name).stat().st_size != completed["artifact_bytes"][name]:
                raise ValueError("pilot pin mismatch: "+name)
        config = json.loads((pilot/"config.json").read_text())
        if digest(args.data/"completed.json") != config["data_completion_sha256"]:
            raise ValueError("data lineage differs")
        data_completion = json.loads((args.data/"completed.json").read_text())
        for name, pin in data_completion["artifact_sha256"].items():
            if digest(args.data/name) != pin:
                raise ValueError("data artifact changed: "+name)
        metadata = json.loads((args.data/"manifest.json").read_text())
        tokens = load_arrays(args.data/"tokens.npz")["development"]
        teacher = load_arrays(pilot/"teacher_trace.npz")
        for name in ("train_hidden", "train_output", "dev_hidden", "dev_output"):
            a = teacher[name]
            expected = (16 if name.startswith("train") else 64, 512, 960)
            if a.shape != expected or not np.isfinite(a).all():
                raise AssertionError("teacher capture shape/finite contract differs")
            rounded = torch.tensor(a).bfloat16().float().numpy()
            if not np.array_equal(a, rounded):
                raise AssertionError("teacher tensor is not an exact BF16 promotion")
        encoder = load_arrays(pilot/"encoder.npz")
        train_h = teacher["train_hidden"].reshape(-1, 960)
        dev_h = teacher["dev_hidden"].reshape(-1, 960)
        train_y = teacher["train_output"].reshape(-1, 960).astype(float)
        dev_y = teacher["dev_output"].reshape(-1, 960).astype(float)
        np.testing.assert_array_equal(encoder["center"], train_h.astype(float).mean(axis=0).astype(np.float32))
        projected = (train_h.astype(float)-encoder["center"].astype(float)) @ encoder["projection"].astype(float)
        np.testing.assert_array_equal(encoder["thresholds"],
            np.quantile(projected, np.arange(1, 16)/16, axis=0, method="linear").T)
        np.testing.assert_array_equal(encoder["mean"], projected.mean(axis=0))
        np.testing.assert_array_equal(encoder["scale"], projected.std(axis=0))
        ortho_error = float(np.max(np.abs(encoder["projection"].astype(float).T @ encoder["projection"].astype(float)-np.eye(16))))
        if ortho_error > 1e-7:
            raise AssertionError("encoder basis is not orthonormal")
        ordered, real = independently_encode(train_h, encoder)
        dordered, dreal = independently_encode(dev_h, encoder)
        result = json.loads((pilot/"results.json").read_text())
        stock_nll = float(teacher["dev_loss"].astype(float).mean())
        if stock_nll != result["stock_nll"]:
            raise AssertionError("stock loss aggregate differs")
        records = result["results"]
        if len(records) != 44 or len({(r["seed"], r["method"], r["stage"]) for r in records}) != 44:
            raise AssertionError("all 42 seed/method/stage records and two nulls required")
        checked_residues, max_error, max_normal_equation = 0, 0., 0.
        replay_cases = []
        for record in records:
            seed, method, stage_name = record["seed"], record["method"], record["stage"]
            if seed is None:
                loss = load_arrays(pilot/f"{method}_losses.npz")["losses"]
                prediction = np.zeros((512, 960)) if method == "zero" else np.broadcast_to(train_y.mean(axis=0), (512, 960))
                replay_cases.append((method, loss[0], prediction))
            else:
                directory = pilot/f"seed-{seed}"
                input_state = load_arrays(directory/"input_state.npz")
                stage = load_arrays(directory/f"{method}_{stage_name}.npz")
                for role, oo, rr, yy in (("training", ordered, real, train_y), ("development", dordered, dreal, dev_y)):
                    phi, count = independently_features(method, oo, rr, input_state, stage)
                    checked_residues += count
                    prediction = phi @ stage["coefficient"]+stage["intercept"]
                    sse = float(np.sum((yy-prediction)**2))
                    norm = float(np.sum(yy*yy))
                    nrmse = float(np.sqrt(sse/norm))
                    error = abs(nrmse-record[role]["nrmse"])
                    max_error = max(error, max_error)
                    if error > 1e-10 or abs(sse-record[role]["sse"]) > 1e-8*max(1., sse):
                        raise AssertionError("independent feature/readout fit differs")
                    if record[role]["constant_features"] != int(np.sum(np.var(phi, axis=0) == 0)):
                        raise AssertionError("constant feature count differs")
                    np.testing.assert_allclose(phi.mean(axis=0), record[role]["feature_means"], atol=1e-12, rtol=1e-12)
                    if role == "training":
                        xc, yc = phi-phi.mean(axis=0), yy-yy.mean(axis=0)
                        gradient = xc.T @ (xc @ stage["coefficient"]-yc)/len(xc)+.001*stage["coefficient"]
                        ge = float(np.max(np.abs(gradient)))
                        max_normal_equation = max(ge, max_normal_equation)
                        if ge > 1e-8:
                            raise AssertionError("ridge normal equation fails")
                    elif "nll" in record:
                        rounded = torch.tensor(prediction).bfloat16().float().numpy().astype(float)
                        bf16_nrmse = float(np.sqrt(np.sum((yy-rounded)**2)/norm))
                        if abs(bf16_nrmse-record[role]["bf16_output_nrmse"]) > 1e-10:
                            raise AssertionError("BF16 output error differs")
                        loss = load_arrays(directory/f"{method}_{stage_name}_losses.npz")["losses"]
                        replay_cases.append((f"{seed}-{method}-{stage_name}", loss[0], prediction[:512]))
                if stage_name == "final":
                    history = json.loads((directory/f"{method}_history.json").read_text())
                    if record["proposals"] != sum(h["proposals"] for h in history) or record["accepted_moves"] != sum(h["accepted"] for h in history):
                        raise AssertionError("optimization ledger differs")
                    expected_proposals = 0 if method in ("unmixed_ball", "real_linear") else 2*32*68
                    if record["proposals"] != expected_proposals:
                        raise AssertionError("finite-search budget differs")
                    for a, b in zip(history, history[1:]):
                        if b["row"] >= 0 and b["sse"] > a["sse"]+max(1e-7, a["sse"]*1e-9):
                            raise AssertionError("accepted hard-coordinate moves increase training SSE")
            if "nll" in record:
                if loss.shape != (64, 511) or not np.isfinite(loss).all():
                    raise AssertionError("all raw development target losses required")
                mean = float(loss.astype(float).mean())
                if mean != record["nll"] or mean-stock_nll != record["delta_nll"]:
                    raise AssertionError("raw token-loss aggregate differs")
        if len(replay_cases) != 26:
            raise AssertionError("26 declared non-stock LM interventions required")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
            torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
        if model_hash(model) != config["model_state_sha256"]:
            raise AssertionError("fresh audit checkpoint differs")
        stock_replay = replay(model, tokens[0], teacher["dev_hidden"][0], None)
        if not np.array_equal(stock_replay, teacher["dev_loss"][0]):
            raise AssertionError("fresh stock losses differ")
        max_loss_difference = 0.
        for name, expected, prediction in replay_cases:
            if time.monotonic()-wall > 1800:
                raise TimeoutError("prospective independent audit allocation exhausted")
            actual = replay(model, tokens[0], teacher["dev_hidden"][0], prediction)
            difference = float(np.max(np.abs(actual.astype(float)-expected.astype(float))))
            max_loss_difference = max(max_loss_difference, difference)
            if difference > 4e-6:
                raise AssertionError("fresh LM token loss differs: "+name)
        by_key = {(r["seed"], r["method"], r["stage"]):r for r in records}
        null_nll = min(by_key[(None, m, "final")]["nll"] for m in ("zero", "mean"))
        gates = []
        for seed in config["seeds"]:
            initial, final = (by_key[(seed, "reverse_ball", s)] for s in ("initial", "final"))
            real_control = by_key[(seed, "real_threshold", "final")]
            criteria = dict(hard_weights_change=final["ring_weight_changes"] > 0,
                training_sse_decreases=final["training"]["sse"] < initial["training"]["sse"],
                stock_development_allowance=final["delta_nll"] <= log(1.10),
                informative_over_nulls=final["nll"] <= null_nll-.005,
                matched_real_allowance=final["nll"] <= real_control["nll"]+log(1.02))
            gates.append(dict(seed=seed, criteria=criteria, all_pass=all(criteria.values())))
        json_write(out/"audit.json", dict(status="ball_ffn_development_pilot_independently_verified",
            created_utc=datetime.now(timezone.utc).isoformat(), pilot_completion_sha256=digest(pilot/"completed.json"),
            audit_source_sha256=digest(__file__), records_checked=len(records), residue_outputs_checked=checked_residues,
            encoder_orthonormal_error=ortho_error, max_nrmse_difference=max_error,
            max_ridge_normal_equation_error=max_normal_equation, fresh_lm_windows=27,
            fresh_loss_max_abs_difference=max_loss_difference, scored_development_targets=64*511,
            previously_reused_development_articles=metadata["development_articles"],
            prospective_progression_criteria=gates, progression_all_seeds=all(g["all_pass"] for g in gates),
            scope="Exploratory fixed small FFN, reused development data; no confirmatory competitiveness or universal p-adic conclusion",
            elapsed_process_seconds=time.monotonic()-wall))
        (out/"audit_ball_ffn.py").write_bytes(Path(__file__).read_bytes())
        print((out/"audit.json").read_text())
    except BaseException as exc:
        json_write(out/"failed.json", dict(status="ball_ffn_independent_audit_failed", type=type(exc).__name__,
            message=str(exc), elapsed_process_seconds=time.monotonic()-wall))
        raise


if __name__ == "__main__":
    main()
