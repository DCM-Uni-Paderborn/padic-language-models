"""Frozen small P1 learning pilot: teacher FFN distillation and real LM loss."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from padic_lm.ball_ffn import (fit_encoder, encode, reversed_digits, features, ridge, discrete_fit)

DATA_HASH = "033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454"
STATE_HASH = "b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583"
SEEDS = (17, 29, 43)
METHODS = ("reverse_ball", "ordered_ball", "shuffled_ball", "reverse_signed",
           "real_threshold", "unmixed_ball", "real_linear")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().contiguous()
        value.update(name.encode()); value.update(str(a.dtype).encode())
        value.update(json.dumps(list(a.shape)).encode())
        value.update(a.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def verify_data(directory):
    completion = json.loads((directory/"completed.json").read_text())
    if digest(directory/"completed.json") != DATA_HASH:
        raise ValueError("requires the pinned corrected learning data")
    for name, pin in completion["artifact_sha256"].items():
        if digest(directory/name) != pin or (directory/name).stat().st_size != completion["artifact_bytes"][name]:
            raise ValueError(f"data mismatch: {name}")
    return json.loads((directory/"manifest.json").read_text())


def capture(model, tokens, check_time):
    mlp = model.model.layers[0].mlp
    h, y, losses = [], [], []
    def collect(module, inputs, output):
        h.append(inputs[0][0].float().cpu().numpy().copy())
        y.append(output[0].float().cpu().numpy().copy())
    hook = mlp.register_forward_hook(collect)
    try:
        with torch.inference_mode():
            for i, window in enumerate(tokens):
                check_time()
                batch = torch.as_tensor(window[None], device="cuda")
                logits = model(batch, use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       batch[:, 1:].reshape(-1), reduction="none")
                losses.append(loss.float().cpu().numpy().copy())
                print(json.dumps(dict(stage="capture", window=i+1, windows=len(tokens))), flush=True)
    finally:
        hook.remove()
    return np.stack(h), np.stack(y), np.stack(losses)


class ReplayMLP(torch.nn.Module):
    """First-layer input is unchanged; enforce this before an FFN intervention."""
    def __init__(self, hidden, output):
        super().__init__()
        self.hidden, self.output, self.index = hidden, output, 0

    def forward(self, hidden):
        if self.index >= len(self.hidden) or not np.array_equal(hidden[0].float().cpu().numpy(), self.hidden[self.index]):
            raise ArithmeticError("intervention input differs from original first-layer teacher input")
        result = torch.as_tensor(self.output[self.index][None], device=hidden.device, dtype=hidden.dtype)
        self.index += 1
        return result


def lm_losses(model, tokens, hidden, output, check_time):
    layer = model.model.layers[0]
    original = layer.mlp
    replay = ReplayMLP(hidden, output)
    layer.mlp = replay
    loss = []
    try:
        with torch.inference_mode():
            for window in tokens:
                check_time()
                batch = torch.as_tensor(window[None], device="cuda")
                logits = model(batch, use_cache=False).logits
                value = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                        batch[:, 1:].reshape(-1), reduction="none")
                loss.append(value.float().cpu().numpy().copy())
    finally:
        layer.mlp = original
    if replay.index != len(tokens):
        raise AssertionError("not every FFN intervention was consumed")
    return np.stack(loss)


def method_input(method, ordered, real, shuffled):
    if method.startswith("real"):
        return real
    if method == "ordered_ball":
        return ordered
    if method == "shuffled_ball":
        return np.column_stack([shuffled[j, ordered[:, j]] for j in range(16)])
    return reversed_digits()[ordered]


def method_features(method, x, stage, real_scale):
    if method == "real_linear":
        return np.pad(x, ((0, 0), (0, 112)))
    kind = "real" if method == "real_threshold" else "signed" if method == "reverse_signed" else "ball"
    return features(x, stage["weights"], stage["bias"], kind, real_scale if kind == "real" else None)


def diagnostic(phi, target, stage):
    output = phi @ stage["coefficient"]+stage["intercept"]
    squared = float(np.sum((target-output)**2))
    norm = float(np.sum(target*target))
    return dict(sse=squared, target_squared_norm=norm, nrmse=float(np.sqrt(squared/norm)),
                feature_rank=int(np.linalg.matrix_rank(phi-phi.mean(axis=0))),
                constant_features=int(np.sum(np.var(phi, axis=0) == 0)),
                feature_means=phi.mean(axis=0).tolist()), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "execution", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("requires a fresh pilot directory")
    execution = json.loads(args.execution.read_text())
    for name, pin in execution["source_sha256"].items():
        if digest(ROOT/name) != pin:
            raise ValueError("prospective source mismatch: "+name)
    if digest(ROOT/execution["protocol"]) != execution["protocol_sha256"]:
        raise ValueError("prospective protocol changed")
    data = args.data.resolve()
    metadata = verify_data(data)
    with np.load(data/"tokens.npz", allow_pickle=False) as a:
        train_tokens, dev_tokens = a["train"].copy(), a["development"].copy()
    if train_tokens.shape != (16, 512) or dev_tokens.shape != (64, 512):
        raise ValueError("frozen token shapes differ")
    output.mkdir(parents=True)
    (output/"sources").mkdir()
    for name in execution["source_sha256"]:
        (output/"sources"/Path(name).name).write_bytes((ROOT/name).read_bytes())
    (output/"prospective_protocol.md").write_bytes((ROOT/execution["protocol"]).read_bytes())
    (output/"execution.json").write_bytes(args.execution.read_bytes())
    wall = time.monotonic()
    def check_time():
        if time.monotonic()-wall > 1800:
            raise TimeoutError("one prospective 1800-second allocation exhausted")
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(8)
        model = AutoModelForCausalLM.from_pretrained(metadata["model"], revision=metadata["model_revision"],
                    torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
        if state_hash(model) != STATE_HASH:
            raise ValueError("original model state differs")
        config = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="prospective_development_pilot",
            model=metadata["model"], revision=metadata["model_revision"], model_state_sha256=STATE_HASH,
            data_completion_sha256=DATA_HASH, execution_sha256=digest(args.execution),
            model_hidden_size=model.config.hidden_size, model_intermediate_size=model.config.intermediate_size,
            teacher_ffn_tensor_bytes=sum(t.numel()*t.element_size() for t in model.model.layers[0].mlp.state_dict().values()),
            seeds=list(SEEDS), methods=list(METHODS), layer=0, p=2, k=4, inputs=16, affine_rows=32,
            features=128, ridge_alpha=.001, sweeps=2, train_tokens=8192, dev_targets=64*511,
            source_sha256=execution["source_sha256"],
            packages={n:importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
            hardware=torch.cuda.get_device_name(), speed_claim=False,
            scope="Previously used development corpus; no untouched confirmation, compression or full-Q_p claim")
        write_json(output/"config.json", config)
        htrain, ytrain, ltrain = capture(model, train_tokens, check_time)
        hdev, ydev, ldev = capture(model, dev_tokens, check_time)
        np.savez_compressed(output/"teacher_trace.npz", train_hidden=htrain, train_output=ytrain,
                            dev_hidden=hdev, dev_output=ydev, train_loss=ltrain, dev_loss=ldev)
        identity = lm_losses(model, dev_tokens[:1], hdev[:1], ydev[:1], check_time)
        if not np.array_equal(identity, ldev[:1]):
            raise ArithmeticError("teacher replay does not reproduce every first-window stock token loss")
        target, devtarget = ytrain.reshape(-1, 960).astype(np.float64), ydev.reshape(-1, 960).astype(np.float64)
        encoder = fit_encoder(htrain.reshape(-1, 960))
        np.savez_compressed(output/"encoder.npz", **encoder)
        ordered, real = encode(htrain.reshape(-1, 960), encoder)
        devordered, devreal = encode(hdev.reshape(-1, 960), encoder)
        results = []
        base_nll = float(ldev.astype(np.float64).mean())
        for name, prediction in (("zero", np.zeros_like(ydev)),
                                 ("mean", np.broadcast_to(target.mean(axis=0), ydev.shape))):
            losses = lm_losses(model, dev_tokens, hdev, prediction, check_time)
            np.savez_compressed(output/f"{name}_losses.npz", losses=losses)
            results.append(dict(seed=None, method=name, stage="final", nll=float(losses.astype(np.float64).mean()),
                                delta_nll=float(losses.astype(np.float64).mean()-base_nll)))
        for seed in SEEDS:
            rng = np.random.Generator(np.random.PCG64(seed))
            original_w = rng.integers(0, 16, (32, 16), dtype=np.uint8)
            original_b = rng.integers(0, 16, 32, dtype=np.uint8)
            shuffled = np.stack([rng.permutation(16) for _ in range(16)]).astype(np.uint8)
            sw = np.where(original_w >= 8, original_w.astype(float)-16, original_w)/8
            real_scale = np.maximum(np.linalg.norm(sw, axis=1), 1e-6)
            directory = output/f"seed-{seed}"
            directory.mkdir()
            np.savez_compressed(directory/"input_state.npz", shuffled=shuffled, real_scale=real_scale,
                                initial_weights=original_w, initial_bias=original_b)
            for method in METHODS:
                check_time()
                x = method_input(method, ordered, real, shuffled)
                dx = method_input(method, devordered, devreal, shuffled)
                w, b = original_w.copy(), original_b.copy()
                kind = "real" if method == "real_threshold" else "signed" if method == "reverse_signed" else "ball"
                if method in ("unmixed_ball", "real_linear"):
                    if method == "unmixed_ball":
                        w[:] = 0
                        w[np.arange(32), np.arange(32) % 16] = 1
                    stage = dict(weights=w, bias=b)
                    phi = method_features(method, x, stage, real_scale)
                    coefficient, intercept = ridge(phi, target)
                    initial = final = dict(**stage, coefficient=coefficient, intercept=intercept)
                    history = []
                else:
                    initial, final, history = discrete_fit(x, target, w, b, kind,
                        real_scale if kind == "real" else None, seed=seed, check_time=check_time)
                write_json(directory/f"{method}_history.json", history)
                for stage_name, stage in (("initial", initial), ("final", final)):
                    np.savez_compressed(directory/f"{method}_{stage_name}.npz", **stage)
                    train_info, _ = diagnostic(method_features(method, x, stage, real_scale), target, stage)
                    dev_info, predicted = diagnostic(method_features(method, dx, stage, real_scale), devtarget, stage)
                    record = dict(seed=seed, method=method, stage=stage_name, training=train_info, development=dev_info,
                        ring_weight_changes=int(np.sum(stage["weights"] != original_w)) if method != "real_linear" else 0,
                        candidate_array_bytes=sum(a.nbytes for a in stage.values())+sum(a.nbytes for a in encoder.values())+
                            shuffled.nbytes+real_scale.nbytes+original_w.nbytes+original_b.nbytes,
                        archived_bundle_bytes=(directory/f"{method}_{stage_name}.npz").stat().st_size+
                            (output/"encoder.npz").stat().st_size+(directory/"input_state.npz").stat().st_size,
                        accepted_moves=sum(h["accepted"] for h in history) if stage_name == "final" else 0,
                        proposals=sum(h["proposals"] for h in history) if stage_name == "final" else 0)
                    if stage_name == "final" or method == "reverse_ball":
                        rounded = torch.as_tensor(predicted).to(torch.bfloat16).float().numpy().astype(np.float64)
                        record["development"]["bf16_output_nrmse"] = float(np.sqrt(np.sum((rounded-devtarget)**2)/np.sum(devtarget**2)))
                        losses = lm_losses(model, dev_tokens, hdev, predicted.reshape(64, 512, 960), check_time)
                        np.savez_compressed(directory/f"{method}_{stage_name}_losses.npz", losses=losses)
                        record.update(nll=float(losses.astype(np.float64).mean()),
                                      delta_nll=float(losses.astype(np.float64).mean()-base_nll))
                    results.append(record)
                write_json(output/"progress.json", dict(completed_seed=seed, completed_method=method,
                                                       elapsed_process_seconds=time.monotonic()-wall))
                print(json.dumps(dict(stage="method_finished", seed=seed, method=method,
                    dev_nrmse=results[-1]["development"]["nrmse"], delta_nll=results[-1].get("delta_nll"))), flush=True)
        write_json(output/"results.json", dict(scope=config["scope"], stock_nll=base_nll, results=results))
        artifacts = sorted(p for p in output.rglob("*") if p.is_file())
        write_json(output/"completed.json", dict(status="ball_ffn_development_pilot_produced_awaiting_audit",
            artifact_sha256={str(p.relative_to(output)):digest(p) for p in artifacts},
            artifact_bytes={str(p.relative_to(output)):p.stat().st_size for p in artifacts},
            identity_token_losses_exact=True, teacher_forward_windows=80, identity_forward_windows=1,
            intervention_forward_windows=(2+3*8)*64, proposals=3*5*2*32*68,
            elapsed_process_seconds=time.monotonic()-wall))
    except BaseException as exc:
        write_json(output/"failed.json", dict(status="ball_ffn_pilot_failed", type=type(exc).__name__,
                                             message=str(exc), elapsed_process_seconds=time.monotonic()-wall))
        raise


if __name__ == "__main__":
    main()
