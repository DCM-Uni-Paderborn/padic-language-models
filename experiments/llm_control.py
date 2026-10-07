"""Development-only pretrained LLM arithmetic control; not a native speed test."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResidueLinear(torch.nn.Module):
    """Symmetric W4/A8 quantization and optional balanced mod-2^K decoding.

    The dot product uses FP32 only when a proof bounds all integer partial sums
    within its exact integer range and TF32 is disabled. This is emulation.
    Scale factors and the real-valued output remain outside the residue ring.
    """

    def __init__(self, original: torch.nn.Linear, residue_bits: int | None):
        super().__init__()
        if original.in_features * 7 * 127 >= 2**24:
            raise ValueError("FP32 exact-integer accumulation bound exceeded")
        if residue_bits is not None and not 2 <= residue_bits <= 52:
            raise ValueError("residue_bits must be in [2,52]")
        self.original = original
        self.residue_bits = residue_bits
        weight = original.weight.detach().float()
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("Weights must be finite")
        amax = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), (amax / 7).clamp_min(torch.finfo(torch.float32).tiny))
        self.register_buffer("weight_scale", scale)
        self.register_buffer("weight_codes", (weight / scale).round().clamp(-7, 7))
        self.reset_statistics()

    def reset_statistics(self):
        self.statistics = {
            "dot_count": 0, "wrap_count": 0,
            "layer_squared_error": 0.0, "layer_reference_squared_norm": 0.0,
            "max_integer_abs": 0, "modular_squared_error": 0.0,
        }

    def forward(self, inputs):
        if torch.is_autocast_enabled(inputs.device.type):
            raise RuntimeError("Exact-integer emulation requires autocast disabled")
        if inputs.is_cuda and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("Exact-integer emulation requires TF32 disabled")
        x = inputs.float()
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Inputs must be finite")
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), (amax / 127).clamp_min(torch.finfo(torch.float32).tiny))
        codes = (x / scale).round().clamp(-127, 127)
        integer_dot = F.linear(codes, self.weight_codes).to(torch.int64)
        decoded = integer_dot
        if self.residue_bits is not None:
            modulus = 1 << self.residue_bits
            residue = integer_dot.remainder(modulus)
            decoded = torch.where(residue >= modulus // 2, residue - modulus, residue)
        output = decoded.float() * scale * self.weight_scale.squeeze(-1)
        if self.original.bias is not None:
            output = output + self.original.bias.float()
        output = output.to(inputs.dtype)
        reference = self.original(inputs)
        stats = self.statistics
        stats["dot_count"] += integer_dot.numel()
        stats["wrap_count"] += int((decoded != integer_dot).sum())
        stats["max_integer_abs"] = max(stats["max_integer_abs"], int(integer_dot.abs().max()))
        stats["modular_squared_error"] += float((decoded.double() - integer_dot.double()).square().sum())
        stats["layer_squared_error"] += float((output.float() - reference.float()).double().square().sum())
        stats["layer_reference_squared_norm"] += float(reference.double().square().sum())
        return output


def evaluate(model, tokens, sequence_length, device, wrapper):
    chunks = tokens.reshape(-1, sequence_length)
    with torch.inference_mode():
        model(chunks[:1].to(device), use_cache=False)
        if wrapper is not None:
            wrapper.reset_statistics()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        losses = []
        for chunk in chunks:
            batch = chunk.unsqueeze(0).to(device)
            logits = model(batch, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1), reduction="mean",
            )
            losses.append(float(loss))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
    mean_loss = sum(losses) / len(losses)
    result = {
        "mean_nll": mean_loss, "perplexity": math.exp(mean_loss),
        "block_mean_nll": losses,
        "predicted_tokens": len(losses) * (sequence_length - 1),
        "wall_seconds_emulation_with_diagnostics": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
    }
    if wrapper is not None:
        result["layer_statistics"] = dict(wrapper.statistics)
        s = wrapper.statistics
        result["layer_nrmse"] = math.sqrt(s["layer_squared_error"] / max(s["layer_reference_squared_norm"], 1e-300))
        result["wrap_fraction"] = s["wrap_count"] / s["dot_count"]
    return result


def capture_attention(model, tokens, sequence_length, device, layer_index, output):
    """Record actual post-RoPE Q/K and repeated GQA values from the baseline."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    attention = model.model.layers[layer_index].self_attn
    if attention.__class__.__name__ != "LlamaAttention":
        raise TypeError("Trace capture currently supports LlamaAttention only")
    traces = {"queries": [], "keys": [], "values": []}

    def before_attention(module, inputs, kwargs):
        hidden = kwargs["hidden_states"]
        shape = (*hidden.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden).view(shape).transpose(1, 2)
        key = module.k_proj(hidden).view(shape).transpose(1, 2)
        value = module.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        for name, tensor in [("queries", query), ("keys", key), ("values", value)]:
            traces[name].append(tensor.detach().float().cpu().numpy()[0])

    handle = attention.register_forward_pre_hook(before_attention, with_kwargs=True)
    try:
        with torch.inference_mode():
            for chunk in tokens.reshape(-1, sequence_length):
                model(chunk.unsqueeze(0).to(device), use_cache=False)
    finally:
        handle.remove()
    arrays = {name: np.stack(values) for name, values in traces.items()}
    np.savez_compressed(output / "attention-trace.npz", **arrays)
    (output / "attention-trace-manifest.json").write_text(json.dumps({
        "layer_index": layer_index,
        "shape": {name: list(array.shape) for name, array in arrays.items()},
        "stage": "unmodified reference model", "rotary_position": "applied",
        "grouped_keys_values": "repeated to query-head count",
        "split": "validation", "calibration_rule": "first chunks reserved by routing diagnostic",
        "sha256": hashlib.sha256((output / "attention-trace.npz").read_bytes()).hexdigest(),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--residue-bits", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--capture-attention", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/llm-control"))
    args = parser.parse_args()
    if args.sequence_length < 2 or args.tokens < args.sequence_length or args.tokens % args.sequence_length:
        raise ValueError("token count must be a positive multiple of sequence length")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory; previous research results are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is not available")
    api = HfApi()
    model_revision = api.model_info(args.model, revision=args.model_revision).sha
    dataset_name = "Salesforce/wikitext"
    dataset_revision = api.dataset_info(dataset_name, revision=args.dataset_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=model_revision, torch_dtype=dtype,
        attn_implementation="eager", use_safetensors=True,
    ).to(args.device).eval()
    stream = load_dataset(
        dataset_name, "wikitext-2-raw-v1", revision=dataset_revision,
        split="validation", streaming=True,
    )
    pieces, chars = [], 0
    # Fetch enough fixed-order validation text before tokenization; never test.
    for example in stream:
        text = example["text"]
        pieces.append(text)
        chars += len(text)
        if chars >= args.tokens * 16:
            break
    text = "\n\n".join(pieces)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(encoded) < args.tokens:
        raise ValueError("Fetched validation text has too few tokens")
    tokens = encoded[:args.tokens].contiguous()
    (args.output / "input-token-ids.json").write_text(json.dumps(tokens.tolist()))
    original = model.model.layers[args.layer_index].mlp.down_proj
    packages = {}
    for package in ["torch", "transformers", "datasets", "numpy", "huggingface-hub"]:
        packages[package] = importlib.metadata.version(package)
    manifest = {
        "status": "pilot_validation_only",
        "model": args.model, "model_revision": model_revision,
        "dataset": dataset_name, "dataset_revision": dataset_revision,
        "dataset_config": "wikitext-2-raw-v1", "split": "validation",
        "tokenizer": args.model, "sequence_length": args.sequence_length,
        "tokens": args.tokens, "seed": 0, "dtype": str(dtype),
        "layer": f"model.layers.{args.layer_index}.mlp.down_proj",
        "weight_codes": [-7, 7], "activation_codes": [-127, 127],
        "weight_scale": "per_output_row_absmax", "activation_scale": "per_token_absmax",
        "tf32": False, "compute": "FP32 exact-integer emulation under checked bound",
        "integer_abs_bound": original.in_features * 7 * 127,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "packages": packages, "platform": platform.platform(),
        "limitations": [
            "one frozen projection, one development sample, no training or statistical confirmation",
            "diagnostic reference matmul is included in modified-mode runtime",
            "quantized codes are stored in float32; no packed-memory or native-throughput claim",
            "separate blocks omit cross-block next-token predictions",
            "ordinary scales, other model layers and readout remain real-valued",
        ],
    }
    if args.device.startswith("cuda"):
        manifest["gpu"] = torch.cuda.get_device_name()
        manifest["gpu_capability"] = torch.cuda.get_device_capability()
        manifest["cuda_build"] = torch.version.cuda
        try:
            manifest["nvidia_smi"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"
            ], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            manifest["nvidia_smi"] = None
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.capture_attention:
        capture_attention(model, tokens, args.sequence_length, args.device, args.layer_index, args.output)
    results = {}
    modes = [("reference", None)] + [("integer_w4a8", None)] + [(f"residue_{k}", k) for k in args.residue_bits]
    for name, k in modes:
        wrapper = None
        if name != "reference":
            wrapper = ResidueLinear(original, k)
            model.model.layers[args.layer_index].mlp.down_proj = wrapper
        else:
            model.model.layers[args.layer_index].mlp.down_proj = original
        results[name] = evaluate(model, tokens, args.sequence_length, args.device, wrapper)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"mode": name, "perplexity": results[name]["perplexity"],
                          "wrap_fraction": results[name].get("wrap_fraction"),
                          "seconds": results[name]["wall_seconds_emulation_with_diagnostics"]}), flush=True)
    for name in results:
        if name.startswith("residue_") and results[name]["layer_statistics"]["wrap_count"] == 0:
            if results[name]["block_mean_nll"] != results["integer_w4a8"]["block_mean_nll"]:
                raise AssertionError("No-wrap residue result disagrees with quantized integer baseline")
    (args.output / "completed.json").write_text(json.dumps({
        "status": "completed", "no_wrap_equivalence_verified": True,
        "source_sha256": manifest["source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
