"""Frozen first-layer causal routing intervention using cached pilot text.

This performs CPU routing and dense GPU QK/value matmuls with a selected-key
mask. Timings and memory are diagnostic emulation measurements only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from padic_lm.routing import (
    PrefixTree, QuantileCodebook, angular_codes, angular_hyperplanes,
    digit_layout, pack_codes, select_causal_angular, select_causal_padic,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def validate_selection_mask(selected: np.ndarray, budget: int):
    """Require the same exact causal budget invariant for every method."""
    selected = np.asarray(selected)
    if selected.ndim != 4 or not all(selected.shape) or selected.shape[-2] != selected.shape[-1] or selected.dtype != bool:
        raise ValueError("selection must be a nonempty boolean [batch,head,sequence,sequence] mask")
    if budget < 1:
        raise ValueError("budget must be positive")
    if not np.all(selected.any(axis=-1)) or np.any(np.triu(selected, k=1)):
        raise AssertionError("selection mask must have nonempty rows and no future keys")
    expected = np.minimum(np.arange(1, selected.shape[-1] + 1), budget)
    if not np.array_equal(selected.sum(axis=-1), np.broadcast_to(expected, selected.shape[:-1])):
        raise AssertionError("selection mask does not match the exact causal key budget")


def build_oracle_selection_mask(stock_logits, *, budget: int, recent_window: int = 8):
    """Maximize dense softmax mass using actual stock-backend score ordering.

    BF16/FP16/FP32 values are copied losslessly to FP32 for deterministic CPU
    selection. No dot product is recomputed on CPU. Every row considers only
    its causal prefix; ties favor the newer key. Recent keys take precedence.
    This is a retained-attention-mass diagnostic, not an output/NLL oracle.
    """
    if stock_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("oracle requires native FP16/BF16/FP32 stock logits")
    if stock_logits.ndim != 4 or stock_logits.shape[-2] != stock_logits.shape[-1] or not all(stock_logits.shape):
        raise ValueError("oracle requires nonempty [batch,head,sequence,sequence] logits")
    if budget < 1 or recent_window < 0:
        raise ValueError("budget must be positive and recent window nonnegative")
    scores = stock_logits.detach().float().cpu().numpy()
    selected = np.zeros(scores.shape, dtype=bool)
    for batch in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            for position in range(scores.shape[2]):
                prefix = scores[batch, head, position, :position + 1]
                if np.any(np.isnan(prefix)) or np.any(np.isposinf(prefix)) or not np.any(np.isfinite(prefix)):
                    raise ValueError("causal oracle scores need finite mass and no NaN/+infinity")
                count = min(position + 1, budget)
                recent = min(recent_window, count)
                mandatory = np.arange(position + 1 - recent, position + 1)
                pool = np.arange(position + 1 - recent)
                rank = np.lexsort((-pool, -prefix[pool]))
                indices = np.concatenate((mandatory, pool[rank[:count - recent]]))
                selected[batch, head, position, indices] = True
    validate_selection_mask(selected, budget)
    return selected, {"oracle_logits": scores}


def build_selection_mask(
    queries: np.ndarray, keys: np.ndarray, *, method: str, budget: int,
    codebook=None, planes=None, recent_window: int = 8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select only each query's causal prefix; encoder fitting is external."""
    queries, keys = np.asarray(queries), np.asarray(keys)
    if queries.ndim != 4 or queries.shape != keys.shape or not all(queries.shape):
        raise ValueError("Q/K must have equal nonempty [batch,heads,sequence,width] shapes")
    if budget < 1 or recent_window < 0:
        raise ValueError("budget must be positive and recent window nonnegative")
    if method not in {"dense", "padic", "trie", "recency", "angular"}:
        raise ValueError("unknown routing method")
    batch, heads, length, _ = queries.shape
    artifacts = {}
    if method in {"padic", "trie"}:
        if codebook is None:
            raise ValueError("p-adic and trie modes require a calibrated codebook")
        qcodes, kcodes = codebook.encode(queries), codebook.encode(keys)
        artifacts = {"query_codes": qcodes, "key_codes": kcodes,
                     "packed_query_codes": pack_codes(qcodes, codebook.digits, prime=codebook.prime),
                     "packed_key_codes": pack_codes(kcodes, codebook.digits, prime=codebook.prime)}
    elif method == "angular":
        if planes is None:
            raise ValueError("angular mode requires fixed hyperplanes")
        qcodes, kcodes = angular_codes(queries, planes), angular_codes(keys, planes)
        artifacts = {"query_codes": qcodes, "key_codes": kcodes}
    selected = np.zeros((batch, heads, length, length), dtype=bool)
    for b in range(batch):
        for head in range(heads):
            tree = PrefixTree(qcodes.shape[-1], codebook.digits, prime=codebook.prime) if method == "trie" else None
            for position in range(length):
                if method == "dense":
                    indices = np.arange(position + 1)
                elif method == "recency":
                    indices = np.arange(max(0, position + 1 - budget), position + 1)
                elif method == "padic":
                    indices = select_causal_padic(qcodes[b, head, position], kcodes[b, head], position,
                                                 budget, digits=codebook.digits, prime=codebook.prime,
                                                 recent_window=recent_window)
                elif method == "trie":
                    tree.append(kcodes[b, head, position])
                    indices = tree.select(qcodes[b, head, position], budget, recent_window)
                else:
                    indices = select_causal_angular(qcodes[b, head, position], kcodes[b, head], position,
                                                   budget, recent_window)
                indices = np.asarray(indices, dtype=np.int64)
                expected_count = position + 1 if method == "dense" else min(position + 1, budget)
                if len(indices) != expected_count or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices > position):
                    raise AssertionError("selector returned an invalid causal key budget")
                selected[b, head, position, indices] = True
    validate_selection_mask(selected, length if method == "dense" else budget)
    return selected, artifacts


def stock_eager_logits(query, key, attention_mask, *, scaling: float, groups: int):
    """Use the original eager interface's native matmul, scale, and bias."""
    from transformers.models.llama.modeling_llama import repeat_kv
    key_states = repeat_kv(key, groups)
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if not attention_mask.dtype.is_floating_point:
            raise ValueError("attention bias must be a floating additive mask")
        logits = logits + attention_mask[:, :, :, :key_states.shape[-2]]
    return logits


def routed_eager_attention(query, key, value, selected, attention_mask, *, scaling: float, groups: int,
                           precomputed_logits=None):
    """Stock eager softmax arithmetic with GQA and a selected-key mask.

    Original Q/K/V dtypes and original scaling are retained. Full QK and value
    matmuls remain dense. Returns [batch,query,head,value_width], probabilities,
    and dense-reference attention mass captured by the selected keys.
    """
    from transformers.models.llama.modeling_llama import repeat_kv
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V must be rank four")
    if key.shape[:3] != value.shape[:3] or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V batch, head, sequence, or key-width mismatch")
    if groups < 1 or query.shape[1] != key.shape[1] * groups:
        raise ValueError("GQA groups do not match query and KV heads")
    expected = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
    if tuple(selected.shape) != expected or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean [batch,query-head,query,key] mask")
    if not bool(selected.any(dim=-1).all()):
        raise ValueError("every query/head must retain at least one key")
    value_states = repeat_kv(value, groups)
    logits = (stock_eager_logits(query, key, attention_mask, scaling=scaling, groups=groups)
              if precomputed_logits is None else precomputed_logits)
    if tuple(logits.shape) != expected or not logits.dtype.is_floating_point or logits.device != query.device:
        raise ValueError("precomputed logits must match the original stock score shape/device")
    dense_probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    retained_mass = (dense_probabilities * selected).sum(dim=-1)
    logits = logits.masked_fill(~selected, torch.finfo(logits.dtype).min)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, value_states).transpose(1, 2).contiguous()
    return output, probabilities, retained_mass


class RoutedLlamaAttention(torch.nn.Module):
    """Frozen original projections and RoPE; replace only the key selection."""
    def __init__(self, original, *, method, budget, codebook=None, planes=None, recent_window=8):
        super().__init__()
        if original.config._attn_implementation != "eager":
            raise ValueError("routing control requires original eager Llama attention")
        self.original = original
        self.method, self.budget = method, budget
        self.codebook, self.planes, self.recent_window = codebook, planes, recent_window
        self.last_diagnostics = None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                cache_position=None, **kwargs):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        if past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise ValueError("this fixed-block research intervention does not support KV caching")
        if self.training or self.original.training:
            raise RuntimeError("routing control requires evaluation mode and zero dropout")
        input_shape = hidden_states.shape[:-1]
        shape = (*input_shape, -1, self.original.head_dim)
        query = self.original.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.original.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.original.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        repeated_key = repeat_kv(key, self.original.num_key_value_groups)
        routing_start = time.monotonic()
        stock_logits = None
        if self.method == "oracle":
            stock_logits = stock_eager_logits(query, key, attention_mask, scaling=self.original.scaling,
                                              groups=self.original.num_key_value_groups)
            selected_np, codes = build_oracle_selection_mask(stock_logits, budget=self.budget,
                                                            recent_window=self.recent_window)
        else:
            q_cpu, k_cpu = query.detach().float().cpu().numpy(), repeated_key.detach().float().cpu().numpy()
            selected_np, codes = build_selection_mask(q_cpu, k_cpu, method=self.method, budget=self.budget,
                                                      codebook=self.codebook, planes=self.planes,
                                                      recent_window=self.recent_window)
        selected = torch.from_numpy(selected_np).to(query.device)
        routing_seconds = time.monotonic() - routing_start
        output, probabilities, retained_mass = routed_eager_attention(
            query, key, value, selected, attention_mask, scaling=self.original.scaling,
            groups=self.original.num_key_value_groups, precomputed_logits=stock_logits,
        )
        output = self.original.o_proj(output.reshape(*input_shape, -1).contiguous())
        self.last_diagnostics = {
            "selection_mask": selected_np, **codes,
            "retained_dense_attention_mass": retained_mass.detach().cpu().numpy(),
            "layer_output": output.detach().float().cpu().numpy(),
            "routing_and_transfer_seconds_emulation": routing_seconds,
        }
        if stock_logits is not None:
            self.last_diagnostics["oracle_logits_original_dtype"] = str(stock_logits.dtype)
        return output, probabilities


def state_dict_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_mode(model, chunks, source_chunk_indices, device, original, wrapper, output_dir):
    records, arrays, layer_outputs = [], {}, []
    captured = []
    hook = None
    if wrapper is None:
        hook = original.register_forward_hook(lambda module, inputs, result: captured.append(result[0].detach().float().cpu().numpy()))
    try:
        with torch.inference_mode():
            for evaluation_index, (chunk, source_index) in enumerate(zip(chunks, source_chunk_indices)):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.monotonic()
                batch = chunk.unsqueeze(0).to(device)
                logits = model(batch, use_cache=False).logits
                token_nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                            batch[:, 1:].reshape(-1), reduction="none")
                predictions = logits[:, :-1].argmax(dim=-1)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                nll_cpu = token_nll.cpu().numpy()
                arrays[f"chunk{source_index}_target_nll"] = nll_cpu
                arrays[f"chunk{source_index}_predicted_ids"] = predictions.cpu().numpy()
                diagnostics = {"layer_output": captured.pop()} if wrapper is None else wrapper.last_diagnostics
                layer_outputs.append(diagnostics["layer_output"])
                entry = {"evaluation_index": evaluation_index, "source_chunk_index": int(source_index),
                         "source_token_start": int(source_index) * chunk.numel(),
                         "predicted_tokens": int(token_nll.numel()), "mean_nll": float(token_nll.mean()),
                         "wall_seconds_emulation_with_diagnostics": elapsed,
                         "target_nll_sha256": hash_array(nll_cpu),
                         "layer_output_sha256": hash_array(diagnostics["layer_output"])}
                for name in ("selection_mask", "query_codes", "key_codes", "packed_query_codes", "packed_key_codes", "oracle_logits", "retained_dense_attention_mass"):
                    if name in diagnostics:
                        arrays[f"chunk{source_index}_{name}"] = diagnostics[name]
                        entry[f"{name}_sha256"] = hash_array(diagnostics[name])
                if wrapper is not None:
                    entry["selected_key_count"] = int(diagnostics["selection_mask"].sum())
                    entry["mean_retained_dense_attention_mass"] = float(diagnostics["retained_dense_attention_mass"].mean())
                    entry["routing_and_transfer_seconds_emulation"] = diagnostics["routing_and_transfer_seconds_emulation"]
                    if "oracle_logits_original_dtype" in diagnostics:
                        entry["oracle_logits_original_dtype"] = diagnostics["oracle_logits_original_dtype"]
                records.append(entry)
                if wrapper is not None:
                    wrapper.last_diagnostics = None
    finally:
        if hook is not None:
            hook.remove()
    arrays["layer_outputs"] = np.concatenate(layer_outputs, axis=0)
    arrays["source_chunk_indices"] = np.asarray(source_chunk_indices, dtype=np.int64)
    np.savez_compressed(output_dir / "outputs.npz", **arrays)
    (output_dir / "chunks.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    loss = float(np.mean([row["mean_nll"] for row in records]))
    return {"mean_nll": loss, "perplexity": math.exp(loss), "block_mean_nll": [r["mean_nll"] for r in records],
            "predicted_tokens": sum(r["predicted_tokens"] for r in records),
            "wall_seconds_emulation_with_diagnostics": sum(r["wall_seconds_emulation_with_diagnostics"] for r in records),
            "outputs_sha256": sha256_file(output_dir / "outputs.npz"),
            "chunks_sha256": sha256_file(output_dir / "chunks.json")}


def main():
    from transformers import AutoModelForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "results/llm-control-spark-20261003")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/llm-routing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--primes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--code-bit-budget", type=int, default=8)
    parser.add_argument("--coordinates", type=int, default=2)
    parser.add_argument("--calibration-chunks", type=int, default=2)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-oracle", action="store_true", help="add the largest-stock-logit retained-mass diagnostic")
    parser.add_argument("--controls-only", action="store_true", help="omit p-adic/trie fitting and evaluation")
    args = parser.parse_args()
    if min(args.budgets) < 1 or args.calibration_chunks < 1 or args.recent_window < 0 or args.seed < 0:
        raise ValueError("budgets/calibration must be positive; recency/seed nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a fresh output directory to preserve research results")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sources").mkdir()
    sources = [Path(__file__).resolve(), PROJECT / "src/padic_lm/routing.py", PROJECT / "tests/test_llm_routing.py"]
    for path in sources:
        shutil.copyfile(path, args.output / "sources" / path.name)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    input_manifest = json.loads((args.input / "manifest.json").read_text())
    trace_manifest = json.loads((args.input / "attention-trace-manifest.json").read_text())
    trace_path = args.input / "attention-trace.npz"
    if sha256_file(trace_path) != trace_manifest["sha256"]:
        raise ValueError("cached attention trace hash does not match its manifest")
    if (trace_manifest["layer_index"] != 0 or trace_manifest["rotary_position"] != "applied"
            or trace_manifest["stage"] != "unmodified reference model"):
        raise ValueError("this experiment requires first-layer post-RoPE reference traces")
    tokens = torch.tensor(json.loads((args.input / "input-token-ids.json").read_text()), dtype=torch.int64)
    if hashlib.sha256(tokens.numpy().tobytes()).hexdigest() != input_manifest["tokens_sha256"]:
        raise ValueError("cached token IDs do not match their manifest")
    length = input_manifest["sequence_length"]
    chunks = tokens.reshape(-1, length)
    with np.load(trace_path, allow_pickle=False) as trace:
        queries, keys = trace["queries"], trace["keys"]
    if queries.shape != keys.shape or queries.shape[0] != len(chunks) or queries.shape[2] != length:
        raise ValueError("trace chunks and token chunks do not align")
    if args.calibration_chunks >= len(chunks):
        raise ValueError("reserve at least one independent evaluation chunk")
    qcal, kcal = queries[:args.calibration_chunks], keys[:args.calibration_chunks]
    books = {} if args.controls_only else {
        p: QuantileCodebook.fit(qcal, kcal, coordinates=args.coordinates, seed=args.seed,
                               prime=p, code_bit_budget=args.code_bit_budget) for p in args.primes}
    planes = angular_hyperplanes(queries.shape[1], queries.shape[-1], args.code_bit_budget, args.seed)
    encoder_artifacts = {"angular_hyperplanes": planes}
    encoder_metadata = {}
    for p, book in books.items():
        encoder_artifacts.update({f"p{p}_center": book.center, f"p{p}_projection": book.projection,
                                  f"p{p}_thresholds": book.thresholds})
        layout = digit_layout(book.digits, args.coordinates, p)
        encoder_metadata[str(p)] = {"prime": p, "digits": list(layout), "coordinates": args.coordinates,
                                    "effective_entropy_bits": sum(layout) * math.log2(p),
                                    "projection_kind": book.projection_kind,
                                    "parameter_array_bytes": book.center.nbytes + book.projection.nbytes + book.thresholds.nbytes,
                                    "calibration_tokens_per_Q_or_K": book.calibration_tokens,
                                    "state_arrays_sha256": {name: hash_array(value) for name, value in
                                                            [("center", book.center), ("projection", book.projection), ("thresholds", book.thresholds)]}}
    np.savez_compressed(args.output / "encoder-state.npz", **encoder_artifacts)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(input_manifest["model"], revision=input_manifest["model_revision"],
                                                torch_dtype=dtype, attn_implementation="eager", use_safetensors=True).to(args.device).eval()
    original = model.model.layers[0].self_attn
    if original.__class__.__name__ != "LlamaAttention":
        raise TypeError("this intervention requires LlamaAttention")
    if queries.shape[1] != model.config.num_attention_heads or queries.shape[-1] != original.head_dim:
        raise ValueError("cached trace head layout does not match model")
    original_state_hash = state_dict_hash(model)
    indices = list(range(args.calibration_chunks, len(chunks)))
    evaluation_chunks = chunks[args.calibration_chunks:]
    (args.output / "evaluation-token-ids.json").write_text(json.dumps(evaluation_chunks.reshape(-1).tolist()) + "\n")
    manifest = {"status": "pilot_validation_only", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": input_manifest["model"], "model_revision": input_manifest["model_revision"],
                "model_loaded_state_sha256": original_state_hash, "dtype": str(dtype), "seed": args.seed,
                "input_directory": str(args.input.resolve()), "input_manifest_sha256": sha256_file(args.input / "manifest.json"),
                "trace_sha256": sha256_file(trace_path), "tokens_sha256": input_manifest["tokens_sha256"],
                "calibration_chunk_indices": list(range(args.calibration_chunks)), "evaluation_chunk_indices": indices,
                "source_chunk_mapping": "trace[c] = tokens[c*sequence_length:(c+1)*sequence_length]",
                "calibration_is_disjoint": True, "dataset": input_manifest["dataset"], "dataset_revision": input_manifest["dataset_revision"],
                "split": input_manifest["split"], "sequence_length": length, "layer_index": 0,
                "query_heads": model.config.num_attention_heads,
                "physical_kv_heads": model.config.num_key_value_heads,
                "gqa_groups": original.num_key_value_groups,
                "budget_scope": "per-query-head attention access; all physical KV states remain retained",
                "budgets": args.budgets, "primes": list(books), "code_bit_budget": args.code_bit_budget,
                "controls_only": args.controls_only,
                "oracle": {"enabled": args.include_oracle,
                           "objective": "largest actual stock logits maximize retained dense attention mass conditional on mandatory recent window",
                           "logits": "native torch.matmul(post-RoPE Q, repeated K transpose) * original scaling + original additive bias",
                           "backend": "same model device and native BF16/FP16/FP32 score operations as stock eager attention; same tensor reused for softmax",
                           "archive": "lossless native BF16/FP16/FP32 score values copied to float32; original score dtype recorded per chunk",
                           "selection": "causal prefix only; mandatory min(recent_window,budget,prefix_length) most recent keys; remaining slots descending score with recency ties",
                           "nll_or_value_error_optimality": False, "efficient_router": False},
                "recent_window": args.recent_window, "encoder": encoder_metadata,
                "encoder_state_sha256": sha256_file(args.output / "encoder-state.npz"),
                "angular_hyperplanes_sha256": hash_array(planes), "tf32": False,
                "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "huggingface-hub"]},
                "platform": platform.platform(), "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "source_sha256": {str(path.relative_to(PROJECT)): sha256_file(path) for path in sources},
                "limitations": ["Frozen first attention layer only; PCA/quantiles trained solely on first calibration chunks",
                                "Remaining validation chunks give descriptive pilot evidence; no held-out test or confirmation",
                                "Finite p-adic ball ranking and same-code radix tree are mathematically equivalent",
                                "Odd primes share an entropy cap; effective alphabets and precision differ",
                                "Working codes use uint16; separately archived packed codes do not imply native GPU packing",
                                "CPU routing with transfers, dense QK/value matmuls, diagnostics: no native speed/memory claim",
                                "Optional oracle maximizes retained dense attention mass only; it is not an output-error/NLL oracle or efficient router",
                                "Context resets per chunk and cross-chunk next-token targets are omitted"]}
    if args.device.startswith("cuda"):
        manifest["gpu"] = torch.cuda.get_device_name()
        manifest["cuda_build"] = torch.version.cuda
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    modes = [("reference", None, None, length), ("dense_control", "dense", None, length)]
    for budget in args.budgets:
        modes.extend([(f"recency_b{budget}", "recency", None, budget), (f"angular_b{budget}", "angular", None, budget)])
        if args.include_oracle:
            modes.append((f"oracle_b{budget}", "oracle", None, budget))
        for p in books:
            modes.extend([(f"padic_p{p}_b{budget}", "padic", p, budget), (f"trie_p{p}_b{budget}", "trie", p, budget)])
    results = {}
    for name, method, p, budget in modes:
        directory = args.output / name
        directory.mkdir()
        wrapper = None if method is None else RoutedLlamaAttention(original, method=method, budget=budget,
                                                                  codebook=books.get(p), planes=planes,
                                                                  recent_window=args.recent_window).eval()
        model.model.layers[0].self_attn = original if wrapper is None else wrapper
        result = evaluate_mode(model, evaluation_chunks, indices, args.device, original, wrapper, directory)
        result.update({"method": method or "reference", "prime": p, "key_budget": budget})
        if name != "reference":
            with np.load(args.output / "reference/outputs.npz", allow_pickle=False) as ref, np.load(directory / "outputs.npz", allow_pickle=False) as actual:
                error = actual["layer_outputs"].astype(np.float64) - ref["layer_outputs"].astype(np.float64)
                reference_norm = float(np.square(ref["layer_outputs"].astype(np.float64)).sum())
                result["attention_layer_nrmse_vs_reference"] = math.sqrt(float(np.square(error).sum()) / max(reference_norm, 1e-300))
        results[name] = result
        (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"mode": name, "mean_nll": result["mean_nll"], "perplexity": result["perplexity"]}), flush=True)
        if method == "trie":
            other = args.output / f"padic_p{p}_b{budget}"
            with np.load(other / "outputs.npz", allow_pickle=False) as left, np.load(directory / "outputs.npz", allow_pickle=False) as right:
                for key in left.files:
                    if not np.array_equal(left[key], right[key]):
                        raise AssertionError(f"same-code p-adic/trie discrepancy in {key}")
            if results[f"padic_p{p}_b{budget}"]["block_mean_nll"] != result["block_mean_nll"]:
                raise AssertionError("same-code p-adic/trie NLL differs")
    if results["dense_control"]["attention_layer_nrmse_vs_reference"] > (0.01 if dtype == torch.bfloat16 else 1e-5):
        raise AssertionError("full-budget routed attention differs from stock eager attention")
    max_dense_nll_difference = max(abs(a - b) for a, b in zip(results["reference"]["block_mean_nll"], results["dense_control"]["block_mean_nll"]))
    if max_dense_nll_difference > (0.01 if dtype == torch.bfloat16 else 1e-5):
        raise AssertionError("full-budget NLL differs from stock eager attention")
    oracle_mass_max_shortfall = 0.0
    oracle_mass_pairs_checked = 0
    if args.include_oracle:
        for budget in args.budgets:
            with np.load(args.output / f"oracle_b{budget}/outputs.npz", allow_pickle=False) as oracle:
                for name, result in results.items():
                    if result["key_budget"] != budget or result["method"] in {"reference", "dense", "oracle"}:
                        continue
                    with np.load(args.output / name / "outputs.npz", allow_pickle=False) as other:
                        for source_index in indices:
                            bound = oracle[f"chunk{source_index}_retained_dense_attention_mass"]
                            candidate = other[f"chunk{source_index}_retained_dense_attention_mass"]
                            oracle_mass_max_shortfall = max(oracle_mass_max_shortfall, float(np.max(candidate - bound)))
                            oracle_mass_pairs_checked += 1
        if oracle_mass_max_shortfall > 1e-6:
            raise AssertionError("oracle failed retained-attention-mass upper control under same causal/recent budget")
    model.model.layers[0].self_attn = original
    if state_dict_hash(model) != original_state_hash:
        raise AssertionError("original model weights changed during frozen evaluation")
    trie_pairs = sum(result["method"] == "trie" for result in results.values())
    (args.output / "completed.json").write_text(json.dumps({"status": "completed", "same_code_trie_equivalence_verified": True if trie_pairs else None,
                                                           "same_code_trie_pairs_checked": trie_pairs,
                                                           "full_budget_stock_eager_verified": True,
                                                           "oracle_conditional_mass_control_verified": True if args.include_oracle else None,
                                                           "oracle_mass_mode_chunk_pairs_checked": oracle_mass_pairs_checked,
                                                           "oracle_mass_max_shortfall": oracle_mass_max_shortfall if args.include_oracle else None,
                                                           "max_dense_block_nll_difference": max_dense_nll_difference,
                                                           "model_weights_unchanged": True,
                                                           "manifest_sha256": sha256_file(args.output / "manifest.json"),
                                                           "results_sha256": sha256_file(args.output / "results.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
