"""Frozen untouched-test group-shared quality evaluation; dense emulation."""
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
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from padic_lm import shared_budget, routing, flat_routing, learned, development, heldout
from padic_lm.shared_budget import (shared_rank_mask, fixed_group_mask, depth_row,
                                   grid_coordinates, grid_row, mass_group_mask, validate_group_mask)
from padic_lm.learned import folded_latents
from padic_lm.routing import pack_codes
from padic_lm.flat_routing import FlatKeyCodebook
from padic_lm.development import paired_quality
import evaluate_qk_bridge as prior_driver
import llm_routing
from evaluate_qk_bridge import (TRAIN_HASH, MODEL_HASH, SEEDS, verify_manifest, arrays, book_from_state,
                                digest, json_write, bf16_bits, baseline_forward, replay, native_metrics)
from llm_routing import state_dict_hash

KINDS = ("p2", "p3", "coarsened_p2", "grid_p2", "grid_p3", "flat", "gaussian", "gaussian_local64")
METHODS = ["full_control", "recency", "sink_recency", "uniform", "mass_upper"] + [f"s{s}_{k}" for s in SEEDS for k in KINDS]


def freeze_books(training):
    result = {}
    for seed in SEEDS:
        d = training / f"seed-{seed}"
        state = arrays(d / "routing_state.npz")
        result[seed] = {"encoder": arrays(d / "final_encoder.npz"),
                        "p2": book_from_state(state, "p2_", seed, 2),
                        "p3": book_from_state(state, "p3_", seed, 3),
                        "flat": FlatKeyCodebook(**{n[5:]: a for n, a in state.items() if n.startswith("flat_")}),
                        "numeric_state": json.loads((d / "results.json").read_text())["standalone_numeric_state_bytes"]}
    return result


def feature_states(q, k, frozen):
    result = {}
    saved = {}
    timing = {}
    for seed, state in frozen.items():
        torch.cuda.synchronize()
        encoder_started = time.monotonic()
        lq = folded_latents(q, state["encoder"], "query").cpu().numpy()[0]
        lk = folded_latents(k, state["encoder"], "key").cpu().numpy()[0]
        timing[seed] = {"real_encoder_and_gpu_to_cpu_seconds": time.monotonic() - encoder_started}
        metadata_started = time.monotonic()
        item = {"lq": lq, "lk": lk}
        saved[f"s{seed}_query_latents"] = lq
        saved[f"s{seed}_key_latents"] = lk
        for kind in ("p2", "p3"):
            book = state[kind]
            qc, kc = book.encode(lq[None])[0], book.encode(lk[None])[0]
            gq, scales = grid_coordinates(qc, book.digits, book.prime)
            gk, _ = grid_coordinates(kc, book.digits, book.prime)
            item[kind] = {"q": qc, "k": kc, "gq": gq, "gk": gk, "book": book, "scales": scales}
            saved[f"s{seed}_{kind}_query_codes"] = pack_codes(qc, book.digits, book.prime)
            saved[f"s{seed}_{kind}_key_codes"] = pack_codes(kc, book.digits, book.prime)
        flat = state["flat"]
        item["flat_q"] = flat.project_queries(lq[None])[0]
        item["flat_k"] = flat.encode_keys(lk[None])[0]
        saved[f"s{seed}_flat_key_codes"] = item["flat_k"]
        # Small query-to-centroid matrix, not a dense query/key distance matrix.
        item["flat_scores"] = np.stack([-((item["flat_q"][h, :, None] - flat.centroids[h, :int(flat.centroid_counts[h])]) ** 2).sum(-1)
                                         for h in range(15)])
        result[seed] = item
        timing[seed]["finite_flat_metadata_seconds"] = time.monotonic() - metadata_started
    return result, saved, timing


def score_provider(item, kind):
    if kind in ("p2", "p3", "coarsened_p2"):
        field = item["p3" if kind == "p3" else "p2"]
        def score(pos, stop):
            depth = depth_row(field["q"][:, pos], field["k"][:, :stop], field["book"].digits, field["book"].prime)
            return depth // 2 if kind == "coarsened_p2" else depth
        return score
    if kind in ("grid_p2", "grid_p3"):
        field = item[kind[5:]]
        return lambda pos, stop: grid_row(field["gq"][:, pos], field["gk"][:, :stop])
    if kind == "flat":
        return lambda pos, stop: np.stack([item["flat_scores"][h, pos, item["flat_k"][h, :stop]] for h in range(15)])
    if kind in ("gaussian", "gaussian_local64"):
        def score(pos, stop):
            delta = item["lq"][:, pos, None] - item["lk"][:, :stop]
            return -(delta * delta).sum(-1) / np.float32(2)
        return score
    raise ValueError("unknown fixed score recipe")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh output required")
    preflight = json.loads(args.preflight.read_text())
    if (preflight["status"] != "heldout_feature_preflight_passed"
            or preflight["test_driver_sha256"] != digest(__file__)
            or preflight["feature_arrays_exact_against_both_original_and_archive"] != 21
            or preflight["score_rows_exact_against_original"] != 96):
        raise ValueError("requires exact-feature preflight for the current test driver")
    data_hash = (args.data / "completed.sha256").read_text().strip()
    verify_manifest(args.data, data_hash)
    verify_manifest(args.training, TRAIN_HASH)
    if (digest(args.development / "completed.json") != heldout.DEVELOPMENT_HASH
            or digest(args.development_audit) != heldout.DEVELOPMENT_AUDIT_HASH):
        raise ValueError("passed development reference differs")
    parent_audit = json.loads(args.development_audit.read_text())
    if parent_audit["status"] != "shared_budget_audit_passed":
        raise ValueError("development must be fully verified")
    data = json.loads((args.data / "manifest.json").read_text())
    data_audit = json.loads(args.data_audit.read_text())
    if data_audit["status"] != "heldout_context_data_verified" or data_audit["data_completion_sha256"] != data_hash:
        raise ValueError("requires independent verification of all frozen test contexts")
    protocol = args.protocol.read_bytes()
    if hashlib.sha256(protocol).hexdigest() != data["protocol_sha256"] or data["protocol_sha256"] != heldout.PROTOCOL_HASH:
        raise ValueError("protocol changed after freezing data")
    count = data["article_count"]
    if count < 1 or count != data["windows"] or data["lengths"] != [2048] or not data["test_split_read"]:
        raise ValueError("all eligible first2048-token test articles required")
    if data_audit["eligible_articles_verified"] != count:
        raise ValueError("test document count differs from its independent verification")
    tokens = arrays(args.data / "tokens.npz")
    if tokens["tokens2048"].shape != (count, 2048):
        raise ValueError("frozen test token shape differs")
    prior_seconds = data_audit["cumulative_gate_seconds"]
    def check_time():
        if prior_seconds + time.monotonic() - started >= 10800:
            raise TimeoutError("held-out cumulative10800-second ceiling reached")
    frozen = freeze_books(args.training)
    paths = [Path(__file__), Path(prior_driver.__file__), Path(shared_budget.__file__), Path(routing.__file__),
             Path(flat_routing.__file__), Path(learned.__file__), Path(development.__file__), Path(llm_routing.__file__), Path(heldout.__file__)]
    sources = {p.name: p.read_bytes() for p in paths}
    parent_completion = json.loads((args.development / "completed.json").read_text())
    if digest(args.development / "config.json") != parent_completion["artifact_sha256"]["config.json"]:
        raise ValueError("development source manifest differs")
    pinned_sources = json.loads((args.development / "config.json").read_text())["source_sha256"]
    for name, content in sources.items():
        if name in pinned_sources and hashlib.sha256(content).hexdigest() != pinned_sources[name]:
            raise ValueError(f"frozen production source differs: {name}")
    args.output.mkdir(parents=True)
    (args.output / "sources").mkdir()
    (args.output / "windows").mkdir()
    for name, content in sources.items():
        (args.output / "sources" / name).write_bytes(content)
    (args.output / "prospective_protocol.txt").write_bytes(protocol)
    np.savez_compressed(args.output / "additional_numeric_state.npz",
                        binary_grid_scales=np.array([1, 1], dtype=np.int64),
                        ternary_grid_scales=np.array([4, 13], dtype=np.int64),
                        coarsened_binary_stride=np.array([2], dtype=np.uint8))
    config = {"stage": "heldout_gqa_quality", "created_utc": datetime.now(timezone.utc).isoformat(),
              "data_completion_sha256": data_hash, "data_audit_sha256": digest(args.data_audit),
              "feature_preflight_sha256": digest(args.preflight),
              "training_completion_sha256": TRAIN_HASH, "development_completion_sha256": heldout.DEVELOPMENT_HASH, "development_audit_sha256": heldout.DEVELOPMENT_AUDIT_HASH,
              "protocol_sha256": hashlib.sha256(protocol).hexdigest(), "model": data["model"], "model_revision": data["model_revision"],
              "methods": METHODS, "seeds": list(SEEDS), "layer": 0, "lengths": [2048], "articles": data["articles"],
              "group_size": 3, "groups": 5, "group_budget": 128, "mandatory_recent": 8, "gaussian_local64_recent": 64,
              "prior_gate_seconds": prior_seconds, "cumulative_wall_limit_seconds": 10800,
              "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
              "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")}, "command": sys.argv,
              "ranking_arithmetic": "CPU exact integer twice-midranks; FP32 NumPy Gaussian direct distances; FP64 centroid scores",
              "scope": "Frozen official-test articles; fixed three seeds; dense numerical readout; no classification before numerical and statistical audit"}
    json_write(args.output / "config.json", config)
    complete = 0
    totals = {}
    feature_timings = []
    try:
        check_time()
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("Spark CUDA evaluation required")
        model = AutoModelForCausalLM.from_pretrained(data["model"], revision=data["model_revision"],
                    torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
        if (state_dict_hash(model) != MODEL_HASH or model.config.max_position_embeddings < 2048
                or (model.config.num_attention_heads, model.config.num_key_value_heads, model.config.num_hidden_layers) != (15, 5, 32)):
            raise ValueError("pinned model differs or context unsupported")
        original = model.model.layers[0].self_attn
        np.savez_compressed(args.output / "output_projection.npz", weight_bits=bf16_bits(original.o_proj.weight))
        from transformers.models.llama.modeling_llama import repeat_kv
        with torch.inference_mode():
            for length in (2048,):
                reference_losses = []
                aggregate = {n: {"loss": [], "pe": [], "ve": [], "mass": [], "overlap": [], "route_seconds": 0., "forward_seconds": 0.} for n in METHODS}
                energies, value_energies = [], []
                for index, token_ids in enumerate(tokens[f"tokens{length}"]):
                    check_time()
                    batch = torch.tensor(token_ids[None], dtype=torch.int64, device="cuda")
                    reference, ref_loss, ref_prediction = baseline_forward(model, batch)
                    q = reference["q"].float()
                    k = repeat_kv(reference["k"], 3).float()
                    v = repeat_kv(reference["v"], 3)
                    for h in range(15):
                        if not torch.equal(k[:, h], k[:, h // 3 * 3]) or not torch.equal(v[:, h], v[:, h // 3 * 3]):
                            raise AssertionError("GQA repeated K/V differ")
                    states, saved, feature_timing = feature_states(q, k, frozen)
                    feature_timings.append(feature_timing)
                    probability = torch.softmax(reference["scores"], dim=-1, dtype=torch.float32).cpu().numpy()[0]
                    saved.update(tokens=token_ids, reference_target_nll=ref_loss, reference_predictions=ref_prediction,
                        reference_queries_bits=bf16_bits(reference["q"])[0], reference_keys_bits=bf16_bits(reference["k"])[0],
                        reference_values_bits=bf16_bits(reference["v"])[0], reference_scores_bits=bf16_bits(reference["scores"])[0],
                        reference_native_values_bits=bf16_bits(reference["values"])[0], reference_projected_bits=bf16_bits(reference["projected"])[0],
                        reference_q_fp32_stride=np.array(q.stride()), reference_k_fp32_stride=np.array(k.stride()))
                    reference_losses.append(ref_loss)
                    recency = fixed_group_mask(length, 5, "recency")
                    upper_mass = None
                    for name in METHODS:
                        check_time()
                        ts = time.monotonic()
                        recent = 64 if name.endswith("gaussian_local64") else 8
                        if name in ("full_control", "recency", "sink_recency", "uniform"):
                            groups = fixed_group_mask(length, 5, "full" if name == "full_control" else name)
                        elif name == "mass_upper":
                            groups = mass_group_mask(probability)
                        else:
                            seed, kind = name.split("_", 1)
                            groups = shared_rank_mask(length, 15, score_provider(states[int(seed[1:])], kind), recent=recent)
                        validate_group_mask(groups, budget=length if name == "full_control" else 128, recent=recent)
                        selected = np.repeat(groups, 3, axis=0)[None]
                        record = aggregate[name]
                        record["route_seconds"] += time.monotonic() - ts
                        torch.cuda.synchronize()
                        ts = time.monotonic()
                        actual, loss, prediction = replay(model, original, batch, selected, reference)
                        torch.cuda.synchronize()
                        record["forward_seconds"] += time.monotonic() - ts
                        if not np.array_equal(loss[:128], ref_loss[:128]):
                            raise AssertionError("unaffected early target losses changed")
                        if name == "full_control" and (not torch.equal(actual["projected"], reference["projected"]) or not torch.equal(actual["values"], reference["values"]) or not np.array_equal(loss, ref_loss)):
                            raise AssertionError("full-budget stock fixture differs")
                        pe, energy, ve, value_energy = native_metrics(actual, reference)
                        mass = actual["mass"].cpu().numpy()[0]
                        group_mass = mass.reshape(5, 3, length).mean(1)
                        if name == "mass_upper":
                            upper_mass = group_mass
                        # Native upper control is evaluated before feature-based methods.
                        if upper_mass is not None and name not in ("full_control", "recency", "sink_recency", "uniform", "mass_upper") and np.any(group_mass > upper_mass + 5e-7):
                            raise AssertionError("group mass upper bound violated")
                        overlap = (groups & recency).sum(-1).astype(np.uint16)
                        record["loss"].append(loss)
                        record["pe"].append(pe)
                        record["ve"].append(ve)
                        record["mass"].append(mass)
                        record["overlap"].append(overlap)
                        native_bits = bf16_bits(actual["values"])[0]
                        saved.update({name + "_target_nll": loss, name + "_predictions": prediction,
                            name + "_group_selection_bits": np.packbits(groups, axis=-1, bitorder="little"),
                            name + "_projected_bits": bf16_bits(actual["projected"])[0], name + "_projected_error": pe,
                            name + "_native_value_error": ve, name + "_retained_mass": mass,
                            name + "_recency_overlap": overlap,
                            name + "_native_values_sha256": np.array(hashlib.sha256(native_bits.tobytes()).hexdigest())})
                    saved["projected_reference_energy"] = energy
                    saved["native_value_reference_energy"] = value_energy
                    saved["mass_upper_covers_all_sparse"] = np.array(all(np.all(saved[n + "_retained_mass"].reshape(5, 3, length).mean(1) <= upper_mass + 5e-7) for n in METHODS if n != "full_control"))
                    if not saved["mass_upper_covers_all_sparse"]:
                        raise AssertionError("upper control misses sparse comparator")
                    energies.append(energy)
                    value_energies.append(value_energy)
                    check_time()
                    np.savez_compressed(args.output / "windows" / f"length-{length}-article-{index:02d}.npz", **saved)
                    complete += 1
                    json_write(args.output / "progress.json", {"complete_windows": complete, "total_windows": count,
                        "length": length, "article_index": index, "cumulative_stage_seconds": prior_seconds + time.monotonic() - started})
                    print(json.dumps({"complete_windows": complete, "length": length, "article": index,
                                      "cumulative_seconds": round(prior_seconds + time.monotonic() - started, 3)}), flush=True)
                reference_losses = np.stack(reference_losses)
                energy, value_energy = np.stack(energies), np.stack(value_energies)
                summaries = {}
                for name, r in aggregate.items():
                    loss, pe, ve, mass, overlap = [np.stack(r[k]) for k in ("loss", "pe", "ve", "mass", "overlap")]
                    state_bytes = key_bytes = 0
                    if name.startswith("s") and name not in ("sink_recency",):
                        seed, kind = name.split("_", 1)
                        states_bytes = frozen[int(seed[1:])]["numeric_state"]
                        gaussian = kind.startswith("gaussian")
                        state_bytes = 15612 if gaussian else states_bytes["flat" if kind == "flat" else "p3" if kind.endswith("p3") else "p2"]
                        state_bytes += 16 if kind.startswith("grid") else 1 if kind == "coarsened_p2" else 0
                        key_bytes = 15 * length * (8 if gaussian else 1)
                    summaries[name] = {**paired_quality(loss, reference_losses, data["articles"]),
                        "affected_projected_nrmse": float(np.sqrt(pe[:, 128:].sum() / energy[:, 128:].sum())),
                        "affected_native_value_nrmse": float(np.sqrt(ve[:, :, 128:].sum() / value_energy[:, :, 128:].sum())),
                        "affected_mean_retained_mass": float(mass[:, :, 128:].astype(np.float64).mean()),
                        "affected_recency_shared_keys": float(overlap[:, :, 128:].mean()),
                        "affected_mean_group_union": (length + 129) / 2 if name == "full_control" else 128,
                        "numeric_state_bytes": state_bytes, "key_metadata_bytes_per_window": key_bytes,
                        "routing_wall_seconds": r["route_seconds"], "model_forward_with_diagnostics_wall_seconds": r["forward_seconds"],
                        "scope": "Frozen test documents and fixed three seeds; statistical classification requires audited document bootstrap"}
                totals[str(length)] = {"reference": paired_quality(reference_losses, reference_losses, data["articles"]), "methods": summaries}
                totals[str(length)]["reference"]["scope"] = "Frozen official-test document reference"
                json_write(args.output / f"results-{length}.json", totals[str(length)])
        if state_dict_hash(model) != MODEL_HASH:
            raise AssertionError("original model parameters changed")
        verify_manifest(args.training, TRAIN_HASH)
        verify_manifest(args.data, data_hash)
        if args.protocol.read_bytes() != protocol or any(p.read_bytes() != sources[p.name] for p in paths):
            raise RuntimeError("executed protocol/source changed during evaluation")
        result = {"status": "heldout_quality_evaluation_complete", "config": config, "length_results": totals,
            "completed_windows": complete, "stock_full_control_exact_windows": count,
            "original_model_state_unchanged": True, "new_fitting_or_tuning": False, "test_split_read": True,
            "evaluation_wall_seconds": time.monotonic() - started, "cumulative_stage_seconds": prior_seconds + time.monotonic() - started,
            "feature_construction_timings_by_window": feature_timings,
            "feature_timing_scope": "Each seed: real affine Q/K maps plus GPU-to-CPU transfers; finite/flat metadata separately. Shared all-readout work, not deployed method latency.",
            "buffer_payloads_by_length": {str(n): {"cpu_group_mask_bool": 5 * n * n,
                "expanded_cpu_head_mask_bool": 15 * n * n, "gpu_head_mask_bool": 15 * n * n,
                "native_bf16_scores": 30 * n * n, "one_fp32_probability_matrix": 60 * n * n,
                "physical_original_bf16_kv": 5 * n * 64 * 4} for n in (2048,)},
            "scope": "Frozen held-out quality evaluation; real KV retained; no statistical classification or native benefit yet"}
        json_write(args.output / "results.json", result)
        check_time()
        artifacts = sorted(p for p in args.output.rglob("*") if p.is_file())
        artifact_hashes = {str(p.relative_to(args.output)): digest(p) for p in artifacts}
        artifact_bytes = {str(p.relative_to(args.output)): p.stat().st_size for p in artifacts}
        check_time()
        # Sample the closing clock after the bulk artifact manifest hashing.
        json_write(args.output / "completed.json", {"status": result["status"], "data_completion_sha256": data_hash,
            "cumulative_stage_seconds": prior_seconds + time.monotonic() - started,
            "artifact_sha256": artifact_hashes, "artifact_bytes": artifact_bytes})
        check_time()
        (args.output / "completed.sha256").write_text(digest(args.output / "completed.json") + "\n")
        print(json.dumps({"status": result["status"], "completed_windows": complete, "cumulative_seconds": result["cumulative_stage_seconds"]}), flush=True)
    except Exception as error:
        (args.output / "completed.json").unlink(missing_ok=True)
        (args.output / "completed.sha256").unlink(missing_ok=True)
        json_write(args.output / "failed.json", {"status": "heldout_quality_evaluation_failed", "type": type(error).__name__,
            "message": str(error), "completed_windows": complete, "cumulative_stage_seconds": prior_seconds + time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
