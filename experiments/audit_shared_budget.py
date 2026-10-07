"""Separate prefix/count-rank/native-readout verification; no production imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

CONTEXT = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k].copy() for k in raw.files}


def native(bits):
    return torch.tensor(bits, dtype=torch.uint16, device="cuda").view(torch.bfloat16)


def bits(tensor):
    return tensor.contiguous().view(torch.uint16).cpu().numpy()


def quantiles(latents, state, prefix, prime):
    heads, length, _ = latents.shape
    digits = state[prefix + "digits"]
    residues = np.empty((heads, length, 2), dtype=np.uint16)
    ordinals = np.empty((heads, length, 2), dtype=np.int64)
    for h in range(heads):
        point = (latents[h].astype(np.float64) - state[prefix + "center"][h]) @ state[prefix + "projection"][h]
        for coordinate, count in enumerate(digits):
            cuts = state[prefix + "thresholds"][h, coordinate, :prime ** int(count) - 1]
            # Explicit inequalities, not the production searchsorted routine.
            rank = (point[:, coordinate, None] >= cuts[None]).sum(-1)
            ordinals[h, :, coordinate] = rank
            reverse = np.zeros(length, dtype=np.int64)
            for _ in range(int(count)):
                reverse = reverse * prime + rank % prime
                rank = rank // prime
            residues[h, :, coordinate] = reverse
    packed = (residues[..., 0] + residues[..., 1] * prime ** int(digits[0])).astype(np.uint8)[..., None]
    return residues, ordinals, packed


def prefix_depth(query, keys, digits, prime):
    # Ordinary common digit-prefix traversal, no p-adic subtraction/modulus test.
    alive = np.ones(keys.shape[:2], dtype=bool)
    result = np.zeros(keys.shape[:2], dtype=np.int16)
    for level in range(max(int(d) for d in digits)):
        for coordinate, count in enumerate(digits):
            if level < int(count):
                alive &= (query[:, coordinate, None] // prime ** level) % prime == (keys[..., coordinate] // prime ** level) % prime
        result += alive
    return result


def count_ranks(scores):
    # Histogram frequencies yield exact midranks without sorting individual keys.
    result = np.empty(scores.shape, dtype=np.int32)
    for h, row in enumerate(scores):
        _, inverse, count = np.unique(row, return_inverse=True, return_counts=True)
        below = np.cumsum(count) - count
        result[h] = (2 * below + count - 1)[inverse]
    return result


def truth_mask(length, name, states, probabilities):
    mask = np.zeros((5, length, length), dtype=bool)
    seed = kind = None
    if name.startswith("s") and name != "sink_recency":
        seed, kind = name.split("_", 1)
        s = states[int(seed[1:])]
    local = 64 if kind == "gaussian_local64" else 8
    for pos in range(length):
        if pos < 128 or name == "full_control":
            mask[:, pos, :pos + 1] = True
            continue
        if name == "recency":
            mask[:, pos, pos - 127:pos + 1] = True
            continue
        if name == "sink_recency":
            mask[:, pos, :4] = True
            mask[:, pos, pos - 123:pos + 1] = True
            continue
        stop = pos + 1 - local
        mask[:, pos, stop:pos + 1] = True
        if name == "uniform":
            indices = [((2 * i + 1) * stop) // 240 for i in range(120)]
            mask[:, pos, indices] = True
            continue
        if name == "mass_upper":
            ranking = probabilities[:, pos, :stop].astype(np.float64).reshape(5, 3, stop).sum(1)
        else:
            if kind in ("p2", "p3", "coarsened_p2"):
                key = "p3" if kind == "p3" else "p2"
                field = s[key]
                score = prefix_depth(field["q"][:, pos], field["k"][:, :stop], field["digits"], field["prime"])
                if kind == "coarsened_p2":
                    score //= 2
            elif kind.startswith("grid"):
                field = s[kind[5:]]
                # Real ordinal differences with the explicitly fixed normalization.
                scales = np.array([1, 1] if kind == "grid_p2" else [4, 13], dtype=np.int64)
                delta = (field["qo"][:, pos, None] - field["ko"][:, :stop]) * scales
                score = -(delta * delta).sum(-1)
            elif kind == "flat":
                score = np.stack([-((s["flat_q"][h, pos] - s["centers"][h, s["flat_k"][h, :stop]]) ** 2).sum(-1) for h in range(15)])
            else:
                delta = s["lq"][:, pos, None] - s["lk"][:, :stop]
                score = -(delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]) * np.float32(0.5)
            ranking = count_ranks(score).reshape(5, 3, stop).sum(1, dtype=np.int32)
        keys = np.arange(stop)
        for group in range(5):
            order = np.lexsort((-keys, -ranking[group]))[:128 - local]
            mask[group, pos, order] = True
    return mask


def reconstructed_states(window, states):
    result = {}
    for seed, state in states.items():
        lq, lk = window[f"s{seed}_query_latents"], window[f"s{seed}_key_latents"]
        item = {"lq": lq, "lk": lk}
        for prime in (2, 3):
            prefix = f"p{prime}_"
            q, qo, qp = quantiles(lq, state, prefix, prime)
            k, ko, kp = quantiles(lk, state, prefix, prime)
            assert np.array_equal(qp, window[f"s{seed}_{prefix}query_codes"])
            assert np.array_equal(kp, window[f"s{seed}_{prefix}key_codes"])
            item[f"p{prime}"] = {"q": q, "k": k, "qo": qo, "ko": ko, "digits": state[prefix + "digits"], "prime": prime}
        fq, fk, codes = [], [], []
        for h in range(15):
            pq = (lq[h].astype(np.float64) - state["flat_center"][h]) @ state["flat_projection"][h]
            pk = (lk[h].astype(np.float64) - state["flat_center"][h]) @ state["flat_projection"][h]
            centers = state["flat_centroids"][h, :int(state["flat_centroid_counts"][h])]
            code = np.argmin(((pk[:, None] - centers) ** 2).sum(-1), axis=-1)
            fq.append(pq)
            fk.append(pk)
            codes.append(code)
        item["flat_q"] = np.stack(fq)
        item["flat_k"] = np.stack(codes)
        item["centers"] = state["flat_centroids"]
        assert np.array_equal(item["flat_k"], window[f"s{seed}_flat_key_codes"])
        result[seed] = item
    return result


def verify_manifest(directory, expected=None):
    c = json.loads((directory / "completed.json").read_text())
    h = digest(directory / "completed.json")
    assert h == (directory / "completed.sha256").read_text().strip()
    if expected:
        assert h == expected
    for name, value in c["artifact_sha256"].items():
        assert digest(directory / name) == value
        assert (directory / name).stat().st_size == c["artifact_bytes"][name]
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "training", "data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh verification directory required")
    args.output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(source)
    CONTEXT.update(output=args.output, started=started, prior=0.)
    complete = verify_manifest(args.input)
    verify_manifest(args.training, "f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1")
    verify_manifest(args.data, complete["data_completion_sha256"])
    result = json.loads((args.input / "results.json").read_text())
    config = result["config"]
    data = json.loads((args.data / "manifest.json").read_text())
    assert result["status"] == "shared_budget_development_complete"
    assert config["articles"] == data["articles"] and len(data["articles"]) == 43
    assert config["data_completion_sha256"] == digest(args.data / "completed.json")
    assert (config["group_budget"], config["group_size"], config["groups"], config["lengths"]) == (128, 3, 5, [512, 2048])
    assert config["protocol_sha256"] == digest(args.input / "prospective_protocol.txt") == data["protocol_sha256"]
    expected = ["full_control", "recency", "sink_recency", "uniform", "mass_upper"]
    kinds = ("p2", "p3", "coarsened_p2", "grid_p2", "grid_p3", "flat", "gaussian", "gaussian_local64")
    for seed in (17, 29, 43):
        expected += [f"s{seed}_{kind}" for kind in kinds]
    assert config["methods"] == expected and len(expected) == 29
    assert not result["test_split_read"] and not result["new_fitting_or_tuning"]
    for name, h in config["source_sha256"].items():
        assert digest(args.input / "sources" / name) == h
    prior = complete["cumulative_stage_seconds"]
    CONTEXT["prior"] = prior
    def check_time():
        if prior + time.monotonic() - started >= 7200:
            raise TimeoutError("separate shared-budget evaluation/audit ceiling reached")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids = load(args.data / "tokens.npz")
    states = {s: load(args.training / f"seed-{s}" / "routing_state.npz") for s in (17, 29, 43)}
    maps = {s: load(args.training / f"seed-{s}" / "final_encoder.npz") for s in states}
    state_bytes = {s: json.loads((args.training / f"seed-{s}" / "results.json").read_text())["standalone_numeric_state_bytes"] for s in states}
    weight = native(load(args.input / "output_projection.npz")["weight_bits"])
    group_rows = native_elements = projected_elements = latent_elements = window_count = summary_checks = 0
    max_metric_difference = 0.
    first = {}
    extra = load(args.input / "additional_numeric_state.npz")
    assert np.array_equal(extra["binary_grid_scales"], [1, 1]) and extra["binary_grid_scales"].nbytes == 16
    assert np.array_equal(extra["ternary_grid_scales"], [4, 13]) and extra["ternary_grid_scales"].nbytes == 16
    assert np.array_equal(extra["coarsened_binary_stride"], [2]) and extra["coarsened_binary_stride"].nbytes == 1
    with torch.inference_mode():
        for length in (512, 2048):
            refs = []
            totals = {name: {key: [] for key in ("loss", "pe", "ve", "mass", "overlap")} for name in expected}
            energies, value_energies = [], []
            causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
            for index in range(43):
                check_time()
                w = load(args.input / "windows" / f"length-{length}-article-{index:02d}.npz")
                assert np.array_equal(w["tokens"], ids[f"tokens{length}"][index])
                refs.append(w["reference_target_nll"])
                q = native(w["reference_queries_bits"])[None]
                k = native(w["reference_keys_bits"]).repeat_interleave(3, dim=0)[None]
                v = native(w["reference_values_bits"]).repeat_interleave(3, dim=0)[None]
                # Preserve the recorded native matrix layout as well as values.
                # The FP32 copies used by the encoder preserve BF16 input strides.
                restored = {}
                for role, value in (("q", q), ("k", k)):
                    t = torch.empty_strided(value.shape, tuple(int(a) for a in w[f"reference_{role}_fp32_stride"]),
                                            dtype=torch.bfloat16, device="cuda")
                    t.copy_(value)
                    restored[role] = t
                q, k = restored["q"], restored["k"]
                scores = (q @ k.transpose(-1, -2)) / 8
                scores += torch.zeros_like(scores).masked_fill(~causal, torch.finfo(torch.bfloat16).min)
                assert np.array_equal(bits(scores)[0], w["reference_scores_bits"])
                probability = torch.softmax(scores, -1, dtype=torch.float32)
                dense = probability.to(torch.bfloat16) @ v
                assert np.array_equal(bits(dense)[0], w["reference_native_values_bits"])
                projected_dense = F.linear(dense.transpose(1, 2).contiguous().reshape(1, length, 960), weight)
                assert np.array_equal(bits(projected_dense)[0], w["reference_projected_bits"])
                native_elements += 15 * length * 64
                projected_elements += length * 960
                real = {}
                for role, value in (("q", q), ("k", k)):
                    t = torch.empty_strided((1, 15, length, 64), tuple(int(a) for a in w[f"reference_{role}_fp32_stride"]),
                                            dtype=torch.float32, device="cuda")
                    t.copy_(value.float())
                    real[role] = t
                for seed, mapping in maps.items():
                    for role, label in (("q", "query"), ("k", "key")):
                        mapped = torch.stack([F.linear(real[role][0, h], torch.tensor(mapping[label + "_weight"][h].T, device="cuda"),
                                             torch.tensor(mapping[label + "_bias"][h], device="cuda")) for h in range(15)])
                        assert np.array_equal(mapped.cpu().numpy(), w[f"s{seed}_{label}_latents"])
                        latent_elements += 15 * length * 2
                recovered = reconstructed_states(w, states)
                p = probability.cpu().numpy()[0]
                recency = truth_mask(length, "recency", recovered, p)
                upper = truth_mask(length, "mass_upper", recovered, p)
                summed_probability = p.astype(np.float64).reshape(5, 3, length, length).sum(1)
                upper_mass = (summed_probability * upper).sum(-1)
                energy = projected_dense.double().square().sum(-1)[0].cpu().numpy()
                value_energy = dense.double().square().sum(-1)[0].cpu().numpy()
                energies.append(energy)
                value_energies.append(value_energy)
                for name in expected:
                    check_time()
                    mask = np.unpackbits(w[name + "_group_selection_bits"], axis=-1, bitorder="little")[..., :length].astype(bool)
                    assert mask.shape == (5, length, length) and not np.any(np.triu(mask, 1))
                    budget = length if name == "full_control" else 128
                    assert np.array_equal(mask.sum(-1), np.broadcast_to(np.minimum(np.arange(1, length + 1), budget), (5, length)))
                    local = 64 if name.endswith("gaussian_local64") else 8
                    for pos in range(length):
                        assert mask[:, pos, max(0, pos + 1 - local):pos + 1].all()
                    truth = upper if name == "mass_upper" else recency if name == "recency" else truth_mask(length, name, recovered, p)
                    assert np.array_equal(mask, truth), (length, index, name, "shared mask differs")
                    group_rows += 5 * length
                    if name != "full_control":
                        assert np.all((summed_probability * mask).sum(-1) <= upper_mass + 1e-12)
                    selected = torch.tensor(np.repeat(mask, 3, axis=0), device="cuda")[None]
                    weights = torch.softmax(scores.masked_fill(~selected, torch.finfo(torch.bfloat16).min), -1, dtype=torch.float32).to(torch.bfloat16)
                    value = weights @ v
                    assert hashlib.sha256(bits(value)[0].tobytes()).hexdigest() == str(w[name + "_native_values_sha256"])
                    projected = F.linear(value.transpose(1, 2).contiguous().reshape(1, length, 960), weight)
                    assert np.array_equal(bits(projected)[0], w[name + "_projected_bits"])
                    native_elements += 15 * length * 64
                    projected_elements += length * 960
                    pe = (projected.double() - projected_dense.double()).square().sum(-1)[0].cpu().numpy()
                    ve = (value.double() - dense.double()).square().sum(-1)[0].cpu().numpy()
                    mass = (probability * selected).sum(-1)[0].cpu().numpy()
                    overlap = (mask & recency).sum(-1).astype(np.uint16)
                    for a, b in ((pe, w[name + "_projected_error"]), (ve, w[name + "_native_value_error"]),
                                 (mass, w[name + "_retained_mass"]), (energy, w["projected_reference_energy"]),
                                 (value_energy, w["native_value_reference_energy"]), (overlap, w[name + "_recency_overlap"])):
                        difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                        max_metric_difference = max(max_metric_difference, difference)
                        assert difference == 0., (length, index, name, "raw metric differs", difference)
                    loss = w[name + "_target_nll"]
                    assert np.array_equal(loss[:128], w["reference_target_nll"][:128])
                    if name == "full_control":
                        assert np.array_equal(loss, w["reference_target_nll"])
                    for key, value_array in (("loss", loss), ("pe", pe), ("ve", ve), ("mass", mass), ("overlap", overlap)):
                        totals[name][key].append(value_array)
                if index == 0:
                    first[length] = w
                window_count += 1
                print(json.dumps({"verified_windows": window_count, "length": length,
                                  "cumulative_seconds": round(prior + time.monotonic() - started, 3)}), flush=True)
            reference = np.stack(refs).astype(np.float64)
            energy, value_energy = np.stack(energies), np.stack(value_energies)
            reference_summary = result["length_results"][str(length)]["reference"]
            assert abs(reference_summary["mean_nll"] - reference.mean()) < 1e-13
            for name, values in totals.items():
                loss, pe, ve, mass, overlap = [np.stack(values[k]) for k in ("loss", "pe", "ve", "mass", "overlap")]
                loss = loss.astype(np.float64)
                summary = result["length_results"][str(length)]["methods"][name]
                expected_summary = {"mean_nll": loss.mean(), "paired_delta_nll": (loss - reference).mean(),
                    "perplexity": np.exp(loss.mean()), "perplexity_ratio": np.exp((loss - reference).mean()),
                    "affected_paired_delta_nll": (loss[:, 128:] - reference[:, 128:]).mean(),
                    "affected_mean_nll": loss[:, 128:].mean(),
                    "affected_projected_nrmse": np.sqrt(pe[:, 128:].sum() / energy[:, 128:].sum()),
                    "affected_native_value_nrmse": np.sqrt(ve[:, :, 128:].sum() / value_energy[:, :, 128:].sum()),
                    "affected_mean_retained_mass": mass[:, :, 128:].astype(np.float64).mean(),
                    "affected_recency_shared_keys": overlap[:, :, 128:].mean()}
                for key, value in expected_summary.items():
                    assert abs(summary[key] - value) < 1e-13, (length, name, key)
                    summary_checks += 1
                assert summary["targets"] == 43 * (length - 1) and summary["affected_targets"] == 43 * (length - 129)
                assert np.allclose(summary["window_mean_nll"], loss.mean(1), atol=1e-13, rtol=0)
                assert np.allclose(summary["window_paired_delta_nll"], (loss - reference).mean(1), atol=1e-13, rtol=0)
                for article_index, article in enumerate(summary["article_records"]):
                    assert article["article_id"] == data["articles"][article_index]["article_id"]
                    assert article["windows"] == [article_index] and article["targets"] == length - 1
                    assert abs(article["nll_sum"] - loss[article_index].sum()) < 1e-10
                    assert abs(article["paired_delta_nll"] - (loss[article_index] - reference[article_index]).mean()) < 1e-13
                if name.startswith("s") and name != "sink_recency":
                    seed, kind = name.split("_", 1)
                    sb = state_bytes[int(seed[1:])]
                    numerical = 15612 if kind.startswith("gaussian") else sb["flat" if kind == "flat" else "p3" if kind.endswith("p3") else "p2"]
                    numerical += 16 if kind.startswith("grid") else 1 if kind == "coarsened_p2" else 0
                    keys = 15 * length * (8 if kind.startswith("gaussian") else 1)
                else:
                    numerical = keys = 0
                assert summary["numeric_state_bytes"] == numerical and summary["key_metadata_bytes_per_window"] == keys
                assert summary["affected_mean_group_union"] == ((length + 129) / 2 if name == "full_control" else 128)
    check_time()
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["model_revision"],
                torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
    attention = model.model.layers[0].self_attn
    stock_forward = attention.forward
    active = {}
    def independent_forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, 64)
        q = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = repeat_kv(k, 3), repeat_kv(v, 3)
        scores = (q @ k.transpose(-1, -2)) / 8 + attention_mask
        scores = scores.masked_fill(~active["selected"], torch.finfo(torch.bfloat16).min)
        probability = torch.softmax(scores, -1, dtype=torch.float32).to(torch.bfloat16)
        value = (probability @ v).transpose(1, 2).contiguous().reshape(1, hidden_states.shape[1], 960)
        return F.linear(value, module.o_proj.weight, module.o_proj.bias), probability
    model_checks = 0
    with torch.inference_mode():
        for length, window in first.items():
            batch = torch.tensor(ids[f"tokens{length}"][0:1], device="cuda")
            attention.forward = stock_forward
            logits = model(batch, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="none").cpu().numpy()
            assert np.array_equal(loss, window["reference_target_nll"])
            model_checks += 1
            attention.forward = types.MethodType(independent_forward, attention)
            try:
                for name in expected:
                    check_time()
                    group = np.unpackbits(window[name + "_group_selection_bits"], axis=-1, bitorder="little")[..., :length].astype(bool)
                    active["selected"] = torch.tensor(np.repeat(group, 3, axis=0), device="cuda")[None]
                    logits = model(batch, use_cache=False).logits
                    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="none").cpu().numpy()
                    assert np.array_equal(loss, window[name + "_target_nll"]), (length, name, "independent complete model fixture differs")
                    model_checks += 1
            finally:
                attention.forward = stock_forward
    verify_manifest(args.input)
    check_time()
    assert Path(__file__).read_bytes() == source
    report = {"status": "shared_budget_audit_passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "development_completion_sha256": digest(args.input / "completed.json"), "audit_source_sha256": digest(__file__),
              "windows_verified": window_count, "group_mask_rows_exact": group_rows, "equivalent_head_mask_rows": group_rows * 3,
              "native_weighted_value_elements_exact_by_sha256": native_elements,
              "projected_output_elements_exact": projected_elements, "folded_latent_elements_exact": latent_elements,
              "maximum_raw_metric_difference": max_metric_difference, "summary_endpoint_checks": summary_checks,
              "complete_model_fixture_windows": 2, "complete_model_fixture_readouts_including_references": model_checks,
              "full_budget_target_loss_identity_windows": 86, "native_group_mass_upper_bound_verified": True,
              "audit_wall_seconds": time.monotonic() - started, "cumulative_gate_seconds": prior + time.monotonic() - started,
              "scope": "Same-operator separate algorithms; all masks/readouts/aggregation; full-model replay first article at both lengths; correlated development, no confirmation"}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT:
            report = {"status": "shared_budget_audit_failed", "type": type(error).__name__, "message": str(error),
                      "audit_wall_seconds": time.monotonic() - CONTEXT["started"],
                      "cumulative_gate_seconds": CONTEXT["prior"] + time.monotonic() - CONTEXT["started"]}
            (CONTEXT["output"] / "failed.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
