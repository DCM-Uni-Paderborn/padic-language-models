"""Re-evaluate the published PG-19 inputs with the frozen native first-layer readouts.

This fresh-run helper retains the scientific operator and writes only compact
per-book endpoint sums. It does not require omitted historical tensor archives.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from verify_materials import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError("fresh output required")
    verify()
    import torch
    from transformers import AutoModelForCausalLM
    from padic_lm.native_quality import encode_states, sequence_component, replay_projected, METHODS
    from evaluate_qk_bridge import baseline_forward, MODEL_HASH
    from llm_routing import state_dict_hash
    if not torch.cuda.is_available(): raise RuntimeError("requires a compatible CUDA environment")
    torch.set_num_threads(1); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    directory = ROOT / "results/pg19-context-data-spark-20261004"
    metadata = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "tokens.npz", allow_pickle=False) as z:
        tokens = {k: z[k].copy() for k in z.files}
    if tokens["tokens4096"].shape != (40, 4096) or not np.array_equal(tokens["tokens2048"], tokens["tokens4096"][:, :2048]):
        raise ValueError("frozen nested input differs")
    training = ROOT / "results/learned-qk-training-spark-20261004"
    frozen = {}
    for seed in (17, 29, 43):
        pair = []
        for name in ("final_encoder.npz", "routing_state.npz"):
            with np.load(training / f"seed-{seed}" / name, allow_pickle=False) as z:
                pair.append({k: z[k].copy() for k in z.files})
        frozen[seed] = tuple(pair)
    model = AutoModelForCausalLM.from_pretrained(metadata["model"], revision=metadata["model_revision"],
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to("cuda").eval()
    if state_dict_hash(model) != MODEL_HASH: raise ValueError("checkpoint state differs")
    projection = model.model.layers[0].self_attn.o_proj.weight
    rows = []
    a.output.mkdir(parents=True)
    with torch.inference_mode():
        for length in (2048, 4096):
            for index, ids in enumerate(tokens[f"tokens{length}"]):
                batch = torch.tensor(ids[None], device="cuda", dtype=torch.int64)
                reference, nll, _ = baseline_forward(model, batch)
                del reference["scores"]
                q, k, v = (reference[n][0] for n in ("q", "k", "v"))
                states, _ = encode_states(q, k, frozen, lambda: None)
                losses = {"eager_original": nll}
                for method in METHODS:
                    projected, head, selected = sequence_component(q, k, v, projection, method, states, lambda: None)
                    loss, _ = replay_projected(model, model.model.layers[0].self_attn, batch, projected, reference)
                    losses[method] = loss
                    del projected, head, selected
                for method, values in losses.items():
                    if not np.isfinite(values).all(): raise ArithmeticError("nonfinite losses")
                    for endpoint, start in (("all", 0), ("affected", 128)):
                        rows.append({"book_index": index, "book_id": metadata["books"][index]["book_id"],
                            "length": length, "method": method, "endpoint": endpoint, "targets": len(values)-start,
                            "loss_sum": float(np.sum(values[start:].astype(np.float64)))})
                print(json.dumps({"length": length, "completed_books": index+1, "books": 40}), flush=True)
                del reference, q, k, v, states, losses
    (a.output / "document-endpoints.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    (a.output / "settings.json").write_text(json.dumps({"model": metadata["model"], "revision": metadata["model_revision"],
        "model_state_sha256": MODEL_HASH, "contexts": [2048, 4096], "books": 40, "methods": ["eager_original", *METHODS],
        "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(0),
        "scope": "Fresh frozen-input reproduction, not the historical execution ledger or a latency benchmark"}, indent=2)+"\n")


if __name__ == "__main__": main()
