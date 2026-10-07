"""Old-data sequence-injection and Flash shape validation, before PG-19 access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
from padic_lm.native_quality import encode_states, sequence_component, replay_projected, METHODS
from padic_lm.native_decode import attention_component
from evaluate_qk_bridge import arrays, digest, bf16_bits, baseline_forward, TRAIN_HASH, MODEL_HASH, verify_manifest
from llm_routing import state_dict_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'development', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError('fresh preflight required')
    a.output.mkdir(parents=True)
    verify_manifest(a.training, TRAIN_HASH)
    fixture = a.development / 'windows/length-512-article-00.npz'
    parent = json.loads((a.development / 'completed.json').read_text())
    assert digest(a.development / 'completed.json') == 'ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8'
    assert digest(fixture) == parent['artifact_sha256']['windows/length-512-article-00.npz']
    raw = arrays(fixture)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M',
        revision='f8027fd0eaeea54caa13c31d31b9fdc459c38b49', torch_dtype=torch.bfloat16,
        attn_implementation='eager', use_safetensors=True).to('cuda').eval()
    assert state_dict_hash(model) == MODEL_HASH
    original = model.model.layers[0].self_attn
    frozen = {s: (arrays(a.training / f'seed-{s}/final_encoder.npz'),
                  arrays(a.training / f'seed-{s}/routing_state.npz')) for s in (17, 29, 43)}
    with torch.inference_mode():
        batch = torch.tensor(raw['tokens'][None], device='cuda', dtype=torch.int64)
        reference, nll, prediction = baseline_forward(model, batch)
        assert np.array_equal(nll, raw['reference_target_nll'])
        assert np.array_equal(prediction, raw['reference_predictions'])
        injected_nll, injected_prediction = replay_projected(model, original, batch, reference['projected'], reference)
        assert np.array_equal(nll, injected_nll) and np.array_equal(prediction, injected_prediction)
        q, k, v = (reference[n][0] for n in ('q', 'k', 'v'))
        states, saved = encode_states(q, k, frozen)
        checked = 0
        for method in METHODS:
            projected, head, ids = sequence_component(q, k, v, original.o_proj.weight, method, states)
            assert projected.shape == (1, 512, 960) and head.shape == (1, 15, 512, 64)
            if ids is not None:
                for pos in range(512):
                    count = min(pos + 1, 128); selected = ids[:, pos, :count]
                    assert np.all(selected <= pos) and np.all(np.diff(selected.astype(int), axis=1) > 0)
                    assert np.all(ids[:, pos, count:] == 65535)
                    assert np.all(selected[:, -min(8, count):] == np.arange(pos + 1 - min(8, count), pos + 1))
            for pos in (0, 255, 511):
                chosen = None if ids is None else torch.tensor(ids[:, pos, :min(pos + 1, 128)].astype(np.int64), device='cuda')
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out, h = attention_component(q[:, pos:pos + 1], k[:, :pos + 1], v[:, :pos + 1], original.o_proj.weight, chosen)
                assert torch.equal(out, projected[:, pos:pos + 1]) and torch.equal(h, head[:, :, pos:pos + 1])
                checked += 1
            loss, pred = replay_projected(model, original, batch, projected, reference)
            assert np.isfinite(loss).all() and loss.shape == nll.shape and pred.shape == prediction.shape
        shape_checks = []
        for length in (2048, 4096):
            q0 = torch.zeros((15, 1, 64), device='cuda', dtype=torch.bfloat16)
            k0 = torch.zeros((5, length, 64), device='cuda', dtype=torch.bfloat16)
            v0 = (torch.arange(length, device='cuda') % 16).to(torch.bfloat16)[None, :, None].expand(5, length, 64).contiguous()
            weight = torch.eye(960, device='cuda', dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION): out, head = attention_component(q0, k0, v0, weight)
            assert torch.all(head == 7.5) and torch.all(out == 7.5)
            shape_checks.append(length)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in
               (Path(__file__), ROOT / 'src/padic_lm/native_quality.py', ROOT / 'src/padic_lm/native_decode.py')}
    report = {'status': 'native_quality_preflight_passed', 'source_sha256': sources,
              'old_fixture_eager_nll_and_predictions_exact': True, 'original_output_injection_exact': True,
              'native_methods_forwarded': len(METHODS), 'direct_one_query_bitwise_readouts': checked,
              'flash_full_prefix_shapes': shape_checks, 'unchanged_qkv_checked_every_injection': True,
              'wall_seconds': time.monotonic() - started, 'torch': torch.__version__,
              'model_state_sha256': MODEL_HASH, 'pg19_text_read': False}
    (a.output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
