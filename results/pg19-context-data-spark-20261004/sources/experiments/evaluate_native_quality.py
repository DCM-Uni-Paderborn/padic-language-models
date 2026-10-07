"""Frozen PG-19 first-layer single-query native-readout quality evaluation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'experiments'))
from padic_lm.native_quality import encode_states, sequence_component, replay_projected, METHODS
from padic_lm.native_study import MODEL, REVISION, sha, write, snapshot, verify, complete, guard, failure
from evaluate_qk_bridge import arrays, bf16_bits, baseline_forward, TRAIN_HASH, MODEL_HASH
from llm_routing import state_dict_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('freeze', 'data', 'data-audit', 'training', 'output'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); prior = 0
    if a.output.exists(): raise FileExistsError('fresh native PG-19 evaluation required')
    a.output.mkdir(parents=True); snapshot(a.output, a.freeze); (a.output / 'windows').mkdir()
    try:
        parent = verify(a.data); verify(a.training, TRAIN_HASH)
        audit = json.loads(a.data_audit.read_text()); data = json.loads((a.data / 'manifest.json').read_text())
        assert audit['status'] == 'pg19_data_independently_verified' and audit['data_completion_sha256'] == sha(a.data / 'completed.json')
        assert data['freeze_sha256'] == sha(a.freeze) and data['protocol_sha256'] == sha(a.output / 'prospective_protocol.txt')
        assert data['book_count'] == audit['selected_books'] == 40 and data['lengths'] == [2048, 4096]
        prior = audit['cumulative_gate_seconds']; guard(started, prior)
        tokens = arrays(a.data / 'tokens.npz')
        assert np.array_equal(tokens['tokens2048'], tokens['tokens4096'][:, :2048]) and tokens['tokens4096'].shape == (40, 4096)
        torch.set_num_threads(1); torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        frozen = {s: (arrays(a.training / f'seed-{s}/final_encoder.npz'),
                      arrays(a.training / f'seed-{s}/routing_state.npz')) for s in (17, 29, 43)}
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
            attn_implementation='eager', use_safetensors=True).to('cuda').eval()
        assert state_dict_hash(model) == MODEL_HASH and model.config.max_position_embeddings >= 4096
        assert (model.config.num_attention_heads, model.config.num_key_value_heads, model.config.num_hidden_layers) == (15, 5, 32)
        original = model.model.layers[0].self_attn
        np.savez_compressed(a.output / 'output_projection.npz', weight_bits=bf16_bits(original.o_proj.weight))
        write(a.output / 'config.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'model': MODEL, 'model_revision': REVISION, 'model_state_sha256': MODEL_HASH,
              'training_completion_sha256': TRAIN_HASH, 'data_completion_sha256': sha(a.data / 'completed.json'),
              'data_audit_sha256': sha(a.data_audit), 'freeze_sha256': sha(a.freeze),
              'methods': ['eager_original', *METHODS], 'contexts': [2048, 4096], 'books': data['books'],
              'backend': 'forced_FLASH_ATTENTION', 'tf32': False, 'prior_gate_seconds': prior,
              'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy')},
              'device': str(torch.cuda.get_device_properties(0)), 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
              'scope': 'First-layer native single-query readout on original uncached sequence QKV; no cached deployment or latency benchmark'})
        summaries, clocks = [], []
        with torch.inference_mode():
            for length in (2048, 4096):
                for index, token_ids in enumerate(tokens[f'tokens{length}']):
                    guard(started, prior); window_started = time.monotonic()
                    batch = torch.tensor(token_ids[None], device='cuda', dtype=torch.int64)
                    reference, nll, predictions = baseline_forward(model, batch)
                    del reference['scores']
                    raw = {'tokens': token_ids, 'eager_original_target_nll': nll, 'eager_original_predictions': predictions,
                           'reference_queries_bits': bf16_bits(reference['q'])[0], 'reference_keys_bits': bf16_bits(reference['k'])[0],
                           'reference_values_bits': bf16_bits(reference['v'])[0], 'eager_original_head_bits': bf16_bits(reference['values']),
                           'eager_original_projected_bits': bf16_bits(reference['projected'])}
                    q, k, v = (reference[n][0] for n in ('q', 'k', 'v'))
                    torch.cuda.synchronize(); encoded_started = time.monotonic()
                    states, saved = encode_states(q, k, frozen, lambda: guard(started, prior)); raw.update(saved)
                    torch.cuda.synchronize(); encoded_seconds = time.monotonic() - encoded_started
                    method_clocks = {}
                    for method in METHODS:
                        method_started = time.monotonic(); guard(started, prior)
                        projected, head, ids = sequence_component(q, k, v, original.o_proj.weight, method, states, lambda: guard(started, prior))
                        torch.cuda.synchronize(); component_seconds = time.monotonic() - method_started
                        loss, prediction = replay_projected(model, original, batch, projected, reference)
                        if not np.isfinite(loss).all(): raise ArithmeticError('nonfinite native losses')
                        raw[method + '_target_nll'] = loss; raw[method + '_predictions'] = prediction
                        raw[method + '_head_bits'] = bf16_bits(head); raw[method + '_projected_bits'] = bf16_bits(projected)
                        if ids is not None: raw[method + '_selected_ids'] = ids
                        torch.cuda.synchronize()
                        method_clocks[method] = {'component_sequence_seconds': component_seconds,
                                                'with_injected_model_forward_seconds': time.monotonic() - method_started}
                        del projected, head
                    for method in ('eager_original', *METHODS):
                        values = raw[method + '_target_nll']
                        for endpoint, start in (('all', 0), ('affected', 128)):
                            loss_sum = float(np.sum(values[start:].astype(np.float64)))
                            summaries.append({'book_index': index, 'book_id': data['books'][index]['book_id'], 'length': length,
                                  'method': method, 'endpoint': endpoint, 'targets': len(values) - start, 'loss_sum': loss_sum})
                    np.savez_compressed(a.output / 'windows' / f'length-{length}-book-{index:02d}.npz', **raw)
                    clocks.append({'length': length, 'book_index': index, 'encoder_seconds': encoded_seconds,
                                   'methods': method_clocks, 'window_wall_seconds': time.monotonic() - window_started})
                    write(a.output / 'progress.json', {'completed_windows': len(clocks), 'total_windows': 80,
                          'cumulative_gate_seconds': prior + time.monotonic() - started})
                    print(json.dumps({'length': length, 'book': index, 'completed_windows': len(clocks),
                                      'cumulative_gate_seconds': prior + time.monotonic() - started}), flush=True)
                    del reference, states, raw, saved, q, k, v
        write(a.output / 'document_endpoints.json', summaries); write(a.output / 'process_clocks.json', clocks)
        guard(started, prior); snapshot_value = json.loads((a.output / 'production_freeze.json').read_text())
        for name, h in snapshot_value['source_sha256'].items():
            if sha(ROOT / name) != h: raise RuntimeError('production source changed during evaluation')
        complete(a.output, 'native_pg19_quality_complete_pending_independent_audit', prior + time.monotonic() - started,
                 {'evaluation_wall_seconds': time.monotonic() - started, 'windows': 80})
        print(json.dumps({'status': 'native_pg19_quality_complete_pending_independent_audit',
                          'completion_sha256': sha(a.output / 'completed.json')}), flush=True)
    except Exception as error: failure(a.output, error, started, prior); raise


if __name__ == '__main__': main()
