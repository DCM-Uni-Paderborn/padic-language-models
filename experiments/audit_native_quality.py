"""Independent GPU bridge replay, exhaustive CPU selection, FP64 probes and model replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torch.nn.attention import SDPBackend, sdpa_kernel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as r: return {n: r[n] for n in r.files}


def decode(bits): return (bits.astype(np.uint32) << 16).view(np.float32)
def tensor(bits): return torch.tensor(bits.copy(), device='cuda').view(torch.bfloat16)
def bits(value): return value.contiguous().view(torch.uint16).cpu().numpy()


def expected_ids(qcodes, kcodes, pos, prime, digits, coarse):
    prefix = pos + 1
    if prefix <= 128: return np.broadcast_to(np.arange(prefix), (5, prefix)).copy()
    stop = prefix - 8
    q = np.stack((qcodes[:, pos] % prime ** digits[0], qcodes[:, pos] // prime ** digits[0]), -1).astype(np.int64)
    k = np.stack((kcodes[:, :stop] % prime ** digits[0], kcodes[:, :stop] // prime ** digits[0]), -1).astype(np.int64)
    delta = q[:, None] - k
    scores = np.zeros((15, stop), dtype=np.int64)
    for level in range(1, max(digits) + 1): scores += np.all(delta % prime ** level == 0, axis=-1)
    scores //= coarse
    ranks = np.empty_like(scores)
    for head in range(15):
        counts = np.bincount(scores[head], minlength=5)
        midranks = np.array([2 * sum(int(v) for v in counts[:level]) + int(counts[level]) - 1 for level in range(5)])
        ranks[head] = midranks[scores[head]]
    aggregate = ranks.reshape(5, 3, stop).sum(1)
    # Stable sort preserves increasing timestamps within ties; the largest120
    # are exactly the most recent ones at the boundary. No GPU top-k/LUT imports.
    old = np.argsort(aggregate, axis=1, kind='stable')[:, -120:]
    return np.sort(np.concatenate((old, np.broadcast_to(np.arange(stop, prefix), (5, 8))), 1), axis=1)


def errors(actual, reference, limit):
    delta = actual.astype(np.float64) - reference
    nrmse = float(np.sqrt(np.sum(delta ** 2) / max(float(np.sum(reference ** 2)), 1e-30)))
    maximum = float(np.max(np.abs(delta))); allowed = float(.03 * np.max(np.abs(reference)) + 1e-4)
    return {'nrmse': nrmse, 'max_abs': maximum, 'allowed_max_abs': allowed,
            'nrmse_limit': limit, 'passed': nrmse <= limit and maximum <= allowed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation', 'training', 'endpoint-diagnosis', 'output'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError('fresh independent numerical audit required')
    a.output.mkdir(parents=True); (a.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    parent = json.loads((a.evaluation / 'completed.json').read_text())
    diagnosis = json.loads(a.endpoint_diagnosis.read_text()); prior = diagnosis['cumulative_gate_seconds']
    def guard():
        if prior + time.monotonic() - started >= 10800: raise TimeoutError('native PG-19 cumulative10800s cap reached')
    try:
        assert sha(a.evaluation / 'completed.json') == (a.evaluation / 'completed.sha256').read_text().strip()
        assert diagnosis['status'] == 'all_native_endpoint_sums_verified_rounding_difference_diagnosed'
        assert diagnosis['evaluation_completion_sha256'] == sha(a.evaluation / 'completed.json')
        assert diagnosis['endpoints_verified'] == diagnosis['recorded_numpy_reductions_bitwise_equal'] == diagnosis['fsum_exact_binary_correct_roundings_verified'] == 2080
        assert sha(a.endpoint_diagnosis.parent / 'diagnostic_source.py') == diagnosis['diagnostic_source_sha256']
        assert prior >= parent['cumulative_gate_seconds']
        for name, h in parent['artifact_sha256'].items():
            guard(); assert sha(a.evaluation / name) == h and (a.evaluation / name).stat().st_size == parent['artifact_bytes'][name]
        config = json.loads((a.evaluation / 'config.json').read_text()); methods = config['methods']
        assert len(methods) == 13 and methods[0] == 'eager_original'
        assert sha(a.training / 'completed.json') == 'f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1'
        training = json.loads((a.training / 'completed.json').read_text()); frozen = {}
        for seed in (17, 29, 43):
            for name in ('final_encoder.npz', 'routing_state.npz'):
                path = a.training / f'seed-{seed}' / name
                assert sha(path) == training['artifact_sha256'][f'seed-{seed}/{name}']
            frozen[seed] = (load(a.training / f'seed-{seed}/final_encoder.npz'), load(a.training / f'seed-{seed}/routing_state.npz'))
        projection_bits = load(a.evaluation / 'output_projection.npz')['weight_bits']
        weight = decode(projection_bits).astype(np.float64); wgpu = tensor(projection_bits)
        endpoint_rows = json.loads((a.evaluation / 'document_endpoints.json').read_text())
        endpoints = {(r['length'], r['book_index'], r['method'], r['endpoint']): r for r in endpoint_rows}
        assert len(endpoints) == 2080
        torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest'); torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['model_revision'],
                  torch_dtype=torch.bfloat16, attn_implementation='eager', use_safetensors=True).to('cuda').eval()
        model_hash = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            value = value.detach().contiguous(); model_hash.update(name.encode()); model_hash.update(str(value.dtype).encode())
            model_hash.update(json.dumps(list(value.shape)).encode()); model_hash.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        assert model_hash.hexdigest() == config['model_state_sha256'] == 'b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583'
        counts = {'code_elements': 0, 'latent_elements': 0, 'selected_group_rows': 0, 'document_endpoints': 0, 'whole_model_replays': 0}
        probes = []; max_endpoint_sum_rounding_difference = 0.
        with torch.inference_mode():
            for length in (2048, 4096):
                for index in range(40):
                    guard(); raw = load(a.evaluation / 'windows' / f'length-{length}-book-{index:02d}.npz')
                    qnp, knp, vnp = (decode(raw['reference_' + n + '_bits']) for n in ('queries', 'keys', 'values'))
                    q, k, v = (tensor(raw['reference_' + n + '_bits']) for n in ('queries', 'keys', 'values'))
                    # Replay the exact declared t=1 contractions independently;
                    # CPU projection contraction need not share GPU FP64 rounding.
                    for seed in (17, 29, 43):
                        encoder, books = frozen[seed]
                        for role, values in (('query', q), ('key', k.repeat_interleave(3, 0))):
                            bias = torch.tensor(encoder[role + '_bias'], device='cuda')[:, None]
                            matrix = torch.tensor(encoder[role + '_weight'], device='cuda')
                            latents = [torch.baddbmm(bias, values[:, pos:pos + 1].float(), matrix) for pos in range(length)]
                            latent = torch.cat(latents, 1)
                            for family, prime, digits in (('p2', 2, (4, 4)), ('p3', 3, (3, 2))):
                                saved_latent = raw[f's{seed}_{family}_{role}_latents']
                                assert np.array_equal(latent.cpu().numpy(), saved_latent); counts['latent_elements'] += latent.numel()
                                center = torch.tensor(books[family + '_center'], device='cuda')[:, None]
                                matrix64 = torch.tensor(books[family + '_projection'], device='cuda')
                                projected = torch.cat([torch.bmm(z.double() - center, matrix64) for z in latents], 1).cpu().numpy()
                                residues = np.empty((15, length, 2), dtype=np.int64)
                                for head in range(15):
                                    for coordinate, digit in enumerate(digits):
                                        cuts = books[family + '_thresholds'][head, coordinate, :prime ** digit - 1]
                                        bins = np.searchsorted(cuts, projected[head, :, coordinate], side='right')
                                        reversed_digits = sum((bins // prime ** j % prime) * prime ** (digit - 1 - j) for j in range(digit))
                                        residues[head, :, coordinate] = reversed_digits
                                packed = (residues[..., 0] + prime ** digits[0] * residues[..., 1]).astype(np.uint8)
                                assert np.array_equal(packed, raw[f's{seed}_{family}_{role}_codes']); counts['code_elements'] += packed.size
                            del latent, latents
                    for method in methods:
                        loss = raw[method + '_target_nll']; assert loss.shape == (length - 1,) and np.isfinite(loss).all()
                        for endpoint, start in (('all', 0), ('affected', 128)):
                            row = endpoints[length, index, method, endpoint]
                            assert row['targets'] == len(loss) - start
                            # Frozen production stores an FP64 NumPy reduction,
                            # not the correctly rounded exact binary sum. Verify
                            # its bits independently and bound rounding using a
                            # separate fsum; do not change the raw losses/endpoint.
                            reduction = float(np.sum(loss[start:].astype(np.float64)))
                            faithful = math.fsum(float(x) for x in loss[start:])
                            assert row['loss_sum'] == reduction
                            nu = (len(loss) - start - 1) * 2. ** -53
                            difference = abs(reduction - faithful)
                            assert difference <= nu / (1 - nu) * faithful + .5 * math.ulp(faithful)
                            max_endpoint_sum_rounding_difference = max(max_endpoint_sum_rounding_difference, difference)
                            counts['document_endpoints'] += 1
                        if method not in ('eager_original', 'native_full'):
                            ids = raw[method + '_selected_ids']; assert ids.shape == (5, length, 128)
                            for pos in range(length):
                                count = min(pos + 1, 128)
                                if method == 'recency': expected = np.broadcast_to(np.arange(pos + 1 - count, pos + 1), (5, count))
                                elif method == 'uniform':
                                    if pos < 128: chosen = np.arange(pos + 1)
                                    else:
                                        stop = pos + 1 - 8
                                        chosen = np.r_[((2 * np.arange(120) + 1) * stop) // 240, np.arange(stop, pos + 1)]
                                    expected = np.broadcast_to(chosen, (5, count))
                                else:
                                    seed = int(method.split('_')[0][1:]); kind = method.split('_', 1)[1]
                                    family = 'p3' if kind == 'p3' else 'p2'; prime, digits = (3, (3, 2)) if family == 'p3' else (2, (4, 4))
                                    expected = expected_ids(raw[f's{seed}_{family}_query_codes'], raw[f's{seed}_{family}_key_codes'], pos,
                                                            prime, digits, 2 if kind == 'coarsened_p2' else 1)
                                assert np.array_equal(ids[:, pos, :count], expected) and np.all(ids[:, pos, count:] == 65535)
                                counts['selected_group_rows'] += 5
                        if method != 'eager_original':
                            for prefix in (length // 2, 3 * length // 4, length):
                                pos = prefix - 1; heads = []
                                for head in range(15):
                                    selected = np.arange(prefix) if method == 'native_full' else raw[method + '_selected_ids'][head // 3, pos, :min(prefix, 128)]
                                    logits = knp[head // 3, selected].astype(np.float64) @ qnp[head, pos].astype(np.float64) / 8
                                    exponent = np.exp(logits - logits.max()); probability = exponent / math.fsum(float(x) for x in exponent)
                                    heads.append(probability @ vnp[head // 3, selected].astype(np.float64))
                                heads = np.asarray(heads)
                                projected = heads.reshape(960) @ weight.T
                                he = errors(decode(raw[method + '_head_bits'])[0, :, pos], heads, .01)
                                pe = errors(decode(raw[method + '_projected_bits'])[0, pos], projected, .015)
                                probes.append({'length': length, 'book_index': index, 'method': method, 'prefix': prefix,
                                               'head': he, 'projected': pe})
                                if not (he['passed'] and pe['passed']):
                                    raise ArithmeticError(f'FP64 tolerance failed: context{length} book{index} method{method} prefix{prefix}')
                        if index == 0:
                            batch = torch.tensor(raw['tokens'][None], device='cuda', dtype=torch.int64)
                            hook = None
                            if method != 'eager_original':
                                projected, heads = [], []
                                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                                    for pos in range(length):
                                        kk, vv = k[:, :pos + 1], v[:, :pos + 1]
                                        if method != 'native_full':
                                            selected = torch.tensor(raw[method + '_selected_ids'][:, pos, :min(pos + 1, 128)].astype(np.int64), device='cuda')
                                            gather = selected[..., None].expand(-1, -1, 64); kk = kk.gather(1, gather); vv = vv.gather(1, gather)
                                        h = F.scaled_dot_product_attention(q[None, :, pos:pos + 1], kk[None], vv[None],
                                                    dropout_p=0., is_causal=False, enable_gqa=True)
                                        heads.append(h); projected.append(F.linear(h.transpose(1, 2).reshape(1, 1, 960), wgpu))
                                joined = torch.cat(projected, 1); joined_heads = torch.cat(heads, 2)
                                assert np.array_equal(bits(joined), raw[method + '_projected_bits'])
                                assert np.array_equal(bits(joined_heads), raw[method + '_head_bits'])
                                hook = model.model.layers[0].self_attn.register_forward_hook(lambda module, args, output: (joined, None))
                            try:
                                logits = model(batch, use_cache=False).logits
                                actual_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction='none').cpu().numpy()
                                actual_prediction = logits[:, :-1].argmax(-1)[0].cpu().numpy()
                            finally:
                                if hook is not None: hook.remove()
                            assert np.array_equal(actual_loss, loss) and np.array_equal(actual_prediction, raw[method + '_predictions'])
                            counts['whole_model_replays'] += 1
                    print(json.dumps({'length': length, 'book': index, 'verified_group_rows': counts['selected_group_rows'],
                                      'cumulative_gate_seconds': prior + time.monotonic() - started}), flush=True)
                    del raw, q, k, v
        assert len(probes) == 2880 and counts['whole_model_replays'] == 26
        (a.output / 'readout_probes.json').write_text(json.dumps(probes, indent=2) + '\n')
        guard()
        report = {'status': 'native_pg19_numerical_audit_passed', 'evaluation_completion_sha256': sha(a.evaluation / 'completed.json'),
                  'audit_source_sha256': sha(__file__), **counts, 'fp64_readout_probes': len(probes),
                  'endpoint_rounding_diagnosis_sha256': sha(a.endpoint_diagnosis),
                  'prior_failed_audit_sha256': diagnosis['failed_audit_record_sha256'],
                  'max_endpoint_sum_rounding_difference': max_endpoint_sum_rounding_difference,
                  'max_head_nrmse': max(r['head']['nrmse'] for r in probes),
                  'max_projected_nrmse': max(r['projected']['nrmse'] for r in probes),
                  'audit_wall_seconds': time.monotonic() - started, 'cumulative_gate_seconds': prior + time.monotonic() - started,
                  'scope': 'No production imports; exhaustive declared GPU bridge replay and CPU ID reconstruction;2880 FP64 probes;26 independently regenerated whole-model replays'}
        (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)
    except Exception as error:
        if 'probes' in locals():
            (a.output / 'readout_probes_partial.json').write_text(json.dumps(probes, indent=2) + '\n')
        (a.output / 'failed.json').write_text(json.dumps({'status': 'native_pg19_numerical_audit_failed', 'message': str(error),
             'type': type(error).__name__, 'cumulative_gate_seconds': prior + time.monotonic() - started}, indent=2) + '\n'); raise


if __name__ == '__main__': main()
