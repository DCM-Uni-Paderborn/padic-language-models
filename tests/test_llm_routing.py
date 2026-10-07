"""Causal intervention tests without downloads; torch/transformers tests skip explicitly."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from padic_lm.routing import QuantileCodebook, angular_hyperplanes

try:
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention, eager_attention_forward
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def load_targets():
    source = PROJECT / "experiments/llm_routing.py"
    spec = importlib.util.spec_from_file_location("routing_intervention_tests", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and Transformers required on experiment host")
class RoutingInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = load_targets()
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    def calibration(self, heads=4, width=8):
        rng = np.random.default_rng(742)
        return rng.normal(size=(2, heads, 12, width)), rng.normal(size=(2, heads, 12, width))

    def test_same_code_padic_tree_masks_equal_and_causal_all_primes(self):
        qcal, kcal = self.calibration()
        rng = np.random.default_rng(193)
        q, k = rng.normal(size=(2, 4, 17, 8)), rng.normal(size=(2, 4, 17, 8))
        for prime in (2, 3, 5):
            book = QuantileCodebook.fit(qcal, kcal, prime=prime, code_bit_budget=8)
            for budget in (1, 5, 9, 32):
                args = {"budget": budget, "codebook": book, "recent_window": 2}
                p_mask, p_codes = self.targets.build_selection_mask(q, k, method="padic", **args)
                t_mask, t_codes = self.targets.build_selection_mask(q, k, method="trie", **args)
                np.testing.assert_array_equal(p_mask, t_mask)
                np.testing.assert_array_equal(p_codes["query_codes"], t_codes["query_codes"])
                self.assertFalse(np.any(np.triu(p_mask, k=1)))
                self.assertTrue(np.all(p_mask.any(axis=-1)))
                np.testing.assert_array_equal(p_mask.sum(axis=-1), np.broadcast_to(np.minimum(np.arange(1, 18), budget), (2, 4, 17)))

    def test_future_key_mutations_do_not_change_prefix_selection(self):
        qcal, kcal = self.calibration()
        book = QuantileCodebook.fit(qcal, kcal, code_bit_budget=8)
        q, k = qcal[:1].copy(), kcal[:1].copy()
        changed = k.copy()
        changed[:, :, 7:] += 100
        for method in ("padic", "trie", "recency", "angular"):
            kwargs = {"method": method, "budget": 5, "codebook": book,
                      "planes": angular_hyperplanes(4, 8, 8, 17), "recent_window": 2}
            original, _ = self.targets.build_selection_mask(q, k, **kwargs)
            modified, _ = self.targets.build_selection_mask(q, changed, **kwargs)
            np.testing.assert_array_equal(original[:, :, :7], modified[:, :, :7])

    def test_dense_budget_and_recency_mask_exact_expected(self):
        q, k = self.calibration()
        dense, _ = self.targets.build_selection_mask(q, k, method="dense", budget=128)
        np.testing.assert_array_equal(dense, np.broadcast_to(np.tri(12, dtype=bool), (2, 4, 12, 12)))
        recent, _ = self.targets.build_selection_mask(q, k, method="recency", budget=3)
        for position in range(12):
            expected = np.zeros(12, dtype=bool)
            expected[max(0, position - 2):position + 1] = True
            np.testing.assert_array_equal(recent[:, :, position], np.broadcast_to(expected, (2, 4, 12)))

    def stock_case(self, dtype=torch.float32 if TORCH_AVAILABLE else None, groups=2, value_width=8):
        torch.manual_seed(845)
        query = torch.randn(2, 4, 7, 8, dtype=dtype)
        key = torch.randn(2, 4 // groups, 7, 8, dtype=dtype)
        value = torch.randn(2, 4 // groups, 7, value_width, dtype=dtype)
        causal = torch.tril(torch.ones(7, 7, dtype=torch.bool)).expand(2, 4, 7, 7)
        bias = torch.zeros(2, 1, 7, 7, dtype=dtype).masked_fill(~causal[:, :1], torch.finfo(dtype).min)
        module = torch.nn.Module()
        module.num_key_value_groups = groups
        module.eval()
        return query, key, value, causal, bias, module

    def test_full_budget_matches_stock_eager_gqa_value_dimensions(self):
        for dtype in (torch.float32, torch.bfloat16):
            for groups in (1, 2, 4):
                query, key, value, causal, bias, module = self.stock_case(dtype, groups, value_width=5)
                expected, probabilities = eager_attention_forward(module, query, key, value, bias, scaling=8**-0.5, dropout=0)
                actual, actual_probabilities, mass = self.targets.routed_eager_attention(query, key, value, causal, bias, scaling=8**-0.5, groups=groups)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_probabilities, probabilities, rtol=0, atol=0)
                self.assertEqual(tuple(actual.shape), (2, 7, 4, 5))
                torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-6, atol=1e-6)

    def test_selected_attention_matches_independent_selected_key_softmax(self):
        query, key, value, causal, bias, _ = self.stock_case(groups=2, value_width=5)
        selected = torch.zeros_like(causal)
        selected[..., 0] = True
        for position in range(7):
            selected[:, :, position, position] = True
        output, probabilities, _ = self.targets.routed_eager_attention(query, key, value, selected, bias, scaling=8**-0.5, groups=2)
        for b in range(2):
            for h in range(4):
                for position in range(7):
                    indices = torch.nonzero(selected[b, h, position]).flatten()
                    logits = (key[b, h // 2, indices] * query[b, h, position]).sum(dim=-1) * (8**-0.5)
                    weights = torch.softmax(logits, dim=-1)
                    expected = (weights[:, None] * value[b, h // 2, indices]).sum(dim=0)
                    torch.testing.assert_close(output[b, position, h], expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool((probabilities[~selected] == 0).all()))

    def test_attention_invalid_shapes_and_empty_rows_rejected(self):
        query, key, value, causal, bias, _ = self.stock_case()
        empty = causal.clone()
        empty[0, 0, 0] = False
        for selected, groups in ((empty, 2), (causal.to(torch.int64), 2), (causal, 3)):
            with self.assertRaises(ValueError):
                self.targets.routed_eager_attention(query, key, value, selected, bias, scaling=8**-0.5, groups=groups)

    def test_no_additive_bias_still_respects_selected_causal_rows(self):
        query, key, value, causal, _, _ = self.stock_case()
        output, probabilities, _ = self.targets.routed_eager_attention(query, key, value, causal, None, scaling=8**-0.5, groups=2)
        self.assertTrue(bool((probabilities[~causal] == 0).all()))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, 4, 7), rtol=1e-6, atol=1e-6)
        for head in range(4):
            torch.testing.assert_close(output[:, 0, head], value[:, head // 2, 0], rtol=0, atol=0)

    def test_oracle_causal_ties_recent_budget_and_exhaustive_mass(self):
        scores = torch.tensor([3.0, 3.0, -1.0, 5.0, 2.0, 0.0]).expand(1, 2, 6, 6).clone()
        # Deliberately large future scores still cannot enter a causal prefix.
        scores = scores.masked_fill(torch.triu(torch.ones(6, 6, dtype=torch.bool), 1), 10000)
        for budget in (1, 3, 12):
            for recent in (0, 1, 8):
                selected, archived = self.targets.build_oracle_selection_mask(scores, budget=budget, recent_window=recent)
                np.testing.assert_array_equal(archived["oracle_logits"], scores.numpy())
                for head in range(2):
                    for position in range(6):
                        count = min(position + 1, budget)
                        mandatory = list(range(position + 1 - min(recent, count), position + 1))
                        pool = [i for i in range(position + 1) if i not in mandatory]
                        ranked = sorted(pool, key=lambda i: (-float(scores[0, head, position, i]), -i))
                        expected = sorted(mandatory + ranked[:count - len(mandatory)])
                        self.assertEqual(np.flatnonzero(selected[0, head, position]).tolist(), expected)
                        weights = scores[0, head, position, :position + 1].softmax(dim=-1)
                        bound = float(weights[expected].sum())
                        for other in itertools.combinations(pool, count - len(mandatory)):
                            candidate = float(weights[mandatory + list(other)].sum())
                            self.assertGreaterEqual(bound + 1e-6, candidate)

    def test_oracle_bfloat16_ties_follow_actual_stock_scores_and_reuse_tensor(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            query = torch.ones(1, 1, 3, 2, dtype=torch.bfloat16, device=device)
            key = torch.tensor([[[[1, 1 / 256], [1, 1 / 512], [0, 0]]]], dtype=torch.bfloat16, device=device)
            value = torch.tensor([[[[10], [20], [30]]]], dtype=torch.bfloat16, device=device)
            causal = torch.tril(torch.ones(3, 3, dtype=torch.bool, device=device))
            bias = torch.zeros(1, 1, 3, 3, dtype=torch.bfloat16, device=device).masked_fill(~causal, torch.finfo(torch.bfloat16).min)
            logits = self.targets.stock_eager_logits(query, key, bias, scaling=1.0, groups=1)
            native = torch.matmul(query, key.transpose(2, 3)) + bias
            torch.testing.assert_close(logits, native, rtol=0, atol=0)
            self.assertEqual(float(logits[0, 0, 2, 0]), float(logits[0, 0, 2, 1]))
            double_scores = torch.matmul(query.double(), key.double().transpose(2, 3))
            self.assertGreater(float(double_scores[0, 0, 2, 0]), float(double_scores[0, 0, 2, 1]))
            selected, archived = self.targets.build_oracle_selection_mask(logits, budget=1, recent_window=0)
            self.assertEqual(np.flatnonzero(selected[0, 0, 2]).tolist(), [1])
            torch.testing.assert_close(torch.from_numpy(archived["oracle_logits"]).to(device), logits.float(), rtol=0, atol=0)
            module = torch.nn.Module()
            module.num_key_value_groups = 1
            module.eval()
            _, stock_probabilities = eager_attention_forward(module, query, key, value, bias, scaling=1.0, dropout=0)
            torch.testing.assert_close(logits.softmax(dim=-1, dtype=torch.float32).to(query.dtype), stock_probabilities, rtol=0, atol=0)
            with mock.patch.object(self.targets, "stock_eager_logits", side_effect=AssertionError("precomputed oracle logits must be reused")):
                output, _, _ = self.targets.routed_eager_attention(query, key, value, torch.from_numpy(selected).to(device), bias,
                                                                  scaling=1.0, groups=1, precomputed_logits=logits)
            self.assertEqual(float(output[0, 2, 0, 0]), 20)

    def test_oracle_includes_actual_additive_bias_and_mandatory_recent_keys(self):
        query = torch.ones(1, 1, 4, 2)
        key = torch.tensor([[[[3, 3], [1, 1], [0, 0], [-1, -1]]]], dtype=torch.float32)
        bias = torch.zeros(1, 1, 4, 4).masked_fill(~torch.tril(torch.ones(4, 4, dtype=torch.bool)), torch.finfo(torch.float32).min)
        bias[..., 0] -= 10
        bias[..., 1] += 10
        logits = self.targets.stock_eager_logits(query, key, bias, scaling=0.5, groups=1)
        selected, _ = self.targets.build_oracle_selection_mask(logits, budget=2, recent_window=1)
        self.assertEqual(np.flatnonzero(selected[0, 0, 3]).tolist(), [1, 3])
        selected, _ = self.targets.build_oracle_selection_mask(logits, budget=1, recent_window=8)
        np.testing.assert_array_equal(selected[0, 0], np.eye(4, dtype=bool))

    def test_oracle_mass_optimum_does_not_imply_value_error_optimum(self):
        # Three old keys and eight mandatory recent keys. Dense value is -.05;
        # the largest-mass old key has a large value-error after renormalizing.
        masses = torch.tensor([0.4, 0.35, 0.05] + [0.025] * 8)
        logits = masses.log().reshape(1, 1, 1, 11).expand(1, 1, 11, 11).clone()
        selected, _ = self.targets.build_oracle_selection_mask(logits, budget=9, recent_window=8)
        oracle = torch.from_numpy(selected[0, 0, 10])
        alternative = torch.tensor([False, False, True] + [True] * 8)
        values = torch.tensor([-1.0, 1.0, 0.0] + [0.0] * 8)
        weights = logits[0, 0, 10].softmax(dim=-1)
        dense_value = float((weights * values).sum())
        oracle_value = float((weights[oracle] * values[oracle]).sum() / weights[oracle].sum())
        alternative_value = float((weights[alternative] * values[alternative]).sum() / weights[alternative].sum())
        self.assertGreater(float(weights[oracle].sum()), float(weights[alternative].sum()))
        self.assertGreater(abs(oracle_value - dense_value), abs(alternative_value - dense_value))

    def make_attention(self, dtype=torch.float32 if TORCH_AVAILABLE else None):
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                             num_key_value_heads=2, num_hidden_layers=1, attention_dropout=0)
        config._attn_implementation = "eager"
        torch.manual_seed(294)
        return LlamaAttention(config, layer_idx=0).to(dtype=dtype).eval()

    def test_full_budget_wrapper_matches_stock_attention_post_rope_output(self):
        for dtype in (torch.float32, torch.bfloat16):
            original = self.make_attention(dtype)
            hidden = torch.randn(2, 7, 32, dtype=dtype)
            angles = torch.linspace(0, 1, 7).reshape(1, 7, 1).expand(2, 7, 8)
            position_embeddings = (angles.cos().to(dtype), angles.sin().to(dtype))
            bias = torch.zeros(2, 1, 7, 7, dtype=dtype).masked_fill(~torch.tril(torch.ones(7, 7, dtype=torch.bool)), torch.finfo(dtype).min)
            wrapper = self.targets.RoutedLlamaAttention(original, method="dense", budget=128).eval()
            with torch.inference_mode():
                expected, expected_weights = original(hidden, position_embeddings, bias)
                actual, actual_weights = wrapper(hidden, position_embeddings, bias)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
            self.assertIs(wrapper.original, original)
            self.assertEqual(wrapper.last_diagnostics["selection_mask"].shape, (2, 4, 7, 7))

    def test_full_budget_oracle_wrapper_matches_stock_and_archives_native_scores(self):
        original = self.make_attention(torch.bfloat16)
        hidden = torch.randn(2, 7, 32, dtype=torch.bfloat16)
        position_embeddings = (torch.ones(2, 7, 8, dtype=torch.bfloat16), torch.zeros(2, 7, 8, dtype=torch.bfloat16))
        causal = torch.tril(torch.ones(7, 7, dtype=torch.bool))
        bias = torch.zeros(2, 1, 7, 7, dtype=torch.bfloat16).masked_fill(~causal, torch.finfo(torch.bfloat16).min)
        wrapper = self.targets.RoutedLlamaAttention(original, method="oracle", budget=128).eval()
        with torch.inference_mode():
            expected, expected_weights = original(hidden, position_embeddings, bias)
            actual, actual_weights = wrapper(hidden, position_embeddings, bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
        self.assertEqual(wrapper.last_diagnostics["oracle_logits_original_dtype"], "torch.bfloat16")
        self.assertEqual(wrapper.last_diagnostics["oracle_logits"].dtype, np.dtype("float32"))

    def test_wrapper_rejects_training_and_cache(self):
        original = self.make_attention()
        wrapper = self.targets.RoutedLlamaAttention(original, method="dense", budget=128)
        hidden = torch.randn(1, 3, 32)
        position_embeddings = (torch.ones(1, 3, 8), torch.zeros(1, 3, 8))
        with self.assertRaises(RuntimeError):
            wrapper(hidden, position_embeddings, None)
        wrapper.eval()
        with self.assertRaises(ValueError):
            wrapper(hidden, position_embeddings, None, past_key_values=object())


if __name__ == "__main__":
    unittest.main()
