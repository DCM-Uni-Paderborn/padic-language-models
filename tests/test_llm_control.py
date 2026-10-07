"""Projection arithmetic tests, without model or dataset downloads.

The production class/function are compiled directly from their source AST to
isolate them from optional Hugging Face imports. No source code is rewritten.
Run on a host with PyTorch using unittest discovery. Missing torch is an explicit
skip; it does not count as validation of the projection or LLM pilot.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def load_projection_targets():
    path = Path(__file__).resolve().parents[1] / "experiments/llm_control.py"
    source = ast.parse(path.read_text(), filename=str(path))
    targets = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {"ResidueLinear", "evaluate"}]
    if {node.name for node in targets} != {"ResidueLinear", "evaluate"}:
        raise AssertionError("Production projection/evaluation targets missing")
    namespace = {"torch": torch, "F": F, "math": math, "time": time, "__name__": "projection_control_tests"}
    module = ast.Module(body=targets, type_ignores=[])
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return namespace["ResidueLinear"], namespace["evaluate"]


@unittest.skipIf(torch is None, "PyTorch unavailable: run this suite on the experiment host")
class ProjectionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ResidueLinear, evaluate = load_projection_targets()
        cls.evaluate = staticmethod(evaluate)
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    def make_projection(self, dimension=13, outputs=5, bias=True, dtype=None, device="cpu"):
        torch.manual_seed(914)
        original = torch.nn.Linear(dimension, outputs, bias=bias, device=device, dtype=dtype or torch.float32)
        return original.eval()

    def independent_expected(self, wrapper, inputs, residue_bits):
        x = inputs.float()
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax == 0, torch.ones_like(amax), (amax / 127).clamp_min(torch.finfo(torch.float32).tiny))
        codes = (x / scale).round().clamp(-127, 127)
        flat = codes.reshape(-1, codes.shape[-1]).to(torch.int64).tolist()
        rows = wrapper.weight_codes.to(torch.int64).tolist()
        exact = [[sum(int(w) * int(v) for w, v in zip(row, vector)) for row in rows] for vector in flat]
        decoded = exact
        if residue_bits is not None:
            modulus = 1 << residue_bits
            decoded = [[s % modulus - modulus if s % modulus >= modulus // 2 else s % modulus for s in vector] for vector in exact]
        exact_tensor = torch.tensor(exact, dtype=torch.int64, device=inputs.device).reshape(*inputs.shape[:-1], len(rows))
        decoded_tensor = torch.tensor(decoded, dtype=torch.float32, device=inputs.device).reshape(*inputs.shape[:-1], len(rows))
        expected = decoded_tensor * scale * wrapper.weight_scale.squeeze(-1)
        if wrapper.original.bias is not None:
            expected = expected + wrapper.original.bias.float()
        return expected.to(inputs.dtype), exact_tensor

    def test_fp32_emulation_matches_independent_python_integer_dots(self):
        original = self.make_projection()
        inputs = torch.randn(2, 3, 13)
        for bits in (None, 4, 8, 12, 32, 52):
            wrapper = self.ResidueLinear(original, bits)
            expected, exact = self.independent_expected(wrapper, inputs, bits)
            actual = wrapper(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(wrapper.statistics["dot_count"], 30)
            self.assertEqual(wrapper.statistics["max_integer_abs"], int(exact.abs().max()))
            if bits is not None:
                modulus = 1 << bits
                decoded = exact.remainder(modulus)
                decoded = torch.where(decoded >= modulus // 2, decoded - modulus, decoded)
                self.assertEqual(wrapper.statistics["wrap_count"], int((decoded != exact).sum()))
                self.assertEqual(wrapper.statistics["modular_squared_error"], float((decoded.double() - exact.double()).square().sum()))

    def test_no_wrap_projection_is_bitwise_identical_to_integer_baseline(self):
        original = self.make_projection(dimension=257, outputs=3)
        inputs = torch.randn(3, 257)
        integer = self.ResidueLinear(original, None)
        residue = self.ResidueLinear(original, 32)
        torch.testing.assert_close(integer(inputs), residue(inputs), rtol=0, atol=0)
        self.assertEqual(residue.statistics["wrap_count"], 0)

    def test_even_half_modulus_endpoint_decodes_negative(self):
        original = self.make_projection(dimension=4, outputs=1, bias=False)
        with torch.no_grad():
            original.weight.fill_(1)
        inputs = torch.tensor([[1.0, 1.0, 1.0, -1.0]])
        # Codes give dot 7*127*(1+1+1-1)=1778. The direct modulo
        # endpoint is independently covered for a constructed code dot below.
        wrapper = self.ResidueLinear(original, 4)
        expected, exact = self.independent_expected(wrapper, inputs, 4)
        torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)
        with torch.no_grad():
            wrapper.weight_codes.fill_(1)
        inputs = torch.ones(1, 4)
        # 4*127 = 508 gives residue 12, hence negative signed recovery.
        expected, exact = self.independent_expected(wrapper, inputs, 4)
        self.assertEqual(int(exact[0, 0]), 508)
        self.assertLess(float(expected[0, 0]), 0)
        torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)
        # Set one weight code to 5: (5+1+1+1)*127=1016 == 8 mod16.
        with torch.no_grad():
            wrapper.weight_codes[0, 0] = 5
        expected, exact = self.independent_expected(wrapper, inputs, 4)
        self.assertEqual(int(exact[0, 0]) % 16, 8)
        torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)
        self.assertLess(float(expected[0, 0]), 0)

    def test_zero_inputs_weights_and_bias_are_finite(self):
        original = self.make_projection(dimension=4, outputs=2)
        with torch.no_grad():
            original.weight.zero_()
            original.bias.copy_(torch.tensor([0.5, -0.25]))
        for bits in (None, 4, 32):
            wrapper = self.ResidueLinear(original, bits)
            result = wrapper(torch.zeros(2, 3, 4))
            self.assertTrue(bool(torch.isfinite(result).all()))
            torch.testing.assert_close(result, original.bias.expand(2, 3, 2), rtol=0, atol=0)
            self.assertEqual(wrapper.statistics["max_integer_abs"], 0)
            self.assertEqual(wrapper.statistics["wrap_count"], 0)

    def test_subnormal_quantization_scales_stay_positive(self):
        original = self.make_projection(dimension=4, outputs=2, bias=False)
        subnormal = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        with torch.no_grad():
            original.weight.zero_()
            original.weight[0, 0] = subnormal
        wrapper = self.ResidueLinear(original, 32)
        self.assertTrue(bool((wrapper.weight_scale > 0).all()))
        self.assertTrue(bool(torch.isfinite(wrapper.weight_codes).all()))
        inputs = torch.zeros(1, 4)
        inputs[0, 0] = subnormal
        result = wrapper(inputs)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_nonfinite_weights_and_inputs_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            original = self.make_projection(dimension=4, outputs=2)
            with torch.no_grad():
                original.weight[0, 0] = bad
            with self.assertRaises(ValueError):
                self.ResidueLinear(original, 32)
            wrapper = self.ResidueLinear(self.make_projection(dimension=4, outputs=2), 32)
            inputs = torch.zeros(1, 4)
            inputs[0, 0] = bad
            with self.assertRaises(ValueError):
                wrapper(inputs)

    def test_proven_accumulator_range_is_enforced(self):
        unsafe_dimension = (2**24 + 888) // 889
        with self.assertRaises(ValueError):
            self.ResidueLinear(self.make_projection(dimension=unsafe_dimension, outputs=1), None)
        self.ResidueLinear(self.make_projection(dimension=unsafe_dimension - 1, outputs=1), None)
        for bits in (1, 53):
            with self.assertRaises(ValueError):
                self.ResidueLinear(self.make_projection(), bits)

    def test_supported_float_input_dtype_and_shape_are_preserved(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            original = self.make_projection(dimension=4, outputs=2, dtype=dtype)
            wrapper = self.ResidueLinear(original, 32)
            for shape in ((4,), (3, 4), (2, 3, 4)):
                inputs = torch.ones(shape, dtype=dtype)
                result = wrapper(inputs)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(tuple(result.shape), (*shape[:-1], 2))
                expected, _ = self.independent_expected(wrapper, inputs, 32)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_mode_replacement_retains_same_unchanged_original(self):
        original = self.make_projection(dimension=4, outputs=2)
        saved_weight, saved_bias = original.weight.detach().clone(), original.bias.detach().clone()
        container = torch.nn.Module()
        for bits in (None, 8, 12, 32, None):
            wrapper = self.ResidueLinear(original, bits)
            container.projection = wrapper
            self.assertIs(wrapper.original, original)
            wrapper(torch.ones(1, 4))
        torch.testing.assert_close(original.weight, saved_weight, rtol=0, atol=0)
        torch.testing.assert_close(original.bias, saved_bias, rtol=0, atol=0)

    def test_statistics_reset_and_evaluation_omit_warmup(self):
        original = self.make_projection(dimension=4, outputs=8)
        wrapper = self.ResidueLinear(original, 32)

        class TinyLanguageModel(torch.nn.Module):
            def __init__(self, projection):
                super().__init__()
                self.projection = projection

            def forward(self, input_ids, use_cache=False):
                features = F.one_hot(input_ids.remainder(4), num_classes=4).float()
                return SimpleNamespace(logits=self.projection(features))

        model = TinyLanguageModel(wrapper).eval()
        tokens = torch.arange(8, dtype=torch.int64)
        result = self.evaluate(model, tokens, 4, "cpu", wrapper)
        self.assertEqual(result["predicted_tokens"], 6)
        self.assertEqual(len(result["block_mean_nll"]), 2)
        self.assertEqual(result["layer_statistics"]["dot_count"], 2 * 4 * 8)
        self.assertEqual(result["wrap_fraction"], 0)
        wrapper.reset_statistics()
        self.assertEqual(wrapper.statistics["dot_count"], 0)
        self.assertEqual(wrapper.statistics["max_integer_abs"], 0)

    def test_autocast_rejected_before_integer_matrix_multiply(self):
        wrapper = self.ResidueLinear(self.make_projection(dimension=4, outputs=2), 32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with self.assertRaises(RuntimeError):
                wrapper(torch.ones(1, 4))

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_reference_and_buffers_with_tf32_disabled(self):
        original = self.make_projection(dimension=13, outputs=5, device="cuda", dtype=torch.bfloat16)
        wrapper = self.ResidueLinear(original, 32)
        self.assertEqual(wrapper.weight_codes.device.type, "cuda")
        self.assertEqual(wrapper.weight_scale.device.type, "cuda")
        inputs = torch.ones(2, 13, device="cuda", dtype=torch.bfloat16)
        expected, _ = self.independent_expected(wrapper, inputs, 32)
        torch.testing.assert_close(wrapper(inputs), expected, rtol=0, atol=0)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with self.assertRaises(RuntimeError):
                wrapper(inputs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False


if __name__ == "__main__":
    unittest.main()
