import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.kernels.amd import nv_custom_kernels_supported, q8_quantize, gated_delta_prefill, QUANT_SIZES, NV_UNSUPPORTED_QUANTS
from test.runtime.test_llm_quantized import QuantLinearMixin

def setUpModule():
  if not nv_custom_kernels_supported(Tensor.empty(1).device): raise unittest.SkipTest("NV with the CUDA renderer required")

NV_QUANTS = [t for t in QUANT_SIZES if t not in NV_UNSUPPORTED_QUANTS]

class TestNVQuantKernels(QuantLinearMixin, unittest.TestCase):
  def test_values_and_scales(self):
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale, gsum = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)
    np.testing.assert_array_equal(gsum.numpy().reshape(2, 2), expected.reshape(2, 2, 16).sum(-1).astype(np.float32))

  def test_quantize_rounding_ties(self):
    values = np.array([-127,127]+[i+0.5 for i in range(-15,15)], dtype=np.float32)
    quant, _, _ = q8_quantize(Tensor(values), 1, 32)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(32).numpy(), np.rint(values).astype(np.int8))

  def test_quant_linear(self):
    # token counts that use 1, 3 (odd), 16 and 32/64 (token blocks of 16) tokens per wave
    for typ in NV_QUANTS:
      self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=768, out_features=32, token_counts=(1, 3, 16, 32, 64))

  def test_quant_linear_symbolic(self):
    for typ in NV_QUANTS:
      self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=48, token_counts=(3, 17), bias=True, symbolic=True)

  def test_unsupported_quants_fall_back(self):
    for typ in NV_UNSUPPORTED_QUANTS:
      self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3), custom=False)

class TestNVGatedDelta(unittest.TestCase):
  def test_gated_delta_random_scan(self):
    rng = np.random.default_rng(42)
    B, H, T, K, V = 1, 2, 8, 128, 128
    q, k = (rng.normal(0, 0.1, (B, H, T, K)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(B, H, T, V)).astype(np.float32)
    beta = rng.uniform(0, 1, (B, H, T)).astype(np.float32)
    initial = rng.normal(0, 0.1, (B, H, V, K)).astype(np.float32)
    for per_channel in (False, True):
      alpha = rng.uniform(0.8, 1, (B, H, T, K) if per_channel else (B, H, T)).astype(np.float32)
      for start in (0, 3):
        with self.subTest(per_channel=per_channel, start=start):
          expected_state, outputs = np.zeros_like(initial) if start == 0 else initial.copy(), []
          for t in range(T):
            expected_state *= alpha[:, :, t, None, :] if per_channel else alpha[:, :, t, None, None]
            delta = (v[:, :, t] - (expected_state*k[:, :, t, None]).sum(-1)) * beta[:, :, t, None]
            expected_state += delta[..., None]*k[:, :, t, None]
            outputs.append((expected_state*q[:, :, t, None]).sum(-1))
          state = Tensor(initial).contiguous().realize()
          out = gated_delta_prefill(Tensor(q), Tensor(k), Tensor(v), Tensor(beta), Tensor(alpha), state,
                                    Tensor(UOp.variable("start_pos", 0, 10).bind(start)))
          np.testing.assert_allclose(out.numpy(), np.stack(outputs, axis=2), rtol=1e-4, atol=1e-5)
          np.testing.assert_allclose(state.numpy(), expected_state, rtol=1e-4, atol=1e-5)

if __name__ == "__main__":
  unittest.main()
