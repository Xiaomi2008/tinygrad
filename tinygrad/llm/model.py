from __future__ import annotations
import enum, functools, itertools, mmap, pathlib
from dataclasses import dataclass, replace
import typing
from typing import Callable, cast
from tinygrad.helpers import prod
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, flash_attention_nv, _device_name, amd_custom_kernels_supported, gated_delta_kernel_supported
from tinygrad.llm.gguf import gguf_load, gguf_load_partial, ggml_data_to_tensor, ggml_nbytes
from tinygrad.uop.ops import resolve

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  # realized: requests with images rewrite it in place (see Transformer.set_rope_layout), and the JITs must read it as a buffer
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device).realize()

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  rope_sections: tuple[int, ...] = ()  # M-RoPE (Qwen3.5): rotary frequencies split between the (t, h, w) positions of image tokens

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

# ******** quantized KV cache ********

def kv_rotation(D:int, device) -> Tensor:
  # random-sign Hadamard matrix / sqrt(D): orthogonal (o = (P (v R)) R^T), and it spreads outlier channels over all of them
  assert D & (D-1) == 0, "head_dim must be a power of two"
  signs = [1 if (i*2654435761 >> 7) & 1 else -1 for i in range(D)]
  # realized here: left lazy, the graph re-uploads it from the host on every call (8 copies per token over the eGPU link)
  return Tensor([[signs[i] * (1 if bin(i & j).count("1") % 2 == 0 else -1) / D**0.5 for j in range(D)] for i in range(D)],
                device=device).contiguous().realize()

def kv_quantize(x:Tensor, bits:int) -> tuple[Tensor, Tensor]:
  # symmetric absmax per vector: 8 bits -> int8 values; 4 bits -> nibble + 8 in 1..15, two per byte (low nibble first)
  levels = 127 if bits == 8 else 7
  scale = (x.abs().max(-1) / levels).maximum(1e-8)
  q = (x / scale.unsqueeze(-1)).round().clip(-levels, levels)
  if bits == 8: return q.cast(dtypes.int8), scale
  n = (q + 8).cast(dtypes.uint8).reshape(*x.shape[:-1], x.shape[-1]//2, 2)
  return n[..., 0] | (n[..., 1] << 4), scale

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    kv_scale, fmt = None, self._kv_fmt(x)
    if fmt == "int8":
      values, scale = kv_quantize(Tensor.stack(k, v).float(), 8)
      store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(values.uop)
    elif fmt == "k8vr4":
      (kq, kscale), (vq, vscale) = kv_quantize(k.float(), 8), kv_quantize(v.float() @ self.kv_rot, 4)
      store = self.cache_kv[:, :, start_pos:start_pos+T, :].uop.store(kq.bitcast(dtypes.uint8).cat(vq, dim=-1).uop)
      scale = Tensor.stack(kscale, vscale)
    if fmt != "fp16":
      scale_store = self.cache_kv_scale[:, :, :, start_pos:start_pos+T].uop.store(scale.cast(dtypes.half).uop)
      kv_scale = Tensor(self.cache_kv_scale.uop.after(scale_store))
    else: store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    # on NV, prefill chunks (and decode: a KV group's heads fill the rows) use the tensor-core flash attention kernel over the KV cache
    if _device_name(x.device) == "NV" and getenv("NV_FLASH_ATTN", 1) and (resolve(T != 1) or getenv("NV_FLASH_DECODE", 1)):
      attn = flash_attention_nv(q, assigned_kv, start_pos+T, kv_scale, fmt)
      if fmt == "k8vr4": attn = attn @ self.kv_rot.T                                   # V was stored rotated
      attn = attn.cast(x.dtype).transpose(1, 2).reshape(B, T, -1)                      # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    if fmt == "k8vr4":
      D = self.config.head_dim
      k = assigned_kv[:, :, 0:start_pos+T, :D].bitcast(dtypes.int8).cast(dtypes.half) * kv_scale[0, :, :, 0:start_pos+T].unsqueeze(-1)
      n = assigned_kv[:, :, 0:start_pos+T, D:]
      n = Tensor.stack(n & 15, n >> 4, dim=-1).reshape(*n.shape[:-1], D).float() - 8
      v = (n * kv_scale[1, :, :, 0:start_pos+T].unsqueeze(-1).float()).cast(dtypes.half)
    else:
      k = assigned_kv[0, :, :, 0:start_pos+T, :]
      v = assigned_kv[1, :, :, 0:start_pos+T, :]
      if kv_scale is not None:
        k = k.cast(dtypes.half) * kv_scale[0, :, :, 0:start_pos+T].unsqueeze(-1)
        v = v.cast(dtypes.half) * kv_scale[1, :, :, 0:start_pos+T].unsqueeze(-1)

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    # build the columns from a static arange: triu's arange over the symbolic start_pos+T lowers to a running-sum reduce that is
    # quadratic in the context length (~100 ms per 8-token chunk at 6k tokens on an RTX 3060 Ti)
    cols = Tensor.arange(self.config.max_context)[:start_pos+T]
    mask = (Tensor.arange(T).unsqueeze(-1) + (start_pos+1) <= cols).where(float("-inf"), 0).cast(x.dtype).reshape(1, 1, T, start_pos+T) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    if fmt == "k8vr4": attn = attn.float() @ self.kv_rot.T
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _kv_fmt(self, x:Tensor) -> str:
    # KV_FMT (NV): int8 (default: int8 K and V, fp16 scale per position) | k8vr4 (int8 K, V rotated by a random-sign Hadamard
    # matrix and stored at 4 bits: 0.75 bytes per value) | fp16. K needs 8 bits (4-bit K moved the next-token distribution ~20x
    # more than 4-bit V); the rotation spreads V's outlier channels, cutting its 4-bit error ~3x.
    if _device_name(x.device) != "NV": return "fp16"
    return str(getenv("KV_FMT", "int8" if getenv("KV_INT8", 1) else "fp16"))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      # KV_INT8 (NV): int8 values with an fp16 scale per (k/v, head, position), half the memory of an fp16 cache
      fmt, B, H, C, D = self._kv_fmt(x), x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim
      # k8vr4: one row per position, D bytes of int8 K then D/2 bytes of 4-bit V (two per byte, low nibble first)
      if fmt == "k8vr4": self.cache_kv = Tensor.zeros(B, H, C, D + D//2, dtype=dtypes.uint8, device=x.device)
      else: self.cache_kv = Tensor.zeros(2, B, H, C, D, dtype=dtypes.int8 if fmt == "int8" else dtypes.half, device=x.device)
      if fmt != "fp16": self.cache_kv_scale = Tensor.zeros(2, B, H, C, dtype=dtypes.half, device=x.device)
      if fmt == "k8vr4": self.kv_rot = kv_rotation(D, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    # build the columns from a static arange: triu's arange over the symbolic start_pos+T lowers to a running-sum reduce that is
    # quadratic in the context length (~100 ms per 8-token chunk at 6k tokens on an RTX 3060 Ti)
    cols = Tensor.arange(self.config.max_context)[:start_pos+T]
    mask = (Tensor.arange(T).unsqueeze(-1) + (start_pos+1) <= cols).where(float("-inf"), 0).cast(x.dtype).reshape(1, 1, T, start_pos+T) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)
    # speculative verify: keep the whole window and the recurrent state after each token, so rollback() can undo rejected tokens
    hist = self.keep_history and isinstance(T, int) and T > 1
    if hist:
      conv_state_store = self.conv_state.uop.after(conv_state_store, self.conv_hist.uop.store(conv_window.cast(self.conv_hist.dtype).uop))

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and gated_delta_kernel_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos), self.rec_hist if hist else None).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
      state = initial.where(0, state.float())
      outs, states = [], []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))
        states.append(state)

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      if hist: state_store = state_store.after(self.rec_hist.uop.store(Tensor.stack(*states[:-1], dim=2).uop))
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  keep_history = False

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()
    if self.keep_history and isinstance(T:=x.shape[1], int) and T > 1 and not hasattr(self, "conv_hist"):
      self.conv_hist = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1+T, self.conv_channels, device=x.device).clone()
      self.rec_hist = Tensor.zeros(x.shape[0], self.num_v_heads, T-1, self.head_v_dim, self.head_k_dim, device=x.device).clone()

  def rollback(self, kept:int) -> list[Tensor]:
    # after a verify pass over T tokens, keep only the first `kept` (1 <= kept < T): the conv state is the window's columns that
    # followed them, the recurrent state the one saved after them
    return [self.conv_state.assign(self.conv_hist[:, kept:kept+self.ssm_conv_kernel-1]), self.recurrent_state.assign(self.rec_hist[:, :, kept-1])]

class HostEmbedding:
  """
  A quantized token embedding left in the GGUF file: each call copies only the packed rows of the tokens in use to the device,
  where they're dequantized like any other weight. Saves the whole table in device memory (0.7 GB for a 248k vocab at 4 bits).
  """
  def __init__(self, path:str|pathlib.Path, offset:int, dims:tuple[int, ...], ggml_type:int):
    (self.dim, self.vocab_size), self.ggml_type = dims, ggml_type
    self.row_bytes = ggml_nbytes(self.dim, ggml_type)
    with open(path, "rb") as f: self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    self.offset = offset

  def rows(self, ids:list[int], pad_to:int, device:str) -> Tensor:
    rb, base = self.row_bytes, self.offset
    data = b"".join(self.mm[base+i*rb:base+(i+1)*rb] for i in ids) + bytes(rb * (pad_to - len(ids)))
    return Tensor(data, device=device).reshape(1, pad_to, rb)

  def __call__(self, raw:Tensor) -> Tensor:
    # symbolic token count: dequantize the whole padded chunk, the garbage rows are sliced off
    T = raw.shape[1]
    if not isinstance(T, int): raw = raw.pad_to(raw.max_shape)
    x = ggml_data_to_tensor(raw.flatten(), prod(raw.shape[:2]) * self.dim, self.ggml_type).reshape(*raw.shape[:2], self.dim)
    return x if isinstance(T, int) else x[:, :T]

class MTPHead:
  """
  Qwen3.5's multi-token-prediction layer (GGUF `blk.N.nextn.*`): from the main model's final normed hidden state at position i and
  the embedding of token i+1, one more transformer block predicts token i+2. Used as the draft model for speculative decoding.
  """
  def __init__(self, config:TransformerConfig):
    self.eh_proj = Linear(2*config.dim, config.dim, bias=False)
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.block = TransformerBlock(replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0,
                                          hidden_dim=config.dense_hidden_dim or config.hidden_dim))

  def __call__(self, hidden:Tensor, embeds:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.eh_proj(self.enorm(embeds).cat(self.hnorm(hidden), dim=-1))
    return self.shared_head_norm(self.block(x, start_pos))

@dataclass
class PromptImage:
  """An image's span in the prompt: tokens[start:start+gh*gw] are placeholders (negative ids, the same for the same image so prefix
  caching works) for a gh x gw grid of embeddings, which embeds() returns as (gh*gw, D), on the host or any device."""
  start: int
  gh: int
  gw: int
  embeds: Callable[[], typing.Any]  # a Tensor or a numpy array
  @property
  def end(self) -> int: return self.start + self.gh * self.gw

class Transformer:
  def __init__(self, config:TransformerConfig):
    self.config = config
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd: nn.Embedding|None = nn.Embedding(config.vocab_size, config.dim)
    self.host_embd: HostEmbedding|None = None
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # recurrent models: a copy of the recurrent state at a stable point of the prompt (e.g. the end of the last user message), so a
    # request whose rendering diverges after that point (chat templates drop past thinking blocks) resumes there instead of at 0
    self._checkpoint_tokens: list[int] = []
    self._checkpoint_state: list[tuple[Tensor, Tensor]] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    # prompt chunks before the last one only fill the caches: no output projection (the 248k-row LM head is ~5% of a chunk)
    self.prefill_body_jit = TinyJit(self.forward_body)
    # prompt chunks with image tokens take embeddings instead of tokens; the RoPE table is laid out for the current prompt's images
    self.prefill_x_jit, self.prefill_body_x_jit = TinyJit(self.forward_x), TinyJit(self.forward_body_x)
    self._rope_key: tuple = ()
    self._rope_default: Tensor|None = None
    # speculative decoding with the MTP layer as the draft (see generate)
    self.mtp: MTPHead|None = None
    self.spec_stats = [0, 0]  # drafts proposed, accepted

  # with a host embedding, `tokens` is the packed embedding rows (B, T, row_bytes) instead of the token ids (B, T)
  def embed(self, tokens:Tensor) -> Tensor:
    return (self.host_embd(tokens) if self.host_embd is not None else cast(nn.Embedding, self.token_embd)(tokens)).float()

  # the *_x variants take the input embeddings (B, T, D) instead of tokens: prompt chunks holding image tokens
  def forward_body(self, tokens:Tensor, start_pos:int|UOp) -> Tensor: return self.forward_body_x(self.embed(tokens), start_pos)
  def forward_body_x(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    for block in self.blk: x = block(x, start_pos)
    return x[:, -1:, :1].contiguous()

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return self.forward_x(self.embed(tokens), start_pos, temperature)
  def forward_x(self, x:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    for block in self.blk: x = block(x, start_pos)        # (B, T, D)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  # ******** speculative decoding ********

  @staticmethod
  def _sample(logits:Tensor, temperature:Tensor) -> Tensor:
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1)

  def forward_hidden(self, tokens:Tensor, start_pos:int|UOp) -> Tensor: return self.forward_hidden_x(self.embed(tokens), start_pos)
  def forward_hidden_x(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    # a prompt chunk that also feeds the MTP layer: the final normed hidden states, no output projection
    for block in self.blk: x = block(x, start_pos)
    return self.output_norm(x).contiguous()

  def forward_hidden_sample(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    # the last prompt chunk: hidden states and the first sampled token
    h = self.forward_hidden(tokens, start_pos)
    return h, self._sample(self.output(h[:, -1:])[:, -1, :], temperature).reshape(1, 1)

  def forward_verify(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    # [last token, draft tokens...] -> a sample at every position (B, T) and the hidden states (B, T, D)
    for b in self._recurrent_blocks(): b.keep_history = True
    try: h = self.forward_hidden(tokens, start_pos)
    finally:
      for b in self._recurrent_blocks(): b.keep_history = False
    return self._sample(self.output(h), temperature).contiguous(), h

  def forward_draft(self, hidden:Tensor, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    # the MTP layer over (hidden[i], token[i+1]) pairs at positions start_pos.., greedy draft from the last one: a deterministic draft
    # keeps sampling exact (accept iff the target's own sample equals it)
    x = cast(MTPHead, self.mtp)(hidden, self.embed(tokens), start_pos)
    return self.output(x[:, -1:])[:, -1, :].argmax(-1).reshape(1, 1)

  def forward_draft_fill(self, hidden:Tensor, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    # prompt chunks: only fill the MTP layer's KV cache
    return cast(MTPHead, self.mtp)(hidden, self.embed(tokens), start_pos)[:, -1:, :1].contiguous()

  def _rollback(self) -> Tensor:
    # keep only the first of the two verified tokens
    return Tensor.stack(*[t.flatten()[0] for b in self._recurrent_blocks() for t in b.rollback(1)])

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # HOST_EMBD: leave the token embedding in the file (see HostEmbedding) and skip the multi-token-prediction layers, which
    # nothing here runs: 0.9 GB of device memory on Qwen3.8-27B
    # SPEC (default on): load the first MTP layer for speculative decoding, from this file or (for quants that dropped it, like
    # GSQ-RCO IQ3_S) from MTP_GGUF, another GGUF of the same model
    def first_mtp(kv:dict) -> int:
      arch = kv['general.architecture']
      return kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    spec = bool(getenv("SPEC", 1))
    host_embd = None
    if getenv("HOST_EMBD", 0) and not isinstance(gguf, Tensor):
      def skip(kv:dict, name:str) -> bool:
        if name == 'token_embd.weight': return True
        if not name.startswith('blk.'): return False
        layer = int(name.split('.')[1])
        return layer >= first_mtp(kv) and not (spec and layer == first_mtp(kv))
      kv, state_dict, skipped = gguf_load_partial(gguf, skip)
      embd_offset, embd_dims, embd_type = skipped['token_embd.weight']
      if 'output.weight' in state_dict: host_embd = HostEmbedding(gguf, embd_offset, embd_dims, embd_type)
      else:  # tied output projection: the table is needed on the device anyway
        n = ggml_nbytes(prod(embd_dims), embd_type)
        raw = Tensor(pathlib.Path(gguf))[embd_offset:embd_offset+n+1].to(None).realize()
        state_dict['token_embd.weight'] = ggml_data_to_tensor(raw, prod(embd_dims), embd_type).reshape(*reversed(embd_dims))
    else:
      # TODO: remove the need for copy to default device
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    # the MTP layer's tensors: blk.N.nextn.* -> mtp.*, the rest of blk.N -> mtp.block.*
    mtp_state: dict[str, Tensor] = {}
    if spec:
      prefix = f"blk.{first_mtp(kv)}."
      if not any(k.startswith(prefix) for k in state_dict) and (mtp_path := getenv("MTP_GGUF", "")):
        def skip_main(kv:dict, name:str) -> bool: return not name.startswith(f"blk.{first_mtp(kv)}.")
        mtp_kv, extra, _ = gguf_load_partial(mtp_path, skip_main)
        assert mtp_kv[f"{mtp_kv['general.architecture']}.embedding_length"] == kv[f"{kv['general.architecture']}.embedding_length"]
        prefix = f"blk.{first_mtp(mtp_kv)}."
        state_dict.update({k:v.cast('float16') if getenv("HALF", 1) else v for k,v in extra.items()})
      for k in [k for k in state_dict if k.startswith(prefix)]:
        name = k[len(prefix):].replace('post_attention_norm', 'ffn_norm')
        mtp_state[name[len('nextn.'):] if name.startswith('nextn.') else 'block.'+name] = state_dict.pop(k)
      if 'eh_proj.weight' not in mtp_state: mtp_state = {}
    else:
      for k in [k for k in state_dict if k.startswith('blk.') and int(k.split('.')[1]) >= first_mtp(kv)]: state_dict.pop(k)

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      rope_sections=tuple(kv.get(f'{arch}.rope.dimension_sections', ())))
    model = Transformer(config)
    if host_embd is not None: model.token_embd, model.host_embd = None, host_embd
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if mtp_state:
      model.mtp = MTPHead(replace(config, qk_norm=config.head_dim) if config.ssm else config)
      nn.state.load_state_dict(model.mtp, mtp_state, verbose=False, consume=True, realize=False)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))
    # a two-chunk prompt captures both prefill paths (cache-filling chunks without the head, and the last chunk), each runs once
    # per prompt; different tokens each time so nothing is reused from the cache
    for i in range(2): list(zip(range(2), self.generate([1+i]*(getenv("PREFILL_CHUNK", 32)+8))))
    if self.mtp is not None and getenv("SPEC", 1):
      # every speculative path (draft over 1 or 2 pairs, verify, rollback) captured before the first request, whatever got accepted
      list(zip(range(8), self.generate([3, 4, 5])))
      v, temp = UOp.variable("start_pos", 0, self.max_context-1), Tensor([0.0])
      for _ in range(3):
        ys, h = self.verify_jit(self._spec_input([1, 2], 2), v.bind(10), temp)
        for n in (1, 2):
          self.draft_h[n].assign(h[:, :n]).realize()
          self.draft_jits[n](self.draft_h[n], self._spec_input([1, 2][:n], n), v.bind(9)).realize()
        self.rollback_jit().realize()
      self.spec_stats = [0, 0]
    self._cached_tokens, self._checkpoint_tokens = [], []

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block: return self._recurrent_resume(tokens)[0]
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def _recurrent_resume(self, tokens:list[int]) -> tuple[int, bool]:
    # (start_pos, restore checkpoint first): recurrent state can't be partially reused after divergence, so resume only where the
    # state is known: the end of the cached tokens, or the saved checkpoint
    def extends(prefix:list[int]) -> bool: return 0 < len(prefix) < len(tokens) and tokens[:len(prefix)] == prefix
    if extends(self._cached_tokens): return len(self._cached_tokens), False
    if extends(self._checkpoint_tokens): return len(self._checkpoint_tokens), True
    return 0, False

  def _recurrent_blocks(self) -> list[GatedDeltaNetBlock]: return [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]

  # ******** images ********

  def set_rope_layout(self, images:list[PromptImage]):
    """
    M-RoPE (Qwen3-VL/3.5): an image token at row i, column j of its grid gets the positions (t, h, w) = (s, s+i, s+j), s being the
    position where the image starts, and the text after it continues from s + max(gh, gw), not from s + gh*gw. Every position is a
    function of the cache index, so instead of passing positions around, the shared RoPE table (one row per cache index, read by
    every attention layer and the MTP layer) is rewritten in place for the prompt's images, and restored without them.
    """
    key = tuple((im.start, im.gh, im.gw) for im in images)
    if key == self._rope_key: return
    table = next((b.freqs_cis for b in self.blk if hasattr(b, "freqs_cis")), None)
    if table is None: return  # no attention layer has run yet (warmup sets them up), and images need M-RoPE sections anyway
    if self._rope_default is None: self._rope_default = table.clone().realize()
    if not key: table.assign(self._rope_default).realize()
    else:
      import numpy as np
      assert self.config.rope_sections, "this model has no M-RoPE sections: images are not supported"
      C, half = self.max_context, table.shape[1] // 2
      pos = np.tile(np.arange(C, dtype=np.float64), (3, 1))
      shift = 0
      for im in images:
        s, n = im.start + shift, im.gh * im.gw
        k = np.arange(n)
        pos[0, im.start:im.end], pos[1, im.start:im.end], pos[2, im.start:im.end] = s, s + k // im.gw, s + k % im.gw
        shift += max(im.gh, im.gw) - n
        pos[:, im.end:] = np.arange(im.end, C) + shift
      # interleaved sections: frequency i follows h if i%3 == 1 (first 3*sections[1]), w if i%3 == 2 (first 3*sections[2]), else t
      sec = [1 if i % 3 == 1 and i < 3*self.config.rope_sections[1] else 2 if i % 3 == 2 and i < 3*self.config.rope_sections[2] else 0
             for i in range(half)]
      inv = 1.0 / (self.config.rope_theta ** (np.arange(0, 2*half, 2, dtype=np.float64) / (2*half)))
      ang = pos[sec].T * inv  # (C, half)
      table.assign(Tensor(np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32), device=table.device)).realize()
    self._rope_key = key

  def _embed_chunk(self, tokens:list[int], lo:int, n:int, pad:int, images:list[PromptImage]) -> Tensor:
    # input embeddings (1, pad, D) of tokens[lo:lo+n], image rows taken from the images' embeddings
    x = self.embed(self._spec_input([t if t >= 0 else 0 for t in tokens[lo:lo+n]], pad))
    for im in images:
      a, b = max(lo, im.start), min(lo + n, im.end)
      if a >= b: continue
      e = im.embeds()[a-im.start:b-im.start]
      rows = (e.to(x.device) if isinstance(e, Tensor) else Tensor(e, device=x.device)).float().reshape(1, b-a, -1)
      x = x[:, :a-lo].cat(rows, x[:, b-lo:], dim=1)
    return x.contiguous().realize()

  @staticmethod
  def _has_image(images:list[PromptImage], lo:int, hi:int) -> bool: return any(im.start < hi and lo < im.end for im in images)

  def _save_checkpoint(self, tokens:list[int]):
    blocks = self._recurrent_blocks()
    if not self._checkpoint_state:
      self._checkpoint_state = [(b.conv_state.clone().realize(), b.recurrent_state.clone().realize()) for b in blocks]
    else:
      for b, (conv, rec) in zip(blocks, self._checkpoint_state): Tensor.realize(conv.assign(b.conv_state), rec.assign(b.recurrent_state))
    self._checkpoint_tokens = list(tokens)

  def _restore_checkpoint(self):
    for b, (conv, rec) in zip(self._recurrent_blocks(), self._checkpoint_state):
      Tensor.realize(b.conv_state.assign(conv), b.recurrent_state.assign(rec))

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0, checkpoint:int|None=None,
               on_prefill:Callable[[int], None]|None=None, images:list[PromptImage]|None=None):
    # on_prefill(start_pos) after each prompt chunk is queued: progress for a status display
    # images: spans of tokens that are image placeholders (see PromptImage), fed as the images' embeddings
    images = images or []
    self.set_rope_layout(images)
    if chunk_size is None: chunk_size = getenv("PREFILL_CHUNK", 32)
    if self.has_recurrent_block and not gated_delta_kernel_supported(self.output_norm.weight.device):
      chunk_size = getenv("RECURRENT_PREFILL_CHUNK", 1)  # the generic scan is unrolled per token, so big chunks make big graphs
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    host, device = self.host_embd, self.output_norm.weight.device
    if host is None: t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # the model's input for tokens[start_pos:start_pos+n_toks]: token ids, or their packed embedding rows padded to the chunk size
    def inputs(start_pos:int, n_toks:int, sp:UOp, nt:UOp) -> Tensor:
      if host is None: return t[:, sp:sp+nt]
      return host.rows(tokens[start_pos:start_pos+n_toks], chunk_size, device)[:, :nt]
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if self.has_recurrent_block:
      if self._recurrent_resume(tokens)[1]: self._restore_checkpoint()
      # positions from start_pos on get rewritten, so a checkpoint past that point no longer matches the caches
      if start_pos < len(self._checkpoint_tokens): self._checkpoint_tokens = []
    if checkpoint is not None and not (start_pos < checkpoint < len(tokens)): checkpoint = None
    if self.mtp is not None and getenv("SPEC", 1):
      yield from self._generate_spec(tokens, start_pos, chunk_size, temp, checkpoint, on_prefill, v_start_pos, v_toks, images)
      return
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos, *([checkpoint - start_pos] if checkpoint is not None and start_pos < checkpoint else []))
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      if start_pos < prompt_len and self._has_image(images, start_pos, start_pos+n_toks):
        x = self._embed_chunk(tokens, start_pos, n_toks, chunk_size, images)[:, :nt].contiguous()
        if start_pos + n_toks < len(tokens): self.prefill_body_x_jit(x, sp).realize()
        else: out = self.prefill_x_jit(x, sp, temp).realize()
      elif start_pos + n_toks < len(tokens) and n_toks > 1 and getenv("PREFILL_SKIP_HEAD", 1):
        self.prefill_body_jit(inputs(start_pos, n_toks, sp, nt).contiguous(), sp).realize()
      elif start_pos < prompt_len or out is None: out = self(inputs(start_pos, n_toks, sp, nt), sp, temp).realize()
      # decode: feed the sampled token back (on the device, or as its embedding row read from the file)
      else: out = self(out if host is None else host.rows(tokens[start_pos:start_pos+1], 1, device), sp, temp).realize()
      start_pos += n_toks
      if self.has_recurrent_block and start_pos == checkpoint: self._save_checkpoint(tokens[:checkpoint])
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens):
        if on_prefill is not None: on_prefill(start_pos)
        continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  def _spec_input(self, ids:list[int], pad_to:int) -> Tensor:
    if self.host_embd is not None: return self.host_embd.rows(ids, pad_to, self.output_norm.weight.device)
    return Tensor(ids + [0] * (pad_to - len(ids)), dtype="int32", device=self.output_norm.weight.device).reshape(1, pad_to)

  def _generate_spec(self, tokens:list[int], start_pos:int, chunk_size:int, temp:Tensor, checkpoint:int|None,
                     on_prefill:Callable[[int], None]|None, v_start_pos:UOp, v_toks:UOp, images:list[PromptImage]):
    """
    Speculative decoding with the MTP layer as a one-token draft. Each step: the MTP layer drafts d from the last hidden state and
    token x, then the main model runs [x, d] and samples at both positions. The draft is greedy, so taking the target's sample y0
    at x's position and accepting d iff y0 == d is exact sampling. Accepted: emit d and the sample after it. Rejected: emit y0 and
    roll the recurrent state back to after x (the attention KV cache past it is simply overwritten later).
    """
    if not hasattr(self, "spec_body_jit"):
      self.spec_body_jit, self.spec_last_jit = TinyJit(self._spec_prefill_body), TinyJit(self._spec_prefill_last)
      self.verify_jit, self.rollback_jit = TinyJit(self.forward_verify), TinyJit(self._rollback)
      self.spec_body_x_jit, self.spec_last_x_jit = TinyJit(self._spec_prefill_body_x), TinyJit(self._spec_prefill_last_x)
      self.draft_jits = {1: TinyJit(self.forward_draft), 2: TinyJit(self.forward_draft)}
      dim, device = self.output_norm.weight.shape[0], self.output_norm.weight.device
      self.draft_h = {n: Tensor.zeros(1, n, dim, device=device).contiguous().realize() for n in (1, 2)}
    # prompt: chunks also fill the MTP layer's cache with (hidden[i], token[i+1]); the last chunk's final pair has no next token yet,
    # so it gets a padding token, rewritten by the first draft step at the same position
    prompt_len = len(tokens)
    while True:
      n_toks = min(chunk_size, prompt_len - start_pos, *([checkpoint - start_pos] if checkpoint is not None and start_pos < checkpoint else []))
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      last = start_pos + n_toks == prompt_len
      if self._has_image(images, start_pos, start_pos+n_toks+1):  # this chunk or its next tokens (the MTP layer's input) hold image tokens
        n_next = min(n_toks, prompt_len - start_pos - 1)
        cur = self._embed_chunk(tokens, start_pos, n_toks, chunk_size, images)
        nxt = self._embed_chunk(tokens, start_pos+1, n_next, chunk_size, images)
        if last: x, h, _ = self.spec_last_x_jit(cur[:, :nt].contiguous(), nxt[:, :nt].contiguous(), sp, temp)
        else: self.spec_body_x_jit(cur[:, :nt].contiguous(), nxt[:, :nt].contiguous(), sp).realize()
      else:
        cur, nxt = self._spec_input(tokens[start_pos:start_pos+n_toks], chunk_size), self._spec_input(tokens[start_pos+1:start_pos+n_toks+1], chunk_size)
        if last: x, h, _ = self.spec_last_jit(cur[:, :nt].contiguous(), nxt[:, :nt].contiguous(), sp, temp)
        else: self.spec_body_jit(cur[:, :nt].contiguous(), nxt[:, :nt].contiguous(), sp).realize()
      start_pos += n_toks
      if self.has_recurrent_block and start_pos == checkpoint: self._save_checkpoint(tokens[:checkpoint])
      if last: break
      if on_prefill is not None: on_prefill(start_pos)
    tok = int(x.item())
    tokens.append(tok)
    self._cached_tokens = tokens[:-1]
    yield tok
    # decode. p: position of the newest token (sampled, not yet run); pending: (hidden, token) pairs the MTP layer hasn't seen
    p, pending_h, pending, mtp_pos = prompt_len, h, [tok], prompt_len-1
    while p + 2 <= self.max_context:
      # the hidden states go through a fixed buffer: JIT outputs and slices of them reach the draft JIT as different views
      hin = self.draft_h[len(pending)]
      hin.assign(pending_h.reshape(hin.shape)).realize()
      draft = int(self.draft_jits[len(pending)](hin, self._spec_input(pending, len(pending)), v_start_pos.bind(mtp_pos)).item())
      ys, h = self.verify_jit(self._spec_input([tokens[-1], draft], 2), v_start_pos.bind(p), temp)
      y0, y1 = ys.tolist()[0]
      self.spec_stats[0] += 1
      if y0 == draft:
        self.spec_stats[1] += 1
        tokens.append(draft)
        self._cached_tokens = tokens[:]
        yield draft
        tokens.append(y1)
        self._cached_tokens = tokens[:-1]
        yield y1
        pending_h, pending, mtp_pos, p = h, [draft, y1], p, p+2
      else:
        self.rollback_jit().realize()
        tokens.append(y0)
        self._cached_tokens = tokens[:-1]
        yield y0
        pending_h, pending, mtp_pos, p = h[:, :1], [y0], p, p+1

  def _spec_prefill_body(self, tokens:Tensor, next_tokens:Tensor, start_pos:int|UOp) -> Tensor:
    return self._spec_prefill_body_x(self.embed(tokens), self.embed(next_tokens), start_pos)
  def _spec_prefill_body_x(self, x:Tensor, next_x:Tensor, start_pos:int|UOp) -> Tensor:
    h = self.forward_hidden_x(x, start_pos)
    return cast(MTPHead, self.mtp)(h, next_x, start_pos)[:, -1:, :1].contiguous()

  def _spec_prefill_last(self, tokens:Tensor, next_tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return self._spec_prefill_last_x(self.embed(tokens), self.embed(next_tokens), start_pos, temperature)
  def _spec_prefill_last_x(self, x:Tensor, next_x:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    h = self.forward_hidden_x(x, start_pos)
    fill = cast(MTPHead, self.mtp)(h, next_x, start_pos)[:, -1:, :1].contiguous()
    last = h[:, -1:]
    x = self._sample(self.output(last)[:, -1, :], temperature).reshape(1, 1)
    return x.contiguous(), last.contiguous(), fill
