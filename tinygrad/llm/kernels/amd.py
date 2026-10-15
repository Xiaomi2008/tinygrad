from __future__ import annotations
import functools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve

BLOCK_M, BLOCK_N, DECODE_HEAD_TILE, WARP_SIZE = 32, 32, 8, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32), math.log2(math.e)
Q4_K, Q5_K, Q6_K, IQ4_XS, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q4_WORDS, Q5_WORDS, Q6_BYTES, IQ4_WORDS = 12, 13, 14, 23, 256, 32, 36, 44, 210, 34
QUANT_SIZES = {Q4_K: Q4_WORDS*4, Q5_K: Q5_WORDS*4, Q6_K: Q6_BYTES, IQ4_XS: IQ4_WORDS*4}  # bytes per 256-weight block

def kernel_var(x:UOp) -> UOp:
  # a Variable is a 0-d ALU BUFFER in the tensor graph; inside kernels it takes the ALU PARAM form (same name keeps the value binding)
  return x.substitute({v: UOp.variable(v.expr, v.vmin, v.vmax, dtype=v.dtype, multiple_of=v.arg.multiple_of, param=True)
                       for v in x.toposort() if v.is_variable})

def _unbind(v:int|UOp) -> int|UOp: return kernel_var(v.unbind_all()[0]) if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  # the custom kernels are tuned for RDNA3 (gfx11): the WMMA register layouts don't match gfx12 (RDNA4)
  # or CDNA (MFMA-only, wave64), and the dp4a builtins and 32-lane wave ops aren't portable either.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] == 11

def _device_name(device:str|tuple[str, ...]|None) -> str|None:
  if isinstance(device, tuple): device = device[0]
  return None if device is None else device.split(":")[0]

def gated_delta_kernel_supported(device:str|tuple[str, ...]|None) -> bool:
  # the fused scan only needs 32-lane cross-lane adds: RDNA3 (ds_swizzle) or NV (__shfl_xor_sync). NV_GDN_KERNEL=0 uses the unrolled scan
  return amd_custom_kernels_supported(device) or (_device_name(device) == "NV" and bool(getenv("NV_GDN_KERNEL", 1)))

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False, cuda:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    # the swizzle pattern is an xor butterfly across the wave, which is __shfl_xor_sync on a 32-lane CUDA warp
    other = UOp(Ops.CUSTOM, dtypes.float, (val,), arg=f"__shfl_xor_sync(0xffffffffu, {{0}}, {offset})" if cuda else
      f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))")
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], slot:int, value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights (Q4_K/Q5_K/Q6_K/IQ4_XS) ********

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in QUANT_SIZES.items()}
    raw = next((u for u in decoded.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return
    raw_offset = raw.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and raw.buf_uop.dtype == dtypes.uint8
    self.ggml_type = packed_sizes[prod(raw.shape)]
    # store a typed buffer view: a lazy BITCAST is decomposed into byte-combining ALU before custom-kernel
    # scheduling and would copy the entire packed weight on every JIT graph
    packed_dtype = dtypes.uint8 if self.ggml_type == Q6_K else dtypes.uint32
    self.weight = Tensor(UOp.from_buffer(cast(Buffer, raw.buf_uop.buffer)
      .view(raw.max_numel() * raw.dtype.itemsize // packed_dtype.itemsize, packed_dtype, raw_offset)))
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and (amd_custom_kernels_supported(self.weight.device) or
                                           (_device_name(self.weight.device) == "NV" and bool(getenv("NV_Q8_LINEAR", 1))))
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None: self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in ((Q4_K, Q5_K, Q6_K) if _device_name(self.weight.device) == "NV" else (Q4_K, Q5_K, Q6_K, IQ4_XS)) and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    # generic path too: with a symbolic token count the matmul can't reuse a dequantized weight across the chunk's tokens
    # (each token re-reads the weights); a static padded shape lets the optimizer upcast over tokens
    if not isinstance(x.numel(), int) and getenv("LINEAR_PAD_TOKS", 1):
      # contiguous: otherwise the shrink folds back into the matmul and its kernel loops over the symbolic token count again
      return super().__call__(x.pad_to(x.max_shape)).contiguous().shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

def _amd_dp4a(a:UOp, b:UOp, c:UOp, cuda:bool=False) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.int(), b.int(), c), arg="__dp4a({}, {}, {})" if cuda else
             "__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)")

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint32, tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg="__builtin_amdgcn_perm({}, {}, {})")

def _amd_load(ptr:UOp, lanes:int|None=None, cuda:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None: return ptr.load() if cuda else UOp(Ops.CUSTOMI, ptr.dtype, (ptr,), arg="__builtin_nontemporal_load({0})")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0, dtypes.weakint))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes, dtypes.weakint))).load(dtype=ptr.dtype)

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp: return (raw[base + offset//4] >> ((offset&3)*8).cast(dtypes.uint32)) & 255
def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
  minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _iq4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  low = _load_byte(raw, base, 4 + subgroup//2)
  scale = ((low >> (4*(subgroup%2)).cast(dtypes.uint32)) & 15) | ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
  return _half(raw[base] & 0xffff), (scale.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float()

@functools.cache
def iq4_half_lut(device:str) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor([x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])],
                dtype=dtypes.float16, device=device).bitcast(dtypes.uint32).contiguous()

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int, cuda:bool=False) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True, cuda=cuda) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane*4+i].float() for i in range(4))
  word = sum(((v/group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8)
              for i,v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  stores = (q[token, group, lane.valid(lane < 8)].store(word), scale[token, group.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features,
                                                                    cuda=_device_name(x.device) == "NV"))[:2]
  return q, scale

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str, cuda:bool=False, token_block:int=1,
                   output_block:int=1) -> UOp:
  # token_block > 1: each warp computes several tokens, so the packed weights it loads (identical UOps) are shared by all of them
  # output_block > 1: each warp computes several output rows, so the activations it loads (and per-token sums) are shared too
  chunks, tokens = (group_count+31)//32, out.shape[0]
  assert tokens % token_block == 0 and out_features % output_block == 0
  oblocks = out_features//output_block
  token_output_chunk, lane = UOp.range(tokens//token_block*oblocks*chunks, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  tblock, oblock, chunk = token_output_chunk // (oblocks*chunks), (token_output_chunk//chunks) % oblocks, token_output_chunk % chunks
  group = lane+chunk*32
  stores = []
  for j in range(token_block):
    token = tblock*token_block + j
    for r in range(output_block):
      output = oblock*output_block + r
      value = group_dot(token, output, group) if group_count % 32 == 0 else \
        (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
      total = warp_reduce(value, full_wave=True, cuda=cuda)
      stores.append(out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)))
  return UOp.group(*stores).end(token_output_chunk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, ggml_type:int, cuda:bool=False,
                         token_block:int=1, output_block:int=1) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _amd_load(xq[token, group, 0], 8, cuda=cuda)
    if ggml_type in (Q4_K, Q5_K):
      base = (output * in_features//GGML_BLOCK_SIZE + block) * (Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
      qs_base, dot, qsum = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8, UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base+word_idx] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: word |= ((raw[base+4+word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot, qsum = _amd_dp4a(word, xwords[word_idx], dot, cuda), _amd_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum, cuda)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      return (dot.float()*d*scale - qsum.float()*dmin*minimum) * xd[token, group]
    if ggml_type == IQ4_XS:
      base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        packed = _amd_load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _amd_dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xd[token, group] * d * scale
    base = (output*in_features//GGML_BLOCK_SIZE+block)*Q6_BYTES
    dots = [UOp.const(0, dtypes.int32)] * 2
    for word_idx in range(8):
      pos, within = subgroup*32 + word_idx*4, (subgroup*32 + word_idx*4)%128
      low = _amd_load(raw[base + (pos//128)*64 + within%64], 4, cuda=cuda) >> ((within//64)*4).cast(dtypes.uint8)
      high = _amd_load(raw[base + 128 + (pos//128)*32 + within%32], 4, cuda=cuda) >> ((within//32)*2).cast(dtypes.uint8)
      quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8) - 32
      word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      dots[word_idx//4] = _amd_dp4a(word, xwords[word_idx], dots[word_idx//4], cuda)
    scales = [raw[base + 192 + subgroup*2+i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base+208].cast(dtypes.uint16) | (raw[base+209].cast(dtypes.uint16) << 8)
    return (dots[0].float()*scales[0] + dots[1].float()*scales[1]) * xd[token, group] * _half(dbits)
  names = {Q4_K: "linear_q4_k", Q5_K: "linear_q5_k", IQ4_XS: "linear_iq4_xs", Q6_K: "linear_q6"}
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type], cuda, token_block, output_block)

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  lane, wave = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  hw_lane = UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="__builtin_amdgcn_mbcnt_lo(-1, 0)").cast(dtypes.weakint)
  col, half = hw_lane % 16, hw_lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, half):
  def values(acc:UOp) -> tuple[UOp, ...]:
    vals = tuple(acc.after(update)[i].load() for i in range(8))
    swapped = tuple(UOp(Ops.CUSTOM, dtypes.float32, (value,),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {0}), 50688))") for value in vals)
    low = half.eq(0)
    return tuple(low.where(vals[i], swapped[i+4]) if j == 0 else low.where(swapped[i], vals[i+4]) for i in range(4) for j in range(2))
  return [out[token, output].store(value) for output,output_accs in zip(outputs, accs)
          for tile_tokens,acc in zip(tokens, output_accs) for token,value in zip(tile_tokens, values(acc))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name):
  x = x.reshape(out.shape[0], in_features)
  _, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  output_words = in_features // GGML_BLOCK_SIZE * type_words
  accs = tuple(tuple(UOp.placeholder((8,), dtypes.float32, slot=ot*(token_tile//16)+tile, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.REDUCE)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in range(16)))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  return UOp.group(*_wmma_stores(out, outputs, tokens, accs, update, physical_half)).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  token_tile, output_tiles = (64, 1) if out_features <= 1024 and out.shape[0] % 64 == 0 else \
    (64, 2) if out.shape[0] % 64 == 0 else (32 if out.shape[0] % 32 == 0 else 16, 2)
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in range(4))
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, Q4_WORDS if ggml_type == Q4_K else Q5_WORDS,
                            _wmma_layout(out, out_features, token_tile, output_tiles), dequant,
                            f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma")

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    (out_features <= 6144 or out_features == 5120 and in_features > 8192) else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, _, _, _, _ = layout
  local_lut = UOp.placeholder((256,), dtypes.uint32, slot=32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  lut = local_lut.after(UOp.group(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items))).barrier())
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, scale = _iq4_scales(raw, base, subgroup)
    scale = scale * d
    if out_features <= 6144:
      pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                    for word in range(4) for byte in range(4))
      return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
    def nibble(packed:UOp, index:int): return (packed >> (8*index+4*half)) & 15
    lut_pairs = (lut[(nibble(packed, i) | nibble(packed, i+1)<<4).cast(dtypes.weakint)]
                 for packed in (raw[base+2+subgroup*4+i] for i in range(4)) for i in (0, 2))
    return tuple((_half((pair >> (i*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in lut_pairs for i in range(2))
  return _quant_linear_wmma(out, x, out_features, in_features, IQ4_WORDS, layout, dequant, "linear_iq4_xs_f16_wmma")

# ******** NV tensor cores: mma.sync m16n8k16 (fp16 in, fp32 acc), weights dequantized straight into the A fragments ********
# A = 16 output rows x 16 k (row-major), B = 16 k x 8 tokens, C = 16 outputs x 8 tokens. Per lane: g = lane//4, t = lane%4.
# a: (g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1) (g,2t+8) (g,2t+9) (g+8,2t+8) (g+8,2t+9); b: (2t,g) (2t+1,g) (2t+8,g) (2t+9,g)
# c: (g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1)
CUDA_MMA_ARG = ((8, 16, 16), 'CUDA', 32)

@functools.cache
def _quant_linear_cuda_mma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, token_tiles:int,
                                  output_tiles:int, ksplit:int=1) -> UOp:
  # one warp: (16*output_tiles) outputs x (8*token_tiles) tokens. Each step stages one 256-weight block of its rows in shared memory
  # with coalesced loads, then runs the block's 16 k-steps from there. ksplit > 1 splits the blocks across warps; out[..., kchunk]
  # is summed afterwards.
  # x arrives in fragment order: [token tile of 8, k-step of 16, lane, 4 halves], so each lane's B fragment is one 8-byte load
  tokens, rows = out.shape[0], 16*output_tiles
  x = x.reshape(tokens//8, in_features//16, WARP_SIZE, 4)
  tblk, oblk, kchunk = UOp.range(tokens//(8*token_tiles), 0), UOp.range(out_features//rows, 1), UOp.range(ksplit, 4)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  # the hardware lane: mma fragments are defined by it, and the local range may be split across threadIdx.x/y/z in any order
  # (the & 31 is a UOp so index math knows the lane is in [0, 32): an opaque value forces 64-bit index arithmetic with modulo)
  hw_lane = (UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="((int)(threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z))+0*{0})")
             & 31).cast(dtypes.weakint)
  g, t = hw_lane//4, hw_lane%4
  accs = tuple(tuple(UOp.placeholder((4,), dtypes.float32, slot=ot*token_tiles+nt, addrspace=AddrSpace.REG) for nt in range(token_tiles))
               for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in row) for row in accs)
  blocks_per_row = in_features // GGML_BLOCK_SIZE
  kb = UOp.range(blocks_per_row // ksplit, 3, AxisType.REDUCE)
  block = kchunk*(blocks_per_row//ksplit) + kb
  # stage the block: unit = 32-bit words (Q4_K/Q5_K) or bytes (Q6_K, 210 bytes per block)
  unit_dtype, W = (dtypes.uint8, Q6_BYTES) if ggml_type == Q6_K else (dtypes.uint32, Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
  smem = UOp.placeholder((rows*W,), unit_dtype, slot=64, addrspace=AddrSpace.LOCAL)
  li = UOp.range(rows*W//WARP_SIZE, 90)
  idx = li*WARP_SIZE + hw_lane
  staged = smem.after(kb)[idx].store(raw[(oblk*rows + idx//W)*blocks_per_row*W + block*W + idx%W]).end(li)
  sm = smem.after(UOp.barrier(staged))

  def row_dequant(r:UOp, sg:int):
    # f(idx) -> (w[idx], w[idx+1]) as halves, idx even within 32-weight subgroup sg of local row r
    base = r*W
    if ggml_type in (Q4_K, Q5_K):
      d, dmin, scale, minimum = _q5_scales(sm, base, UOp.const(sg, dtypes.weakint))
      mul, sub = d*scale, dmin*minimum
      qs_base = base + (4 if ggml_type == Q4_K else 12) + (sg//2)*8
      def pair(i0:UOp):
        word = sm[qs_base + i0//4] >> ((i0%4)*8 + (sg&1)*4).cast(dtypes.uint32)
        if ggml_type == Q5_K: hi = sm[base + 4 + i0//4] >> ((i0%4)*8 + sg).cast(dtypes.uint32)
        vals = []
        for i in range(2):
          q = (word >> (8*i)) & 15
          if ggml_type == Q5_K: q = q | (((hi >> (8*i)) & 1) << 4)
          vals.append((q.float()*mul - sub).cast(dtypes.float16))
        return vals
      return pair
    d = _half(sm[base+208].cast(dtypes.uint16) | (sm[base+209].cast(dtypes.uint16) << 8))
    def pair(i0:UOp):
      vals = []
      for i in range(2):
        k = sg*32 + i0 + i  # within the 256-weight block
        within = k % 128
        low = sm[base + (k//128)*64 + within%64] >> ((within//64)*4).cast(dtypes.uint8)
        high = sm[base + 128 + (k//128)*32 + within%32] >> ((within//32)*2).cast(dtypes.uint8)
        q = ((low & 15) | ((high & 3) << 4)).cast(dtypes.int32) - 32
        vals.append((q.float()*sm[base + 192 + k//16].bitcast(dtypes.int8).float()*d).cast(dtypes.float16))
      return vals
    return pair

  wmma_accs = [[acc.after(kb) for acc in row] for row in accs]
  for sg in range(8):
    for h in range(2):
      k0 = h*16 + t*2  # within the subgroup
      ks = block*(GGML_BLOCK_SIZE//16) + sg*2 + h
      bfrags = [UOp.stack(*(x[tblk*token_tiles + nt, ks, hw_lane, i] for i in range(4))) for nt in range(token_tiles)]
      for ot in range(output_tiles):
        p0, p1 = row_dequant(ot*16 + g, sg), row_dequant(ot*16 + g + 8, sg)
        afrag = UOp.stack(*p0(k0), *p1(k0), *p0(k0+8), *p1(k0+8))
        for nt in range(token_tiles):
          wmma_accs[ot][nt] = UOp.wmma(afrag, bfrags[nt], wmma_accs[ot][nt], *CUDA_MMA_ARG)
  update = UOp.group(*(acc.store(v) for row,vrow in zip(accs, wmma_accs) for acc,v in zip(row, vrow))).barrier().end(kb)
  stores = []
  for ot in range(output_tiles):
    for nt in range(token_tiles):
      acc = accs[ot][nt].after(update)
      for e in range(4):
        output = oblk*rows + ot*16 + g + 8*(e//2)
        token = tblk*(8*token_tiles) + nt*8 + t*2 + (e%2)
        stores.append(out[token, output, kchunk].store(acc[e].load()))
  names = {Q4_K: "linear_q4_k_mma", Q5_K: "linear_q5_k_mma", Q6_K: "linear_q6_mma"}
  return UOp.group(*stores).end(tblk, oblk, kchunk, lane).sink(arg=KernelInfo(name=names[ggml_type], opts_to_apply=()))

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in (Q4_K, Q5_K, Q6_K, IQ4_XS)
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  def run(fxn:Callable[..., UOp], out:UOp, *srcs:UOp) -> Tensor:
    all_srcs = (out,)+srcs
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
    kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
    result = Tensor(out.after(kernel))
    if len(result.shape) == 3: result = result.sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  cuda = _device_name(x.device) == "NV"
  if not cuda and tokens % 16 == 0 and out_features % 16 == 0 and layer.ggml_type in (Q4_K, Q5_K, IQ4_XS):
    fxn = _iq4_linear_f16_wmma_kernel if layer.ggml_type == IQ4_XS else functools.partial(_q5_linear_f16_wmma_kernel, ggml_type=layer.ggml_type)
    extra = (iq4_half_lut(str(x.device)).uop,) if layer.ggml_type == IQ4_XS else ()
    return run(fxn, out, raw, x.cast(dtypes.float16).contiguous().uop, *extra)
  if cuda and getenv("NV_MMA_LINEAR", 1) and tokens % 8 == 0 and tokens >= 16 and out_features % 16 == 0 and in_features % 256 == 0:
    nt = max(b for b in (1, 2, 4) if tokens % (8*b) == 0 and b <= getenv("NV_MMA_TT", 4))
    ot = max(b for b in (1, 2, 4) if out_features % (16*b) == 0 and b <= getenv("NV_MMA_OT", 1))
    groups, warps = in_features // 32, (tokens // (8*nt)) * (out_features // (16*ot))
    ksplit = max(b for b in (1, 2, 4, 8, 16) if groups % (8*b) == 0 and groups // (8*b) >= 2 and (b == 1 or warps*b <= getenv("NV_MMA_WARPS", 2048)))
    ksplit = getenv("NV_MMA_KSPLIT", ksplit)
    fxn = functools.partial(_quant_linear_cuda_mma_kernel, ggml_type=layer.ggml_type, token_tiles=nt, output_tiles=ot, ksplit=ksplit)
    out = Tensor.empty(tokens, out_features, ksplit, dtype=dtypes.float32, device=x.device).uop
    # fragment order: lane (g, t) of token tile tb, k-step ks holds x[tb*8+g, ks*16 + (2t, 2t+1, 2t+8, 2t+9)]
    xf = x.reshape(tokens, in_features).cast(dtypes.float16).reshape(tokens//8, 8, in_features//16, 2, 4, 2).permute(0, 2, 1, 4, 3, 5)
    return run(fxn, out, raw, xf.contiguous().uop)
  xq, xd = q8_quantize(x, tokens, in_features)
  # tokens x output rows per warp, tuned on an RTX 3060 Ti with 32-token prefill chunks (Q6_K's byte-wise unpacking wants more tokens)
  max_tb, max_ob = getenv("NV_Q8_TOKEN_BLOCK", 16 if layer.ggml_type == Q6_K else 8), getenv("NV_Q8_OUT_BLOCK", 4 if layer.ggml_type == Q6_K else 8)
  token_block = max(b for b in (1, 2, 4, 8, 16, 32) if tokens % b == 0 and b <= max_tb) if cuda else 1
  output_block = max(b for b in (1, 2, 4, 8) if out_features % b == 0 and b <= max_ob) if cuda else 1
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type, cuda=cuda, token_block=token_block, output_block=output_block)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  return run(decode, out, raw, xq.uop, xd.uop)

# ******** flash attention on the KV cache ********

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DV, heads_per_wave = H // H_KV, block_n, D // WARP_SIZE, 2
  head_tile = min(DECODE_HEAD_TILE, G)  # share each KV stream across two GQA heads per wave
  assert G % head_tile == 0 and head_tile % heads_per_wave == 0
  decode_waves, decode_group = head_tile // heads_per_wave, 4
  block_bhkv = UOp.range(B*H_KV*(G//head_tile), 0, AxisType.GLOBAL)
  valid_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  group_count = min(valid_chunks, out.shape[2]) if isinstance(valid_chunks, int) else valid_chunks.minimum(out.shape[2])
  block_n, lane = UOp.range(group_count, 1, AxisType.GLOBAL), UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  wave = UOp.range(decode_waves, 3, axis_type=AxisType.LOCAL)
  head_group, bhkv = block_bhkv % (G//head_tile), block_bhkv // (G//head_tile)
  b, kv_head = bhkv // H_KV, bhkv % H_KV
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))
  acc, row_max, row_sum = _reg((heads_per_wave, DV), 0, 0), _reg((heads_per_wave,), 1, -math.inf), _reg((heads_per_wave,), 2, 0)
  groups_per_chunk, offset = CHUNK // decode_group, UOp.range(((valid_chunks+group_count-1)//group_count)*(CHUNK//decode_group), 100, AxisType.REDUCE)
  chunk = block_n + (offset // groups_per_chunk) * group_count
  keys = tuple(chunk*CHUNK + (offset % groups_per_chunk)*decode_group + i for i in range(decode_group))
  valid = tuple(key < valid_kv_len for key in keys)
  kvals, vvals = (tuple(tuple(is_valid.where(cache_kv[kv, b, kv_head, key, d].float(), UOp.const(0, dtypes.float)) for d in dims)
    for key,is_valid in zip(keys, valid)) for kv in range(2))
  q_heads = tuple(kv_head*G + head_group*head_tile + wave*heads_per_wave + head for head in range(heads_per_wave))
  updates:list[UOp] = []
  for head,q_head in enumerate(q_heads):
    scores = tuple(warp_reduce(sum((q[b, q_head, 0, d].float()*k for d,k in zip(dims, key_kvals)),
                                   UOp.const(0, dtypes.float)), full_wave=True) / math.sqrt(D) for key_kvals in kvals)
    prev_acc, prev_max, prev_sum = acc.after(offset)[head], row_max.after(offset)[head], row_sum.after(offset)[head]
    new_max = functools.reduce(lambda a,vs:a.maximum(vs[0].where(vs[1], UOp.const(-math.inf, dtypes.float))), zip(valid, scores), prev_max)
    alpha = ((prev_max-new_max)*LOG2E).exp2()
    betas = tuple(is_valid.where(((score-new_max)*LOG2E).exp2(), UOp.const(0, dtypes.float)) for is_valid,score in zip(valid, scores))
    updates += [acc[head].store(prev_acc*alpha + sum((UOp.stack(*value)*beta for value,beta in zip(vvals, betas)), acc[head].const_like(0))),
                row_sum[head].store(prev_sum*alpha + sum(betas, UOp.const(0, dtypes.float))), row_max[head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)
  stores = [out[b, q_head, block_n, d].store(acc[head, i]) for head,q_head in enumerate(q_heads) for i,d in enumerate(dims)] + \
    [stats[b, q_head.valid(lane.eq(0)), block_n, i].store(x[head]) for head,q_head in enumerate(q_heads) for i,x in enumerate((row_max, row_sum))]
  return UOp.group(*stores).end(lane, wave, block_n, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int) -> Tensor:
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  chunks = min(64, max_kv_len // 128)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=128)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len+127)//128
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  partial, stats = partial[:, :, :live], stats[:, :, :live]
  weights = ((stats[..., 0]-stats[..., 0].max(2, keepdim=True))*LOG2E).exp2()
  return ((partial*weights.unsqueeze(-1)).sum(2) / (stats[..., 1]*weights).sum(2, keepdim=True)).unsqueeze(2)

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int) and isinstance(valid_kv_len, int): assert M % BLOCK_M == 0 and valid_kv_len % BLOCK_N == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q, o = (x.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m] for x in (q, o))
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.placeholder((BLOCK_M, D + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)
  n_tile = UOp.range((q_base + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N, 100, AxisType.REDUCE)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90)
  kval = k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
  Q_lds, KV_lds_k = Q_lds.after(qk_load_barrier), KV_lds.after(qk_load_barrier)
  S_reg = _reg((TM, TN), 6, 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk)
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
  q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm * LANES_PER_WAVE_M + lane_m
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  S_reg = S_reg.after(S_reg[rm, rn].store((k_idx <= q_idx).where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), 7, -math.inf, n_tile), UOp.range(TN, 261, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], maximum=True)).end(ri_w))
  tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
  S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
  p_local, ri_ws = _reg((TM,), 8, 0, n_tile), UOp.range(TM, 295)
  p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, TM, LANES_PER_WAVE_M, 1, TN, LANES_PER_WAVE_N, 1).permute((1, 0, 3, 6, 2, 4, 5, 7)) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i, ri4, rj4 = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
  m_new = m_i[ri4].maximum(m_ij[ri4])
  alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
  correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                         m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
  acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.placeholder((D, BLOCK_N + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390)
  vval = v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v].float()
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds, V_lds = P_lds.after(pv_barrier), V_lds.after(pv_barrier)
  pv_acc = _reg((TM, TD), 10, 0, n_tile).after(pv_barrier)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).barrier().end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TD, LANES_PER_WAVE_N, 1) \
    .permute((0, 4, 2, 6, 1, 3, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  if resolve(T_real == 1): return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, assigned_kv.shape[3]))
  if isinstance(T_real, UOp):
    # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
    T_pad = q.max_shape[2]
    assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
    q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
  B, H, T, D = q.shape
  out = Tensor.empty(B*H, T, D, dtype="float32", device=q.device)
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start)
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** NV prefill flash attention: mma.sync m16n8k16, split over the KV cache ********
# warp = 16 queries of one head x one of S interleaved KV splits (16-key tiles); scores, online softmax and P@V stay in registers (the C fragment
# of S is the A fragment of P). Splits write unnormalized O with (max, sum); flash_attention_nv combines them.

def _quad_reduce(val:UOp, maximum:bool) -> UOp:
  for offset in (1, 2):
    other = UOp(Ops.CUSTOM, dtypes.float, (val,), arg=f"__shfl_xor_sync(0xffffffffu, {{0}}, {offset})")
    val = val.maximum(other) if maximum else val + other
  return val

def _nv_flash_attention_kernel(o:UOp, ml:UOp, q:UOp, cache:UOp, *scales:UOp, valid_kv_len:UOp, q_start:UOp, tokens:int,
                               kv_fmt:str="fp16") -> UOp:
  # scales: a quantized cache's fp16 scale per (k/v, head, position); K's scales the scores, V's is folded into P.
  # kv_fmt k8vr4: cache rows are [D bytes of int8 K | D/2 bytes of 4-bit V (nibble - 8, low nibble first)]
  # rows are (query head within the KV group, token): row r is token r % tokens, so a KV group's heads share each K/V load
  BH, Tp, S, D = o.shape
  if kv_fmt == "k8vr4":
    B, H_KV, max_ctx, W = cache.shape
    k = v = cache.reshape(B*H_KV, max_ctx, W)
  else:
    _, B, H_KV, max_ctx, _ = cache.shape
    k, v = cache[0].reshape(B*H_KV, max_ctx, D), cache[1].reshape(B*H_KV, max_ctx, D)
  ks, vs = (scales[0][0].reshape(B*H_KV, max_ctx), scales[0][1].reshape(B*H_KV, max_ctx)) if scales else (None, None)
  def kval(src:UOp, h:UOp, key:UOp, d:UOp, is_v:bool=False) -> UOp:
    if kv_fmt == "fp16": return src[h, key, d]
    if kv_fmt == "int8": return src[h, key, d].cast(dtypes.float16)
    if not is_v: return src[h, key, d].bitcast(dtypes.int8).cast(dtypes.float16)
    nib = (src[h, key, D + d//2] >> ((d%2)*4).cast(dtypes.uint8)) & 15
    return (nib.cast(dtypes.float32) - 8).cast(dtypes.float16)
  bh, qt, sp = UOp.range(BH, 0), UOp.range(Tp//16, 1), UOp.range(S, 2)
  lane = UOp.range(WARP_SIZE, 3, axis_type=AxisType.LOCAL)
  hw_lane = (UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="((int)(threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z))+0*{0})")
             & 31).cast(dtypes.weakint)
  g, t = hw_lane//4, hw_lane%4
  kvh = bh
  q0 = qt*16
  scale = LOG2E / math.sqrt(D)
  m, l = _reg((2,), 0, -math.inf), _reg((2,), 1, 0)
  O = [_reg((4,), 2+dn, 0) for dn in range(D//8)]
  # interleaved splits: split sp takes key tiles sp, sp+S, sp+2S, ... (never-negative trip count, balanced work)
  # every split runs the same ceil(tiles/S) iterations (tiles past valid_kv_len are masked and their loads clamped) and writes its
  # result on the last one: nothing runs after the loop (the linearizer can define shared subexpressions inside it)
  n_iter = ((valid_kv_len + 15)//16 + (S - 1)) // S
  kt = UOp.range(n_iter, 10, AxisType.REDUCE)
  kbase = (sp + kt*S)*16
  # S = Q K^T for 16 keys (two n-tiles of 8)
  s_regs = [UOp.placeholder((4,), dtypes.float, slot=100+nb, addrspace=AddrSpace.REG) for nb in range(2)]
  s_acc = [r.after(r.after(kt).store(r.const_like(0))) for r in s_regs]
  qrow = lambda r, dd: q[bh, q0 + r, dd]
  for ds in range(D//16):
    d0 = ds*16 + t*2
    a = UOp.stack(qrow(g, d0), qrow(g, d0+1), qrow(g+8, d0), qrow(g+8, d0+1), qrow(g, d0+8), qrow(g, d0+9), qrow(g+8, d0+8), qrow(g+8, d0+9))
    for nb in range(2):
      key = (kbase + nb*8 + g).minimum(max_ctx-1)
      b = UOp.stack(*(kval(k, kvh, key, d0+o_) for o_ in (0, 1, 8, 9)))
      s_acc[nb] = UOp.wmma(a, b, s_acc[nb], *CUDA_MMA_ARG)
  s_done = [r.after(r.store(v)) for r,v in zip(s_regs, s_acc)]
  # masked, scaled (log2 domain) scores; element e of tile nb: row g+8*(e//2), key kbase+nb*8+2t+e%2
  sc = {}
  for nb in range(2):
    for e in range(4):
      key, qpos = kbase + nb*8 + t*2 + e%2, q_start + (q0 + g + 8*(e//2)) % tokens
      sv = s_done[nb][e].load()*scale
      if ks is not None: sv = sv * ks[kvh, key.minimum(max_ctx-1)].float()
      sc[(nb, e)] = ((key <= qpos) & (key < valid_kv_len)).where(sv, -math.inf)
  m_old = [m.after(kt)[r].load() for r in range(2)]
  l_old = [l.after(kt)[r].load() for r in range(2)]
  m_new, corr, p, l_new = [], [], {}, []
  for r in range(2):
    es = (0, 1) if r == 0 else (2, 3)
    rowmax = _quad_reduce(functools.reduce(lambda x, y: x.maximum(y), [sc[(nb, e)] for nb in range(2) for e in es]), True)
    mn = m_old[r].maximum(rowmax)
    safe = mn.eq(-math.inf).where(0.0, mn)
    corr.append((m_old[r] - safe).exp2())
    for nb in range(2):
      for e in es: p[(nb, e)] = (sc[(nb, e)] - safe).exp2()
    l_new.append(l_old[r]*corr[r] + _quad_reduce(sum(p[(nb, e)] for nb in range(2) for e in es), False))
    m_new.append(mn)
  def pv(nb, e):
    if vs is None: return p[(nb, e)]
    return p[(nb, e)] * vs[kvh, (kbase + nb*8 + t*2 + e%2).minimum(max_ctx-1)].float()
  pa = UOp.stack(*(pv(nb, e).cast(dtypes.float16) for nb in range(2) for e in range(4)))
  stores = [m[r].store(m_new[r]) for r in range(2)] + [l[r].store(l_new[r]) for r in range(2)]
  for dn in range(D//8):
    cur = O[dn].after(kt)
    scaled = cur.after(UOp.group(*(cur[e].store(cur[e].load()*corr[e//2]) for e in range(4))))
    vb = UOp.stack(*(kval(v, kvh, (kbase + t*2 + o_).minimum(max_ctx-1), dn*8+g, is_v=True) for o_ in (0, 1, 8, 9)))
    stores.append(O[dn].store(UOp.wmma(pa, vb, scaled, *CUDA_MMA_ARG)))
  last = kt.eq(n_iter - 1)
  regs_done = UOp.group(*stores)
  outs = []
  for dn in range(D//8):
    acc = O[dn].after(regs_done)
    for e in range(4): outs.append(o[bh, q0 + g + 8*(e//2), sp, (dn*8 + t*2 + e%2).valid(last)].store(acc[e].load()))
  for r in range(2):
    for i, src in enumerate((m, l)):
      outs.append(ml[bh, q0 + g + 8*r, sp, UOp.const(i, dtypes.weakint).valid(t.eq(0) & last)].store(src.after(regs_done)[r].load()))
  outs = [UOp.group(*outs).end(kt)]
  return UOp.group(*outs).end(bh, qt, sp, lane).sink(arg=KernelInfo(name="flash_attention_nv", opts_to_apply=()))

def flash_attention_nv(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp, kv_scale:Tensor|None=None, kv_fmt:str|None=None) -> Tensor:
  # q (B,H,T,D), T symbolic (padded to its max) or static; returns (B,H,T,D) float. The heads of a KV group are packed into the rows.
  T_real, S = q.shape[2], getenv("NV_FA_SPLITS", 32)
  B, H, _, D = q.shape
  if kv_fmt is None: kv_fmt = "fp16" if kv_scale is None else "int8"
  H_KV = cast(int, assigned_kv.shape[1 if kv_fmt == "k8vr4" else 2])
  gqa = H // H_KV
  T_pad = q.max_shape[2] if isinstance(T_real, UOp) else T_real
  while (gqa*T_pad) % 16: T_pad += 1
  padded = isinstance(T_real, UOp) or T_pad != T_real
  if padded: q = q.pad_to((B, H, T_pad, D))
  rows = gqa*T_pad
  o = Tensor.empty(B*H_KV, rows, S, D, dtype="float32", device=q.device)
  ml = Tensor.empty(B*H_KV, rows, S, 2, dtype="float32", device=q.device)
  fxn = functools.partial(_nv_flash_attention_kernel, valid_kv_len=_unbind(valid_end), q_start=_unbind(valid_end - T_real), tokens=T_pad,
                          kv_fmt=kv_fmt)
  o, ml = Tensor.custom_kernel(o, ml, q.half().reshape(B*H_KV, rows, D).contiguous(), assigned_kv,
                               *([kv_scale] if kv_scale is not None else []), fxn=fxn)[:2]
  mx, sm = ml[..., 0], ml[..., 1]
  w = (mx - mx.max(-1, keepdim=True)).exp2()
  out = ((o * w.unsqueeze(-1)).sum(2) / (sm * w).sum(-1, keepdim=True)).reshape(B, H, T_pad, D)
  return out[:, :, :T_real] if padded else out

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                cuda:bool=False) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.placeholder((row_tile*key_dim//32,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    av, bv = alpha[bh, token, row if alpha_dim > 1 else 0].load(), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)), full_wave=True, cuda=cuda)
    state_q = warp_reduce(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)), full_wave=True, cuda=cuda)
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, value_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  cuda = _device_name(q.device) == "NV"
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, cuda=cuda))[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  # the bound start_pos reaches the graph through the state AFTER chain, like the flash kernels' valid_end
  call = _gated_delta_prefill_kernel(*params, kernel_var(start_pos.uop.src[0]), cuda=cuda).call(*contig)
  return Tensor(contig[0].after(call))
