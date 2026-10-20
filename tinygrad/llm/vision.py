from __future__ import annotations
import base64, hashlib, io, math, pathlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from tinygrad import Tensor, TinyJit, Context, dtypes, getenv
from tinygrad.llm.gguf import gguf_load_partial

def smart_resize(height:int, width:int, factor:int, min_pixels:int, max_pixels:int) -> tuple[int, int]:
  # Qwen2-VL's image processor: the nearest size with both sides a multiple of factor and the pixel count within bounds
  h, w = max(factor, round(height / factor) * factor), max(factor, round(width / factor) * factor)
  if h * w > max_pixels:
    beta = math.sqrt(height * width / max_pixels)
    h, w = max(factor, math.floor(height / beta / factor) * factor), max(factor, math.floor(width / beta / factor) * factor)
  elif h * w < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
  return h, w

def load_image(url:str):
  from PIL import Image
  if url.startswith("data:"): data = base64.b64decode(url.split(",", 1)[1])
  elif url.startswith(("http://", "https://")):
    with urllib.request.urlopen(url, timeout=30) as r: data = r.read()
  else: data = pathlib.Path(url.removeprefix("file://")).read_bytes()
  img = Image.open(io.BytesIO(data))
  img.load()
  return img, data

@dataclass
class Image:
  """An image in the prompt: its merged-token grid (gh x gw tokens) and a key that identifies its content for prefix caching."""
  img: object
  key: str
  gh: int
  gw: int
  @property
  def n_tokens(self) -> int: return self.gh * self.gw

class VisionEncoder:
  """
  Qwen3-VL/Qwen3.5 vision tower from an mmproj GGUF (projector qwen3vl_merger): a ViT over 16x16 patches with learned position
  embeddings (bilinearly resized from a 48x48 grid) and 2D RoPE, then a merger that turns each 2x2 patch block into one
  LLM token. Runs on its own device (VISION_DEVICE, default METAL) so its weights stay out of the LLM GPU's memory.

  All of its device work runs on one long-lived thread: the server's request threads exit after each request, and a thread that
  used Metal crashes on exit when its autorelease pool frees objects tinygrad already released.
  """
  def __init__(self, path:str|pathlib.Path, device:str):
    self.worker = ThreadPoolExecutor(1, thread_name_prefix="vision")
    self.worker.submit(self._load, path, device).result()

  def _load(self, path:str|pathlib.Path, device:str):
    kv, _, tensors = gguf_load_partial(path, lambda kv, name: True)  # nothing copied to the default device: read below
    mm = np.memmap(path, mode="r")
    def load(name:str) -> np.ndarray:
      off, dims, typ = tensors[name]
      assert typ in (0, 1), f"{name}: only F32/F16 mmproj files are supported, got ggml type {typ}"
      dt = np.float32 if typ == 0 else np.float16
      return np.frombuffer(mm, dtype=dt, count=math.prod(dims), offset=off).reshape(*reversed(dims))
    assert kv.get("clip.projector_type") == "qwen3vl_merger", f"unsupported projector {kv.get('clip.projector_type')}"
    self.device = device
    self.dim, self.heads = kv["clip.vision.embedding_length"], kv["clip.vision.attention.head_count"]
    self.patch, self.merge = kv["clip.vision.patch_size"], kv["clip.vision.spatial_merge_size"]
    self.eps = kv["clip.vision.attention.layer_norm_epsilon"]
    self.out_dim = kv["clip.vision.projection_dim"]
    def T(a:np.ndarray, half=False) -> Tensor: return Tensor(np.ascontiguousarray(a.astype(np.float16 if half else np.float32)), device=device).realize()
    # the conv3d's two temporal slices see the same frame (a still image is repeated): sum them into one 2D patch projection
    self.patch_w = T((load("v.patch_embd.weight").astype(np.float32) + load("v.patch_embd.weight.1").astype(np.float32)).reshape(self.dim, -1), half=True)
    self.patch_b = T(load("v.patch_embd.bias"))
    self.pos_embd = load("v.position_embd.weight").astype(np.float32)  # (side*side, dim), interpolated per image on the host
    self.pos_side = int(math.isqrt(self.pos_embd.shape[0]))
    self.blocks = []
    for i in range(kv["clip.vision.block_count"]):
      p = f"v.blk.{i}."
      self.blocks.append({k: T(load(p + k), half=k.endswith("weight") and not k.startswith("ln")) for k in
        ("ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias",
         "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias")})
    self.post_ln_w, self.post_ln_b = T(load("v.post_ln.weight")), T(load("v.post_ln.bias"))
    self.mm0_w, self.mm0_b = T(load("mm.0.weight"), half=True), T(load("mm.0.bias"))
    self.mm2_w, self.mm2_b = T(load("mm.2.weight"), half=True), T(load("mm.2.bias"))
    self.min_pixels, self.max_pixels = getenv("VISION_MIN_PIXELS", 256*256), getenv("VISION_MAX_PIXELS", 1024*1024)
    self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
    # one JIT per padded patch count: images are padded (and the padding masked out of attention) to a multiple of BUCKET patches,
    # so most image sizes share a few JITs; eagerly, scheduling the 27 blocks costs ~10 s per image
    self.bucket = getenv("VISION_BUCKET", 512)
    self.jits: dict[int, TinyJit] = {}

  def prepare(self, url:str) -> Image:
    img, data = load_image(url)
    w, h = img.size
    f = self.patch * self.merge
    rh, rw = smart_resize(h, w, f, self.min_pixels, self.max_pixels)
    return Image(img, hashlib.sha1(data + f"{rh}x{rw}".encode()).hexdigest(), rh // f, rw // f)

  def encode(self, image:Image) -> np.ndarray:
    """(gh*gw, out_dim) float32 embeddings in row-major merged-token order, on the host. The last few are cached by content."""
    if (hit := self.cache.get(image.key)) is not None:
      self.cache.move_to_end(image.key)
      return hit
    out = self.worker.submit(lambda: self._encode(image).numpy()).result()
    self.cache[image.key] = out
    while len(self.cache) > getenv("VISION_CACHE", 8): self.cache.popitem(last=False)
    return out

  def _encode(self, image:Image) -> Tensor:
    from PIL import Image as PILImage
    p, m, D = self.patch, self.merge, self.dim
    gh, gw = image.gh * m, image.gw * m  # in patches
    img = image.img.convert("RGB").resize((gw * p, gh * p), PILImage.BICUBIC)  # type: ignore[attr-defined]
    a = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5  # (H, W, 3), mean = std = 0.5
    # patches grouped so each 2x2 merge block is 4 consecutive rows: (block row, block col, row in block, col in block, C, py, px)
    a = a.reshape(gh // m, m, p, gw // m, m, p, 3).transpose(0, 3, 1, 4, 6, 2, 5).reshape(-1, 3 * p * p)
    # patch (row, col) in the same order
    br, bc, ir, ic = np.meshgrid(np.arange(gh // m), np.arange(gw // m), np.arange(m), np.arange(m), indexing="ij")
    rows, cols = (br * m + ir).reshape(-1), (bc * m + ic).reshape(-1)

    # learned position embeddings, bilinear (align_corners) from the side x side grid to gh x gw
    s = self.pos_side
    def axis(n:int):
      x = np.linspace(0, s - 1, n)
      lo = np.floor(x).astype(np.int64)
      return lo, np.minimum(lo + 1, s - 1), x - lo
    (h0, h1, dh), (w0, w1, dw) = axis(gh), axis(gw)
    hr0, hr1, dhr, wc0, wc1, dwc = h0[rows], h1[rows], dh[rows, None], w0[cols], w1[cols], dw[cols, None]
    E = self.pos_embd
    pos = (E[hr0*s + wc0] * (1-dhr)*(1-dwc) + E[hr0*s + wc1] * (1-dhr)*dwc + E[hr1*s + wc0] * dhr*(1-dwc) + E[hr1*s + wc1] * dhr*dwc)

    # 2D RoPE: the first half of each head's rotary angles from the patch row, the second half from the column
    hd = D // self.heads
    inv = 1.0 / (10000.0 ** (np.arange(0, hd // 2, 2, dtype=np.float64) / (hd // 2)))
    ang = np.concatenate([rows[:, None] * inv, cols[:, None] * inv], axis=1)  # (N, hd/2)

    N = a.shape[0]
    Np = -(-N // self.bucket) * self.bucket
    def dev(v:np.ndarray) -> Tensor: return Tensor(np.pad(v.astype(np.float32), ((0, Np - N), (0, 0))), device=self.device).realize()
    mask = Tensor(np.where(np.arange(Np) < N, 0, -np.inf).astype(np.float32), device=self.device).realize()
    jit = self.jits.setdefault(Np, TinyJit(self._forward))
    # no BEAM search: it would tune the Metal kernels for minutes per new image size, for a tower that already runs in a few seconds
    with Context(JITBEAM=0, BEAM=0): return jit(dev(a), dev(pos), dev(np.cos(ang)), dev(np.sin(ang)), mask)[:N // (m*m)]

  def _forward(self, patches:Tensor, pos:Tensor, cos:Tensor, sin:Tensor, mask:Tensor) -> Tensor:
    m, D, Np = self.merge, self.dim, patches.shape[0]
    x = (self._lin(patches, self.patch_w, self.patch_b) + pos).realize()
    for b in self.blocks: x = self._block(x, b, cos, sin, mask, Np, D // self.heads).realize()
    x = x.layernorm(eps=self.eps) * self.post_ln_w + self.post_ln_b
    x = self._lin(x.reshape(Np // (m*m), m*m*D), self.mm0_w, self.mm0_b)
    x = x * 0.5 * (1.0 + (x / math.sqrt(2)).erf())  # exact GELU
    return self._lin(x, self.mm2_w, self.mm2_b).realize()

  @staticmethod
  def _lin(x:Tensor, w:Tensor, b:Tensor) -> Tensor: return (x.cast(dtypes.half) @ w.T).float() + b

  def _block(self, x:Tensor, b:dict[str, Tensor], cos:Tensor, sin:Tensor, mask:Tensor, N:int, hd:int) -> Tensor:
    h = x.layernorm(eps=self.eps) * b["ln1.weight"] + b["ln1.bias"]
    qkv = self._lin(h, b["attn_qkv.weight"], b["attn_qkv.bias"]).reshape(N, 3, self.heads, hd).permute(1, 2, 0, 3)  # (3, H, N, hd)
    def rope(t:Tensor) -> Tensor:
      t1, t2 = t.chunk(2, dim=-1)
      return (t1 * cos - t2 * sin).cat(t2 * cos + t1 * sin, dim=-1)
    q, k, v = rope(qkv[0]), rope(qkv[1]), qkv[2]
    # a few heads at a time: the full (H, N, N) score matrix is 1 GB for a 4096-patch image
    g = max(1, min(self.heads, (1 << 28) // (N * N * 4)))
    attn = Tensor.cat(*[q[i:i+g].scaled_dot_product_attention(k[i:i+g], v[i:i+g], attn_mask=mask).realize() for i in range(0, self.heads, g)])
    x = x + self._lin(attn.permute(1, 0, 2).reshape(N, self.dim), b["attn_out.weight"], b["attn_out.bias"])
    h = x.layernorm(eps=self.eps) * b["ln2.weight"] + b["ln2.bias"]
    return x + self._lin(self._lin(h, b["ffn_up.weight"], b["ffn_up.bias"]).gelu(), b["ffn_down.weight"], b["ffn_down.bias"])

def warmup(model, vision:VisionEncoder):
  """Capture the LLM's image-chunk JITs (prompt body and last chunk, with and without speculative decoding) before the first request."""
  from PIL import Image as PILImage
  from tinygrad.llm.model import PromptImage
  f = vision.patch * vision.merge
  for i in range(3):
    im = Image(PILImage.new("RGB", (16*f, 16*f), (40*i, 90, 160)), f"warmup{i}", 16, 16)
    emb = vision.encode(im)
    # image tokens in a middle chunk and in the last one
    tokens = [1+i]*8 + [-1-i]*im.n_tokens + [2]*3
    list(zip(range(2), model.generate(tokens, images=[PromptImage(8, im.gh, im.gw, lambda emb=emb: emb)])))
  model.set_rope_layout([])
  model._cached_tokens, model._checkpoint_tokens = [], []
  vision.cache.clear()

