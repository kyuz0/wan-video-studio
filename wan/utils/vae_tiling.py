# wan/utils/vae_tiling.py
import math
import functools
import contextlib
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

# WAN-2.1 VAE strides (time,height,width -> latent)
STRIDE_T = 4
STRIDE_H = 8
STRIDE_W = 8

__all__ = ["tiled_encode", "tiled_decode", "pixel_to_latent_tiles"]

def pixel_to_latent_tiles(tile_px: int) -> int:
    return max(1, int(tile_px) // STRIDE_H)

def _pad_to_stride(x: torch.Tensor, sh: int, sw: int, mode: str = "reflect") -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    ph = (math.ceil(H / sh) * sh) - H
    pw = (math.ceil(W / sw) * sw) - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x

@functools.lru_cache(maxsize=256)
def _feather_mask_cached(h: int, w: int, oy: int, ox: int, device_type: str, dtype_str: str):
    device = torch.device(device_type)
    dtype = getattr(torch, dtype_str)
    def ramp(n, o):
        if o <= 0:
            return torch.ones(n, device=device, dtype=dtype)
        r = torch.arange(n, device=device, dtype=dtype)
        l = torch.clamp((r / (o + 1)), 0, 1)
        r2 = torch.clamp(((n - 1 - r) / (o + 1)), 0, 1)
        edge = torch.minimum(l, r2)
        return 0.5 - 0.5*torch.cos(edge * math.pi)
    wy = ramp(h, oy); wx = ramp(w, ox)
    return wy.view(1,1,h,1) * wx.view(1,1,1,w)

def _build_mask(h, w, oy, ox, ref: torch.Tensor):
    m = _feather_mask_cached(h, w, oy, ox, ref.device.type, str(ref.dtype).split('.')[-1])
    if m.device != ref.device:
        m = m.to(ref.device)
    return m

def _supports_direct_model_api(vae) -> bool:
    return hasattr(vae, "model") and hasattr(vae, "scale") and callable(getattr(vae.model, "encode", None)) and callable(getattr(vae.model, "decode", None))

def _autocast_ctx(vae):
    if vae is not None and hasattr(vae, "device") and torch.device(vae.device).type == "cuda":
        return amp.autocast(device_type="cuda", dtype=getattr(vae, "dtype", torch.float16))
    return contextlib.nullcontext()

@torch.no_grad()
def _encode_many(vae, tiles, use_batch: bool, max_bs: int):
    if not tiles:
        return []
    # If we can't batch, fall back to list API (safe for mixed sizes)
    if not (use_batch and _supports_direct_model_api(vae)):
        return vae.encode(tiles)  # list in -> list out  :contentReference[oaicite:0]{index=0}
    # Bucket by size so we only stack equal shapes
    outs = [None] * len(tiles)
    buckets = {}
    for i, t in enumerate(tiles):
        key = (t.shape[1], t.shape[2], t.shape[3])  # (T,H,W)
        buckets.setdefault(key, []).append(i)
    with _autocast_ctx(vae):
        for key, idxs in buckets.items():
            for s in range(0, len(idxs), max_bs):
                batch_idx = idxs[s:s+max_bs]
                x = torch.stack([tiles[j] for j in batch_idx], 0)  # [B,C,T,H,W]
                z = vae.model.encode(x, vae.scale).float()  # [B,Cz,Tl,Hl,Wl]  :contentReference[oaicite:1]{index=1}
                for k, j in enumerate(batch_idx):
                    outs[j] = z[k]
    return outs

@torch.no_grad()
def _decode_many(vae, zs, use_batch: bool, max_bs: int):
    if not zs:
        return []
    # If we can't batch, use list API (handles mixed latent sizes)
    if not (use_batch and _supports_direct_model_api(vae)):
        return vae.decode(zs)  # list in -> list out  :contentReference[oaicite:2]{index=2}
    # Bucket by latent shape (Tl,Hl,Wl) so stacks are valid
    outs = [None] * len(zs)
    buckets = {}
    for i, z in enumerate(zs):
        key = (z.shape[2], z.shape[3], z.shape[4])
        buckets.setdefault(key, []).append(i)
    with _autocast_ctx(vae):
        for key, idxs in buckets.items():
            for s in range(0, len(idxs), max_bs):
                batch_idx = idxs[s:s+max_bs]
                z = torch.stack([zs[j] for j in batch_idx], 0)  # [B,Cz,Tl,Hl,Wl]
                x = vae.model.decode(z, vae.scale).float().clamp_(-1, 1)  # [B,3,T,H,W]  :contentReference[oaicite:3]{index=3}
                for k, j in enumerate(batch_idx):
                    outs[j] = x[k]
    return outs

@torch.no_grad()
def tiled_encode(
    vae,
    video: torch.Tensor,
    tile_px: int = 256,
    overlap_px: int = 64,
    batch_tiles: int = 8,
    use_batch: bool = True
) -> torch.Tensor:
    # [3,T,H,W] or [1,3,T,H,W] -> latent [Cz,Tl,HL,WL]
    if video.ndim == 5: video = video[0]
    assert video.ndim == 4 and video.shape[0] in (3, 4)
    C, T, H, W = video.shape

    th = tw = int(tile_px)
    oy = ox = int(overlap_px)
    th = max(th, STRIDE_H); tw = max(tw, STRIDE_W)
    oy = (oy // STRIDE_H) * STRIDE_H
    ox = (ox // STRIDE_W) * STRIDE_W

    out = None
    wsum = None

    for y0 in range(0, H, th):
        y_src0 = max(0, y0 - oy)
        h_pix = min(th + (oy if y0>0 else 0) + (oy if y0+th < H else 0), H - y_src0)

        pending_tiles = []
        pending_meta  = []

        for x0 in range(0, W, tw):
            x_src0 = max(0, x0 - ox)
            w_pix = min(tw + (ox if x0>0 else 0) + (ox if x0+tw < W else 0), W - x_src0)

            tile = video[:, :, y_src0:y_src0 + h_pix, x_src0:x_src0 + w_pix].contiguous()
            tile = _pad_to_stride(tile, STRIDE_H, STRIDE_W, mode="reflect")
            pending_tiles.append(tile)
            pending_meta.append((y0, x0, y_src0, x_src0))

            if len(pending_tiles) == batch_tiles or x0 + tw >= W:
                z_list = _encode_many(vae, pending_tiles, use_batch, batch_tiles)

                for z_tile, (y0i,x0i,ys0,xs0) in zip(z_list, pending_meta):
                    h_core = min(th, H - y0i)
                    w_core = min(tw, W - x0i)
                    hL_core = math.ceil(h_core / STRIDE_H)
                    wL_core = math.ceil(w_core / STRIDE_W)
                    topL  = (y0i - ys0) // STRIDE_H
                    leftL = (x0i - xs0) // STRIDE_W
                    z_core = z_tile[:, :, topL:topL + hL_core, leftL:leftL + wL_core]

                    if out is None:
                        Cz, Tl = z_tile.shape[:2]
                        HL_full = math.ceil(H / STRIDE_H)
                        WL_full = math.ceil(W / STRIDE_W)
                        out  = torch.zeros((Cz, Tl, HL_full, WL_full), device=video.device, dtype=z_tile.dtype)
                        wsum = torch.zeros_like(out)

                    yL0 = y0i // STRIDE_H
                    xL0 = x0i // STRIDE_W
                    oyL = max(1, oy // STRIDE_H) if y0i>0 else 0
                    oxL = max(1, ox // STRIDE_W) if x0i>0 else 0
                    m = _build_mask(hL_core, wL_core, oyL if y0i>0 else 0, oxL if x0i>0 else 0, z_core)

                    out[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += z_core * m
                    wsum[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += m

                pending_tiles.clear()
                pending_meta.clear()

    out = out / (wsum.clamp_min(1e-8))
    return out

@torch.no_grad()
def tiled_decode(
    vae,
    latent: torch.Tensor,
    latent_tile: int = 32,    # for tile_px=256, stride=8
    latent_overlap: int = 8,  # for overlap_px=64, stride=8
    batch_tiles: int = 8,
    use_batch: bool = True
) -> torch.Tensor:
    # latent [Cz,Tl,HL,WL] -> video [3,T,H,W]
    Cz, Tl, HL, WL = latent.shape
    step = max(1, int(latent_tile))
    ov  = max(0, int(latent_overlap))

    # Single probe (once) to get decode scaling (T, hp, wp)
    y1 = min(step, HL); x1 = min(step, WL)
    probe = _decode_many(vae, [latent[:, :, :y1, :x1].contiguous()], use_batch=False, max_bs=1)[0]
    C, Tpix, hp, wp = probe.shape
    H = HL * hp // y1
    W = WL * wp // x1

    out  = torch.zeros((C, Tpix, H, W), device=latent.device, dtype=probe.dtype)
    wsum = torch.zeros_like(out)

    pending_chunks = []
    pending_meta   = []

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            yb0 = max(0, y0 - ov); yb1 = min(HL, y0 + step + ov)
            xb0 = max(0, x0 - ov); xb1 = min(WL, x0 + step + ov)

            z = latent[:, :, yb0:yb1, xb0:xb1].contiguous()
            pending_chunks.append(z)
            pending_meta.append((yb0, xb0))

            flush = (len(pending_chunks) == batch_tiles) or (x0 + step >= WL and y0 + step >= HL)
            if flush:
                tiles = _decode_many(vae, pending_chunks, use_batch, batch_tiles)
                for tile, (yb0i, xb0i) in zip(tiles, pending_meta):
                    yp0 = yb0i * (tile.shape[-2] // (tile.shape[-2] // hp))  # equals yb0i * (hp per latent cell)
                    xp0 = xb0i * (tile.shape[-1] // (tile.shape[-1] // wp))
                    yp0 = yb0i * (hp // (hp // hp))  # simplify to yb0i*hp when hp is per-cell height
                    xp0 = xb0i * (wp // (wp // wp))  # simplify to xb0i*wp
                    yp0 = yb0i * hp
                    xp0 = xb0i * wp
                    yp1 = yp0 + tile.shape[-2]
                    xp1 = xp0 + tile.shape[-1]

                    oy = ov * hp
                    ox = ov * wp
                    m = _build_mask(tile.shape[-2], tile.shape[-1], oy, ox, tile)

                    out[:, :, yp0:yp1, xp0:xp1] += tile * m
                    wsum[:, :, yp0:yp1, xp0:xp1] += m

                pending_chunks.clear()
                pending_meta.clear()

    out = out / wsum.clamp_min(1e-8)
    return out
