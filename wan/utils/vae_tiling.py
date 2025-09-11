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

# --- keep the rest of the file as-is above this line ---

@torch.no_grad()
def _decode_many(vae, zs, use_batch: bool, max_bs: int):
    if not zs:
        return []
    # Safe fallback for mixed shapes
    if not (use_batch and _supports_direct_model_api(vae)):
        return vae.decode(zs)

    outs = [None] * len(zs)
    buckets = {}
    for i, z in enumerate(zs):
        # z is [Cz, Tl, Hl, Wl]
        key = (z.shape[-3], z.shape[-2], z.shape[-1])  # (Tl, Hl, Wl)
        buckets.setdefault(key, []).append(i)

    with _autocast_ctx(vae):
        for key, idxs in buckets.items():
            for s in range(0, len(idxs), max_bs):
                batch_idx = idxs[s:s+max_bs]
                z = torch.stack([zs[j] for j in batch_idx], 0)  # [B,Cz,Tl,Hl,Wl]
                x = vae.model.decode(z, vae.scale).float().clamp_(-1, 1)  # [B,3,T,H,W]
                for k, j in enumerate(batch_idx):
                    outs[j] = x[k]
    return outs

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

    # Probe once to infer per-latent pixel scale robustly
    y1 = min(step, HL); x1 = min(step, WL)
    probe = _decode_many(vae, [latent[:, :, :y1, :x1].contiguous()], use_batch=False, max_bs=1)[0]
    C, Tpix, hp, wp = probe.shape
    per_lat_h_probe = max(1, hp // y1)
    per_lat_w_probe = max(1, wp // x1)
    H = HL * per_lat_h_probe
    W = WL * per_lat_w_probe

    out  = torch.zeros((C, Tpix, H, W), device=latent.device, dtype=probe.dtype)
    wsum = torch.zeros_like(out)

    pending_chunks, pending_meta = [], []

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            yb0 = max(0, y0 - ov); yb1 = min(HL, y0 + step + ov)
            xb0 = max(0, x0 - ov); xb1 = min(WL, x0 + step + ov)

            z = latent[:, :, yb0:yb1, xb0:xb1].contiguous()
            pending_chunks.append(z)
            pending_meta.append((yb0, yb1, xb0, xb1))

            flush = (len(pending_chunks) == batch_tiles) or (x0 + step >= WL and y0 + step >= HL)
            if flush:
                tiles = _decode_many(vae, pending_chunks, use_batch, batch_tiles)

                for tile, (yb0i, yb1i, xb0i, xb1i) in zip(tiles, pending_meta):
                    th, tw = tile.shape[-2], tile.shape[-1]
                    # Compute per-chunk pixel scale (handles edge tiles cleanly)
                    per_lat_h = max(1, th // (yb1i - yb0i))
                    per_lat_w = max(1, tw // (xb1i - xb0i))

                    yp0 = yb0i * per_lat_h
                    xp0 = xb0i * per_lat_w
                    yp1 = yp0 + th
                    xp1 = xp0 + tw

                    oy_px = ov * per_lat_h
                    ox_px = ov * per_lat_w
                    m = _build_mask(th, tw, oy_px, ox_px, tile)

                    out[:, :, yp0:yp1, xp0:xp1] += tile * m
                    wsum[:, :, yp0:yp1, xp0:xp1] += m

                pending_chunks.clear()
                pending_meta.clear()

    out = out / wsum.clamp_min(1e-8)
    return out
