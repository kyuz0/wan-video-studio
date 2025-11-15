# wan/utils/vae_tiling.py
import math
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.amp import autocast as torch_autocast
except ImportError:  # fallback for older torch
    try:
        from torch.cuda.amp import autocast as torch_autocast  # type: ignore
    except ImportError:
        torch_autocast = None  # type: ignore

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

# -------- helpers --------

def _vae_autocast_context(device_type: str, dtype: torch.dtype):
    if torch_autocast is None:
        return nullcontext()
    enabled = device_type in {"cuda", "hip"}
    return torch_autocast(device_type if device_type != "hip" else "cuda", dtype=dtype, enabled=enabled)


def _run_vae_encode(vae, batch: torch.Tensor) -> torch.Tensor:
    model = getattr(vae, "model", None)
    scale = getattr(vae, "scale", None)
    target_dtype = getattr(vae, "dtype", batch.dtype)
    device_type = batch.device.type
    if model is None or scale is None:
        encoded = [vae.encode([sample])[0] for sample in batch]
        return torch.stack(encoded, dim=0)
    ctx = _vae_autocast_context(device_type, target_dtype)
    with ctx:
        encoded = model.encode(batch.to(target_dtype), scale)
    return encoded.float()


def _run_vae_decode(vae, batch: torch.Tensor) -> torch.Tensor:
    model = getattr(vae, "model", None)
    scale = getattr(vae, "scale", None)
    target_dtype = getattr(vae, "dtype", batch.dtype)
    device_type = batch.device.type
    if model is None or scale is None:
        decoded = vae.decode([sample for sample in batch])
        return torch.stack(decoded, dim=0)
    ctx = _vae_autocast_context(device_type, target_dtype)
    with ctx:
        decoded = model.decode(batch.to(target_dtype), scale)
    return decoded.float().clamp_(-1, 1)


# -------- minimal, fast tiling with small overlap --------

@torch.no_grad()
def tiled_encode(
    vae,
    video: torch.Tensor,
    tile_px: int = 128,        # same as your original
    overlap_px: int = 16,      # small halo; quantized to stride
    tile_batch_size: int = 4,
) -> torch.Tensor:
    """
    [3,T,H,W] or [1,3,T,H,W] -> latent [Cz,Tl,HL,WL]
    Minimal change vs original: small reflect-halo + sum/count stitch in latent.
    """
    if video.ndim == 5:  # [1,3,T,H,W]
        video = video[0]
    assert video.ndim == 4 and video.shape[0] in (3, 4), "expected [3,T,H,W]"

    _, T, H, W = video.shape
    th = tw = int(tile_px)

    oy = (int(overlap_px) // STRIDE_H) * STRIDE_H
    ox = (int(overlap_px) // STRIDE_W) * STRIDE_W

    out_sum = None
    out_cnt = None
    batch_limit = max(1, int(tile_batch_size))

    encode_groups: dict[tuple[int, int], list] = defaultdict(list)

    def flush_group(key):
        nonlocal out_sum, out_cnt
        jobs = encode_groups[key]
        if not jobs:
            return
        batch = torch.stack([job.pop("tile") for job in jobs], dim=0)
        z_batch = _run_vae_encode(vae, batch)
        for job, z_tile in zip(jobs, z_batch.unbind(0)):
            z_core = z_tile[:, :, job["topL"]:job["topL"] + job["hL_core"],
                            job["leftL"]:job["leftL"] + job["wL_core"]]
            if out_sum is None:
                Cz, Tl = z_tile.shape[:2]
                HL_full = math.ceil(H / STRIDE_H)
                WL_full = math.ceil(W / STRIDE_W)
                dev = video.device
                dtype = z_tile.dtype
                out_sum = torch.zeros((Cz, Tl, HL_full, WL_full), device=dev, dtype=dtype)
                out_cnt = torch.zeros((1, 1, HL_full, WL_full), device=dev, dtype=dtype)
            yL0 = job["yL0"]
            xL0 = job["xL0"]
            hL = job["hL_core"]
            wL = job["wL_core"]
            out_sum[:, :, yL0:yL0 + hL, xL0:xL0 + wL] += z_core
            out_cnt[:, :, yL0:yL0 + hL, xL0:xL0 + wL] += 1
        jobs.clear()

    for y0 in range(0, H, th):
        ys0 = max(0, y0 - oy)
        ys1 = min(H, y0 + th + (oy if y0 + th < H else 0))
        for x0 in range(0, W, tw):
            xs0 = max(0, x0 - ox)
            xs1 = min(W, x0 + tw + (ox if x0 + tw < W else 0))

            tile = video[:, :, ys0:ys1, xs0:xs1].contiguous()
            tile = _pad_to_stride(tile, STRIDE_H, STRIDE_W, mode="reflect")

            h_core = min(th, H - y0)
            w_core = min(tw, W - x0)
            hL_core = math.ceil(h_core / STRIDE_H)
            wL_core = math.ceil(w_core / STRIDE_W)

            topL  = (y0 - ys0) // STRIDE_H
            leftL = (x0 - xs0) // STRIDE_W
            job = dict(
                tile=tile,
                topL=topL,
                leftL=leftL,
                hL_core=hL_core,
                wL_core=wL_core,
                yL0=y0 // STRIDE_H,
                xL0=x0 // STRIDE_W,
            )
            key = (tile.shape[-2], tile.shape[-1])
            encode_groups[key].append(job)
            if len(encode_groups[key]) >= batch_limit:
                flush_group(key)

    for key in list(encode_groups.keys()):
        flush_group(key)

    out = out_sum / out_cnt.clamp_min(1.0)
    return out

def _linear_mask(h: int, w: int, oy: int, ox: int, device, dtype):
    """
    Very cheap separable linear feather on borders:
    - oy/ox is overlap width in **pixels** to feather.
    """
    if oy <= 0 and ox <= 0:
        return None  # treat as all-ones
    # y
    if oy > 0:
        wy = torch.ones(h, device=device, dtype=dtype)
        ramp = torch.linspace(0.0, 1.0, oy + 1, device=device, dtype=dtype)[1:]  # length oy
        wy[:oy] = ramp                      # top ramp 0->1
        wy[-oy:] = ramp.flip(0)             # bottom ramp 1->0
    else:
        wy = torch.ones(h, device=device, dtype=dtype)
    # x
    if ox > 0:
        wx = torch.ones(w, device=device, dtype=dtype)
        ramp = torch.linspace(0.0, 1.0, ox + 1, device=device, dtype=dtype)[1:]
        wx[:ox] = ramp
        wx[-ox:] = ramp.flip(0)
    else:
        wx = torch.ones(w, device=device, dtype=dtype)
    return wy.view(1,1,h,1) * wx.view(1,1,1,w)

@torch.no_grad()
def tiled_decode(
    vae,
    latent: torch.Tensor,
    latent_tile: int = 16,     # ≈ 128px // 8
    latent_overlap: int = 2,   # ≈ overlap_px // stride
    latent_batch_size: int = 4,
) -> torch.Tensor:
    """
    latent [Cz,Tl,HL,WL] -> video [3,T,H,W]
    Minimal change vs original: small latent halo + **linear feather** in pixel space.
    """
    Cz, Tl, HL, WL = latent.shape
    step = max(1, int(latent_tile))
    ov   = max(0, int(latent_overlap))

    # probe to get per-latent pixel scale
    y1 = min(step, HL); x1 = min(step, WL)
    probe = vae.decode([latent[:, :, :y1, :x1]])[0]  # [3,T,hp,wp]
    C, T, hp, wp = probe.shape
    per_lat_h = max(1, hp // y1)
    per_lat_w = max(1, wp // x1)
    H = HL * per_lat_h
    W = WL * per_lat_w

    out_sum = torch.zeros((C, T, H, W), device=latent.device, dtype=probe.dtype)
    out_wsum = torch.zeros((1, 1, H, W), device=latent.device, dtype=probe.dtype)

    # place first probe (feathering applied the same way for consistency)
    oy_px = ov * per_lat_h
    ox_px = ov * per_lat_w
    m0 = _linear_mask(probe.shape[-2], probe.shape[-1], oy_px, ox_px, probe.device, probe.dtype)
    if m0 is None:
        out_sum[:, :, :hp, :wp] += probe
        out_wsum[:, :, :hp, :wp] += 1
    else:
        out_sum[:, :, :hp, :wp] += probe * m0
        out_wsum[:, :, :hp, :wp] += m0

    decode_groups: dict[tuple[int, int], list] = defaultdict(list)
    latent_batch_limit = max(1, int(latent_batch_size))

    def flush_decode(key):
        jobs = decode_groups[key]
        if not jobs:
            return
        batch = torch.stack([job.pop("latent") for job in jobs], dim=0)
        tiles = _run_vae_decode(vae, batch)
        for job, tile in zip(jobs, tiles.unbind(0)):
            th, tw = tile.shape[-2], tile.shape[-1]
            y_span = max(1, job["yb1"] - job["yb0"])
            x_span = max(1, job["xb1"] - job["xb0"])
            per_h = max(1, th // y_span)
            per_w = max(1, tw // x_span)
            yp0 = job["yb0"] * per_h
            xp0 = job["xb0"] * per_w
            yp1 = yp0 + th
            xp1 = xp0 + tw
            oy_px = ov * per_h
            ox_px = ov * per_w
            m = _linear_mask(th, tw, oy_px, ox_px, tile.device, tile.dtype)
            if m is None:
                out_sum[:, :, yp0:yp1, xp0:xp1] += tile
                out_wsum[:, :, yp0:yp1, xp0:xp1] += 1
            else:
                out_sum[:, :, yp0:yp1, xp0:xp1] += tile * m
                out_wsum[:, :, yp0:yp1, xp0:xp1] += m
        jobs.clear()

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            if y0 == 0 and x0 == 0:
                continue
            yb0 = max(0, y0 - ov); yb1 = min(HL, y0 + step + ov)
            xb0 = max(0, x0 - ov); xb1 = min(WL, x0 + step + ov)

            z = latent[:, :, yb0:yb1, xb0:xb1].contiguous()
            job = dict(
                latent=z,
                yb0=yb0,
                yb1=yb1,
                xb0=xb0,
                xb1=xb1,
            )
            key = (z.shape[-2], z.shape[-1])
            decode_groups[key].append(job)
            if len(decode_groups[key]) >= latent_batch_limit:
                flush_decode(key)

    for key in list(decode_groups.keys()):
        flush_decode(key)

    out = out_sum / out_wsum.clamp_min(1e-8)
    return out
