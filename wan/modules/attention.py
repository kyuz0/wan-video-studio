# wan/utils/vae_tiling.py
import math
import torch
import torch.nn.functional as F

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

@torch.no_grad()
def tiled_encode(
    vae,
    video: torch.Tensor,
    tile_px: int = 128,        # keep your original default
    overlap_px: int = 16       # small, fast overlap to kill seams
) -> torch.Tensor:
    """
    Accepts [3,T,H,W] or [1,3,T,H,W]; returns latent [Cz,Tl,HL,WL].
    Minimal change vs original: small halo (reflect), sum/count stitch in latent.
    """
    if video.ndim == 5:  # [1,3,T,H,W]
        video = video[0]
    assert video.ndim == 4 and video.shape[0] in (3, 4), "expected [3,T,H,W]"

    _, T, H, W = video.shape
    th = tw = int(tile_px)

    # quantize overlap to stride (so cropping aligns cleanly)
    oy = (int(overlap_px) // STRIDE_H) * STRIDE_H
    ox = (int(overlap_px) // STRIDE_W) * STRIDE_W

    out_sum = None
    out_cnt = None

    for y0 in range(0, H, th):
        # source region with halo
        ys0 = max(0, y0 - oy)
        ys1 = min(H, y0 + th + (oy if y0 + th < H else 0))
        for x0 in range(0, W, tw):
            xs0 = max(0, x0 - ox)
            xs1 = min(W, x0 + tw + (ox if x0 + tw < W else 0))

            tile = video[:, :, ys0:ys1, xs0:xs1].contiguous()
            tile = _pad_to_stride(tile, STRIDE_H, STRIDE_W, mode="reflect")

            # encode one tile -> [Cz,Tl,HL_pad,WL_pad]
            z_tile = vae.encode([tile])[0]

            # core (non-halo) size (clip at edges)
            h_core = min(th, H - y0)
            w_core = min(tw, W - x0)
            hL_core = math.ceil(h_core / STRIDE_H)
            wL_core = math.ceil(w_core / STRIDE_W)

            # crop out the halo in latent space
            topL  = (y0 - ys0) // STRIDE_H
            leftL = (x0 - xs0) // STRIDE_W
            z_core = z_tile[:, :, topL:topL + hL_core, leftL:leftL + wL_core]

            # allocate outputs
            if out_sum is None:
                Cz, Tl = z_tile.shape[:2]
                HL_full = math.ceil(H / STRIDE_H)
                WL_full = math.ceil(W / STRIDE_W)
                dev = video.device
                dtype = z_tile.dtype
                out_sum = torch.zeros((Cz, Tl, HL_full, WL_full), device=dev, dtype=dtype)
                out_cnt = torch.zeros((1, 1, HL_full, WL_full), device=dev, dtype=dtype)

            yL0 = y0 // STRIDE_H
            xL0 = x0 // STRIDE_W

            out_sum[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += z_core
            out_cnt[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += 1

    out = out_sum / out_cnt.clamp_min(1.0)
    return out

@torch.no_grad()
def tiled_decode(
    vae,
    latent: torch.Tensor,
    latent_tile: int = 16,     # keep close to original behavior (128px // 8)
    latent_overlap: int = 2    # small latent halo (≈ overlap_px/STRIDE)
) -> torch.Tensor:
    """
    Accepts latent [Cz,Tl,HL,WL]; returns video [3,T,H,W].
    Minimal change vs original: small halo, sum/count stitch in pixel space.
    """
    Cz, Tl, HL, WL = latent.shape
    step = max(1, int(latent_tile))
    ov   = max(0, int(latent_overlap))

    # do one small decode to infer per-latent pixel scale
    y1 = min(step, HL); x1 = min(step, WL)
    probe = vae.decode([latent[:, :, :y1, :x1]])[0]  # [3,T,hp,wp]
    C, T, hp, wp = probe.shape
    per_lat_h = max(1, hp // y1)
    per_lat_w = max(1, wp // x1)
    H = HL * per_lat_h
    W = WL * per_lat_w

    out_sum = torch.zeros((C, T, H, W), device=latent.device, dtype=probe.dtype)
    out_cnt = torch.zeros((1, 1, H, W), device=latent.device, dtype=probe.dtype)

    # place first probe
    out_sum[:, :, :hp, :wp] += probe
    out_cnt[:, :, :hp, :wp] += 1

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            if y0 == 0 and x0 == 0:
                continue
            yb0 = max(0, y0 - ov); yb1 = min(HL, y0 + step + ov)
            xb0 = max(0, x0 - ov); xb1 = min(WL, x0 + step + ov)

            z = latent[:, :, yb0:yb1, xb0:xb1].contiguous()
            tile = vae.decode([z])[0]  # [3,T,th,tw]

            # scale per chunk (edge tiles smaller)
            th, tw = tile.shape[-2], tile.shape[-1]
            per_h = max(1, th // (yb1 - yb0))
            per_w = max(1, tw // (xb1 - xb0))

            yp0 = yb0 * per_h
            xp0 = xb0 * per_w
            yp1 = yp0 + th
            xp1 = xp0 + tw

            out_sum[:, :, yp0:yp1, xp0:xp1] += tile
            out_cnt[:, :, yp0:yp1, xp0:xp1] += 1

    out = out_sum / out_cnt.clamp_min(1.0)
    return out
