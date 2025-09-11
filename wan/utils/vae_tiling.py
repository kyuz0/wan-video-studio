# wan/utils/vae_tiling.py
import math
import torch
import torch.nn.functional as F

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
        # pad (left, right, top, bottom)
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x

def _feather_mask(h: int, w: int, oy: int, ox: int, iy: int, ix: int, device, dtype):
    """
    Build a separable cosine ramp mask sized [1,1,h,w] where inner core [iy..h-iy, ix..w-ix] ≈ 1
    and ramps down to edges over overlap size oy/ox. Used to blend tiles.
    """
    def ramp(n, o, i):
        # n total, inner margin i, overlap o
        v = torch.ones(n, device=device, dtype=dtype)
        if o > 0:
            # left ramp
            if i > 0: v[:i] = 0.0
            r = torch.arange(o, device=device, dtype=dtype)
            v[i:i+o] = 0.5 - 0.5*torch.cos(math.pi*(r+1)/(o+1))
            # right ramp
            if i > 0: v[-i:] = 0.0
            v[-(o+i):-i if i>0 else None] = v[i:i+o].flip(0)
        else:
            if i > 0:
                v[:] = 0.0
                v[i:-i] = 1.0
        return v

    wy = ramp(h, oy, iy)
    wx = ramp(w, ox, ix)
    mask = wy.view(1,1,h,1) * wx.view(1,1,1,w)
    return mask

@torch.no_grad()
def tiled_encode(vae, video: torch.Tensor, tile_px: int = 128, overlap_px: int = 64) -> torch.Tensor:
    """
    [3,T,H,W] or [1,3,T,H,W] -> latent [Cz,Tl,HL,WL]
    Adds spatial overlap + feathered blending to avoid seams.
    """
    if video.ndim == 5:
        video = video[0]
    assert video.ndim == 4 and video.shape[0] in (3, 4)
    C, T, H, W = video.shape

    th = tw = int(tile_px)
    oy = ox = int(overlap_px)
    # enforce multiples of stride for core placement
    th = max(th, STRIDE_H); tw = max(tw, STRIDE_W)
    oy = (oy // STRIDE_H) * STRIDE_H
    ox = (ox // STRIDE_W) * STRIDE_W

    out = None
    wsum = None

    for y0 in range(0, H, th):
        h_pix = min(th + (oy if y0>0 else 0) + (oy if y0+th < H else 0), H - max(0, y0-oy))
        y_src0 = max(0, y0 - oy)
        for x0 in range(0, W, tw):
            x_src0 = max(0, x0 - ox)
            w_pix = min(tw + (ox if x0>0 else 0) + (ox if x0+tw < W else 0), W - x_src0)

            tile = video[:, :, y_src0:y_src0 + h_pix, x_src0:x_src0 + w_pix].contiguous()
            tile = _pad_to_stride(tile, STRIDE_H, STRIDE_W, mode="reflect")
            z_tile = vae.encode([tile])[0]  # [Cz,Tl,hL_pad,wL_pad]

            # compute valid (center) region in latent units to crop away halo
            h_core = min(th, H - y0)
            w_core = min(tw, W - x0)
            hL_core = math.ceil(h_core / STRIDE_H)
            wL_core = math.ceil(w_core / STRIDE_W)

            # total latent size of the encoded (haloed) tile
            hL_tot = math.ceil(h_pix / STRIDE_H)
            wL_tot = math.ceil(w_pix / STRIDE_W)

            # halo in latent cells
            topL  = (y0 - y_src0) // STRIDE_H
            leftL = (x0 - x_src0) // STRIDE_W
            z_core = z_tile[:, :, topL:topL + hL_core, leftL:leftL + wL_core]

            if out is None:
                Cz, Tl = z_tile.shape[:2]
                HL_full = math.ceil(H / STRIDE_H)
                WL_full = math.ceil(W / STRIDE_W)
                out = torch.zeros((Cz, Tl, HL_full, WL_full),
                                  device=video.device, dtype=z_tile.dtype)
                wsum = torch.zeros_like(out)

            yL0 = y0 // STRIDE_H
            xL0 = x0 // STRIDE_W

            # blend weights over core in latent space (use small latent feathering)
            oyL = max(1, oy // STRIDE_H) if y0>0 else 0
            oxL = max(1, ox // STRIDE_W) if x0>0 else 0
            iyL = 0; ixL = 0  # inner “dead” band not needed in latent here
            m = _feather_mask(hL_core, wL_core, oyL if y0>0 else 0, oxL if x0>0 else 0, iyL, ixL,
                              device=video.device, dtype=z_core.dtype)

            out[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += z_core * m
            wsum[:, :, yL0:yL0 + hL_core, xL0:xL0 + wL_core] += m

    out = out / (wsum.clamp_min(1e-8))
    return out

@torch.no_grad()
def tiled_decode(vae, latent: torch.Tensor, latent_tile: int = 16, latent_overlap: int = 4) -> torch.Tensor:
    """
    latent [Cz,Tl,HL,WL] -> video [3,T,H,W]
    Decode in latent tiles with overlap and cosine blending in **pixel** space.
    """
    Cz, Tl, HL, WL = latent.shape
    step = max(1, int(latent_tile))
    ov  = max(0, int(latent_overlap))

    # probe one tiny decode to get pixel scaling
    y1 = min(step, HL); x1 = min(step, WL)
    probe = vae.decode([latent[:, :, :y1, :x1]])[0]  # [3,T,hp,wp]
    C, T, hp, wp = probe.shape
    scale_h = hp / y1
    scale_w = wp / x1
    H = int(round(HL * scale_h))
    W = int(round(WL * scale_w))

    out  = torch.zeros((C, T, H, W), device=latent.device, dtype=probe.dtype)
    wsum = torch.zeros_like(out)

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            yb0 = max(0, y0 - ov); yb1 = min(HL, y0 + step + ov)
            xb0 = max(0, x0 - ov); xb1 = min(WL, x0 + step + ov)

            z = latent[:, :, yb0:yb1, xb0:xb1].contiguous()
            tile = vae.decode([z])[0]  # [3,T,hp,wp]

            yp0 = int(round(yb0 * scale_h))
            xp0 = int(round(xb0 * scale_w))
            yp1 = yp0 + tile.shape[-2]
            xp1 = xp0 + tile.shape[-1]

            # build feather mask in pixel space (blend on the borders)
            oy = int(round(ov * scale_h))
            ox = int(round(ov * scale_w))
            # valid core relative to the block (exclude the halo from weight 1.0)
            iy = 0; ix = 0
            m = _feather_mask(tile.shape[-2], tile.shape[-1], oy, ox, iy, ix, device=tile.device, dtype=tile.dtype)

            out[:, :, yp0:yp1, xp0:xp1] += tile * m
            wsum[:, :, yp0:yp1, xp0:xp1] += m

    out = out / wsum.clamp_min(1e-8)
    return out
