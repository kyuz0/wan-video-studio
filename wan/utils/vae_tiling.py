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

def _pad_to_stride(x: torch.Tensor, sh: int, sw: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    ph = (math.ceil(H / sh) * sh) - H
    pw = (math.ceil(W / sw) * sw) - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0)
    return x

@torch.no_grad()
def tiled_encode(vae, video: torch.Tensor, tile_px: int = 128) -> torch.Tensor:
    """
    Accepts [3,T,H,W] or [1,3,T,H,W]; returns latent [Cz,Tl,HL,WL].
    """
    if video.ndim == 5:  # [1,3,T,H,W]
        video = video[0]
    assert video.ndim == 4 and video.shape[0] in (3, 4), "expected [3,T,H,W]"

    _, T, H, W = video.shape
    th = tw = int(tile_px)

    out = None
    for y0 in range(0, H, th):
        h_pix = min(th, H - y0)
        for x0 in range(0, W, tw):
            w_pix = min(tw, W - x0)
            tile = video[:, :, y0:y0 + h_pix, x0:x0 + w_pix].contiguous()
            tile = _pad_to_stride(tile, STRIDE_H, STRIDE_W)
            z_tile = vae.encode([tile])[0]  # [Cz,Tl,hL_pad,wL_pad]

            hL = math.ceil(h_pix / STRIDE_H)
            wL = math.ceil(w_pix / STRIDE_W)
            z_tile = z_tile[:, :, :hL, :wL]

            if out is None:
                Cz, Tl = z_tile.shape[:2]
                HL_full = math.ceil(H / STRIDE_H)
                WL_full = math.ceil(W / STRIDE_W)
                out = torch.empty((Cz, Tl, HL_full, WL_full),
                                  device=video.device, dtype=z_tile.dtype)

            yL0 = y0 // STRIDE_H
            xL0 = x0 // STRIDE_W
            out[:, :, yL0:yL0 + hL, xL0:xL0 + wL] = z_tile
    return out

@torch.no_grad()
def tiled_decode(vae, latent: torch.Tensor, latent_tile: int = 16) -> torch.Tensor:
    """
    Accepts latent [Cz,Tl,HL,WL]; returns video [3,T,H,W].
    """
    Cz, Tl, HL, WL = latent.shape
    step = max(1, int(latent_tile))

    y1 = min(step, HL); x1 = min(step, WL)
    probe = vae.decode([latent[:, :, :y1, :x1]])[0]  # [3,T,hp,wp]
    C, T, hp, wp = probe.shape
    H = HL * hp // y1
    W = WL * wp // x1
    out = torch.empty((C, T, H, W), device=latent.device, dtype=probe.dtype)
    out[:, :, :hp, :wp] = probe

    for y0 in range(0, HL, step):
        for x0 in range(0, WL, step):
            if y0 == 0 and x0 == 0:
                continue
            y1 = min(y0 + step, HL); x1 = min(x0 + step, WL)
            z = latent[:, :, y0:y1, x0:x1].contiguous()
            tile = vae.decode([z])[0]  # [3,T,hp,wp]
            yp0 = y0 * (tile.shape[-2] // (y1 - y0))
            xp0 = x0 * (tile.shape[-1] // (x1 - x0))
            out[:, :, yp0:yp0 + tile.shape[-2], xp0:xp0 + tile.shape[-1]] = tile
    return out
