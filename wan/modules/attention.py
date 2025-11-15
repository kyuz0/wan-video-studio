# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import sys
import logging
import torch

# -----------------------------------------------------------------------------
# Minimal, production logging: print exactly once which attention backend is used
# -----------------------------------------------------------------------------
logger = logging.getLogger("wan.attn")
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, os.getenv("WAN_LOGLEVEL", "INFO").upper(), logging.INFO))
_LOGGED_BACKEND = False
def _log_backend_once(label: str):
    global _LOGGED_BACKEND
    if not _LOGGED_BACKEND:
        logger.info(f"Using attention backend: {label}")
        _LOGGED_BACKEND = True

__all__ = ["flash_attention", "attention"]

# -----------------------------------------------------------------------------
# Torch SDPA entry
# -----------------------------------------------------------------------------
def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: int | None = None,
):
    """
    Inputs are [B, L*, H, C]; callers pad sequences to equal length upstream, so
    q_lens/k_lens are currently unused and kept only for signature compatibility.
    """

    _ = (q_lens, k_lens, deterministic, version)

    if window_size != (-1, -1):
        raise NotImplementedError("window_size is not supported with torch SDPA")

    _log_backend_once("Torch SDPA")

    out_dtype = q.dtype
    target_dtype = dtype or torch.bfloat16

    q = q.to(target_dtype)
    k = k.to(target_dtype)
    v = v.to(target_dtype)

    if q_scale is not None:
        q = q * q_scale

    x = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    ).transpose(1, 2).contiguous()

    return x.type(out_dtype)

# -----------------------------------------------------------------------------
# Compatibility wrapper. We route everything through flash_attention() so the
# env override is respected even if callers invoke this function instead.
# -----------------------------------------------------------------------------
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    causal: bool = False,
    window_size=None,
    dtype: torch.dtype | None = None,
    dropout_p: float = 0.0,
    fa_version: int | None = None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=None,
        q_scale=None,
        causal=causal,
        window_size=window_size if window_size is not None else (-1, -1),
        deterministic=False,
        dtype=(dtype or torch.bfloat16),
        version=None,
    )
