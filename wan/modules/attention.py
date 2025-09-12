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

# -----------------------------------------------------------------------------
# Availability flags
# -----------------------------------------------------------------------------
try:
    import flash_attn_interface  # FA3
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn  # FA2
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = ["flash_attention", "attention"]

# -----------------------------------------------------------------------------
# Flash / SDPA unified entry (varlen-compatible). Env override supported:
#   WAN_ATTENTION_BACKEND in {"sdpa", "sdpa_math", "fa2", "fa3", ""(auto)}
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
    Inputs are [B, L*, H, C]; varlen via q_lens/k_lens (optional).
    """

    # --- ENV OVERRIDE: route to PyTorch SDPA instead of FlashAttention ---
    forced = os.environ.get("WAN_ATTENTION_BACKEND", "").lower()
    if forced in ("sdpa", "sdpa_math"):
        # One-time log
        _log_backend_once("SDPA (math)" if forced == "sdpa_math" else "SDPA")

        out_dtype = q.dtype
        if dtype is not None:
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        # SDPA expects (B, H, L, D)
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        ).transpose(1, 2).contiguous()
        return x.type(out_dtype)
    # ---------------------------------------------------------------------

    # Decide FA version (auto: prefer v3 when available)
    ver = 3 if ((version in (None, 3)) and FLASH_ATTN_3_AVAILABLE) else 2
    _log_backend_once(f"FlashAttention v{ver}")

    # Sanity
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # Params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # Apply FA
    if ver == 3 and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p/window_size unsupported in FA3
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE, "FlashAttention v2 not available"
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

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
    forced = os.environ.get("WAN_ATTENTION_BACKEND", "").lower()
    ver = fa_version
    if ver is None:
        if forced == "fa2":
            ver = 2
        elif forced == "fa3":
            ver = 3
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
        version=ver,
    )
