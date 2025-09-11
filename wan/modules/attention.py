# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import os, sys, logging, warnings

print(f"[wan.attn] loaded from {__file__}")
logger = logging.getLogger("wan.attn")
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, os.getenv("WAN_LOGLEVEL", "INFO").upper(), logging.INFO))

# env: WAN_ATTENTION_BACKEND in {"sdpa","sdpa_math","fa2","fa3",""}  ("" = auto)
_WAN_ATTN = os.environ.get("WAN_ATTENTION_BACKEND", "").lower()

# --- detect FlashAttention availability ---
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

logger.info(f"[wan.attn] FA2={FLASH_ATTN_2_AVAILABLE} FA3={FLASH_ATTN_3_AVAILABLE} "
            f"WAN_ATTENTION_BACKEND={os.getenv('WAN_ATTENTION_BACKEND','') or '(auto)'}")

__all__ = ["flash_attention", "attention"]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    causal: bool = False,
    window_size=None,
    dtype=None,
    dropout_p: float = 0.0,
    fa_version: int | None = None,
):
    forced = os.environ.get("WAN_ATTENTION_BACKEND", "").lower()
    fa_dtype = dtype or torch.bfloat16  # FA asserts half dtype; default safely

    def _choose_backend(has_fa2: bool, has_fa3: bool) -> str:
        if forced in ("sdpa", "sdpa_math"): return "sdpa"
        if forced == "fa3" and has_fa3:    return "fa3"
        if forced == "fa2" and has_fa2:    return "fa2"
        if fa_version == 3 and has_fa3:    return "fa3"
        if fa_version == 2 and has_fa2:    return "fa2"
        if has_fa3: return "fa3"
        if has_fa2: return "fa2"
        return "sdpa"

    backend = _choose_backend(FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE)
    logger.info(
        f"[wan.attn] backend={backend} "
        f"q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)} "
        f"causal={causal} win={window_size} dtype={q.dtype}"
    )

    if backend.startswith("fa"):
        ver = 3 if (backend == "fa3" and FLASH_ATTN_3_AVAILABLE) else 2
        if fa_version in (2, 3):
            ver = fa_version
        logger.debug(f"[wan.attn.fa] v{ver} launching")
        try:
            return flash_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                causal=causal, window_size=window_size,
                dtype=fa_dtype, version=ver, dropout_p=dropout_p
            )
        except Exception as e:
            logger.error(f"[wan.attn.fa] CRASH v{ver}: {e}")
            raise

    # SDPA path
    logger.debug("[wan.attn.sdpa] launching")
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can impact performance.'
        )
    q_ = q.transpose(1, 2).to(dtype)
    k_ = k.transpose(1, 2).to(dtype)
    v_ = v.transpose(1, 2).to(dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()
