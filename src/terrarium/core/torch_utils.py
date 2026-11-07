"""Device selection and mixed precision that work on CUDA, Apple MPS and CPU."""

from __future__ import annotations

import contextlib

import torch


def pick_device(requested: str | None = None) -> torch.device:
    """``requested`` = cuda | mps | cpu | None (auto: cuda > mps > cpu)."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed-precision dtype, or None for full fp32.

    CUDA: bf16 where supported (Ampere+), else fp16 (Turing, e.g. RTX 2060).
    MPS / CPU: fp32 — half-precision training on MPS is still patchy.
    """
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast(device: torch.device):
    dt = amp_dtype(device)
    if dt is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dt)


def make_scaler(device: torch.device):
    """Gradient scaler, needed only for fp16."""
    return torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"cuda:{p.name} ({p.total_memory / 2**30:.0f} GB, amp={amp_dtype(device)})"
    return f"{device.type} (fp32)"
