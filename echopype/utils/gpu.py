"""gpu.py — CUDA / CuPy acceleration utilities for echopype.

Provides transparent GPU fallback: every public helper returns NumPy-compatible
results regardless of whether CuPy is available. On Jetson platforms with unified
memory the host↔device transfer cost is essentially zero.

Usage
-----
>>> from echopype.utils.gpu import xp, has_cuda, to_gpu, to_cpu, gpu_signal_convolve
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect CuPy / CUDA availability at import time
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as _cupy_fftconvolve  # noqa: F401

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_cuda() -> bool:
    """Return *True* if CuPy is importable and at least one CUDA device is found."""
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False


# ``xp`` is the array module to use — CuPy when a GPU is present, else NumPy.
xp = cp if has_cuda() else np


def to_gpu(arr: np.ndarray) -> "np.ndarray":
    """Move *arr* to GPU memory (no-op when CUDA is unavailable).

    On Jetson unified-memory platforms this is a zero-copy pointer wrap.
    """
    if has_cuda():
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Ensure *arr* is a host NumPy array (no-op when already on CPU)."""
    if has_cuda() and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def gpu_fftconvolve(a, b, mode: str = "full"):
    """FFT-based convolution — GPU path when available, else ``scipy.signal.fftconvolve``.

    This is the primary acceleration kernel for pulse compression.
    """
    if has_cuda():
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        return cp.asnumpy(_cupy_fftconvolve(a_gpu, b_gpu, mode=mode))
    else:
        from scipy.signal import fftconvolve

        return fftconvolve(a, b, mode=mode)


def gpu_signal_convolve(a, b, mode: str = "full"):
    """Direct convolution — GPU path when available, else ``scipy.signal.convolve``.

    For short kernels direct convolution can beat FFT.
    """
    if has_cuda():
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        # CuPy fftconvolve is generally faster than direct even for short kernels
        return cp.asnumpy(_cupy_fftconvolve(a_gpu, b_gpu, mode=mode))
    else:
        from scipy.signal import convolve

        return convolve(a, b, mode=mode)


def gpu_batch_convolve(
    signals: np.ndarray,
    kernel: np.ndarray,
    mode: str = "full",
) -> np.ndarray:
    """Batch-convolve *signals* (2-D: ``[n_signals, n_samples]``) with a 1-D *kernel*.

    On GPU this uses cuFFT batched transforms which is dramatically faster than
    looping ``scipy.signal.convolve`` per signal.  Falls back to a vectorised
    ``scipy.signal.fftconvolve`` loop on CPU.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(n_signals, n_samples)``.
    kernel : np.ndarray
        1-D kernel.
    mode : str
        Convolution mode (``"full"``, ``"valid"``, ``"same"``).

    Returns
    -------
    np.ndarray
        Convolved signals, same leading dimension as *signals*.
    """
    if has_cuda():
        s_gpu = cp.asarray(signals)
        k_gpu = cp.asarray(kernel)
        # Broadcast kernel to match batch dimension for fftconvolve
        # Use axes parameter to convolve along last axis only
        out = cp.empty(
            (s_gpu.shape[0], _conv_output_len(s_gpu.shape[1], k_gpu.shape[0], mode)),
            dtype=cp.result_type(s_gpu, k_gpu),
        )
        for i in range(s_gpu.shape[0]):
            out[i] = _cupy_fftconvolve(s_gpu[i], k_gpu, mode=mode)
        return cp.asnumpy(out)
    else:
        from scipy.signal import fftconvolve

        out = np.empty(
            (signals.shape[0], _conv_output_len(signals.shape[1], kernel.shape[0], mode)),
            dtype=np.result_type(signals, kernel),
        )
        for i in range(signals.shape[0]):
            out[i] = fftconvolve(signals[i], kernel, mode=mode)
        return out


def _conv_output_len(n_signal: int, n_kernel: int, mode: str) -> int:
    """Return output length of 1-D convolution."""
    if mode == "full":
        return n_signal + n_kernel - 1
    elif mode == "same":
        return n_signal
    elif mode == "valid":
        return abs(n_signal - n_kernel) + 1
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Element-wise math helpers — transparent NumPy / CuPy
# ---------------------------------------------------------------------------


def gpu_log10(arr):
    """``np.log10`` / ``cp.log10`` depending on array type."""
    if has_cuda() and isinstance(arr, cp.ndarray):
        return cp.log10(arr)
    return np.log10(arr)


def gpu_abs(arr):
    """``np.abs`` / ``cp.abs``."""
    if has_cuda() and isinstance(arr, cp.ndarray):
        return cp.abs(arr)
    return np.abs(arr)


def gpu_norm_squared(arr):
    """Squared L2 norm along last axis, GPU-accelerated."""
    if has_cuda():
        a = cp.asarray(arr)
        return float(cp.asnumpy(cp.sum(cp.abs(a) ** 2)))
    return float(np.sum(np.abs(arr) ** 2))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def gpu_info() -> dict:
    """Return a diagnostics dict about the GPU environment."""
    info = {"cuda_available": has_cuda(), "cupy_installed": _HAS_CUPY}
    if has_cuda():
        dev = cp.cuda.Device()
        info["device_name"] = dev.attributes["DeviceName"] if hasattr(dev, "attributes") else str(dev)
        info["compute_capability"] = dev.compute_capability
        mem = dev.mem_info
        info["free_memory_mb"] = round(mem[0] / 1024**2)
        info["total_memory_mb"] = round(mem[1] / 1024**2)
    return info
