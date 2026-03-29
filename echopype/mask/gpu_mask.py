"""gpu_mask.py — GPU-accelerated masking and frequency-differencing operations.

Provides GPU paths for common mask operations:
- Frequency differencing (channel subtraction + thresholding)
- Boolean mask composition (AND/OR reduction)
- SNR thresholding
"""

from __future__ import annotations

import numpy as np

from ..utils.gpu import has_cuda
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def freq_diff_mask_gpu(
    Sv_ch1: np.ndarray,
    Sv_ch2: np.ndarray,
    threshold_low: float,
    threshold_high: float,
) -> np.ndarray:
    """Compute frequency-differencing mask on GPU.

    Parameters
    ----------
    Sv_ch1, Sv_ch2 : np.ndarray
        Sv values for two different frequency channels (same shape).
    threshold_low, threshold_high : float
        dB thresholds; mask is True where ``threshold_low <= diff <= threshold_high``.

    Returns
    -------
    np.ndarray (bool)
        Boolean mask, same shape.
    """
    if has_cuda():
        return _freq_diff_cupy(Sv_ch1, Sv_ch2, threshold_low, threshold_high)
    return _freq_diff_numpy(Sv_ch1, Sv_ch2, threshold_low, threshold_high)


def _freq_diff_cupy(Sv_ch1, Sv_ch2, threshold_low, threshold_high):
    import cupy as cp

    c1 = cp.asarray(Sv_ch1)
    c2 = cp.asarray(Sv_ch2)
    diff = c1 - c2
    mask = (diff >= threshold_low) & (diff <= threshold_high)
    return cp.asnumpy(mask)


def _freq_diff_numpy(Sv_ch1, Sv_ch2, threshold_low, threshold_high):
    diff = Sv_ch1 - Sv_ch2
    return (diff >= threshold_low) & (diff <= threshold_high)


def compose_masks_gpu(*masks: np.ndarray, operation: str = "and") -> np.ndarray:
    """Combine multiple boolean masks with AND or OR on GPU.

    Parameters
    ----------
    *masks : np.ndarray
        Boolean arrays of identical shape.
    operation : str
        ``"and"`` or ``"or"``.

    Returns
    -------
    np.ndarray (bool)
        Combined mask.
    """
    if has_cuda():
        return _compose_cupy(masks, operation)
    return _compose_numpy(masks, operation)


def _compose_cupy(masks, operation):
    import cupy as cp

    gpu_masks = [cp.asarray(m) for m in masks]
    if operation == "and":
        result = gpu_masks[0]
        for m in gpu_masks[1:]:
            result = cp.logical_and(result, m)
    elif operation == "or":
        result = gpu_masks[0]
        for m in gpu_masks[1:]:
            result = cp.logical_or(result, m)
    else:
        raise ValueError(f"Unknown operation: {operation!r}")
    return cp.asnumpy(result)


def _compose_numpy(masks, operation):
    if operation == "and":
        result = masks[0]
        for m in masks[1:]:
            result = np.logical_and(result, m)
    elif operation == "or":
        result = masks[0]
        for m in masks[1:]:
            result = np.logical_or(result, m)
    else:
        raise ValueError(f"Unknown operation: {operation!r}")
    return result


def snr_mask_gpu(
    Sv: np.ndarray,
    Sv_noise: np.ndarray,
    snr_threshold: float = 3.0,
) -> np.ndarray:
    """Create an SNR-based mask on GPU.

    Parameters
    ----------
    Sv : np.ndarray
        Volume backscattering strength (dB).
    Sv_noise : np.ndarray
        Noise estimate (dB), same shape.
    snr_threshold : float
        Minimum acceptable SNR in dB.

    Returns
    -------
    np.ndarray (bool)
        True where SNR >= threshold.
    """
    if has_cuda():
        import cupy as cp

        sv = cp.asarray(Sv)
        sn = cp.asarray(Sv_noise)
        return cp.asnumpy((sv - sn) >= snr_threshold)
    return (Sv - Sv_noise) >= snr_threshold


def noise_corrected_sv_gpu(
    Sv: np.ndarray,
    Sv_noise: np.ndarray,
) -> np.ndarray:
    """Compute noise-corrected Sv on GPU.

    Sv_corrected = 10 * log10(10^(Sv/10) - 10^(Sv_noise/10))
    Where the subtraction is negative, the result is NaN.

    Parameters
    ----------
    Sv : np.ndarray
        Volume backscattering strength (dB).
    Sv_noise : np.ndarray
        Noise estimate (dB), same shape.

    Returns
    -------
    np.ndarray
        Noise-corrected Sv (dB), with NaN where correction fails.
    """
    if has_cuda():
        import cupy as cp

        sv = cp.asarray(Sv)
        sn = cp.asarray(Sv_noise)
        linear_diff = cp.power(10.0, sv / 10.0) - cp.power(10.0, sn / 10.0)
        linear_diff = cp.where(linear_diff > 0, linear_diff, cp.nan)
        result = 10.0 * cp.log10(linear_diff)
        return cp.asnumpy(result)

    linear_diff = 10.0 ** (Sv / 10.0) - 10.0 ** (Sv_noise / 10.0)
    linear_diff = np.where(linear_diff > 0, linear_diff, np.nan)
    return 10.0 * np.log10(linear_diff)
