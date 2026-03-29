"""gpu_mvbs.py — GPU-accelerated MVBS (Mean Volume Backscattering Strength).

Provides a fast, GPU-accelerated path for index-based MVBS computation.
The algorithm is identical to ``compute_MVBS_index_binning``:
  - Convert Sv (dB) → linear
  - Coarsen (bin-average) along ping_time and range_sample
  - Convert back linear → dB
"""

from __future__ import annotations

import numpy as np

from ..utils.gpu import has_cuda
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def mvbs_index_binning_gpu(
    Sv: np.ndarray,
    ping_num: int = 100,
    range_sample_num: int = 100,
) -> np.ndarray:
    """Compute MVBS via index binning on GPU (or CPU fallback).

    Parameters
    ----------
    Sv : np.ndarray
        Volume backscattering strength in dB, shape ``(n_pings, n_range)``.
    ping_num : int
        Number of pings per bin.
    range_sample_num : int
        Number of range samples per bin.

    Returns
    -------
    np.ndarray
        MVBS in dB, shape ``(n_ping_bins, n_range_bins)``.
    """
    if has_cuda():
        return _mvbs_cupy(Sv, ping_num, range_sample_num)
    return _mvbs_numpy(Sv, ping_num, range_sample_num)


def _mvbs_cupy(Sv: np.ndarray, ping_num: int, range_sample_num: int) -> np.ndarray:
    import cupy as cp

    sv = cp.asarray(Sv)
    n_ping, n_range = sv.shape

    # Trim to integer multiples
    n_p = (n_ping // ping_num) * ping_num
    n_r = (n_range // range_sample_num) * range_sample_num

    # Linear domain
    linear = cp.power(10.0, sv[:n_p, :n_r] / 10.0)

    # Reshape and mean
    linear = linear.reshape(
        n_p // ping_num, ping_num, n_r // range_sample_num, range_sample_num
    )
    mvbs_linear = cp.nanmean(linear, axis=(1, 3))

    # Back to dB
    mvbs_db = 10.0 * cp.log10(mvbs_linear)

    return cp.asnumpy(mvbs_db)


def _mvbs_numpy(Sv: np.ndarray, ping_num: int, range_sample_num: int) -> np.ndarray:
    n_ping, n_range = Sv.shape

    n_p = (n_ping // ping_num) * ping_num
    n_r = (n_range // range_sample_num) * range_sample_num

    linear = 10.0 ** (Sv[:n_p, :n_r] / 10.0)
    linear = linear.reshape(
        n_p // ping_num, ping_num, n_r // range_sample_num, range_sample_num
    )
    mvbs_linear = np.nanmean(linear, axis=(1, 3))
    mvbs_db = 10.0 * np.log10(mvbs_linear)

    return mvbs_db


def nasc_index_binning_gpu(
    Sv: np.ndarray,
    echo_range: np.ndarray,
    dist_bin_edges: np.ndarray,
    range_bin_edges: np.ndarray,
) -> np.ndarray:
    """GPU-accelerated NASC computation via binned integration.

    Computes the Nautical Area Scattering Coefficient following the formula:
      NASC = 4π × 1852² × ∫ sv × dr  (integrated over range bins, averaged over distance bins)

    Parameters
    ----------
    Sv : np.ndarray
        Volume backscattering strength in dB, shape ``(n_pings, n_range)``.
    echo_range : np.ndarray
        Echo range in metres, shape ``(n_pings, n_range)`` or ``(n_range,)``.
    dist_bin_edges : np.ndarray
        Distance bin edges in metres, shape ``(n_dist_bins + 1,)``.
    range_bin_edges : np.ndarray
        Range (depth) bin edges in metres, shape ``(n_range_bins + 1,)``.

    Returns
    -------
    np.ndarray
        NASC values, shape ``(n_dist_bins, n_range_bins)``.
    """
    if has_cuda():
        return _nasc_cupy(Sv, echo_range, dist_bin_edges, range_bin_edges)
    return _nasc_numpy(Sv, echo_range, dist_bin_edges, range_bin_edges)


def _nasc_core(xp, Sv, echo_range, dist_bin_edges, range_bin_edges):
    """Shared NASC binning logic for both NumPy and CuPy."""
    # Convert Sv to linear sv
    sv_linear = xp.power(10.0, Sv / 10.0)

    # Ensure echo_range is 2D
    if echo_range.ndim == 1:
        echo_range = xp.broadcast_to(echo_range[xp.newaxis, :], sv_linear.shape)

    n_dist = len(dist_bin_edges) - 1
    n_range = len(range_bin_edges) - 1
    nasc = xp.full((n_dist, n_range), xp.nan, dtype=xp.float64)

    FOUR_PI_NM2 = 4.0 * xp.pi * 1852.0**2

    for ri in range(n_range):
        r_lo, r_hi = float(range_bin_edges[ri]), float(range_bin_edges[ri + 1])
        dr = r_hi - r_lo

        for di in range(n_dist):
            d_lo, d_hi = float(dist_bin_edges[di]), float(dist_bin_edges[di + 1])
            # Ping indices within distance bin — simplified to uniform spacing
            p_lo = int(d_lo)
            p_hi = min(int(d_hi), sv_linear.shape[0])
            if p_hi <= p_lo:
                continue

            # Range mask
            er_slice = echo_range[p_lo:p_hi, :]
            sv_slice = sv_linear[p_lo:p_hi, :]
            mask = (er_slice >= r_lo) & (er_slice < r_hi)
            vals = xp.where(mask, sv_slice, xp.nan)
            mean_sv = xp.nanmean(vals)
            if not xp.isnan(mean_sv):
                nasc[di, ri] = float(mean_sv) * dr * FOUR_PI_NM2

    return nasc


def _nasc_cupy(Sv, echo_range, dist_bin_edges, range_bin_edges):
    import cupy as cp

    result = _nasc_core(
        cp,
        cp.asarray(Sv),
        cp.asarray(echo_range),
        cp.asarray(dist_bin_edges),
        cp.asarray(range_bin_edges),
    )
    return cp.asnumpy(result)


def _nasc_numpy(Sv, echo_range, dist_bin_edges, range_bin_edges):
    return _nasc_core(np, Sv, echo_range, dist_bin_edges, range_bin_edges)
