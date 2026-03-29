"""gpu_noise_est.py — GPU-accelerated background noise estimation.

Drop-in replacement for the heavy compute paths in
:class:`echopype.clean.noise_est.NoiseEst` using CuPy when available,
with transparent CPU fallback.

The core algorithm is identical to De Robertis & Higginbottom (2007):
1. Remove TVG (transmission loss) from Sv → calibrated power (linear).
2. Coarsen (bin-average) the linear power.
3. Take the minimum across range bins → noise floor per ping bin.
4. Forward-fill to original ping grid and add back TVG → Sv_noise.
5. Threshold on SNR.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..utils.gpu import has_cuda, to_cpu, to_gpu
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def estimate_noise_gpu(
    Sv: np.ndarray,
    echo_range: np.ndarray,
    spreading_loss: np.ndarray,
    absorption_loss: np.ndarray,
    ping_num: int,
    range_sample_num: int,
    noise_max: float | None = None,
) -> np.ndarray:
    """Estimate background noise on GPU (or CPU fallback).

    All inputs and outputs are plain NumPy arrays — the GPU transfer is handled
    internally.  On Jetson unified memory this is zero-copy.

    Parameters
    ----------
    Sv : np.ndarray
        Volume backscattering strength, shape ``(n_pings, n_range)``.
    echo_range : np.ndarray
        Echo range in metres, same shape as *Sv*.
    spreading_loss : np.ndarray
        ``20 * log10(echo_range)``, same shape.
    absorption_loss : np.ndarray
        ``2 * absorption * echo_range``, same shape.
    ping_num, range_sample_num : int
        Bin sizes for coarsening.
    noise_max : float or None
        Upper cap on noise estimate (dB).

    Returns
    -------
    np.ndarray
        Noise estimate (Sv_noise), same shape as input *Sv*.
    """
    if has_cuda():
        return _estimate_noise_cupy(
            Sv, spreading_loss, absorption_loss, ping_num, range_sample_num, noise_max
        )
    return _estimate_noise_numpy(
        Sv, spreading_loss, absorption_loss, ping_num, range_sample_num, noise_max
    )


# ---------------------------------------------------------------------------
# CuPy implementation
# ---------------------------------------------------------------------------


def _estimate_noise_cupy(
    Sv: np.ndarray,
    spreading_loss: np.ndarray,
    absorption_loss: np.ndarray,
    ping_num: int,
    range_sample_num: int,
    noise_max: float | None,
) -> np.ndarray:
    import cupy as cp

    sv = cp.asarray(Sv)
    sl = cp.asarray(spreading_loss)
    al = cp.asarray(absorption_loss)

    # 1. Calibrated power (linear, TVG removed)
    power_cal = cp.power(10.0, (sv - sl - al) / 10.0)

    # 2. Coarsen: bin-average along both axes
    n_ping, n_range = power_cal.shape
    # Trim to integer multiples of bin sizes
    n_ping_trim = (n_ping // ping_num) * ping_num
    n_range_trim = (n_range // range_sample_num) * range_sample_num
    pc = power_cal[:n_ping_trim, :n_range_trim]
    pc = pc.reshape(n_ping_trim // ping_num, ping_num, n_range_trim // range_sample_num, range_sample_num)
    # nanmean along bin axes
    binned = cp.nanmean(pc, axis=(1, 3))
    binned_db = 10.0 * cp.log10(binned)

    # 3. Min across range → noise floor per ping bin
    noise_bin = cp.nanmin(binned_db, axis=1)  # shape: (n_ping_bins,)

    if noise_max is not None:
        noise_bin = cp.minimum(noise_bin, noise_max)

    # 4. Forward-fill to original ping grid
    noise_expanded = cp.repeat(noise_bin, ping_num)
    # Handle remainder pings
    remainder = n_ping - n_ping_trim
    if remainder > 0:
        noise_expanded = cp.concatenate([noise_expanded, cp.full(remainder, noise_bin[-1])])

    # 5. Add back TVG → Sv_noise (broadcast to 2-D)
    Sv_noise = noise_expanded[:, cp.newaxis] + sl + al

    return cp.asnumpy(Sv_noise)


# ---------------------------------------------------------------------------
# NumPy fallback (vectorised, no xarray overhead)
# ---------------------------------------------------------------------------


def _estimate_noise_numpy(
    Sv: np.ndarray,
    spreading_loss: np.ndarray,
    absorption_loss: np.ndarray,
    ping_num: int,
    range_sample_num: int,
    noise_max: float | None,
) -> np.ndarray:
    # 1. Calibrated power (linear, TVG removed)
    power_cal = 10.0 ** ((Sv - spreading_loss - absorption_loss) / 10.0)

    # 2. Coarsen
    n_ping, n_range = power_cal.shape
    n_ping_trim = (n_ping // ping_num) * ping_num
    n_range_trim = (n_range // range_sample_num) * range_sample_num
    pc = power_cal[:n_ping_trim, :n_range_trim]
    pc = pc.reshape(n_ping_trim // ping_num, ping_num, n_range_trim // range_sample_num, range_sample_num)
    binned = np.nanmean(pc, axis=(1, 3))
    binned_db = 10.0 * np.log10(binned)

    # 3. Min across range
    noise_bin = np.nanmin(binned_db, axis=1)

    if noise_max is not None:
        noise_bin = np.minimum(noise_bin, noise_max)

    # 4. Forward-fill
    noise_expanded = np.repeat(noise_bin, ping_num)
    remainder = n_ping - n_ping_trim
    if remainder > 0:
        noise_expanded = np.concatenate([noise_expanded, np.full(remainder, noise_bin[-1])])

    # 5. Add back TVG
    Sv_noise = noise_expanded[:, np.newaxis] + spreading_loss + absorption_loss

    return Sv_noise
