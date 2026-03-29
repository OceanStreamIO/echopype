"""gpu_cal.py — GPU-accelerated Sv computation for EK80 CW complex data.

Provides a fast path that takes the raw numpy arrays already extracted by
:class:`CalibrateEK80` and runs the heavy element-wise math on GPU via CuPy.

The algorithm is identical to ``CalibrateEK80._cal_complex_samples`` for CW Sv,
but operates on contiguous numpy arrays instead of xarray objects, eliminating
the overhead of xarray coordinate alignment and deferred computation.

Falls back transparently to NumPy when CuPy is unavailable.
"""

from __future__ import annotations

import numpy as np

from ..utils.gpu import has_cuda
from ..utils.log import _init_logger

logger = _init_logger(__name__)


def compute_sv_from_complex_gpu(
    backscatter_r: np.ndarray,
    backscatter_i: np.ndarray,
    tvg_mod_range: np.ndarray,
    sound_speed: np.ndarray,
    absorption: np.ndarray,
    frequency: np.ndarray,
    transmit_power: np.ndarray,
    gain_correction: np.ndarray,
    equivalent_beam_angle: np.ndarray,
    sa_correction: np.ndarray,
    tau_effective: np.ndarray,
    z_et: float,
    z_er: float,
    n_beams: int,
) -> np.ndarray:
    """Compute Sv from EK80 CW complex samples on GPU.

    All inputs are plain NumPy arrays. The channel dimension is handled by the
    caller (loop over channels or pre-broadcast).

    Parameters
    ----------
    backscatter_r, backscatter_i : np.ndarray
        Real and imaginary backscatter, shape ``(n_pings, n_range, n_beams)``.
    tvg_mod_range : np.ndarray
        TVG-corrected range in metres, shape ``(n_pings, n_range)``.
    sound_speed : np.ndarray
        Sound speed, broadcastable to ``(n_pings,)`` or scalar.
    absorption : np.ndarray
        Absorption coefficient, broadcastable to ``(n_pings,)`` or scalar.
    frequency : np.ndarray
        Centre frequency in Hz, broadcastable to ``(n_pings,)`` or scalar.
    transmit_power : np.ndarray
        Transmit power in watts, broadcastable to ``(n_pings,)`` or scalar.
    gain_correction : np.ndarray
        Gain correction in dB, broadcastable to ``(n_pings,)`` or scalar.
    equivalent_beam_angle : np.ndarray
        Equivalent beam angle in dB, broadcastable to ``(n_pings,)`` or scalar.
    sa_correction : np.ndarray
        Sa correction in dB, broadcastable to ``(n_pings,)`` or scalar.
    tau_effective : np.ndarray
        Effective pulse length in seconds, broadcastable to ``(n_pings,)`` or scalar.
    z_et : float
        Transducer impedance (ohm).
    z_er : float
        Transceiver impedance (ohm).
    n_beams : int
        Number of transducer sectors (beams).

    Returns
    -------
    np.ndarray
        Sv in dB, shape ``(n_pings, n_range)``.
    """
    if has_cuda():
        return _compute_sv_cupy(
            backscatter_r, backscatter_i, tvg_mod_range,
            sound_speed, absorption, frequency, transmit_power,
            gain_correction, equivalent_beam_angle, sa_correction,
            tau_effective, z_et, z_er, n_beams,
        )
    return _compute_sv_numpy(
        backscatter_r, backscatter_i, tvg_mod_range,
        sound_speed, absorption, frequency, transmit_power,
        gain_correction, equivalent_beam_angle, sa_correction,
        tau_effective, z_et, z_er, n_beams,
    )


def _prx_from_complex(xp, bs_r, bs_i, z_et, z_er, n_beams):
    """Compute received power from complex backscatter.

    bs_r, bs_i: shape (n_pings, n_range, n_beams)
    Returns: shape (n_pings, n_range)
    """
    # Mean across beams — avoid xp.mean(axis=-1) which triggers a slow
    # generic reduction kernel in CuPy.  Instead, transpose to beam-first
    # layout (contiguous 2D slices) and accumulate with element-wise ops.
    nb = bs_r.shape[-1]
    if nb == 1:
        mean_r = bs_r[..., 0]
        mean_i = bs_i[..., 0]
    else:
        br_T = xp.ascontiguousarray(bs_r.transpose(2, 0, 1))  # (beams, pings, range)
        bi_T = xp.ascontiguousarray(bs_i.transpose(2, 0, 1))
        sum_r = br_T[0].copy()
        sum_i = bi_T[0].copy()
        for b in range(1, nb):
            sum_r += br_T[b]
            sum_i += bi_T[b]
        mean_r = sum_r / nb
        mean_i = sum_i / nb

    # |mean|^2
    abs_sq = mean_r ** 2 + mean_i ** 2

    # Power: n_beams * |mean(bs)|^2 / (2√2)^2 * (|z_er+z_et|/z_er)^2 / z_et
    impedance_fac = (abs(z_er + z_et) / z_er) ** 2 / z_et
    prx = n_beams * abs_sq / 8.0 * impedance_fac

    return prx


def _sv_from_prx(xp, prx, tvg_mod_range, sound_speed, absorption,
                 frequency, transmit_power, gain_correction,
                 equivalent_beam_angle, sa_correction, tau_effective):
    """Compute Sv from received power and calibration parameters.

    All array params are broadcastable to (n_pings, n_range).
    """
    # Ensure tvg_mod_range > 0, replace <=0 with nan
    tvg_r = xp.where(tvg_mod_range > 0, tvg_mod_range, xp.nan)

    prx_safe = xp.where(prx > 0, prx, xp.nan)

    # Reshape scalar/1D params to broadcast with (n_pings, n_range)
    def _col(a):
        """Reshape to (n_pings, 1) if 1-D."""
        a = xp.asarray(a, dtype=xp.float64)
        if a.ndim == 1:
            return a[:, None]
        return a

    spreading_loss = 20.0 * xp.log10(tvg_r)
    absorption_loss = 2.0 * _col(absorption) * tvg_r

    wavelength = _col(sound_speed) / _col(frequency)

    Sv = (
        10.0 * xp.log10(prx_safe)
        + spreading_loss
        + absorption_loss
        - 10.0 * xp.log10(
            wavelength ** 2 * _col(transmit_power) * _col(sound_speed) / (32.0 * xp.pi ** 2)
        )
        - 2.0 * _col(gain_correction)
        - 10.0 * xp.log10(_col(tau_effective))
        - _col(equivalent_beam_angle)
        - 2.0 * _col(sa_correction)
    )

    return Sv


def _compute_sv_cupy(bs_r, bs_i, tvg_mod_range, sound_speed, absorption,
                     frequency, transmit_power, gain_correction,
                     equivalent_beam_angle, sa_correction, tau_effective,
                     z_et, z_er, n_beams):
    import cupy as cp

    bs_r_g = cp.asarray(bs_r)
    bs_i_g = cp.asarray(bs_i)
    tvg_g = cp.asarray(tvg_mod_range)

    prx = _prx_from_complex(cp, bs_r_g, bs_i_g, z_et, z_er, n_beams)

    Sv = _sv_from_prx(
        cp, prx, tvg_g,
        cp.asarray(sound_speed), cp.asarray(absorption),
        cp.asarray(frequency), cp.asarray(transmit_power),
        cp.asarray(gain_correction), cp.asarray(equivalent_beam_angle),
        cp.asarray(sa_correction), cp.asarray(tau_effective),
    )

    return cp.asnumpy(Sv)


def _compute_sv_numpy(bs_r, bs_i, tvg_mod_range, sound_speed, absorption,
                      frequency, transmit_power, gain_correction,
                      equivalent_beam_angle, sa_correction, tau_effective,
                      z_et, z_er, n_beams):
    prx = _prx_from_complex(np, bs_r, bs_i, z_et, z_er, n_beams)
    return _sv_from_prx(
        np, prx, tvg_mod_range,
        sound_speed, absorption,
        frequency, transmit_power,
        gain_correction, equivalent_beam_angle,
        sa_correction, tau_effective,
    )
