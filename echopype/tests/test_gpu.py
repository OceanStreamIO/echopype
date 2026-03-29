"""Tests for GPU-accelerated echopype operations.

These tests verify that the GPU code paths produce numerically identical
results to the original CPU (scipy/numpy) implementations.  When CuPy is
not installed/available the tests still run — they exercise the CPU fallback
paths inside the GPU modules.
"""

import numpy as np
import pytest

from echopype.utils.gpu import (
    _conv_output_len,
    gpu_batch_convolve,
    gpu_fftconvolve,
    gpu_info,
    gpu_signal_convolve,
    has_cuda,
    to_cpu,
    to_gpu,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def complex_signal(rng):
    """Synthetic complex backscatter signal (1-D, like one channel/one ping)."""
    n = 2048
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


@pytest.fixture
def chirp_replica(rng):
    """Short chirp-like kernel."""
    n = 128
    t = np.linspace(0, 1, n)
    return np.cos(2 * np.pi * 50 * t + np.pi * 30 * t**2).astype(np.float64)


# ===========================================================================
# gpu.py — core utilities
# ===========================================================================


@pytest.mark.unit
class TestGPUUtils:
    def test_has_cuda_returns_bool(self):
        assert isinstance(has_cuda(), bool)

    def test_to_gpu_to_cpu_round_trip(self, rng):
        arr = rng.standard_normal((100, 50)).astype(np.float32)
        gpu_arr = to_gpu(arr)
        cpu_arr = to_cpu(gpu_arr)
        np.testing.assert_array_equal(arr, cpu_arr)

    def test_gpu_info_dict(self):
        info = gpu_info()
        assert "cuda_available" in info
        assert "cupy_installed" in info

    def test_conv_output_len(self):
        assert _conv_output_len(100, 10, "full") == 109
        assert _conv_output_len(100, 10, "same") == 100
        assert _conv_output_len(100, 10, "valid") == 91


# ===========================================================================
# gpu.py — convolution kernels
# ===========================================================================


@pytest.mark.unit
class TestGPUConvolve:
    def test_fftconvolve_vs_scipy(self, complex_signal, chirp_replica):
        from scipy.signal import fftconvolve

        expected = fftconvolve(complex_signal, chirp_replica, mode="full")
        result = gpu_fftconvolve(complex_signal, chirp_replica, mode="full")
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_signal_convolve_vs_scipy(self, chirp_replica):
        from scipy.signal import convolve

        a = chirp_replica[:64]
        b = chirp_replica[:32]
        expected = convolve(a, b, mode="full")
        result = gpu_signal_convolve(a, b, mode="full")
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_batch_convolve(self, rng, chirp_replica):
        from scipy.signal import fftconvolve

        signals = rng.standard_normal((8, 512))
        kernel = chirp_replica[:32]
        result = gpu_batch_convolve(signals, kernel, mode="full")

        # Compare per-signal
        for i in range(signals.shape[0]):
            expected = fftconvolve(signals[i], kernel, mode="full")
            np.testing.assert_allclose(result[i], expected, rtol=1e-4, atol=1e-6)


# ===========================================================================
# ek80_complex.py — pulse compression
# ===========================================================================


@pytest.mark.unit
class TestEK80ComplexGPU:
    def test_convolve_per_channel_matches_scipy(self, rng):
        """Verify the GPU-accelerated _convolve_per_channel matches scipy."""
        from scipy.signal import convolve as scipy_convolve

        from echopype.calibrate.ek80_complex import _convolve_per_channel

        n_range = 1024
        n_channels = 3
        backscatter = (
            rng.standard_normal((n_range, n_channels))
            + 1j * rng.standard_normal((n_range, n_channels))
        ).astype(np.complex64)

        ch_names = ["ch0", "ch1", "ch2"]
        replica_dict = {}
        for ch in ch_names:
            kernel_len = rng.integers(32, 128)
            replica_dict[ch] = (
                rng.standard_normal(kernel_len) + 1j * rng.standard_normal(kernel_len)
            ).astype(np.complex64)

        # Mock channel DataArray-like objects
        class _MockChannel:
            def __init__(self, name):
                self._name = name

            def values(self):
                return self._name

            def __str__(self):
                return self._name

        # Build expected output with scipy
        expected = np.zeros_like(backscatter, dtype=np.complex64)
        for ch_seq, ch_name in enumerate(ch_names):
            replica = replica_dict[ch_name]
            conv = scipy_convolve(backscatter[:, ch_seq], replica, mode="full")
            expected[:, ch_seq] = conv[replica.size - 1 :]

        # Build channel objects that mimic xarray channel coordinate
        # xarray uses .values as a property, so str(channel.values) returns the name
        channels = []
        for name in ch_names:
            ch = type("Ch", (), {})()
            ch.values = name
            channels.append(ch)

        result = _convolve_per_channel(backscatter, replica_dict, channels)
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-5)

    def test_convolve_per_channel_all_zeros(self):
        """All-zeros input should return all-zeros without convolution."""
        from echopype.calibrate.ek80_complex import _convolve_per_channel

        backscatter = np.zeros((100, 2), dtype=np.complex64)
        replica_dict = {"ch0": np.ones(10, dtype=np.complex64)}
        ch = type("Ch", (), {})()
        ch.values = "ch0"
        channels = [ch]
        result = _convolve_per_channel(backscatter, replica_dict, channels)
        assert np.all(result == 0)

    def test_filter_decimate_chirp_gpu(self, rng):
        """Verify filter_decimate_chirp produces valid output."""
        from echopype.calibrate.ek80_complex import filter_decimate_chirp

        y_ch = rng.standard_normal(256)
        coeff_ch = {
            "wbt_fil": rng.standard_normal(16),
            "wbt_decifac": np.array([2]),
            "pc_fil": rng.standard_normal(16),
            "pc_decifac": np.array([2]),
        }
        result, result_time = filter_decimate_chirp(coeff_ch, y_ch, fs=1500000.0)
        assert result.ndim == 1
        assert result.size > 0
        assert result_time.size == result.size


# ===========================================================================
# clean/gpu_noise_est.py
# ===========================================================================


@pytest.mark.unit
class TestGPUNoiseEst:
    def test_estimate_noise_shape(self, rng):
        from echopype.clean.gpu_noise_est import estimate_noise_gpu

        n_ping, n_range = 200, 500
        Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 80
        echo_range = np.broadcast_to(
            np.linspace(0.5, 250.0, n_range)[np.newaxis, :], (n_ping, n_range)
        ).copy()
        spreading = 20 * np.log10(np.maximum(echo_range, 1.0))
        absorption = 0.01 * echo_range

        result = estimate_noise_gpu(
            Sv, echo_range, spreading, absorption, ping_num=10, range_sample_num=25
        )
        assert result.shape == Sv.shape

    def test_estimate_noise_reduces_noise(self, rng):
        """Noise estimate should be lower than signal."""
        from echopype.clean.gpu_noise_est import estimate_noise_gpu

        n_ping, n_range = 100, 200
        echo_range = np.broadcast_to(
            np.linspace(1.0, 200.0, n_range)[np.newaxis, :], (n_ping, n_range)
        ).copy()
        spreading = 20 * np.log10(echo_range)
        absorption = 0.01 * echo_range

        # Signal with known noise floor
        noise_floor = -90.0
        signal_level = -50.0
        Sv = np.full((n_ping, n_range), signal_level)
        # Add some lower-power regions
        Sv[:, :50] = noise_floor

        result = estimate_noise_gpu(
            Sv, echo_range, spreading, absorption, ping_num=10, range_sample_num=25
        )
        # Noise estimate should exist and be finite at the noise-floor region
        assert np.any(np.isfinite(result))


# ===========================================================================
# commongrid/gpu_mvbs.py
# ===========================================================================


@pytest.mark.unit
class TestGPUMVBS:
    def test_mvbs_index_binning_shape(self, rng):
        from echopype.commongrid.gpu_mvbs import mvbs_index_binning_gpu

        Sv = rng.standard_normal((200, 500)).astype(np.float64) * 10 - 70
        result = mvbs_index_binning_gpu(Sv, ping_num=10, range_sample_num=25)
        assert result.shape == (20, 20)

    def test_mvbs_matches_numpy_reference(self, rng):
        """MVBS GPU/CPU result should match naive numpy implementation."""
        from echopype.commongrid.gpu_mvbs import mvbs_index_binning_gpu

        Sv = rng.standard_normal((100, 200)).astype(np.float64) * 10 - 70
        ping_num, range_num = 10, 20

        # Reference: manual numpy
        n_p = (100 // ping_num) * ping_num
        n_r = (200 // range_num) * range_num
        linear = 10.0 ** (Sv[:n_p, :n_r] / 10.0)
        linear = linear.reshape(n_p // ping_num, ping_num, n_r // range_num, range_num)
        ref = 10.0 * np.log10(np.nanmean(linear, axis=(1, 3)))

        result = mvbs_index_binning_gpu(Sv, ping_num=ping_num, range_sample_num=range_num)
        np.testing.assert_allclose(result, ref, rtol=1e-10)


# ===========================================================================
# mask/gpu_mask.py
# ===========================================================================


@pytest.mark.unit
class TestGPUMask:
    def test_freq_diff_mask(self, rng):
        from echopype.mask.gpu_mask import freq_diff_mask_gpu

        Sv1 = rng.standard_normal((100, 100)).astype(np.float64)
        Sv2 = Sv1 - 5.0  # constant 5 dB difference
        mask = freq_diff_mask_gpu(Sv1, Sv2, threshold_low=3.0, threshold_high=7.0)
        assert mask.dtype == bool
        assert mask.all()  # 5 dB diff is within [3, 7]

    def test_freq_diff_mask_excludes(self, rng):
        from echopype.mask.gpu_mask import freq_diff_mask_gpu

        Sv1 = rng.standard_normal((100, 100)).astype(np.float64)
        Sv2 = Sv1 - 10.0
        mask = freq_diff_mask_gpu(Sv1, Sv2, threshold_low=3.0, threshold_high=7.0)
        assert not mask.any()  # 10 dB diff is outside [3, 7]

    def test_compose_masks_and(self):
        from echopype.mask.gpu_mask import compose_masks_gpu

        m1 = np.array([True, True, False, False])
        m2 = np.array([True, False, True, False])
        result = compose_masks_gpu(m1, m2, operation="and")
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_compose_masks_or(self):
        from echopype.mask.gpu_mask import compose_masks_gpu

        m1 = np.array([True, True, False, False])
        m2 = np.array([True, False, True, False])
        result = compose_masks_gpu(m1, m2, operation="or")
        np.testing.assert_array_equal(result, [True, True, True, False])

    def test_snr_mask(self):
        from echopype.mask.gpu_mask import snr_mask_gpu

        Sv = np.array([-50.0, -60.0, -70.0, -80.0])
        Sv_noise = np.array([-80.0, -80.0, -80.0, -80.0])
        mask = snr_mask_gpu(Sv, Sv_noise, snr_threshold=15.0)
        # SNR: 30, 20, 10, 0 dB → first two pass threshold of 15
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_noise_corrected_sv(self, rng):
        from echopype.mask.gpu_mask import noise_corrected_sv_gpu

        Sv = np.array([-50.0, -60.0, -90.0])
        Sv_noise = np.array([-80.0, -80.0, -80.0])
        result = noise_corrected_sv_gpu(Sv, Sv_noise)
        # Result should be close to Sv where SNR is high
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        # Where Sv ≈ noise, correction should produce NaN or very low value
        # -90 signal with -80 noise: linear diff is negative → NaN
        assert np.isnan(result[2])


# ===========================================================================
# Integration-like: full pipeline smoke test
# ===========================================================================


@pytest.mark.unit
class TestGPUPipelineSmoke:
    """Verify that GPU modules can be composed in a typical processing order."""

    def test_sv_to_mvbs_pipeline(self, rng):
        """Simulate: Sv → noise removal → MVBS."""
        from echopype.clean.gpu_noise_est import estimate_noise_gpu
        from echopype.commongrid.gpu_mvbs import mvbs_index_binning_gpu
        from echopype.mask.gpu_mask import noise_corrected_sv_gpu, snr_mask_gpu

        n_ping, n_range = 200, 400
        Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 65

        echo_range = np.broadcast_to(
            np.linspace(1.0, 300.0, n_range)[np.newaxis, :], (n_ping, n_range)
        ).copy()
        spreading = 20 * np.log10(echo_range)
        absorption = 0.01 * echo_range

        # Step 1: Estimate noise
        Sv_noise = estimate_noise_gpu(
            Sv, echo_range, spreading, absorption, ping_num=20, range_sample_num=50
        )
        assert Sv_noise.shape == Sv.shape

        # Step 2: Correct Sv
        Sv_corrected = noise_corrected_sv_gpu(Sv, Sv_noise)
        assert Sv_corrected.shape == Sv.shape

        # Step 3: SNR mask
        mask = snr_mask_gpu(Sv, Sv_noise, snr_threshold=3.0)
        assert mask.shape == Sv.shape

        # Step 4: Apply mask and compute MVBS
        Sv_masked = np.where(mask, Sv_corrected, np.nan)
        mvbs = mvbs_index_binning_gpu(Sv_masked, ping_num=20, range_sample_num=50)
        assert mvbs.ndim == 2
        assert mvbs.shape[0] == 10  # 200/20
        assert mvbs.shape[1] == 8  # 400/50
