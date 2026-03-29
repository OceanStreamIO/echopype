#!/usr/bin/env python3
"""benchmark_gpu.py — Compare CPU vs GPU performance for echopype operations.

Run on Jetson Orin NX (or any CUDA-capable system) to measure speedups.

Usage
-----
    python benchmark_gpu.py                  # Run all benchmarks
    python benchmark_gpu.py --only pulse     # Run only pulse compression
    python benchmark_gpu.py --sizes small    # Quick test with small arrays

The script produces a summary table at the end and optionally writes
results to ``benchmark_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timer(func, *args, warmup: int = 1, repeat: int = 5, **kwargs):
    """Time *func* and return (mean_seconds, std_seconds, result)."""
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), result


# ---------------------------------------------------------------------------
# Benchmark: Pulse Compression Convolution
# ---------------------------------------------------------------------------


def bench_pulse_compression(n_range: int = 8192, n_channels: int = 4, kernel_len: int = 256):
    """Benchmark the core convolution kernel used in pulse compression."""
    from scipy.signal import convolve as scipy_convolve
    from scipy.signal import fftconvolve as scipy_fftconvolve

    from echopype.utils.gpu import gpu_fftconvolve, has_cuda

    rng = np.random.default_rng(42)
    signal = (rng.standard_normal(n_range) + 1j * rng.standard_normal(n_range)).astype(
        np.complex64
    )
    kernel = (rng.standard_normal(kernel_len) + 1j * rng.standard_normal(kernel_len)).astype(
        np.complex64
    )

    results = {}

    # CPU scipy.signal.convolve (direct)
    t_cpu_direct, s_cpu_direct, ref = _timer(scipy_convolve, signal, kernel, mode="full")
    results["cpu_scipy_convolve"] = t_cpu_direct

    # CPU scipy.signal.fftconvolve
    t_cpu_fft, s_cpu_fft, _ = _timer(scipy_fftconvolve, signal, kernel, mode="full")
    results["cpu_scipy_fftconvolve"] = t_cpu_fft

    # GPU path (will be CPU fallback if no CUDA)
    t_gpu, s_gpu, gpu_result = _timer(gpu_fftconvolve, signal, kernel, mode="full")
    results["gpu_fftconvolve"] = t_gpu

    # Verify correctness
    np.testing.assert_allclose(gpu_result, ref, rtol=1e-3, atol=1e-4)

    label = "GPU" if has_cuda() else "CPU-fallback"
    speedup = t_cpu_direct / t_gpu if t_gpu > 0 else float("inf")
    print(f"  scipy.convolve (CPU):      {t_cpu_direct*1000:8.2f} ms")
    print(f"  scipy.fftconvolve (CPU):   {t_cpu_fft*1000:8.2f} ms")
    print(f"  gpu_fftconvolve ({label}): {t_gpu*1000:8.2f} ms  ({speedup:.1f}x vs direct)")

    return results


# ---------------------------------------------------------------------------
# Benchmark: Filter-Decimate Chirp
# ---------------------------------------------------------------------------


def bench_filter_decimate():
    """Benchmark the chirp filter-decimate chain."""
    from echopype.calibrate.ek80_complex import filter_decimate_chirp

    rng = np.random.default_rng(42)
    y_ch = rng.standard_normal(512)
    coeff_ch = {
        "wbt_fil": rng.standard_normal(32) + 1j * rng.standard_normal(32),
        "wbt_decifac": np.array([2]),
        "pc_fil": rng.standard_normal(32) + 1j * rng.standard_normal(32),
        "pc_decifac": np.array([2]),
    }

    t, s, _ = _timer(filter_decimate_chirp, coeff_ch, y_ch, 1500000.0)
    print(f"  filter_decimate_chirp:     {t*1000:8.2f} ms")
    return {"filter_decimate": t}


# ---------------------------------------------------------------------------
# Benchmark: Noise Estimation
# ---------------------------------------------------------------------------


def bench_noise_estimation(n_ping: int = 2000, n_range: int = 4000):
    """Benchmark GPU noise estimation vs xarray-based NoiseEst."""
    from echopype.clean.gpu_noise_est import estimate_noise_gpu

    rng = np.random.default_rng(42)
    Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 70
    echo_range = np.broadcast_to(
        np.linspace(1.0, 500.0, n_range)[np.newaxis, :], (n_ping, n_range)
    ).copy()
    spreading = 20 * np.log10(np.maximum(echo_range, 1.0))
    absorption = 0.01 * echo_range

    t, s, _ = _timer(
        estimate_noise_gpu,
        Sv,
        echo_range,
        spreading,
        absorption,
        ping_num=20,
        range_sample_num=50,
    )
    print(f"  noise estimation ({n_ping}×{n_range}): {t*1000:8.2f} ms")
    return {"noise_estimation": t}


# ---------------------------------------------------------------------------
# Benchmark: MVBS Index Binning
# ---------------------------------------------------------------------------


def bench_mvbs(n_ping: int = 2000, n_range: int = 4000):
    """Benchmark GPU MVBS computation."""
    from echopype.commongrid.gpu_mvbs import mvbs_index_binning_gpu

    rng = np.random.default_rng(42)
    Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 70

    t, s, _ = _timer(mvbs_index_binning_gpu, Sv, ping_num=20, range_sample_num=50)
    print(f"  MVBS index binning ({n_ping}×{n_range}): {t*1000:8.2f} ms")
    return {"mvbs_index_binning": t}


# ---------------------------------------------------------------------------
# Benchmark: Frequency Differencing Mask
# ---------------------------------------------------------------------------


def bench_freq_diff(n_ping: int = 2000, n_range: int = 4000):
    """Benchmark GPU frequency differencing."""
    from echopype.mask.gpu_mask import freq_diff_mask_gpu

    rng = np.random.default_rng(42)
    Sv1 = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 60
    Sv2 = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 65

    t, s, _ = _timer(freq_diff_mask_gpu, Sv1, Sv2, 2.0, 12.0)
    print(f"  freq diff mask ({n_ping}×{n_range}): {t*1000:8.2f} ms")
    return {"freq_diff_mask": t}


# ---------------------------------------------------------------------------
# Benchmark: Noise-Corrected Sv
# ---------------------------------------------------------------------------


def bench_noise_corrected(n_ping: int = 2000, n_range: int = 4000):
    """Benchmark GPU noise-corrected Sv computation."""
    from echopype.mask.gpu_mask import noise_corrected_sv_gpu

    rng = np.random.default_rng(42)
    Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 60
    Sv_noise = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 85

    t, s, _ = _timer(noise_corrected_sv_gpu, Sv, Sv_noise)
    print(f"  noise-corrected Sv ({n_ping}×{n_range}): {t*1000:8.2f} ms")
    return {"noise_corrected_sv": t}


# ---------------------------------------------------------------------------
# Benchmark: Full Pipeline (Sv → noise → MVBS)
# ---------------------------------------------------------------------------


def bench_full_pipeline(n_ping: int = 2000, n_range: int = 4000):
    """Benchmark the full GPU pipeline end-to-end."""
    from echopype.clean.gpu_noise_est import estimate_noise_gpu
    from echopype.commongrid.gpu_mvbs import mvbs_index_binning_gpu
    from echopype.mask.gpu_mask import noise_corrected_sv_gpu, snr_mask_gpu

    rng = np.random.default_rng(42)
    Sv = rng.standard_normal((n_ping, n_range)).astype(np.float64) * 10 - 65
    echo_range = np.broadcast_to(
        np.linspace(1.0, 500.0, n_range)[np.newaxis, :], (n_ping, n_range)
    ).copy()
    spreading = 20 * np.log10(np.maximum(echo_range, 1.0))
    absorption = 0.01 * echo_range

    def _pipeline():
        Sv_noise = estimate_noise_gpu(
            Sv, echo_range, spreading, absorption, ping_num=20, range_sample_num=50
        )
        Sv_corr = noise_corrected_sv_gpu(Sv, Sv_noise)
        mask = snr_mask_gpu(Sv, Sv_noise, snr_threshold=3.0)
        Sv_masked = np.where(mask, Sv_corr, np.nan)
        return mvbs_index_binning_gpu(Sv_masked, ping_num=20, range_sample_num=50)

    t, s, _ = _timer(_pipeline)
    print(f"  full pipeline ({n_ping}×{n_range}): {t*1000:8.2f} ms")
    return {"full_pipeline": t}


# ===========================================================================
# Main
# ===========================================================================

BENCHMARKS = {
    "pulse": bench_pulse_compression,
    "filter": bench_filter_decimate,
    "noise": bench_noise_estimation,
    "mvbs": bench_mvbs,
    "freq_diff": bench_freq_diff,
    "noise_corr": bench_noise_corrected,
    "pipeline": bench_full_pipeline,
}

SIZE_PRESETS = {
    "small": {"n_ping": 200, "n_range": 400},
    "medium": {"n_ping": 1000, "n_range": 2000},
    "large": {"n_ping": 4000, "n_range": 8000},
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU for echopype")
    parser.add_argument("--only", choices=list(BENCHMARKS.keys()), help="Run only one benchmark")
    parser.add_argument(
        "--sizes", choices=list(SIZE_PRESETS.keys()), default="medium", help="Array size preset"
    )
    parser.add_argument("--output", type=Path, help="Write results JSON to file")
    args = parser.parse_args()

    from echopype.utils.gpu import gpu_info

    info = gpu_info()
    print("=" * 60)
    print("Echopype GPU Benchmark")
    print("=" * 60)
    print(f"CUDA available:  {info['cuda_available']}")
    print(f"CuPy installed:  {info['cupy_installed']}")
    if info["cuda_available"]:
        print(f"Device:          {info.get('device_name', 'unknown')}")
        print(f"Memory:          {info.get('free_memory_mb', '?')} / {info.get('total_memory_mb', '?')} MB")
    print(f"Array sizes:     {args.sizes} → {SIZE_PRESETS[args.sizes]}")
    print("=" * 60)

    all_results = {"gpu_info": info, "size_preset": args.sizes}
    sizes = SIZE_PRESETS[args.sizes]

    benchmarks_to_run = {args.only: BENCHMARKS[args.only]} if args.only else BENCHMARKS
    for name, func in benchmarks_to_run.items():
        print(f"\n[{name}]")
        import inspect

        sig = inspect.signature(func)
        # Pass size kwargs only if function accepts them
        kwargs = {k: v for k, v in sizes.items() if k in sig.parameters}
        try:
            result = func(**kwargs)
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in all_results.items():
        if name in ("gpu_info", "size_preset"):
            continue
        if isinstance(result, dict) and "error" not in result:
            for key, val in result.items():
                print(f"  {key:35s} {val*1000:10.2f} ms")

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
