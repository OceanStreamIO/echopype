---
description: "Echopype ocean sonar library — architecture, GPU acceleration, batch loading, and data pipeline patterns"
---

# Echopype — Domain Knowledge

Ocean sonar data processing library. Converts raw echosounder data (EK60, EK80, AZFP, AZFP6, AD2CP) to SONAR-netCDF4 v1.0 format and provides calibration, noise removal, regridding, and masking tools.

## Data Flow

```
Raw file → open_raw() → EchoData → compute_Sv() → xr.Dataset → analysis
                                                                 (clean, commongrid, mask, metrics)
```

For batch processing multiple files:
```
List[raw files] → open_raw_multi() → EchoData (combined)
```

## Key Modules

| Module | Purpose | GPU file |
|--------|---------|----------|
| `convert/` | Parse raw sonar files into `EchoData` | `api.py` (`open_raw_multi`) |
| `echodata/` | `EchoData` class — xarray `DataTree` wrapper | — |
| `calibrate/` | Compute Sv (volume backscattering) and TS | `gpu_cal.py` |
| `clean/` | Noise removal (background, transient, impulse) | `gpu_noise_est.py` |
| `commongrid/` | Regridding: MVBS, NASC | `gpu_mvbs.py` |
| `mask/` | Boolean masks, frequency differencing | `gpu_mask.py` |
| `consolidate/` | Add depth, GPS, split-beam angles | — |
| `utils/` | I/O, logging, GPU helpers, encoding | `gpu.py` |

## GPU / CUDA Acceleration

### Design Principles

- **Transparent fallback**: Every GPU-accelerated function accepts `use_gpu="auto"` (default). Auto-detects CUDA via `has_cuda()` and falls back to CPU seamlessly.
- **CuPy as backend**: `cupy-cuda12x` provides NumPy-compatible GPU arrays.
- **Array-module dispatch**: GPU modules use `xp = cp if has_cuda() else np` pattern.
- **Jetson unified memory**: `to_gpu()` / `to_cpu()` are zero-copy pointer wraps on unified-memory platforms.

### Core GPU Utilities (`utils/gpu.py`)

- `has_cuda()` — Check CuPy + CUDA device availability
- `to_gpu(arr)` / `to_cpu(arr)` — Move arrays between host/device (zero-copy on Jetson)
- `resolve_use_gpu(use_gpu)` — Resolve `"auto"` / `True` / `False` to bool
- `gpu_fftconvolve(a, b)` — FFT convolution (pulse compression)
- `gpu_batch_convolve(signals, kernel)` — Batched 1-D convolution
- `gpu_info()` — Dict of device properties

### GPU-Accelerated Functions

| Function | Module | What it does |
|----------|--------|-------------|
| `compute_Sv(ed, use_gpu=)` | `calibrate/api.py` | EK80 CW complex Sv — element-wise math on GPU via `gpu_cal.py` |
| `remove_background_noise(ds, use_gpu=)` | `clean/api.py` | Noise estimation + subtraction per channel |
| `compute_MVBS_index_binning(ds, use_gpu=)` | `commongrid/api.py` | Ping × range binning on GPU via `gpu_mvbs.py` |
| Frequency differencing | `mask/gpu_mask.py` | Boolean mask operations on GPU |

### Writing New GPU Kernels

1. Create `gpu_<name>.py` in the relevant module directory
2. Import CuPy conditionally: `from ..utils.gpu import has_cuda`
3. Accept plain NumPy arrays as input, return NumPy arrays as output
4. Add `use_gpu="auto"` parameter to the public API function in `api.py`
5. Use `resolve_use_gpu(use_gpu)` to decide the code path
6. Add tests in `tests/test_gpu.py` with `@pytest.mark.integration`
7. Avoid `cp.mean(axis=-1)` on Jetson — it's pathologically slow; use transpose + element-wise accumulation instead

## Batch File Loading (`open_raw_multi`)

`open_raw_multi()` in `convert/api.py` loads multiple raw files into a single `EchoData` object efficiently:

1. Parse all files → accumulate raw numpy arrays per channel
2. Cross-file numpy concatenation with NaN padding for varying range_sample sizes
3. Build a synthetic combined parser object
4. Call SetGroups **once** on the combined data → full SONAR-netCDF4 compliant EchoData

Key helpers:
- `_accumulate_parser_data()` — Extract per-file parsed data into accumulator lists
- `_concat_pad()` — Cross-file concatenation with NaN padding
- `_build_combined_parser()` — Synthetic parser for SetGroups

Performance: ~2.6x faster than sequential `open_raw()` + `combine_echodata()`.

## EchoData Structure

`EchoData` wraps an xarray `DataTree` with groups following SONAR-netCDF4:
- `Top-level` — Dataset metadata
- `Environment` — Sound speed, absorption, temperature, salinity
- `Platform` — Position, heading, pitch, roll
- `Platform/NMEA` — Raw NMEA sentences
- `Sonar` — Transducer/transceiver config
- `Sonar/Beam_group1` — Backscatter data (main group for Sv computation)
- `Vendor_specific` — Instrument-specific parameters (gain, sa_correction, etc.)
- `Provenance` — Processing history

## Code Patterns

- Public API lives in each module's `api.py`, re-exported via `__init__.py`
- Logging: `_init_logger(__name__)` from `utils/log.py` — never bare `print()`
- Type aliases: `SonarModelsHint`, `PathHint`, `FileFormatHint`, `EngineHint` from `core.py`
- xarray throughout; dask for lazy evaluation; fsspec for cloud storage
- NumPy style docstrings with `Parameters`, `Returns`, `Raises` sections
- black (line-length=100), isort, flake8

## Testing

- Markers: `@pytest.mark.unit` (fast) or `@pytest.mark.integration` (needs data/Docker)
- GPU tests: `tests/test_gpu.py` — all `@pytest.mark.integration`
- Test data: Pooch from GitHub Release Assets (`ECHOPYPE_DATA_VERSION=v0.11.1a2`)
- xarray assertions: `xr.testing.assert_allclose()` / `assert_equal()`
- Fixtures: `test_path` (data dirs), `dump_output_dir` (temp output)

```bash
pytest -vvv -rx --numprocesses=2 --cov=echopype --log-cli-level=WARNING
pytest -m unit          # Fast subset
pytest -m integration   # Full suite (needs data + Docker)
```
