# Echopype — Project Guidelines

Ocean sonar data processing library for interoperability and scalability.
Converts raw echosounder data to a standardized format (SONAR-netCDF4 v1.0) and provides analysis tools.

## Fork Context

This is a fork of [echopype](https://github.com/OSOceanAcoustics/echopype) maintained by the Oceanstream team. We enhance it with functionality for our own oceanographic data processing library (`./oceanstream-library`), which builds pipelines for USV-collected data (echosounder, GPS tracks, multibeam, ADCP) on top of echopype. Some of our changes diverge from upstream echopype's goals; others we aim to contribute back to the main project. When making changes, consider whether they are **upstream-compatible** (general-purpose improvements suitable for a PR back to echopype) or **fork-only** (Oceanstream-specific features).

## Architecture

| Module | Purpose |
|--------|---------|
| `convert/` | Parse raw sonar files (EK60, EK80, AZFP, AZFP6, AD2CP) into `EchoData` |
| `echodata/` | `EchoData` class — xarray `DataTree` wrapper following SONAR-netCDF4 groups |
| `calibrate/` | Compute Sv (volume backscattering) and TS (target strength) |
| `clean/` | Noise removal: transient noise, impulse noise |
| `commongrid/` | Regridding: MVBS, NASC, general `regrid()` |
| `consolidate/` | Add depth, GPS location, split-beam angles, swap dims |
| `mask/` | Boolean masks, frequency differencing, seafloor/shoal detection |
| `qc/` | Quality control (time reversal fixes) |
| `metrics/` | Summary statistics on Sv |
| `utils/` | I/O (fsspec), logging, provenance, encoding, alignment |

**Data flow:** Raw file → `open_raw()` → `EchoData` → `compute_Sv()` → xarray `Dataset` → analysis functions

**Key registries:** `SONAR_MODELS` in `core.py` maps model names to parser/group-setter/validator classes.

## Code Style

- **Formatter:** black (line-length=100)
- **Import sorting:** isort (black-compatible, `combine_as_imports=true`)
- **Linting:** flake8 with builtins/comprehensions/mutable/print plugins
- **Docstrings:** NumPy style — use `Parameters`, `Returns`, `Raises` sections
- **Type hints:** Use throughout; use `TYPE_CHECKING` blocks for import-cycle-prone types
- **Type aliases:** `SonarModelsHint`, `PathHint`, `FileFormatHint`, `EngineHint` from `core.py`
- **Logging:** `_init_logger(__name__)` from `utils/log.py` — never bare `print()`
- **Line length:** 100 characters

## Data Handling Patterns

- All array data uses **xarray** (`Dataset`, `DataArray`, `DataTree`)
- Lazy evaluation via **dask** — preserve chunking, avoid `.compute()` unless necessary
- File I/O supports **netCDF4** and **Zarr** engines
- Cloud storage via **fsspec** — pass `storage_options` dicts, never hardcode paths
- Encoding/compression settings in `utils/coding.py`
- Provenance tracked via `add_processing_level()` decorator and `echopype_prov_attrs()`

## Build and Test

```bash
# Install for development
pip install -e ".[plot]"
pip install -r requirements-dev.txt

# Run tests (parallel)
pytest -vvv -rx --numprocesses=2 --cov=echopype --log-cli-level=WARNING

# Markers
pytest -m unit          # Fast, no data downloads
pytest -m integration   # Needs test data + Docker services
```

- **Test data:** Pooch-based, fetched from GitHub Release Assets (`ECHOPYPE_DATA_VERSION=v0.11.1a2`)
- **Integration services:** MinIO (S3 mock, port 9000) + HTTP server (port 8080) via Docker
- **Pre-commit:** `pre-commit run --all-files` (black, isort, flake8, codespell)

## Conventions

- Public API functions live in each module's `api.py` and are re-exported via `__init__.py`
- Calibration/parsing uses abstract base classes (`CalibrateBase`, `ParseBase`, `SetGroupsBase`) — subclass per sonar model
- `EchoData` group names follow SONAR-netCDF4: `Top-level`, `Environment`, `Platform`, `Platform/NMEA`, `Sonar`, `Sonar/Beam_groupN`, `Vendor_specific`, `Provenance`
- Test fixtures provide `test_path` dict for data directories and `dump_output_dir` for temp output
- Python 3.11+ required
