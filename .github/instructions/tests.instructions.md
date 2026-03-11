---
applyTo: "echopype/tests/**"
description: "Echopype test conventions: fixtures, markers, Pooch data, assertions"
---

# Test Conventions

## Markers

Mark every test with `@pytest.mark.unit` or `@pytest.mark.integration`:
- **unit** — Fast, no network/data downloads, no Docker services
- **integration** — Requires Pooch test data and/or MinIO + HTTP Docker services

## Fixtures (session-scoped, from `conftest.py`)

- `test_path` — Dict mapping sonar model keys to data directories:
  `EK60`, `EK80`, `EK80_NEW`, `ES60`, `ES70`, `ES80`, `EA640`, `AZFP`, `AZFP6`, `AD2CP`,
  `EK80_MULTIPLEX`, `EK80_DUPE_PING`, `EK80_MISSING_SOUND`, `EK80_INVALID_ENV`,
  `EK80_SEQUENCE`, `EK80_CAL`, `EK80_EXT`, `ECS`, `LEGACY_DATATREE`, plus `ROOT`
- `dump_output_dir` — Temp directory for test output files
- `minio_bucket` — S3 config dict for MinIO (`endpoint_url`, `key`, `secret`)

## Test Data

- Fetched via **Pooch** from GitHub Release Assets when `USE_POOCH=True`
- Version: `ECHOPYPE_DATA_VERSION=v0.11.1a2`
- Cache: `$XDG_CACHE_HOME/echopype` (managed by pooch)
- Do not commit test data files to the repo

## Patterns

- Test files mirror source structure: `echopype/calibrate/` → `echopype/tests/calibrate/`
- Use `xr.testing.assert_allclose()` / `assert_equal()` for xarray comparisons
- Use `tmp_path` (pytest built-in) for throwaway files; use `dump_output_dir` for larger integration outputs
- Prefer parametrize over copy-pasted test functions

## Running Tests

```bash
pytest -vvv -rx --numprocesses=2 --cov=echopype --log-cli-level=WARNING
pytest -m unit          # Fast subset
pytest -m integration   # Full suite (needs data + Docker)
```
