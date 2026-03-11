# Echopype Upstream Contribution Status

**Fork:** OceanStreamIO/echopype (branch: `oceanstream-integration`)
**Base:** OSOceanAcoustics/echopype `main` @ `f8213393`
**Date:** 2026-03-11
**Test baseline:** 921 passed, 0 failed, 16 skipped, 53 xfailed (Python 3.12, xarray 2026.2, zarr 3.1.5)

---

## Summary

We maintain an echopype fork for the [Oceanstream](https://github.com/OceanStreamIO) project,
processing echosounder data from Saildrone autonomous vehicles. During integration with
real-world EK80 data (including partial recordings, mixed beam types, and sparse GPS), we
identified and fixed several robustness issues in echopype's conversion, calibration,
consolidation, and commongrid modules.

All fixes are backward-compatible, do not change output semantics for valid data, and pass
the full upstream test suite. We're starting with **3 focused PRs** — the highest-value,
easiest-to-review changes — with more to follow based on maintainer feedback.

---

## PRs (Round 1)

| # | Branch | Title | Files | +/- |
|---|--------|-------|-------|-----|
| 1 | `oceanstream/fix-type-hint-and-scalar-extraction` | Type annotation + scalar extraction | 2 | +2/−2 |
| 2 | `oceanstream/fix-zero-length-dims` | Zero-length dims in conversion | 3 | +168/−95 |
| 3 | `oceanstream/fix-splitbeam-mixed-types` | Mixed beam types in split-beam (+ test) | 2 | +79/−11 |

**Total:** 7 files changed, ~249 insertions, ~108 deletions

---

## PR Details

### PR 1: Trivial bug fixes — type annotation + xarray scalar extraction

**Branch:** `oceanstream/fix-type-hint-and-scalar-extraction`
**Files:**
- `echopype/clean/api.py` — `SNR_threshold: float` → `SNR_threshold: str` (type hint doesn't match default `"3.0dB"`)
- `echopype/calibrate/cal_params.py` — `.data.tolist().upper()` → `.values.item().upper()` (canonical xarray scalar extraction)

**Rationale:** Two obvious one-liner fixes. No behavioral change.

---

### PR 2: Handle zero-length dimensions in raw file conversion

**Branch:** `oceanstream/fix-zero-length-dims`
**Files:**
- `echopype/convert/api.py` — `has_zero_length_dim()` and `remove_zero_length_vars()` utilities; guards on every group save; `.sizes.values()` for xarray 2026.2 compat
- `echopype/convert/set_groups_base.py` — Empty ping time guard in `_nan_timestamp_handler()`
- `echopype/utils/coding.py` — NaN time skip in `set_time_encodings()`; zero-length/scalar chunk handling in `set_zarr_encodings()`

**Rationale:** Partial or corrupted raw files (common in autonomous vehicle deployments like Saildrone) produce groups with zero-length dimensions. Save operations crash on chunk calculations (division by zero). These guards prevent crashes while preserving all valid data.

**Reproduction:** Process a truncated EK80 `.raw` file where some channels have no pings recorded.

---

### PR 3: Support mixed beam types in split-beam angle computation

**Branch:** `oceanstream/fix-splitbeam-mixed-types`
**Files:**
- `echopype/consolidate/split_beam_angle.py` — Main logic
- `echopype/tests/consolidate/test_consolidate_integration.py` — New test

**Changes:**
- Add `SUPPORTED_BEAM_TYPES` constant (types 1, 17, 49, 65, 81)
- Use dask-safe scalar extraction (`.values.item()`) for `beam_type`
- Skip unsupported beam types with `logger.warning()` instead of crashing
- Track `valid_channels` for partial results

**Test:** `test_add_splitbeam_angle_partial_valid_channels` — creates a mixed dataset with both split-beam and single-beam channels, verifies the function processes valid channels and skips invalid ones.

**Rationale:** Real EK80 data routinely mixes split-beam and single-beam channels. The current code crashes on unsupported beam types.

---

## Deferred PRs (Round 2)

The following changes are ready but held back to keep review load manageable. All are independent and can be submitted after Round 1 is merged:

| Branch | Title | Scope | Files |
|--------|-------|-------|-------|
| `oceanstream/fix-ek80-filter-robustness` | EK80 filter coefficient robustness | ~32 lines | `calibrate/ek80_complex.py` |
| `oceanstream/fix-ek80-partial-data` | EK80 partial data conversion guards | ~22 lines | `convert/set_groups_ek80.py` |
| `oceanstream/feat-ctd-env-priority` | CTD-enriched env param priority | ~12 lines | `calibrate/env_params.py` |
| `oceanstream/fix-commongrid-robustness` | Commongrid + distance calculation | ~33 lines | `commongrid/api.py`, `commongrid/utils.py` |
| `oceanstream/fix-bin-dim-zarr-logging` | Dynamic bin dim + Zarr guard + logging | ~18 lines | `clean/utils.py`, `echodata/echodata.py`, `utils/log.py` |

---

## Excluded from Upstream

The following fork-specific changes are intentionally **not** included in any upstream PR:

| Change | Reason |
|--------|--------|
| `@add_processing_level` decorator removals | Oceanstream pipeline incompatibility |
| `Sv_corrected` → `Sv` output variable renaming | Fork-specific convention |
| Validation/assertion removals | Fork-specific permissiveness |
| `coarsen(depth=...)` changes | Fork-specific depth handling |
| `tests/conftest.py` enhancements | Repo/CI-specific; could be proposed separately |
| `PORTING_REPORT.md` | Fork-internal documentation |
| `echopype/clean/noise_est.py` | New file, fork-specific |

---

## Testing

All PRs were verified against the full test suite:

```
pytest -vvv -rx --numprocesses=2 --cov=echopype --log-cli-level=WARNING
# 921 passed, 16 skipped, 53 xfailed, 2 xpassed
```

Each PR branch contains only its targeted changes and can be tested independently:

```bash
# PR-specific test commands
pytest echopype/tests/calibrate/ -v -k "ek80"              # PRs 3, 6
pytest echopype/tests/convert/ -v                            # PRs 2, 4
pytest echopype/tests/consolidate/ -v -k "splitbeam"        # PR 5
pytest echopype/tests/commongrid/ -v                         # PR 7
pytest echopype/tests/clean/ -v                              # PRs 1, 8
```
