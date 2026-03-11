# Echopype Port Analysis Report

**Source:** `echopype-dev` branch `oceanstream-iotedge` (commit `808bb55d`)  
**Target:** `echopype-main` branch `oceanstream-integration`  
**Fork point:** `6052344a295764ae7f016963ce15d82d42275b2d`  
**Date:** 2025-03-06  
**Test results:** 921 passed, 0 failed, 16 skipped, 53 xfailed, 2 xpassed (299s)

---

## 1. Executive Summary

All functional changes from the `echopype-dev` fork (`oceanstream-iotedge` branch) have been
analyzed and ported to `echopype-main`. The fork contained 16 modified files (420 insertions, 219
deletions). After porting, `echopype-main` has 16 modified files (411 insertions, 144 deletions).

The difference in insertion/deletion counts is explained by:
- **Fork-specific changes intentionally NOT ported** — decorator removals, output renaming, and
  validation removals that broke upstream tests
- **Improvements made during porting** — upstream API adaptations (xarray 2026.2, zarr 3.x),
  enhanced test infrastructure, and improved `_set_verbose` implementation
- **Cosmetic-only changes skipped** — indentation reformatting in `commongrid/utils.py` that had
  no functional impact

---

## 2. File-by-File Comparison

| File | echopype-dev | echopype-main | Status |
|------|:---:|:---:|--------|
| `calibrate/cal_params.py` | 2±1 | 2±1 | ✅ Fully ported |
| `calibrate/ek80_complex.py` | 47±5 | 37±5 | ✅ Functionally equivalent |
| `calibrate/env_params.py` | 14±3 | 14±3 | ✅ Fully ported |
| `clean/api.py` | 34±16 | 2±2 | ⚠️ Partial — fork-specific changes excluded |
| `clean/utils.py` | 15±6 | 7±4 | ⚠️ Partial — `argmin` changes excluded |
| `commongrid/api.py` | 9±3 | 6±2 | ⚠️ Partial — decorator removal excluded |
| `commongrid/utils.py` | 96±30 | 33±7 | ✅ Substantive changes ported; cosmetic skipped |
| `consolidate/split_beam_angle.py` | 62±13 | 51±13 | ✅ Fully ported (expanded) |
| `convert/api.py` | 198±72 | 169±72 | ✅ Fully ported (adapted for upstream) |
| `convert/set_groups_base.py` | 23±10 | 25±13 | ✅ Fully ported |
| `convert/set_groups_ek80.py` | 35±11 | 30±11 | ✅ Functionally equivalent |
| `echodata/echodata.py` | 1±0 | 10±5 | ✅ Ported + expanded with Zarr guard |
| `mask/api.py` | 0±1 | — | ❌ Excluded — fork-only (decorator removal) |
| `tests/.../test_consolidate_integration.py` | 29±0 | 39±0 | ✅ Ported + expanded |
| `utils/coding.py` | 64±32 | 69±32 | ✅ Fully ported |
| `utils/log.py` | 9±4 | 11±8 | ✅ Ported + improved |
| `tests/conftest.py` | — | 50±4 | ➕ New — test infrastructure |

---

## 3. Detailed Change Analysis

### 3.1 `calibrate/cal_params.py`

**Change:** `.data.tolist().upper()` → `.values.item().upper()`

**Rationale:** The original `tolist()` call returns a scalar string from xarray, but `.values.item()`
is the canonical xarray way to extract a scalar from a 0-d DataArray. With newer xarray versions,
`tolist()` on a bytes-type DataArray may not produce a string, causing `.upper()` to fail.

**Verification:** All calibration tests pass. Specifically tested EK80 calibration paths that
exercise transceiver type lookup.

---

### 3.2 `calibrate/ek80_complex.py`

**Change:** Robustness improvements to `filter_decimate_chirp()`:
- `signal.convolve()` wrapped in try/except to flatten filter arrays when needed
- `wbt_decifac` and `pc_decifac` handled as potentially ndarray (not just scalar)
- Extracted `total_decimation_factor` for clarity in time computation

**Rationale:** Saildrone EK80 data stores filter coefficients as multi-element arrays. When these
reach `signal.convolve()`, the convolution fails unless arrays are flattened. The decimation factors
may also be stored as 1-element arrays rather than scalars, causing integer indexing (`ytx_wbt[0::factor]`)
to fail.

**echopype-dev vs echopype-main:** The dev fork had 47 changed lines with verbose if/else chains
for `wbt_decifac` handling (checking array length, single vs multiple elements). The main port
simplified this to extract `np.unique()` directly, reducing 15 lines to 4 while preserving identical
behavior. The `get_vend_filter_EK80()` function's `filter_channel` dimension was a merge artifact
in echopype-dev and was NOT ported — upstream uses `channel` consistently.

**Verification:** All EK80 complex-sample calibration tests pass, including BB pulse compression.

---

### 3.3 `calibrate/env_params.py`

**Change:** Sound speed and absorption parameter resolution now follows a 3-tier priority:
1. User-supplied values (`user_dict`)
2. Measured CTD values in the Environment group (`env.data_vars`)
3. Calculated from formulas (existing behavior)

**Rationale:** When CTD enrichment adds `sound_speed` and `sound_absorption` variables to the
Environment group (common in Saildrone workflows), these should be preferred over formula-based
calculations. Without this change, CTD-enriched data is silently ignored.

**Verification:** All env_params and calibration tests pass. The change only adds a new intermediate
branch in the priority chain; existing paths (user_dict, formula calculation) remain unchanged.

---

### 3.4 `clean/api.py`

**Ported (1 change):**
- `SNR_threshold: float = "3.0dB"` → `SNR_threshold: str = "3.0dB"` — type annotation bug fix

**NOT ported (fork-specific, 32 changes):**
- Removal of `@add_processing_level("L*B")` decorator from `remove_background_noise`
- Removal of `range_var` validation checks in `mask_transient_noise` and `mask_attenuated_signal`
- Commenting out `_parse_x_bin()` calls for upper/lower SL limits
- `coarsen(range_sample=...)` → `coarsen(depth=...)` + `.min(dim="range_sample")` → `.min(dim="depth")`
- Output renamed from `Sv_corrected` to `Sv` with reduced attribute assignments
- Commenting out `insert_input_processing_level()`
- Removal of `add_processing_level` import

**Why excluded:** These changes alter the clean module's public API (output variable names, processing
level tracking, validation behavior). They break 8 tests that expect the upstream output format
(`Sv_corrected` variable, processing level attributes, range_var validation). These are Oceanstream
pipeline-specific adaptations that assume the clean module operates in a different workflow context.

**Verification:** With only the type annotation fix ported, all 8 clean module tests pass.

---

### 3.5 `clean/utils.py`

**Ported (2 changes):**
- Dynamic `bin_dim = f"{range_var}_bins"` instead of hardcoded `"depth_bins"` in
  `downsample_upsample_along_depth()` — fixes the function when `range_var="echo_range"`
  (the bin dimension becomes `"echo_range_bins"` not `"depth_bins"`)
- Corresponding rename in `.assign_coords()` and `.rename()` calls

**NOT ported (2 changes):**
- `np.argmin((ds_Sv[range_var] <= exclude_above).data)` replacing the xarray `.argmin().values`
  pattern in `index_binning_pool_Sv()` — this change addresses a dask compatibility issue but
  `.data` forces computation unnecessarily. The existing code works.
- `.data` suffix added to `abs(range_var[...] - upper_limit_sl)` in `echopy_attenuated_signal_mask`
  — same pattern, force-computing to avoid xarray FutureWarning

**Why excluded:** The `.data` changes are minor dask/xarray API workarounds that don't fix any
current test failures. They may become necessary in a future xarray version, but currently all tests
pass without them.

**Verification:** All clean and transient noise tests pass.

---

### 3.6 `commongrid/api.py`

**Ported (2 changes):**
- `_get_reduced_positions()` call in `compute_NASC` wrapped in try/except with warning log —
  prevents NASC computation from failing when position data is incomplete
- `ds_NASC.attrs["distance_max"]` attribute added to NASC output

**NOT ported (2 changes):**
- Removal of `@add_processing_level("L3*")` decorator from `compute_MVBS`
- Commenting out `insert_input_processing_level(ds_MVBS, input_ds=ds_Sv)`

**Why excluded:** Decorator and processing level tracking are part of echopype's provenance system.
Removing them breaks the upstream contract that `compute_MVBS` output carries an `L3*` processing
level attribute. Tests verify this attribute exists.

**Verification:** All commongrid tests pass, including MVBS and NASC computation tests.

---

### 3.7 `commongrid/utils.py`

**Ported (substantive changes):**
- `get_distance_from_latlon()` completely rewritten:
  - Added NumPy-style docstring
  - Replaced `.to_dataframe().join()` with explicit `pd.merge()` on `ping_time`
  - Added separate latitude/longitude DataFrame extraction with `reset_index()`
  - Fixed trailing comma in `distance.distance()` call arguments

**NOT ported (cosmetic-only, ~63 lines):**
- Parameter indentation reformatting across 6 functions (`compute_raw_MVBS`, `compute_raw_NASC`,
  `_convert_bins_to_interval_index`, `_setup_and_validate`, `_get_reduced_positions`,
  `_groupby_x_along_channels`) — changing from 4-space to 8-space indentation on function parameter
  lines
- `1852**2` → `1852 ** 2` whitespace formatting in NASC formula

**Why excluded:** The indentation changes are style-only and touch function signatures throughout
the file without changing behavior. They create unnecessary diff noise and make future upstream
syncs harder. The `1852**2` spacing is cosmetic.

**Verification:** All NASC and MVBS tests pass, including distance-from-latlon calculations.

---

### 3.8 `consolidate/split_beam_angle.py`

**Change:** Major robustness improvements for split-beam angle computation:
- Added `SUPPORTED_BEAM_TYPES` constants (1, 17, 49, 65, 81) for documented beam type handling
- Dask-safe angle computation: `sens` and `offset` values are `.values`-extracted when backed by
  dask arrays to avoid lazy-computation issues in division
- Unsupported beam types (e.g., single-beam channels) are skipped with a warning instead of
  crashing
- `valid_channels` list tracks which channels produced angles, so the output DataArray has correct
  channel coordinates

**Rationale:** Real-world EK80 data often mixes split-beam and single-beam channels in the same
file. The original code crashes on `beam_type=0` (single-beam). Saildrone data frequently has
dask-backed sensitivity/offset arrays that fail in division operations.

**Verification:** All consolidate tests pass, including the new
`test_add_splitbeam_angle_partial_valid_channels` test that simulates mixed beam types.

---

### 3.9 `convert/api.py`

**Change:** Zero-length dimension handling throughout `_save_groups_to_file()`:
- New utility functions: `has_zero_length_dim(dataset)` and `remove_zero_length_vars(dataset)`
- Every SONAR-netCDF4 group save is guarded:
  1. Check for zero-length dimensions
  2. Strip chunking encoding from affected variables
  3. Skip saving entirely or save cleaned dataset
- Groups affected: Environment, Platform, Platform/NMEA, Provenance, Sonar, Sonar/Beam_groupX,
  Vendor_specific
- Beam_group1 gets a `is not None` guard before saving

**echopype-main additions (not in echopype-dev):**
- `.sizes.values()` used instead of `.dims.values()` (xarray 2026.2 compatibility — `.dims` now
  returns `Frozen` of dim names, not sizes)
- `encoding.pop('chunks', None)` instead of `encoding['chunks'] = None` (netCDF4 engine rejects
  explicit `None` chunks)

**Rationale:** Certain raw files (especially partial/corrupted Saildrone EK80 recordings) produce
EchoData groups with zero-length dimensions. Saving these with Zarr or netCDF4 causes crashes in
chunking/encoding logic (division by zero in chunk calculations, empty arrays in encoding).

**Verification:** All 41 convert tests pass, including EK80, EK60, AZFP, and AD2CP conversions.

---

### 3.10 `convert/set_groups_base.py`

**Change:** `_nan_timestamp_handler()` guards against empty ping time arrays:
- Checks if `ping_times` list is empty or contains empty arrays before calling `np.array().min()`
- Returns `[np.nan]` for empty ping times instead of crashing
- Flattened control flow: removed nested if/else, used early returns

**Rationale:** Incomplete raw files may have channels with no pings. The original code crashes with
`ValueError: zero-size array` when `ping_times` is empty.

**Verification:** All convert tests pass.

---

### 3.11 `convert/set_groups_ek80.py`

**Changes:**
1. Empty sound velocity profile returns `[]` instead of `[np.nan]` — avoids polluting
   Environment group with NaN placeholder values
2. `ds_invariant_power` and `ds_invariant_complex` initialized to `None` before conditional
   assignment — prevents `UnboundLocalError` when no complex or power data exists
3. `ds_beam is not None` guard before calling `beam_groups_to_convention()` — prevents crash
   when beam dataset is empty
4. Filter coefficients use `xr.merge([dataset, filter_ds])` instead of `dataset.assign()` —
   avoids dimension conflicts when filter data has a `channel` dimension that differs from
   the dataset's existing `channel` coordinate

**echopype-dev difference:** The dev fork used a `filter_channel` dimension for filter coefficients
to avoid the dimension conflict. This was a merge artifact from an abandoned upstream approach —
the tests expect `channel` as the dimension name. Our port uses `xr.merge()` with
`combine_attrs="override"` to solve the same conflict without renaming the dimension.

**Verification:** All EK80 conversion tests pass, including filter coefficient handling.

---

### 3.12 `echodata/echodata.py`

**Ported from echopype-dev (1 change):**
- `MutableMapping` import from `typing`

**Additional changes in echopype-main (not in echopype-dev):**
- `from_file()` method guards `_check_path`, `_sanitize_path`, and `_check_suffix` behind
  `if not isinstance(converted_raw_path, MutableMapping)` — allows opening Zarr stores from
  in-memory MutableMapping objects (e.g., `fsspec` mapped stores)

**Rationale:** The `MutableMapping` import was added in echopype-dev but never used. Our port
completes the intent by adding the Zarr store guard, enabling `EchoData.from_file()` to accept
in-memory Zarr stores for cloud-native workflows.

**Verification:** All echodata tests pass.

---

### 3.13 `mask/api.py`

**NOT ported:**
- Removal of `@add_processing_level("L3*")` from `apply_mask()`

**Why excluded:** Same rationale as commongrid/api.py — processing level decorators are part of
upstream's provenance tracking system. Tests verify the processing level attribute is set.

---

### 3.14 `tests/consolidate/test_consolidate_integration.py`

**Change:** New test `test_add_splitbeam_angle_partial_valid_channels` that:
1. Opens EK80 calibration data
2. Manually overrides one channel's `beam_type` to 0 (unsupported/single-beam)
3. Computes Sv and adds split-beam angles
4. Verifies that angles are computed for valid channels only (N-1 of N)

**echopype-main expansion:** The test includes an additional assertion that uses `sv_channel_count`
(Sv dataset's channel count) rather than hard-coding counts from the beam group.

**Verification:** Test passes and exercises the new `SUPPORTED_BEAM_TYPES` filtering logic.

---

### 3.15 `utils/coding.py`

**Changes:**
1. `set_time_encodings()`:
   - Skips all-NaN time variables with `if np.isnan(da).all(): continue`
   - Wraps `_encode_time_dataarray` in try/except ValueError (re-raises, but provides a clean
     stack trace point)
2. `set_zarr_encodings()`:
   - Zero-length dimension check: sets `chunks = None` for variables with any zero-sized dimension
   - Scalar variable handling: variables with `len(val.shape) == 0` get `chunks = None`

**Rationale:** Zero-length dimensions and all-NaN time arrays occur in partial/corrupted raw files.
Without these guards, the encoding functions crash during save operations (division by zero in
chunk calculations, NaT conversion failures).

**Verification:** All encoding-related tests pass. These changes work in concert with the
`convert/api.py` zero-length dimension guards.

---

### 3.16 `utils/log.py`

**echopype-dev change:**
```python
def _set_verbose(is_verbose):
    logger = logging.getLogger(__name__)
    if is_verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
```

**echopype-main change (improved):**
```python
def _set_verbose(is_verbose):
    package_name = __name__.split(".")[0]
    level = logging.INFO if is_verbose else logging.ERROR
    for logger in _get_all_loggers():
        if package_name in logger.name:
            logger.setLevel(level)
```

**Improvement:** The dev fork only sets the level on a single logger (`echopype.utils.log`). Our
port iterates ALL registered loggers and sets the level on every echopype logger. This correctly
controls verbosity across all modules (calibrate, convert, clean, etc.).

**Verification:** All logging tests pass. The `_get_all_loggers()` function already existed in
the codebase.

---

### 3.17 `tests/conftest.py` (echopype-main only)

**Change:** Enhanced test infrastructure (not from echopype-dev):
- `TEST_DATA_FOLDER` initialized to `None` with fallback
- Diagnostic message when `USE_POOCH` is not set
- `dump_output_dir` fixture uses `tmp_path_factory` when no test data
- `test_path` fixture validates data directories with file counts
- Reports missing/present test data directories with counts

**Rationale:** Improves developer experience when setting up tests for the first time. Without
these changes, missing test data causes cryptic `NoneType` errors deep in test execution.

**Verification:** All 921 tests pass with these infrastructure improvements.

---

## 4. Gap Analysis

### 4.1 Coverage Matrix

| Category | echopype-dev changes | Ported? | Notes |
|----------|---------------------|---------|-------|
| EK80 filter robustness | `ek80_complex.py` | ✅ | Simplified implementation |
| xarray API compat | `cal_params.py` | ✅ | |
| CTD env params priority | `env_params.py` | ✅ | |
| SNR type annotation | `clean/api.py` | ✅ | |
| Dynamic bin_dim | `clean/utils.py` | ✅ | |
| NASC error handling | `commongrid/api.py` | ✅ | |
| distance_max attr | `commongrid/api.py` | ✅ | |
| get_distance_from_latlon rewrite | `commongrid/utils.py` | ✅ | |
| Split-beam robustness | `split_beam_angle.py` | ✅ | |
| Zero-length dim handling | `convert/api.py` | ✅ | |
| Empty ping_time guard | `set_groups_base.py` | ✅ | |
| EK80 group init/guards | `set_groups_ek80.py` | ✅ | |
| MutableMapping import | `echodata.py` | ✅ | Extended with Zarr guard |
| NaN time skip | `utils/coding.py` | ✅ | |
| Zero-length chunk handling | `utils/coding.py` | ✅ | |
| Scalar chunk handling | `utils/coding.py` | ✅ | |
| Verbose logging | `utils/log.py` | ✅ | Improved vs dev |
| Partial beam type test | `test_consolidate_integration.py` | ✅ | Extended |
| `@add_processing_level` removal | `clean/api.py`, `commongrid/api.py`, `mask/api.py` | ❌ | Fork-specific |
| Output var renaming (`Sv_corrected` → `Sv`) | `clean/api.py` | ❌ | Fork-specific |
| Validation removal | `clean/api.py` | ❌ | Fork-specific |
| `coarsen(depth=...)` change | `clean/api.py` | ❌ | Fork-specific |
| `.data` argmin pattern | `clean/utils.py` | ❌ | Not needed |
| Parameter indentation | `commongrid/utils.py` | ❌ | Cosmetic only |

### 4.2 Untracked Files

- `echopype/clean/noise_est.py` — exists on disk in BOTH repos but is **untracked** (not committed)
  in either. Not part of the fork's changes. This file was removed from git tracking in upstream
  echopype at some point. It should NOT be committed.

### 4.3 Conclusion

**All functional improvements from echopype-dev have been successfully ported.** The excluded
changes fall into three categories:

1. **Fork-specific pipeline adaptations** (decorator removals, output renaming, validation removal)
   — these break upstream tests and are specific to Oceanstream's processing pipeline
2. **Cosmetic formatting** (parameter indentation) — no functional impact
3. **Preemptive dask workarounds** (`.data` on argmin) — not currently needed

No gaps exist in the port. The echopype-main port additionally includes improvements not present
in echopype-dev:
- xarray 2026.2 API compatibility (`.sizes` vs `.dims`)
- netCDF4 encoding fix (`pop` vs assign `None`)
- Improved `_set_verbose` (all loggers, not just one)
- Zarr MutableMapping guard in `EchoData.from_file()`
- Enhanced test infrastructure with data validation

---

## 5. Test Verification

### 5.1 Full Test Suite Results

```
921 passed, 16 skipped, 53 xfailed, 2 xpassed in 299.64s
```

### 5.2 Test Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.12 |
| xarray | 2026.2.0 |
| zarr | 3.1.5 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| pandas | 3.0.1 |
| netCDF4 | 1.7.4 |
| dask | 2025.4.0 |
| pytest | 9.0.2 |

### 5.3 Test Data

21 Pooch bundles at `~/Library/Caches/echopype/v0.11.1a2/` covering:
EK60, EK80, ES70, ES80, EA640, AZFP, AZFP6, AD2CP

### 5.4 Integration Services

- MinIO (S3 mock) on port 9000
- HTTP server on port 8080
- Docker Compose v5.0.1

---

## 6. Recommendations

1. **Commit the port** — all changes are verified, no test regressions
2. **Do NOT commit `noise_est.py`** — it is an untracked legacy file
3. **Future upstream PRs** — consider contributing back:
   - `ek80_complex.py` filter robustness (general-purpose improvement)
   - `split_beam_angle.py` multi-beam-type handling (general-purpose)
   - `convert/api.py` zero-length dimension guards (general-purpose)
   - `env_params.py` CTD priority chain (general-purpose with CTD use case)
4. **Keep fork-specific** — clean/api.py output renaming and decorator removal should stay in
   the Oceanstream pipeline layer, not in the echopype library
