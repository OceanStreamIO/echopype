# Dask-native enhancements for transient noise detection

**Type:** Feature / Enhancement
**Labels:** `processing functions`, `enhancement`
**Related:** Extends `echopype.clean.detect_transient` (Fielding method), adds Ryan et al. (2015) method

## Summary

Propose dask-native performance enhancements and an additional transient noise detection method (Ryan et al. 2015) for `echopype.clean`. The current Fielding and Matecho implementations operate on in-memory NumPy arrays, which limits scalability for large multi-day datasets common in autonomous vehicle and long-duration surveys.

## Motivation

Transient noise detection is one of the most computationally expensive denoising steps because it requires ping-neighborhood statistics across the full time axis. For large datasets (multi-day Saildrone deployments, long cruises):

- Current NumPy-based kernels require loading entire datasets into memory
- No ability to leverage Dask distributed clusters for parallel processing
- Missing "unfeasible" mask output — users don't know where detection confidence is limited (time edges, deep NaN regions)

## Proposed enhancements

### 1. Dask-native Fielding implementation via `map_overlap`

Our working implementation uses `dask.array.map_overlap` to apply the Fielding kernel with a time-only halo, enabling:
- Chunk-parallel execution across the ping_time axis
- No full-dataset materialization required
- `bottleneck` acceleration for rolling medians when available

### 2. Ryan et al. (2015) transient noise method

A complementary algorithm that uses 2D rolling percentile blocks in the linear domain:

```python
def mask_transient_noise_ryan(
    ds_Sv: xr.Dataset,
    range_var: str = "depth",
    ping_window: int = 30,
    range_window: int = 20,
    threshold: str = "10.0dB",
    exclude_above: str = "0.0m",
    percentile: float = 50.0,
    min_pings: int = 5,
    min_samples: int = 5,
) -> xr.DataArray
```

This method is useful as a second opinion alongside Fielding — it detects transients with different spatial characteristics (broader vertical extent).

### 3. Unfeasible mask output

Both methods would return an optional secondary mask indicating regions where detection is unreliable:
- Time edges (insufficient ping neighbors)
- Deep regions with all-NaN blocks
- Shallow exclusion zones

This allows downstream pipelines to track detection confidence without assuming uniform reliability.

### Proposed API addition

Add `"ryan"` as a third method to `detect_transient`:

```python
# Existing dispatcher gains new method
detect_transient(ds, method="ryan", params={...})
```

Or as a standalone function:
```python
echopype.clean.mask_transient_noise_ryan(ds_Sv, ...)
```

## Implementation notes

- All operations remain dask-lazy
- `bottleneck.move_median` used as fast path when available, falls back to `scipy.ndimage`
- Compatible with existing `mask_transient_noise` API conventions (dB-string parsing, `range_var` parameter)

## References

- Ryan, T. E., Downie, R. A., Kloser, R. J., & Keith, G. (2015). Reducing bias due to noise and attenuation in open-ocean echo integration data. *ICES Journal of Marine Science*, 72(8), 2482-2493.
- Fielding, S. et al. — Fielding transient noise detection algorithm
- Current echopype implementations: `transient_noise_fielding`, `transient_noise_matecho`
