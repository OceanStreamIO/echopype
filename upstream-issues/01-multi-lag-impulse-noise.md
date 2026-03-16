# Multi-lag impulse noise detector with vote aggregation

**Type:** Feature / Enhancement
**Labels:** `processing functions`, `enhancement`
**Related:** Extends `echopype.clean.mask_impulse_noise`

## Summary

Propose adding a multi-lag impulse noise detection algorithm with k-of-n vote consensus to `echopype.clean`. The current `mask_impulse_noise` uses a single-lag two-sided ping comparison (derived from echopy), which can produce false positives on noisy platforms or miss multi-ping spikes.

## Motivation

On autonomous vehicles (Saildrone USVs, gliders), wave-induced motion and electrical interference generate impulse noise that often spans 2-3 consecutive pings rather than isolated single-ping spikes. The single-lag approach:

- **False positives**: A single outlier neighbor ping can trigger a false spike detection
- **Missed detections**: Multi-ping spike bursts are only partially caught since adjacent pings are also elevated

This affects any deployment on moving platforms, not just USVs — towed bodies, drifting buoys, and vessels in rough seas all exhibit similar patterns.

## Proposed algorithm

We have a working, dask-native implementation in our [OceanStream](https://github.com/OceanStreamIO) project that we'd like to contribute. The algorithm:

1. **Multi-lag differencing**: For each ping, compute forward and backward Sv differences at multiple lag distances (e.g., lags 1, 2, 3) instead of just lag-1
2. **Vote aggregation**: Flag a sample as impulsive only if ≥ k out of n lags agree it exceeds the threshold (e.g., 2-of-3 vote). This consensus approach reduces false positives from single noisy neighbors
3. **Optional vertical bin pooling**: Average Sv in vertical bins (meters or samples) before differencing to improve SNR of the detection step
4. **Post-dilation**: Optional morphological dilation (configurable in pings and samples) via efficient shift-OR operations to catch spike shoulders

### Proposed API extension

```python
def mask_impulse_noise(
    ds_Sv: xr.Dataset,
    depth_bin: str = "5m",
    num_side_pings: int = 2,
    impulse_noise_threshold: str = "10.0dB",
    range_var: str = "depth",
    use_index_binning: bool = False,
    # --- new parameters ---
    ping_lags: list[int] | None = None,       # e.g. [1, 2, 3]; None = current single-lag behavior
    vote_k_of_n: int | None = None,            # minimum agreeing lags; None = all must agree
    vertical_bin_size: str | None = None,       # e.g. "5m"; pre-detection vertical pooling
    post_dilate_pings: int = 0,                 # morphological dilation in ping direction
    post_dilate_samples: int = 0,               # morphological dilation in range direction
) -> xr.DataArray
```

Backward-compatible: when `ping_lags=None`, behavior is identical to current implementation.

## Implementation notes

- Entire pipeline is dask-lazy (no `.compute()` until the caller needs it)
- Shift-OR dilation is O(k) shifts, no convolution kernel needed
- Produces an additional "unfeasible" mask for edge pings where lags extend beyond data bounds, so downstream code knows where detection confidence is limited

## References

- Current echopype implementation: `echopype.clean.mask_impulse_noise`
- De Robertis, A. & Higginbottom, I. (2007). A post-processing technique to estimate the signal-to-noise ratio and remove echosounder background noise. *ICES Journal of Marine Science*, 64(6), 1282-1291.
