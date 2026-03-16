# Enhanced seabed detection with morphological cleanup and max-Sv method

**Type:** Feature / Enhancement
**Labels:** `processing functions`, `enhancement`
**Related:** Directly extends #1522 (Bottom detection algorithms)

## Summary

Propose contributing two enhancements to echopype's seabed detection (#1522):

1. **Morphological post-processing pipeline** applicable to any detection method (basic, blackwell, or future methods)
2. **Max-Sv seabed detection method** — a third algorithm alongside the existing basic and blackwell methods
3. **Automatic 38 kHz channel selection** utility

## 1. Morphological post-processing pipeline

The current `bottom_basic` and `bottom_blackwell` return raw bottom picks that can be noisy — especially on autonomous platforms, in rough weather, or with scattering layer interference. We propose a configurable post-processing chain:

### Steps (each independently toggleable)

1. **Median pre-filtering**: Smooth Sv data before detection to reduce per-sample noise
2. **Depth gating**: Restrict detection to a physically plausible depth range (`r0_m` to `r1_m`)
3. **Threshold with percentile floor**: Use channel-adaptive percentile of Sv values (e.g., 75th percentile) as detection threshold, with a configurable absolute floor
4. **Morphological erosion + dilation**: Remove isolated false picks (erosion), then restore true bottom extent (dilation) — configurable kernel sizes
5. **Ping-continuity enforcement**: Reject bottom candidates that are discontinuous across consecutive pings 
6. **Shallowest-return selection**: When multiple candidates exist, pick the shallowest (first bottom return) — critical for distinguishing true bottom from sub-bottom reflections
7. **Fallback picks**: For pings where all candidates are rejected, interpolate from neighbors
8. **Savitzky-Golay smoothing**: Final de-spiking of the bottom line using a smooth polynomial fit

### Proposed API

```python
def postprocess_bottom_line(
    bottom_depth: xr.DataArray,       # raw bottom picks from any detector
    ds_Sv: xr.Dataset | None = None,  # optional, needed for some steps
    erosion_kernel: tuple = (1, 3),
    dilation_kernel: tuple = (3, 5),
    savgol_window: int = 11,
    savgol_order: int = 2,
    continuity_max_jump_m: float = 50.0,
    fill_gaps: bool = True,
) -> xr.DataArray
```

## 2. Max-Sv seabed detection method

A third algorithm for `detect_seafloor(method="max_sv", ...)`:

### Algorithm

1. Within the depth gate (`depth_min` to `depth_max`), find the sample with maximum Sv per ping — this is the initial bottom candidate
2. From the max-Sv sample, trace **upward** until Sv drops below a secondary threshold — this finds the true onset of the bottom return rather than the peak
3. Optional ping-median smoothing across neighboring pings to reject outlier picks

### Proposed signature

```python
def bottom_max_sv(
    ds: xr.Dataset,
    var_name: str = "Sv",
    channel: str = ...,
    depth_min: float = 10.0,
    depth_max: float = 1000.0,
    threshold: tuple = (-40.0, -60.0),  # (primary, secondary for upward trace)
    smooth_pings: int = 5,
    range_offset: float = 0.0,
) -> xr.DataArray
```

### When to use

- **Basic**: Simple threshold, fast, good for clean data
- **Blackwell**: Uses split-beam angles, best for calibrated split-beam transducers
- **Max-Sv** (proposed): Robust for noisy data / moving platforms where the bottom return is strong but variable; doesn't require split-beam angles

## 3. Automatic 38 kHz channel selection

A small utility that most seabed detection users need:

```python
def find_channel_by_frequency(
    ds: xr.Dataset,
    target_freq_hz: float = 38000.0,
    tolerance_hz: float = 500.0,
) -> str | None
```

Searches `frequency_nominal` coordinate first, falls back to parsing channel labels. Returns the channel string for the closest matching frequency, or `None` if not found.

Useful because:
- 38 kHz is the standard frequency for bottom detection
- Channel naming varies across instruments and firmware versions
- Users currently write this boilerplate in every project

## Implementation notes

- We have working implementations of all three components in our [OceanStream](https://github.com/OceanStreamIO) project, battle-tested on real EK80 data from Saildrone tropical Pacific deployments
- All operations are xarray/dask-compatible
- Post-processing pipeline is method-agnostic — works on output from `bottom_basic`, `bottom_blackwell`, or any future detector
- Max-Sv method doesn't require split-beam angles, making it accessible for single-beam deployments

## References

- Related issue: #1522 (Bottom detection algorithms) — specifically the roadmap for "second bottom detection" and optimization
- Ariza, A. et al. — Ariza bottom detection method
- Blackwell, R. et al. — Blackwell split-beam bottom detection
