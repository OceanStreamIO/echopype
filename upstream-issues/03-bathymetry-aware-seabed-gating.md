# Bathymetry-aware seabed detection pre-check

**Type:** Feature
**Labels:** `processing functions`, `enhancement`
**Related:** Complements #1522 (Bottom detection algorithms), `echopype.mask.detect_seafloor`

## Summary

Propose adding a bathymetry-aware gating utility that queries known water depth before running seabed detection algorithms. When the true water depth exceeds the echosounder's effective range, bottom detection should be skipped or flagged rather than producing spurious picks from scattering layers, DSL, or noise.

## Motivation

This is relevant for **any open-ocean acoustic survey**, not just autonomous platforms:

- **Tropical Pacific cruises** (TPOS): typical depths 4000–5000 m, EK60/EK80 effective range ~1000–1500 m
- **Trans-oceanic transects**: research vessels routinely cross deep basins where bottom is unreachable
- **Deep-water fisheries surveys**: continental slope crossings where depth transitions past instrument range
- **Coastal-to-open-ocean crossings**: the transition zone is exactly where false bottom picks accumulate
- **Glider deployments**: limited echosounder power/range in deep water

Without bathymetric gating, `detect_seafloor` algorithms (basic, blackwell) will pick the strongest return in the water column — typically the deep scattering layer, thermocline, or random noise — and report it as "bottom." This creates downstream artifacts in seafloor masks, habitat classification, and integration products.

## Proposed design

### Two-tier bathymetry lookup

1. **Local GEBCO grid** (preferred): Query `GEBCO_2024.nc` (or later) via nearest-neighbor lookup. Fast, offline, ~450 m resolution — sufficient for gating decisions.
2. **NOAA NCEI web API** (fallback): REST query to NOAA's ArcGIS `identify` endpoint when no local grid is available.

### Integration with `detect_seafloor`

```python
def detect_seafloor(
    ds: xr.Dataset,
    method: str,
    params: Dict,
    # --- new parameter ---
    bathymetry_check: bool = False,
    max_echosounder_range: float | None = None,  # meters; inferred from data if None
) -> xr.DataArray
```

When `bathymetry_check=True`:
1. Extract median lat/lon from dataset
2. Query bathymetry (GEBCO → NOAA fallback)
3. If `known_depth > max_echosounder_range`: return NaN bottom line + warning log, skip detection
4. If `known_depth ≤ max_echosounder_range`: proceed normally with detection

### Standalone utility

Also useful independently:

```python
def get_bathymetry(lat: float, lon: float, gebco_path: str | None = None) -> float | None
def estimate_echosounder_range(ds: xr.Dataset) -> float  # from max depth/range_sample coordinate
```

## Implementation notes

- GEBCO dataset opened lazily and cached (one-time ~200 ms, subsequent queries ~1 ms)
- Web API fallback is optional and can be disabled for air-gapped environments
- No new hard dependencies — uses `xarray` for NetCDF read (already a dependency), `requests` for web API (already commonly available)
- Backward-compatible: `bathymetry_check=False` preserves current behavior

## Example use case

```python
import echopype as ep

# Deep tropical Pacific data — EK80 can't reach bottom at 4500 m
ds_Sv = ep.calibrate.compute_Sv(ed)
bottom = ep.mask.detect_seafloor(
    ds_Sv, method="basic", 
    params={"var_name": "Sv", "channel": "GPT  38 kHz", "threshold": -50},
    bathymetry_check=True,
)
# Returns NaN array with warning: "Known water depth (4523 m) exceeds echosounder range (1200 m). Skipping bottom detection."
```

## References

- GEBCO Compilation Group (2024). GEBCO 2024 Grid. doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f
- NOAA NCEI bathymetry services: https://www.ncei.noaa.gov/maps/bathymetry/
- Related echopype issue: #1522 (Bottom detection algorithms)
