# Configurable stage-based denoising mask composition engine

**Type:** Feature
**Labels:** `processing functions`, `enhancement`
**Related:** Builds on `echopype.clean` and `echopype.mask.apply_mask`

## Summary

Propose adding a composable multi-stage mask pipeline to `echopype.clean` (or a new `echopype.clean.pipeline` submodule) that orchestrates multiple denoising steps — impulse noise, transient noise, background noise, attenuation — into a single reproducible, configurable workflow.

## Motivation

Currently, users who want to apply multiple cleaning masks must:

1. Call each `mask_*` function independently
2. Manually combine masks with correct boolean logic (OR/AND)
3. Handle coordinate alignment between masks from different functions
4. Track which regions are "unfeasible" (insufficient data for reliable detection)
5. Repeat per-channel when different channels need different parameters
6. Manage pulse-mode variations (short/long pulse) across channels

This is error-prone and creates inconsistent workflows across projects. Every production echopype user ends up writing their own mask composition boilerplate.

## Proposed design

### Pipeline configuration

A denoising pipeline defined as a list of stages, each with a method reference and per-channel/per-pulse parameters:

```python
from echopype.clean.pipeline import DenoisePipeline

pipeline = DenoisePipeline(stages=[
    {
        "name": "impulse",
        "function": "mask_impulse_noise",
        "params": {
            "38000": {"impulse_noise_threshold": "10.0dB", "num_side_pings": 2},
            "200000": {"impulse_noise_threshold": "8.0dB", "num_side_pings": 3},
        },
    },
    {
        "name": "transient",
        "function": "mask_transient_noise",
        "params": {
            "38000": {"transient_noise_threshold": "12.0dB", "num_side_pings": 25},
        },
    },
    {
        "name": "background",
        "function": "remove_background_noise",
        "params": {
            "default": {"ping_num": 20, "range_sample_num": 50, "SNR_threshold": "3.0dB"},
        },
    },
])
```

### Execution

```python
ds_masked, report = pipeline.run(ds_Sv)
# ds_masked: dataset with combined mask applied
# report: per-stage mask statistics (flagged %, unfeasible %, per channel)
```

### Key features

1. **Per-channel parameter dispatch**: Different frequencies can have different thresholds/windows
2. **Pulse-aware selection**: Parameters can be keyed by pulse class (`short_pulse`, `long_pulse`) when channels operate in different modes
3. **Stage composition**: Masks from all stages are OR-merged per channel with proper coordinate alignment
4. **Unfeasible propagation**: Each stage's unfeasible regions propagate through and are reported separately
5. **Serializable config**: Pipeline definition is a plain dict/JSON — enables reproducible processing and provenance tracking
6. **Frequency inheritance**: Missing frequency params inherit from a default key, reducing config verbosity

### Config-driven usage

```python
import json
from echopype.clean.pipeline import DenoisePipeline

# Load from JSON config file
with open("denoise_config.json") as f:
    config = json.load(f)

pipeline = DenoisePipeline.from_dict(config)
ds_masked, report = pipeline.run(ds_Sv)
```

## Implementation notes

- Each stage calls existing `echopype.clean` functions — no algorithm reimplementation
- Mask composition is lazy (dask-compatible) using broadcast-safe boolean OR
- Coordinate rebinding between stages prevents alignment issues from rolling-window operations
- Fully backward-compatible: existing standalone `mask_*` functions remain unchanged

## Alternative considered

An alternative is to only provide a `compose_masks(mask_list, method="or")` utility without the full pipeline. This is simpler but doesn't solve the per-channel parameter dispatch or configuration reproducibility problems. We believe the pipeline approach provides more value, but even the simpler `compose_masks` utility would be a welcome addition.

## References

- Current echopype denoising functions: `mask_impulse_noise`, `mask_transient_noise`, `remove_background_noise`, `mask_attenuated_signal`
- Current mask application: `echopype.mask.apply_mask`
