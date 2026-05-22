# Data Contract

This document defines the expected structure of exported training data.

## File Format

The preferred format for Phase 1 is HDF5 because it supports multiple named arrays and metadata in one file.

Alternative `.npy` tensor dumps are acceptable for quick experiments, but HDF5 should be the stable interchange format.

## HDF5 Layout

Recommended file layout:

```text
/features/world_normals              float16  [H, W, 3]
/features/depth_motion               float16  [H, W, 3]
/features/material                   uint8    [H, W, 2]
/features/temporal_luminance_var     float16  [H, W, 1]

/reference/rgb                       float16  [H, W, 3]
/labels/tile_class                   uint8    [H / T, W / T]
/labels/tile_budget                  uint8    [H / T, W / T]

/metadata/frame_id                   string or int
/metadata/scene_id                   string
/metadata/camera_id                  string or int
/metadata/tile_size                  int
/metadata/budget_classes             uint8    [6]
```

Where `T` is the chosen tile size.

## Feature Channels

Initial feature plan:

| Feature | Channels | Type | Notes |
| --- | ---: | --- | --- |
| World normal | 3 | FP16 | Expected normalized world-space normal |
| Depth | 1 | FP16 | Prefer linear depth |
| Motion | 2 | FP16 | Screen-space motion vector |
| Roughness | 1 | UNORM8 | Decode to `[0, 1]` for training |
| Metallic | 1 | UNORM8 | Decode to `[0, 1]` for training |
| Temporal luminance variance | 1 | FP16 | Rolling luminance variance |

This raw list totals 9 scalar channels. If the model target remains 7 input channels, the final packing must explicitly define which channels are included or compressed.

## Model Tensor Layout

PyTorch tensors should use channel-first format:

```text
[B, C, H / T, W / T]
```

The dataset is responsible for converting full-resolution feature buffers into tile-space tensors.

## Labels

The class-to-budget mapping is:

| Class index | Ray budget |
| ---: | ---: |
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |

`tile_class` stores the class index. `tile_budget` stores the actual ray count for debugging and renderer inspection.

## Validation Requirements

Every exported frame should pass these checks:

- Feature, reference, and label dimensions are mutually aligned.
- `H` and `W` are divisible by `tile_size`.
- No NaN or Inf values in floating point datasets.
- Normal vectors are either zero for invalid pixels or approximately unit length.
- Label classes are in `[0, 5]`.
- Metadata contains enough information to reproduce the frame.

