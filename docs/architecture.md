# Architecture

This document describes the end-to-end architecture for intelligent ray budget allocation.

## System Overview

The system learns a mapping from cheap per-frame rendering features to a tile-level ray budget map. The map selects one of six sample count classes for each screen tile:

```text
{0, 1, 2, 4, 8, 16}
```

The trained model should run in the renderer after G-buffer generation and before expensive ray work is scheduled.

## Phase 1: Data Generation

### Renderer Feature Capture

The renderer provides aligned per-frame buffers:

- World normals
- Depth
- Motion vectors
- Roughness
- Metallic
- Temporal luminance variance

These buffers are exported into a stable tensor layout for offline training.

### Reference Rendering

Each training frame needs a high-quality reference render, currently planned as:

```text
4096 samples per pixel
```

The reference frame is used to compute perceptual error and derive per-tile labels.

### Tile Labelling

The labeller compares low-budget candidate renders against the reference image, aggregates error per tile, and assigns the cheapest acceptable ray budget class.

The intended output is a tile grid:

```text
[ceil(W / tile_size), ceil(H / tile_size)]
```

Each tile stores a class index corresponding to `{0, 1, 2, 4, 8, 16}`.
For the Phase 1 exporter, 1920x1080 with 16x16 tiles is stored as `[120, 68]`, with horizontal tile coordinate first.

## Phase 2: ML Training

### Dataset

The dataset consumes exported `.hdf5` or `.npy` dumps and produces:

```text
inputs:  [B, C, H / tile_size, W / tile_size]
labels:  [B, ceil(W / tile_size), ceil(H / tile_size)]
```

The initial expected input channel count is `7`, subject to final G-buffer packing.

### Model

The first model target is a compact U-Net or FCN operating in tile space:

```text
input:   [B, 7, H / 16, W / 16]
output:  [B, 6, H / 16, W / 16]
```

The output dimension `6` corresponds to the six ray budget classes.

### Loss

The training objective combines:

- Classification loss over tile budget classes.
- Image-space perceptual pressure from SSIM.
- L1 reconstruction pressure.
- Render-cost weighting to prefer cheaper budgets when quality is comparable.

## Deployment

The deployment target is:

```text
PyTorch checkpoint -> ONNX -> TensorRT engine
```

The intended runtime precision is FP16, with INT8 as a later optimization path if calibration data is stable.

## Runtime Integration

At runtime, the renderer should:

1. Generate the required G-buffer features.
2. Downsample or pack them into tile-space model input.
3. Run model inference.
4. Convert class logits into ray budget counts.
5. Schedule ray work per tile.
6. Apply temporal/spatial stabilization if the budget map flickers.
