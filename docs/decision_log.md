# Decision Log

This document records project decisions and unresolved choices.

## Confirmed

### Ray Budget Classes

Use six discrete ray budget classes:

```text
{0, 1, 2, 4, 8, 16}
```

### Training Output

The model predicts tile-level logits:

```text
[B, 6, H / T, W / T]
```

### Reference Quality Target

Use high-sample ground truth renders for labelling:

```text
4096 spp
```

## Open

### Tile Size

Confirmed Phase 1 assumption:

```text
T = 16
```

Options:

- `8x8`: more granular, potentially better quality, higher scheduling pressure.
- `16x16`: coarser, safer for renderer scheduling, lower overhead.

### Tile Grid Layout

Phase 1 stores tile labels as:

```text
[tile_x, tile_y] = [120, 68]
```

Training code may transpose this to row-major `[68, 120]` at the PyTorch dataset boundary if needed, but the HDF5 contract remains `[120, 68]`.

### Feature Channel Count

The handoff mentions a model input of `7` channels, while the listed raw G-buffer fields expand to more scalar channels if unpacked directly.

This needs a final packing decision before implementing `dataset.py` and `model.py`.

### Data Generation Source

Two possible Phase 1 entry points exist:

- Vulkan G-buffer extraction first.
- Mitsuba/offline renderer first.

The documentation currently assumes offline generation can start first, while Vulkan integration follows once the data contract is stable.

## Proposed Defaults

Until explicitly changed:

- Tile size: `16x16`.
- Storage: HDF5.
- Model: compact U-Net baseline.
- First training loss: weighted cross entropy plus render-cost regularization.
- Deployment path: PyTorch -> ONNX -> TensorRT FP16.
