# Training Plan

This document describes the Phase 2 training path.

## Dataset

`dataset.py` should implement a PyTorch `Dataset` that reads exported HDF5 files and returns:

```python
{
    "features": FloatTensor[C, Ht, Wt],
    "labels": LongTensor[Wt, Ht],
    "metadata": dict,
}
```

Where:

```text
Ht = ceil(H / tile_size)
Wt = ceil(W / tile_size)
```

## Preprocessing

Recommended preprocessing:

- Decode UNORM material channels to float `[0, 1]`.
- Normalize depth per scene or camera range.
- Preserve motion vector sign.
- Clamp extreme temporal luminance variance before normalization.
- Aggregate full-resolution features to tile-space with mean pooling, except normals, which may need normalized average pooling.

## Model

Initial model:

- Compact U-Net or FCN.
- Input channels: `7` once final packing is confirmed.
- Output channels: `6`.
- Output resolution: tile-space.
- Phase 1 label exports use `[Wt, Ht] = [120, 68]`; training code may transpose to framework-preferred `[Ht, Wt]` if documented at the dataset boundary.
- Activation: raw logits for training, softmax only for inference/debugging.

## Loss

Initial loss components:

- Cross entropy over tile budget class.
- Cost regularization to prefer lower ray counts.
- Optional SSIM/L1 proxy if reconstructed candidate images are available during training.

Because labels already encode quality-vs-cost decisions, start with classification plus class/cost weighting before adding expensive image-space loss terms.

## Metrics

Track:

- Tile classification accuracy.
- Mean absolute budget error.
- Over-budget and under-budget rates.
- Estimated render cost ratio against full budget.
- Perceptual error against reference if candidate renders are available.

## Checkpoints

Save:

```text
outputs/checkpoints/
  latest.pt
  best_val_loss.pt
  best_cost_quality.pt
```

Each checkpoint should include:

- Model weights.
- Optimizer state.
- Epoch and global step.
- Training config.
- Class-to-budget mapping.
- Feature channel contract version.

## Export

`export.py` should support:

```text
PyTorch .pt -> ONNX -> TensorRT .engine
```

Export validation should compare PyTorch and ONNX/TensorRT logits on a fixed batch before accepting the artifact.
