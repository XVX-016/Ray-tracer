# Implementation Roadmap

## Phase 0: Documentation and Contracts

- Create project documentation.
- Define feature channel contract.
- Decide tile size.
- Decide final storage format.
- Define class-to-budget mapping.

## Phase 1: Data Generation

1. Implement `scene_config.py`.
   - Configure Mitsuba 3 scenes.
   - Render G-buffer style feature outputs.
   - Render high-sample reference frames.

2. Implement `exporter.py`.
   - Write HDF5 outputs.
   - Store features, reference, labels, and metadata.
   - Validate shape and dtype consistency.

3. Implement `tile_labeller.py`.
   - Compute perceptual or L1/SSIM error maps.
   - Aggregate error by tile.
   - Assign cheapest acceptable budget class.

## Phase 2: Model Training

1. Implement `dataset.py`.
   - Read HDF5 exports.
   - Normalize and pack feature channels.
   - Return tile-space tensors.

2. Implement `model.py`.
   - Add compact U-Net or FCN baseline.
   - Keep output shape `[B, 6, Ht, Wt]`.

3. Implement `loss.py`.
   - Start with cross entropy and cost weighting.
   - Add SSIM/L1 terms when image reconstruction targets are available.

4. Implement `train.py`.
   - Mixed precision training.
   - Validation split.
   - Checkpointing.
   - Metric logging.

5. Implement `export.py`.
   - ONNX export.
   - TensorRT engine generation.
   - Numeric parity checks.

## Phase 3: Renderer Integration

- Add Vulkan-side feature extraction.
- Add model inference dispatch.
- Convert logits to per-tile ray budgets.
- Add temporal stability controls.
- Profile GPU cost and quality tradeoffs.

## Suggested First Build Order

Start with the data contract and exporter before the model. A stable dataset format will make the renderer, labeller, and trainer independently testable.

