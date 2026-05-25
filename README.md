# Intelligent Ray Budget Allocation

Documentation-first project scaffold for an ML-guided ray budget allocator. The goal is to predict per-tile sample budgets from renderer G-buffer features, then feed those budgets back into a real-time Vulkan rendering path.

The current handover artifact is:

- `intelligent_ray_budget_allocation_handover.pdf`

## Project Goal

Build a two-phase pipeline:

1. **Phase 1: Data generation**
   - Render or extract G-buffer features.
   - Generate high-sample reference frames.
   - Export aligned feature and reference tensors.
   - Convert perceptual reconstruction error into tile-level ray budget labels.

2. **Phase 2: ML training and deployment**
   - Train a compact U-Net or FCN on tile-space tensors.
   - Predict one of six ray budget classes: `{0, 1, 2, 4, 8, 16}`.
   - Export the trained model to ONNX and TensorRT for renderer integration.

## Expected Pipeline

```text
Vulkan Render Pass
       |
       v
G-Buffer Extraction
       |-- World Normals
       |-- Depth + Motion
       |-- Roughness / Metallic
       |-- Temporal Luminance Variance
       v
Offline Exporter
       |
       v
.hdf5 / .npy tensor dumps
       |
       v
Ground Truth Renderer
       |
       v
4096 spp reference frames
       |
       v
U-Net / FCN
       |
       v
Softmax over {0, 1, 2, 4, 8, 16}
       |
       v
Tile ray budget map
```

## Documentation Map

- [Architecture](docs/architecture.md)
- [Data Contract](docs/data_contract.md)
- [Training Plan](docs/training_plan.md)
- [Implementation Roadmap](docs/roadmap.md)
- [Decision Log](docs/decision_log.md)

## Proposed Source Layout

```text
src/
  data/
    scene_config.py
    exporter.py
    tile_labeller.py
  training/
    dataset.py
    model.py
    loss.py
    train.py
    export.py
configs/
  dataset.yaml
  training.yaml
outputs/
  datasets/
  checkpoints/
  exports/
```

## Implemented Phase 1 Modules

The initial Phase 1 implementation lives in `src/data/`:

- `scene_config.py`: Mitsuba 3 dual-pass renderer and G-buffer packer.
- `tile_labeller.py`: perceptual error map generation and 16x16 tile labelling.
- `exporter.py`: HDF5 sequence writer and lightweight reader.

## Open Project Decisions

The main unresolved design choice is tile size:

- **8x8 tiles**: higher spatial granularity, more scheduling overhead, potentially better quality allocation.
- **16x16 tiles**: safer renderer scheduling, lower overhead, coarser allocation.

Documentation and code now use **16x16 tile-space** as the Phase 1 default. The exported label grid is `[120, 68]`, using horizontal tile index first and vertical tile index second.
