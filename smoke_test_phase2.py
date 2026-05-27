"""
Phase 2 Block 1 smoke test.

Run from project root:
    python smoke_test_phase2.py --batch 2
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from src.training.dataset import (
    GBUFFER_CHANNELS,
    IMG_H,
    IMG_W,
    KEY_GBUFFERS,
    KEY_LABELS,
    NUM_CLASSES,
    TILE_X,
    TILE_Y,
    GBufferHDF5Dataset,
    build_dataloaders,
)
from src.training.model import RayBudgetFCN


def make_synthetic_hdf5(path: Path, samples: int) -> None:
    rng = np.random.default_rng(7)
    with h5py.File(path, "w", libver="latest") as hf:
        gbuffers = hf.create_dataset(
            KEY_GBUFFERS,
            shape=(samples, GBUFFER_CHANNELS, IMG_H, IMG_W),
            dtype=np.float32,
            chunks=(1, GBUFFER_CHANNELS, 270, 480),
        )
        labels = hf.create_dataset(
            KEY_LABELS,
            shape=(samples, TILE_X, TILE_Y),
            dtype=np.int64,
            chunks=(1, TILE_X, TILE_Y),
        )

        for i in range(samples):
            sample = np.empty((GBUFFER_CHANNELS, IMG_H, IMG_W), dtype=np.float32)
            sample[0:3] = rng.uniform(-1.0, 1.0, size=(3, IMG_H, IMG_W)).astype(np.float32)
            sample[3] = rng.exponential(scale=4.0, size=(IMG_H, IMG_W)).astype(np.float32)
            sample[4:7] = rng.uniform(0.0, 1.0, size=(3, IMG_H, IMG_W)).astype(np.float32)
            gbuffers[i] = sample
            labels[i] = rng.integers(0, NUM_CLASSES, size=(TILE_X, TILE_Y), dtype=np.int64)

        hf.swmr_mode = True


def run(batch_size: int) -> None:
    print("Phase 2 Block 1 Smoke Test")
    print("=" * 32)

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "phase2_smoke.h5"
        samples = max(batch_size * 3, 6)

        print("[1/5] Creating synthetic HDF5...")
        make_synthetic_hdf5(h5_path, samples)
        print(f"      ok: {h5_path}")

        print("[2/5] Reading one dataset sample...")
        dataset = GBufferHDF5Dataset(h5_path, depth_calibration_samples=samples)
        x, y = dataset[0]
        assert x.shape == (GBUFFER_CHANNELS, IMG_H, IMG_W), tuple(x.shape)
        assert y.shape == (TILE_X, TILE_Y), tuple(y.shape)
        assert x.dtype == torch.float32, x.dtype
        assert y.dtype == torch.int64, y.dtype
        assert torch.isfinite(x).all()
        assert int(y.min()) >= 0 and int(y.max()) < NUM_CLASSES
        print(f"      ok: x={tuple(x.shape)} y={tuple(y.shape)}")

        print("[3/5] Building DataLoader batch...")
        train_loader, _ = build_dataloaders(
            h5_path,
            batch_size=batch_size,
            val_fraction=0.25,
            num_workers=0,
            pin_memory=False,
            depth_calibration_samples=samples,
        )
        bx, by = next(iter(train_loader))
        assert bx.shape == (batch_size, GBUFFER_CHANNELS, IMG_H, IMG_W), tuple(bx.shape)
        assert by.shape == (batch_size, TILE_X, TILE_Y), tuple(by.shape)
        print(f"      ok: bx={tuple(bx.shape)} by={tuple(by.shape)}")

        print("[4/5] Running model forward pass...")
        model = RayBudgetFCN().eval()
        params = sum(p.numel() for p in model.parameters())
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(bx)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        expected = (batch_size, NUM_CLASSES, TILE_X, TILE_Y)
        assert logits.shape == expected, tuple(logits.shape)
        assert logits.dtype == torch.float32
        print(f"      ok: logits={tuple(logits.shape)} params={params:,} cpu_ms={elapsed_ms:.1f}")

        print("[5/5] Checking output sanity...")
        assert torch.isfinite(logits).all()
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)
        print("      ok: finite logits, softmax sums to 1")

    print("=" * 32)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    run(batch_size=args.batch)

