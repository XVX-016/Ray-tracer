"""
Phase 2 HDF5 streaming dataset for ray budget allocation.

Input samples are Phase 1 G-buffer tensors:
    x: [7, 1080, 1920] float32

Targets are Phase 1 tile labels:
    y: [120, 68] int64

Channel layout:
    0: normal_x
    1: normal_y
    2: normal_z
    3: depth
    4: albedo_r
    5: albedo_g
    6: albedo_b
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

IMG_H = 1080
IMG_W = 1920
GBUFFER_CHANNELS = 7
TILE_X = 120
TILE_Y = 68
NUM_CLASSES = 6

KEY_GBUFFERS = "gbuffers"
KEY_LABELS = "labels"


@dataclass(frozen=True)
class DepthStats:
    """Log-depth normalization statistics."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(f"Depth mean must be finite, got {self.mean}")
        if not math.isfinite(self.std) or self.std <= 0.0:
            raise ValueError(f"Depth std must be positive and finite, got {self.std}")


class GBufferNormalizer:
    """
    Normalize Phase 1 G-buffer channels into a stable network input range.

    Normals are mapped from [-1, 1] to [0, 1].
    Depth is clamped non-negative, log1p-compressed, then standardized.
    Albedo is clamped to [0, 1].
    """

    def __init__(self, depth_stats: DepthStats) -> None:
        self.depth_stats = depth_stats

    @staticmethod
    def fit_depth_stats(
        h5_path: str | Path,
        indices: Optional[Sequence[int]] = None,
        max_samples: int = 64,
        pixel_stride: int = 16,
        seed: int = 1234,
    ) -> DepthStats:
        """
        Estimate depth normalization stats from a bounded calibration pass.

        The pass reads only channel 3 and spatially subsamples depth values to
        avoid pulling many full 1080p maps into memory.
        """
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if pixel_stride <= 0:
            raise ValueError("pixel_stride must be positive")

        path = Path(h5_path)
        rng = np.random.default_rng(seed)

        with _open_hdf5_read(path) as hf:
            _validate_hdf5_schema(hf)
            total = int(hf[KEY_GBUFFERS].shape[0])
            source_indices = list(range(total)) if indices is None else list(indices)
            if not source_indices:
                raise ValueError("Cannot fit depth stats from an empty index set")

            n = min(max_samples, len(source_indices))
            chosen = rng.choice(np.asarray(source_indices), size=n, replace=False)
            chunks: list[np.ndarray] = []
            for idx in chosen:
                depth = hf[KEY_GBUFFERS][int(idx), 3, ::pixel_stride, ::pixel_stride]
                depth = np.asarray(depth, dtype=np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=1.0e6, neginf=0.0)
                chunks.append(np.log1p(np.clip(depth, 0.0, None)).reshape(-1))

        values = np.concatenate(chunks, axis=0)
        mean = float(values.mean(dtype=np.float64))
        std = float(values.std(dtype=np.float64))
        return DepthStats(mean=mean, std=max(std, 1.0e-6))

    def __call__(self, gbuffer: torch.Tensor) -> torch.Tensor:
        if gbuffer.shape != (GBUFFER_CHANNELS, IMG_H, IMG_W):
            raise ValueError(
                f"Expected gbuffer shape {(GBUFFER_CHANNELS, IMG_H, IMG_W)}, "
                f"got {tuple(gbuffer.shape)}"
            )
        if gbuffer.dtype != torch.float32:
            gbuffer = gbuffer.float()

        out = torch.empty_like(gbuffer, dtype=torch.float32)
        out[0:3] = (gbuffer[0:3].clamp(-1.0, 1.0) * 0.5) + 0.5
        out[3] = (
            torch.log1p(gbuffer[3].clamp_min(0.0)) - self.depth_stats.mean
        ) / self.depth_stats.std
        out[4:7] = gbuffer[4:7].clamp(0.0, 1.0)
        return out


class GBufferHDF5Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Streaming dataset for Phase 1 HDF5 exports.

    The HDF5 handle is opened lazily per worker with swmr=True. This avoids
    sharing an h5py handle across forked/spawned DataLoader workers.
    """

    def __init__(
        self,
        h5_path: str | Path,
        indices: Optional[Sequence[int]] = None,
        normalizer: Optional[GBufferNormalizer] = None,
        depth_calibration_samples: int = 64,
        transform: Optional[
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with _open_hdf5_read(self.h5_path) as hf:
            _validate_hdf5_schema(hf)
            total = int(hf[KEY_GBUFFERS].shape[0])

        self.indices = list(range(total)) if indices is None else [int(i) for i in indices]
        if not self.indices:
            raise ValueError("Dataset index set cannot be empty")
        if min(self.indices) < 0 or max(self.indices) >= total:
            raise IndexError(f"Indices must be within [0, {total - 1}]")

        self.normalizer = normalizer or GBufferNormalizer(
            GBufferNormalizer.fit_depth_stats(
                self.h5_path,
                indices=self.indices,
                max_samples=depth_calibration_samples,
            )
        )
        self.transform = transform
        self._hf: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        if item < 0 or item >= len(self.indices):
            raise IndexError(f"Index {item} out of range for dataset of size {len(self)}")

        file_index = self.indices[item]
        hf = self._get_hdf5()

        gbuffer_np = np.asarray(hf[KEY_GBUFFERS][file_index], dtype=np.float32)
        labels_np = np.asarray(hf[KEY_LABELS][file_index], dtype=np.int64)

        if gbuffer_np.shape != (GBUFFER_CHANNELS, IMG_H, IMG_W):
            raise ValueError(
                f"gbuffers[{file_index}] shape {gbuffer_np.shape}, "
                f"expected {(GBUFFER_CHANNELS, IMG_H, IMG_W)}"
            )
        if labels_np.shape != (TILE_X, TILE_Y):
            raise ValueError(
                f"labels[{file_index}] shape {labels_np.shape}, "
                f"expected {(TILE_X, TILE_Y)}"
            )

        labels_min = int(labels_np.min())
        labels_max = int(labels_np.max())
        if labels_min < 0 or labels_max >= NUM_CLASSES:
            raise ValueError(
                f"labels[{file_index}] out of range: min={labels_min}, max={labels_max}"
            )

        gbuffer = torch.from_numpy(np.ascontiguousarray(gbuffer_np))
        labels = torch.from_numpy(np.ascontiguousarray(labels_np))
        x = self.normalizer(gbuffer)

        if self.transform is not None:
            x, labels = self.transform(x, labels)

        assert x.shape == (GBUFFER_CHANNELS, IMG_H, IMG_W)
        assert x.dtype == torch.float32
        assert labels.shape == (TILE_X, TILE_Y)
        assert labels.dtype == torch.int64
        return x, labels

    def close(self) -> None:
        if self._hf is not None:
            self._hf.close()
            self._hf = None

    def _get_hdf5(self) -> h5py.File:
        if self._hf is None:
            self._hf = _open_hdf5_read(self.h5_path)
        return self._hf

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_hf"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_dataloaders(
    h5_path: str | Path,
    batch_size: int = 2,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    depth_calibration_samples: int = 64,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build deterministic train/validation DataLoaders over one HDF5 export.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    path = Path(h5_path)
    with _open_hdf5_read(path) as hf:
        _validate_hdf5_schema(hf)
        total = int(hf[KEY_GBUFFERS].shape[0])
    if total < 2 and val_fraction > 0.0:
        raise ValueError("Need at least 2 samples for a validation split")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(total).astype(int).tolist()
    n_val = 0 if val_fraction == 0.0 else max(1, int(round(total * val_fraction)))
    n_val = min(n_val, total - 1) if total > 1 else 0
    val_indices = shuffled[:n_val]
    train_indices = shuffled[n_val:]

    depth_stats = GBufferNormalizer.fit_depth_stats(
        path,
        indices=train_indices,
        max_samples=depth_calibration_samples,
        seed=seed,
    )
    normalizer = GBufferNormalizer(depth_stats)

    train_ds = GBufferHDF5Dataset(path, indices=train_indices, normalizer=normalizer)
    val_ds = (
        GBufferHDF5Dataset(path, indices=val_indices, normalizer=normalizer)
        if val_indices
        else None
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common) if val_ds is not None else None
    return train_loader, val_loader


def _open_hdf5_read(path: str | Path) -> h5py.File:
    try:
        return h5py.File(path, mode="r", swmr=True)
    except (OSError, ValueError):
        return h5py.File(path, mode="r")


def _validate_hdf5_schema(hf: h5py.File) -> None:
    if KEY_GBUFFERS not in hf:
        raise KeyError(f"Missing dataset '{KEY_GBUFFERS}'")
    if KEY_LABELS not in hf:
        raise KeyError(f"Missing dataset '{KEY_LABELS}'")

    g_shape = tuple(hf[KEY_GBUFFERS].shape)
    y_shape = tuple(hf[KEY_LABELS].shape)
    if len(g_shape) != 4 or g_shape[1:] != (GBUFFER_CHANNELS, IMG_H, IMG_W):
        raise ValueError(
            f"'{KEY_GBUFFERS}' shape {g_shape}, expected "
            f"[N, {GBUFFER_CHANNELS}, {IMG_H}, {IMG_W}]"
        )
    if len(y_shape) != 3 or y_shape[1:] != (TILE_X, TILE_Y):
        raise ValueError(
            f"'{KEY_LABELS}' shape {y_shape}, expected [N, {TILE_X}, {TILE_Y}]"
        )
    if g_shape[0] != y_shape[0]:
        raise ValueError(
            f"Sample count mismatch: gbuffers N={g_shape[0]}, labels N={y_shape[0]}"
        )
    if g_shape[0] <= 0:
        raise ValueError("HDF5 file contains zero samples")

