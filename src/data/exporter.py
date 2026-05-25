"""
exporter.py
───────────
Phase 1 – HDF5 Dataset Serialiser
Project Alpha-Ray | Intelligent Ray Budget Allocation

Responsibilities:
  1. Receive fully processed outputs from scene_config.py (RenderResult)
     and tile_labeller.py (TileLabelResult) and serialise them into a
     structured, stream-optimised HDF5 dataset file.

  2. G-Buffer tensor shape: [7, 1080, 1920], float32
       Channels: [N_x, N_y, N_z, Depth, Alb_R, Alb_G, Alb_B]

  3. Target label shape:  [120, 68], int64 (class indices 0–5)

  4. Additional per-sample tensors stored for diagnostic use:
       gt_rgb      : [3, 1080, 1920], float32  – ground-truth beauty
       noisy_rgb   : [3, 1080, 1920], float32  – 1-spp noisy beauty

  5. File structure uses chunked, GZIP-compressed HDF5 datasets with
     pre-allocated extendable axes, supporting O(1) sequential appends
     across arbitrarily long capture sessions (thousands of frames).

  6. A dedicated '/metadata' group tracks per-sample provenance:
       sequence_index, scene_name, timestamp, tile_error statistics.

  7. All file handles are managed via context managers and explicit
     flush/close calls to prevent handle leakage across long sessions.
     Memory is released deterministically after each serialisation call.

HDF5 layout (all datasets resizable on axis 0):
  /gbuffers          [N, 7,   1080, 1920]  float32  chunks=(1, 7, 270, 480)
  /labels            [N, 120,   68]        int64    chunks=(1, 120, 68)
  /gt_rgb            [N, 3,   1080, 1920]  float32  chunks=(1, 3, 270, 480)
  /noisy_rgb         [N, 3,   1080, 1920]  float32  chunks=(1, 3, 270, 480)
  /metadata/
      sequence_index [N]                  int64
      scene_names    [N]                  bytes (variable-length string)
      timestamps     [N]                  float64 (POSIX UTC)
      tile_err_mean  [N]                  float32
      tile_err_max   [N]                  float32

Author : Principal Graphics & AI Engineer – Project Alpha-Ray
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional

import h5py
import numpy as np
import torch

log = logging.getLogger(__name__)

# ── Dimension constants (must match scene_config.py and tile_labeller.py) ─────
WIDTH: int = 1920
HEIGHT: int = 1080
TILE_COLS: int = 120
TILE_ROWS: int = 68
GBUFFER_CHANNELS: int = 7
RGB_CHANNELS: int = 3
NUM_LABEL_CLASSES: int = 6

# ── HDF5 dataset keys ─────────────────────────────────────────────────────────
KEY_GBUFFERS: str = "gbuffers"
KEY_LABELS: str = "labels"
KEY_GT_RGB: str = "gt_rgb"
KEY_NOISY_RGB: str = "noisy_rgb"

META_GROUP: str = "metadata"
KEY_SEQ_IDX: str = f"{META_GROUP}/sequence_index"
KEY_SCENE_NAMES: str = f"{META_GROUP}/scene_names"
KEY_TIMESTAMPS: str = f"{META_GROUP}/timestamps"
KEY_TILE_ERR_MEAN: str = f"{META_GROUP}/tile_err_mean"
KEY_TILE_ERR_MAX: str = f"{META_GROUP}/tile_err_max"

# ── Compression and chunk configuration ───────────────────────────────────────
# Chunks are sized to align with typical sequential access patterns:
# one sample at a time, spatially tiled for cache efficiency.
# Chunk spatial dims are WIDTH/4 × HEIGHT/4 to keep chunk size ≈ 4MB per
# channel, staying within h5py's recommended 1–8MB per chunk range.
_CHUNK_SPATIAL_W: int = WIDTH // 4    # 480
_CHUNK_SPATIAL_H: int = HEIGHT // 4   # 270
_COMPRESSION: str = "gzip"
_COMPRESSION_OPTS: int = 4            # gzip level 4: good balance of speed vs ratio


# ── Schema definition ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSchema:
    """
    Defines the HDF5 schema: dtype, chunk layout, and max-shape for each dataset.
    max_shape uses None on axis 0 to allow unlimited appends.
    """
    name: str
    dtype: np.dtype
    sample_shape: tuple          # shape of one sample (without batch dimension)
    chunk_shape: tuple           # chunk shape INCLUDING batch axis (size 1)


DATASET_SCHEMAS: List[DatasetSchema] = [
    DatasetSchema(
        name=KEY_GBUFFERS,
        dtype=np.float32,
        sample_shape=(GBUFFER_CHANNELS, HEIGHT, WIDTH),
        chunk_shape=(1, GBUFFER_CHANNELS, _CHUNK_SPATIAL_H, _CHUNK_SPATIAL_W),
    ),
    DatasetSchema(
        name=KEY_LABELS,
        dtype=np.int64,
        sample_shape=(TILE_COLS, TILE_ROWS),
        chunk_shape=(1, TILE_COLS, TILE_ROWS),
    ),
    DatasetSchema(
        name=KEY_GT_RGB,
        dtype=np.float32,
        sample_shape=(RGB_CHANNELS, HEIGHT, WIDTH),
        chunk_shape=(1, RGB_CHANNELS, _CHUNK_SPATIAL_H, _CHUNK_SPATIAL_W),
    ),
    DatasetSchema(
        name=KEY_NOISY_RGB,
        dtype=np.float32,
        sample_shape=(RGB_CHANNELS, HEIGHT, WIDTH),
        chunk_shape=(1, RGB_CHANNELS, _CHUNK_SPATIAL_H, _CHUNK_SPATIAL_W),
    ),
]

METADATA_SCHEMAS: List[DatasetSchema] = [
    DatasetSchema(
        name=KEY_SEQ_IDX,
        dtype=np.int64,
        sample_shape=(),
        chunk_shape=(256,),
    ),
    DatasetSchema(
        name=KEY_TIMESTAMPS,
        dtype=np.float64,
        sample_shape=(),
        chunk_shape=(256,),
    ),
    DatasetSchema(
        name=KEY_TILE_ERR_MEAN,
        dtype=np.float32,
        sample_shape=(),
        chunk_shape=(256,),
    ),
    DatasetSchema(
        name=KEY_TILE_ERR_MAX,
        dtype=np.float32,
        sample_shape=(),
        chunk_shape=(256,),
    ),
]


# ── Tensor → NumPy conversion with explicit validation ───────────────────────

def _to_numpy_validated(
    tensor: torch.Tensor,
    expected_shape: tuple,
    expected_dtype: np.dtype,
    name: str,
) -> np.ndarray:
    """
    Convert a CPU torch.Tensor to a NumPy array after validating shape and dtype.
    Raises ValueError on any mismatch.

    Notes:
    - If tensor is on CUDA, it is moved to CPU first (.cpu()).
    - Contiguous layout is enforced to avoid stride-related copy issues in HDF5.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"[{name}] Shape mismatch: expected {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )

    arr: np.ndarray = tensor.numpy()

    # Cast dtype if needed (e.g. torch.long → np.int64).
    if arr.dtype != expected_dtype:
        arr = arr.astype(expected_dtype, copy=False)

    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    return arr


# ── HDF5 file initialisation ──────────────────────────────────────────────────

def _initialise_hdf5(
    hf: h5py.File,
    compression: str = _COMPRESSION,
    compression_opts: int = _COMPRESSION_OPTS,
) -> None:
    """
    Create all extendable datasets and groups in an empty HDF5 file.
    Datasets are initialised with 0 samples and max_shape=(None, ...) to
    allow unlimited resizable appends.

    Parameters
    ----------
    hf               : h5py.File  – opened file handle (must be empty or new)
    compression      : str        – HDF5 compression filter name
    compression_opts : int        – compression filter parameter (gzip level)
    """
    # ── Main tensor datasets ──────────────────────────────────────────────────
    for schema in DATASET_SCHEMAS:
        initial_shape = (0,) + schema.sample_shape
        max_shape = (None,) + schema.sample_shape
        hf.create_dataset(
            schema.name,
            shape=initial_shape,
            maxshape=max_shape,
            dtype=schema.dtype,
            chunks=schema.chunk_shape,
            compression=compression,
            compression_opts=compression_opts,
            # shuffle filter improves gzip compression on float data.
            shuffle=True,
        )
        log.debug("Created HDF5 dataset '%s' | chunks=%s", schema.name, schema.chunk_shape)

    # ── Metadata group ────────────────────────────────────────────────────────
    hf.require_group(META_GROUP)

    for schema in METADATA_SCHEMAS:
        initial_shape = (0,)
        max_shape = (None,)
        hf.create_dataset(
            schema.name,
            shape=initial_shape,
            maxshape=max_shape,
            dtype=schema.dtype,
            chunks=schema.chunk_shape,
            compression=compression,
            compression_opts=compression_opts,
        )

    # Variable-length UTF-8 string dataset for scene names.
    vlen_str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset(
        KEY_SCENE_NAMES,
        shape=(0,),
        maxshape=(None,),
        dtype=vlen_str_dtype,
        chunks=(256,),
    )

    # ── File-level attributes ─────────────────────────────────────────────────
    hf.attrs["created_at"] = time.time()
    hf.attrs["version"] = "1.0"
    hf.attrs["project"] = "alpha-ray"
    hf.attrs["gbuffer_channels"] = "N_x,N_y,N_z,Depth,Alb_R,Alb_G,Alb_B"
    hf.attrs["tile_size"] = 16
    hf.attrs["tile_grid"] = f"{TILE_COLS}x{TILE_ROWS}"
    hf.attrs["num_label_classes"] = NUM_LABEL_CLASSES
    hf.attrs["ray_budget_options"] = np.array([0, 1, 2, 4, 8, 16], dtype=np.int32)
    hf.attrs["image_width"] = WIDTH
    hf.attrs["image_height"] = HEIGHT

    log.info("HDF5 file initialised with %d datasets.", len(DATASET_SCHEMAS) + len(METADATA_SCHEMAS) + 1)


def _append_row(
    dataset: h5py.Dataset,
    data: np.ndarray,
) -> None:
    """
    Extend a resizable HDF5 dataset by one sample along axis 0 and write data.
    This is an O(chunk_size) operation — each append writes exactly one chunk.

    Parameters
    ----------
    dataset : h5py.Dataset  – resizable HDF5 dataset (maxshape[0] = None)
    data    : np.ndarray    – data for this one sample, shape = sample_shape
    """
    current_n = dataset.shape[0]
    new_n = current_n + 1

    if data.ndim == 0:
        # Scalar metadata fields.
        dataset.resize((new_n,))
        dataset[current_n] = data
    else:
        new_shape = (new_n,) + data.shape
        dataset.resize(new_shape)
        dataset[current_n] = data


# ── Main exporter class ───────────────────────────────────────────────────────

class HDF5Exporter:
    """
    Manages an open HDF5 dataset file for sequential sample export.

    Designed for long-running sessions (thousands of frames). The file is
    opened once, kept open across many append() calls, and explicitly
    closed via close() or by using the exporter as a context manager.

    Usage — context manager (recommended):
    ----------------------------------------
    with HDF5Exporter("dataset.h5") as exporter:
        for render_result, label_result in pipeline:
            exporter.append(render_result, label_result)

    Usage — manual lifecycle:
    ----------------------------------------
    exporter = HDF5Exporter("dataset.h5")
    exporter.open()
    exporter.append(render_result, label_result)
    exporter.close()

    Parameters
    ----------
    output_path      : Path | str  – destination .h5 file path
    overwrite        : bool        – if True, existing file is replaced.
                                    If False and file exists, appends to it.
    flush_interval   : int         – flush to disk every N samples.
                                    Lower values = more durability, more I/O.
    compression      : str         – HDF5 compression filter.
    compression_opts : int         – filter parameter.
    """

    def __init__(
        self,
        output_path: "Path | str",
        overwrite: bool = False,
        flush_interval: int = 10,
        compression: str = _COMPRESSION,
        compression_opts: int = _COMPRESSION_OPTS,
    ) -> None:
        self.output_path = Path(output_path)
        self.overwrite = overwrite
        self.flush_interval = flush_interval
        self.compression = compression
        self.compression_opts = compression_opts

        self._hf: Optional[h5py.File] = None
        self._sample_count: int = 0
        self._session_start: float = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> "HDF5Exporter":
        """
        Open the HDF5 file. If overwrite=True or file does not exist, creates
        a new file and initialises schema. Otherwise opens for appending.
        """
        if self._hf is not None:
            raise RuntimeError("HDF5Exporter is already open. Call close() first.")

        if self.output_path.exists() and not self.overwrite:
            log.info("Appending to existing HDF5: %s", self.output_path)
            self._hf = h5py.File(self.output_path, mode="a")
            # Recover sample count from the file's current state.
            self._sample_count = int(self._hf[KEY_GBUFFERS].shape[0])
            log.info("Resuming from sample index %d.", self._sample_count)
        else:
            if self.output_path.exists() and self.overwrite:
                log.warning("Overwriting existing file: %s", self.output_path)
                self.output_path.unlink()

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Creating new HDF5 dataset: %s", self.output_path)
            self._hf = h5py.File(self.output_path, mode="w")
            _initialise_hdf5(self._hf, self.compression, self.compression_opts)
            self._sample_count = 0

        return self

    def close(self) -> None:
        """
        Flush all pending writes and close the HDF5 file handle.
        Safe to call multiple times (idempotent).
        """
        if self._hf is not None:
            try:
                self._hf.flush()
                # Update session statistics in file attributes.
                self._hf.attrs["total_samples"] = self._sample_count
                self._hf.attrs["last_updated_at"] = time.time()
                self._hf.close()
                log.info(
                    "HDF5Exporter closed. Total samples: %d. File: %s",
                    self._sample_count, self.output_path,
                )
            except Exception as exc:
                log.error("Error during HDF5 close: %s", exc)
                raise
            finally:
                self._hf = None

    def __enter__(self) -> "HDF5Exporter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        # Do not suppress exceptions.
        return False

    def __del__(self) -> None:
        """Defensive finaliser: close handle if user forgot to call close()."""
        if self._hf is not None:
            log.warning(
                "HDF5Exporter was not explicitly closed. "
                "Closing in __del__ — this may lose unflushed data. "
                "Always use 'with HDF5Exporter(...) as exporter:' or call close()."
            )
            self.close()

    # ── Core append operation ─────────────────────────────────────────────────

    def append(
        self,
        gbuffer: torch.Tensor,
        class_labels: torch.Tensor,
        gt_rgb: torch.Tensor,
        noisy_rgb: torch.Tensor,
        tile_errors: torch.Tensor,
        scene_name: str = "unknown",
    ) -> int:
        """
        Serialise one complete (G-Buffer, label, GT, noisy) sample into the HDF5 file.

        Parameters
        ----------
        gbuffer       : torch.Tensor [7, 1080, 1920], float32
        class_labels  : torch.Tensor [120, 68],       int64  (class indices 0–5)
        gt_rgb        : torch.Tensor [3, 1080, 1920], float32
        noisy_rgb     : torch.Tensor [3, 1080, 1920], float32
        tile_errors   : torch.Tensor [120, 68],       float32  (raw pre-quantized)
        scene_name    : str   – human-readable scene identifier for metadata

        Returns
        -------
        int : the sequence index of this sample (0-based)
        """
        if self._hf is None:
            raise RuntimeError("HDF5Exporter is not open. Call open() or use 'with' block.")

        # ── Convert all tensors to validated NumPy arrays. ─────────────────
        gbuffer_np = _to_numpy_validated(
            gbuffer,
            expected_shape=(GBUFFER_CHANNELS, HEIGHT, WIDTH),
            expected_dtype=np.float32,
            name="gbuffer",
        )
        labels_np = _to_numpy_validated(
            class_labels,
            expected_shape=(TILE_COLS, TILE_ROWS),
            expected_dtype=np.int64,
            name="class_labels",
        )
        gt_np = _to_numpy_validated(
            gt_rgb,
            expected_shape=(RGB_CHANNELS, HEIGHT, WIDTH),
            expected_dtype=np.float32,
            name="gt_rgb",
        )
        noisy_np = _to_numpy_validated(
            noisy_rgb,
            expected_shape=(RGB_CHANNELS, HEIGHT, WIDTH),
            expected_dtype=np.float32,
            name="noisy_rgb",
        )

        # ── Compute per-sample metadata statistics. ─────────────────────────
        tile_err_np = _to_numpy_validated(
            tile_errors,
            expected_shape=(TILE_COLS, TILE_ROWS),
            expected_dtype=np.float32,
            name="tile_errors",
        )
        tile_err_mean = float(tile_err_np.mean())
        tile_err_max = float(tile_err_np.max())
        timestamp = time.time()
        seq_idx = self._sample_count

        # ── Append to each dataset in the file. ─────────────────────────────
        _append_row(self._hf[KEY_GBUFFERS], gbuffer_np)
        _append_row(self._hf[KEY_LABELS], labels_np)
        _append_row(self._hf[KEY_GT_RGB], gt_np)
        _append_row(self._hf[KEY_NOISY_RGB], noisy_np)

        # Metadata scalars.
        _append_row(self._hf[KEY_SEQ_IDX], np.int64(seq_idx))
        _append_row(self._hf[KEY_TIMESTAMPS], np.float64(timestamp))
        _append_row(self._hf[KEY_TILE_ERR_MEAN], np.float32(tile_err_mean))
        _append_row(self._hf[KEY_TILE_ERR_MAX], np.float32(tile_err_max))

        # Variable-length scene name string.
        ds_names = self._hf[KEY_SCENE_NAMES]
        ds_names.resize((seq_idx + 1,))
        ds_names[seq_idx] = scene_name

        self._sample_count += 1

        # ── Periodic flush to guard against data loss. ───────────────────────
        if self._sample_count % self.flush_interval == 0:
            self._hf.flush()
            log.debug("HDF5 flushed at sample %d.", self._sample_count)

        # ── Deterministic memory release. ────────────────────────────────────
        # Explicitly delete NumPy views before returning to prevent
        # reference cycles keeping the underlying buffers alive.
        del gbuffer_np, labels_np, gt_np, noisy_np, tile_err_np
        gc.collect()

        log.info(
            "Sample %d written | scene='%s' | tile_err_mean=%.4f | tile_err_max=%.4f",
            seq_idx, scene_name, tile_err_mean, tile_err_max,
        )
        return seq_idx

    # ── Convenience wrapper for RenderResult / TileLabelResult ───────────────

    def append_from_results(
        self,
        render_result: "RenderResult",
        label_result: "TileLabelResult",
    ) -> int:
        """
        Convenience overload that accepts the dataclass outputs from
        scene_config.SceneRenderer and tile_labeller.TileLabeller directly.

        Parameters
        ----------
        render_result : scene_config.RenderResult
        label_result  : tile_labeller.TileLabelResult

        Returns
        -------
        int : sequence index of the written sample
        """
        return self.append(
            gbuffer=render_result.gbuffer,
            class_labels=label_result.class_labels,
            gt_rgb=render_result.gt_rgb,
            noisy_rgb=render_result.noisy_rgb,
            tile_errors=label_result.tile_errors,
            scene_name=render_result.scene_name,
        )

    # ── Inspection utilities ──────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Return a dict summarising the current state of the open HDF5 file.
        Safe to call at any point during an active session.
        """
        if self._hf is None:
            return {"status": "closed", "path": str(self.output_path)}

        return {
            "status": "open",
            "path": str(self.output_path),
            "total_samples": self._sample_count,
            "gbuffers_shape": tuple(self._hf[KEY_GBUFFERS].shape),
            "labels_shape": tuple(self._hf[KEY_LABELS].shape),
            "file_size_mb": round(self.output_path.stat().st_size / (1024 ** 2), 2)
            if self.output_path.exists()
            else None,
        }


# ── Standalone dataset reader (for DataLoader use in Phase 2) ─────────────────

class HDF5DatasetReader:
    """
    Read-only accessor for a completed HDF5 dataset produced by HDF5Exporter.
    Intended for use by the Phase 2 PyTorch Dataset class.

    Returns per-sample tensors lazily; does not load the full file into memory.

    Usage
    -----
    reader = HDF5DatasetReader("dataset.h5")
    n = len(reader)
    gbuffer, label = reader[0]   # torch tensors, float32 / int64
    reader.close()
    """

    def __init__(self, path: "Path | str") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")
        # Open in SWMR (Single Writer Multiple Readers) mode for safe
        # concurrent reads during training while a capture session is active.
        try:
            self._hf = h5py.File(self.path, mode="r", swmr=True)
        except Exception:
            # SWMR requires the file was written with SWMR=True; fall back.
            self._hf = h5py.File(self.path, mode="r")
        self._n: int = int(self._hf[KEY_GBUFFERS].shape[0])
        log.info("HDF5DatasetReader opened: %s | %d samples", self.path, self._n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one (gbuffer, label) pair by sequential index.

        Returns
        -------
        gbuffer : torch.Tensor [7, 1080, 1920], float32
        label   : torch.Tensor [120, 68],       int64
        """
        if index < 0 or index >= self._n:
            raise IndexError(
                f"Sample index {index} out of range [0, {self._n - 1}]."
            )

        # Read via NumPy; HDF5 returns a numpy array slice.
        gbuffer_np: np.ndarray = self._hf[KEY_GBUFFERS][index]   # [7, H, W]
        label_np: np.ndarray = self._hf[KEY_LABELS][index]       # [COLS, ROWS]

        return (
            torch.from_numpy(np.ascontiguousarray(gbuffer_np)),
            torch.from_numpy(np.ascontiguousarray(label_np)),
        )

    def get_metadata(self, index: int) -> dict:
        """Return metadata dict for a given sample index."""
        return {
            "sequence_index": int(self._hf[KEY_SEQ_IDX][index]),
            "scene_name": str(self._hf[KEY_SCENE_NAMES][index]),
            "timestamp": float(self._hf[KEY_TIMESTAMPS][index]),
            "tile_err_mean": float(self._hf[KEY_TILE_ERR_MEAN][index]),
            "tile_err_max": float(self._hf[KEY_TILE_ERR_MAX][index]),
        }

    def close(self) -> None:
        if self._hf is not None:
            self._hf.close()
            self._hf = None
            log.info("HDF5DatasetReader closed: %s", self.path)

    def __enter__(self) -> "HDF5DatasetReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        if self._hf is not None:
            self.close()


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("[exporter] Running smoke test with synthetic tensors...")

    torch.manual_seed(0)

    def _make_synthetic_sample(scene_id: int) -> tuple:
        gbuffer = torch.rand(GBUFFER_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
        labels = torch.randint(0, NUM_LABEL_CLASSES, (TILE_COLS, TILE_ROWS), dtype=torch.long)
        gt_rgb = torch.rand(RGB_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
        noisy_rgb = torch.rand(RGB_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
        tile_errors = torch.rand(TILE_COLS, TILE_ROWS, dtype=torch.float32)
        scene_name = f"synthetic_scene_{scene_id:04d}"
        return gbuffer, labels, gt_rgb, noisy_rgb, tile_errors, scene_name

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_dataset.h5"
        NUM_SAMPLES = 3

        print(f"  Writing {NUM_SAMPLES} synthetic samples to: {h5_path}")

        with HDF5Exporter(h5_path, overwrite=True, flush_interval=2) as exporter:
            for i in range(NUM_SAMPLES):
                gbuf, labs, gt, noisy, terr, sname = _make_synthetic_sample(i)
                idx = exporter.append(
                    gbuffer=gbuf,
                    class_labels=labs,
                    gt_rgb=gt,
                    noisy_rgb=noisy,
                    tile_errors=terr,
                    scene_name=sname,
                )
                print(f"    Written sample index={idx} | scene='{sname}'")
            summary = exporter.summary()
            print(f"  Summary while open: {summary}")

        # Re-open as reader and verify round-trip.
        print("  Verifying round-trip read...")
        with HDF5DatasetReader(h5_path) as reader:
            print(f"  Dataset length: {len(reader)}")
            for i in range(NUM_SAMPLES):
                gbuf_r, lab_r = reader[i]
                meta = reader.get_metadata(i)
                assert gbuf_r.shape == (GBUFFER_CHANNELS, HEIGHT, WIDTH), f"gbuffer shape @ {i}: {gbuf_r.shape}"
                assert lab_r.shape == (TILE_COLS, TILE_ROWS), f"label shape @ {i}: {lab_r.shape}"
                assert lab_r.dtype == torch.int64, f"label dtype @ {i}: {lab_r.dtype}"
                print(
                    f"    Sample {i}: gbuffer={tuple(gbuf_r.shape)} "
                    f"label={tuple(lab_r.shape)} "
                    f"scene='{meta['scene_name']}' "
                    f"tile_err_mean={meta['tile_err_mean']:.4f}"
                )

    print("[exporter] Smoke test PASSED")
