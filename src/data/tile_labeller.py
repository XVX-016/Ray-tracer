"""
tile_labeller.py
────────────────
Phase 1 – Tile-Level Ray Budget Label Generator
Project Alpha-Ray | Intelligent Ray Budget Allocation

Responsibilities:
  1. Receive a 1-spp noisy RGB frame and a 4096-spp ground-truth reference
     frame (both [3, H, W] float32 torch tensors).
  2. Compute a spatially-dense perceptual error map combining:
       - Per-pixel SSIM structural error (luminance, contrast, structure)
       - L1 absolute radiance error
     The composite error is weighted to emphasise high-frequency perceptual
     differences over low-frequency bias.
  3. Max-pool the full-resolution error map into 16×16 tiles,
     producing a spatial grid of [120, 68] tile-level error scalars.
     (120 = ceil(1920/16), 68 = ceil(1080/16))
  4. Quantize each tile's error scalar into one of 6 discrete
     ray budget class labels:
       Class 0 → 0  rays per pixel  (trivially flat, no re-sampling needed)
       Class 1 → 1  ray  per pixel
       Class 2 → 2  rays per pixel
       Class 3 → 4  rays per pixel
       Class 4 → 8  rays per pixel
       Class 5 → 16 rays per pixel  (maximally complex / high-variance region)
  5. Return a torch.Tensor of shape [120, 68] with dtype=torch.long
     (class indices, not raw ray counts). Suitable for nn.CrossEntropyLoss.

Implementation notes:
  - SSIM computation is implemented from scratch to avoid external dependencies
    and to operate in a fully vectorised, tile-aware manner.
  - Padding is applied before pooling to handle the non-divisible 1080/16 = 67.5
    case so all tiles are complete 16×16 blocks.
  - Error thresholds are computed adaptively per frame using percentile
    statistics, making the quantization distribution balanced even when
    scene-wide error magnitude varies across different environments.

Author : Principal Graphics & AI Engineer – Project Alpha-Ray
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ── Grid constants ─────────────────────────────────────────────────────────────
WIDTH: int = 1920
HEIGHT: int = 1080
TILE_SIZE: int = 16

# Padded dimensions to make W and H exact multiples of TILE_SIZE.
# 1920 / 16 = 120.0  → no horizontal padding needed.
# 1080 / 16 = 67.5   → pad to 1088 (68 complete tiles).
PADDED_W: int = int(np.ceil(WIDTH / TILE_SIZE)) * TILE_SIZE    # 1920
PADDED_H: int = int(np.ceil(HEIGHT / TILE_SIZE)) * TILE_SIZE   # 1088

TILE_COLS: int = PADDED_W // TILE_SIZE   # 120, horizontal tile index
TILE_ROWS: int = PADDED_H // TILE_SIZE   # 68, vertical tile index
TILE_GRID_SHAPE: Tuple[int, int] = (TILE_COLS, TILE_ROWS)

# Discrete ray budget options and their class indices.
# Index i in this list corresponds to class label i.
RAY_BUDGET_OPTIONS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16)
NUM_CLASSES: int = len(RAY_BUDGET_OPTIONS)  # 6

# SSIM window size (Gaussian kernel standard deviation and window width).
_SSIM_WINDOW_SIZE: int = 11
_SSIM_SIGMA: float = 1.5
_SSIM_C1: float = (0.01 ** 2)   # (K1 * L)^2, L=1 for normalised [0,1] images
_SSIM_C2: float = (0.03 ** 2)   # (K2 * L)^2

# Composite error weights: how much SSIM vs L1 contribute to the final map.
_WEIGHT_SSIM: float = 0.7
_WEIGHT_L1: float = 0.3


# ── SSIM helpers ──────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """
    Build a 1D Gaussian kernel of given size and sigma.
    Returns shape [size], float32.
    """
    x = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """
    Build a separable 2D Gaussian kernel [1, 1, size, size], float32.
    Used as a conv2d weight for local mean/variance estimation.
    """
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = torch.outer(k1d, k1d)
    return k2d.unsqueeze(0).unsqueeze(0)   # [1, 1, size, size]


# Pre-build and cache the Gaussian kernel to avoid recomputing per-call.
_GAUSS_KERNEL: torch.Tensor = _gaussian_kernel_2d(_SSIM_WINDOW_SIZE, _SSIM_SIGMA)


def _ssim_map(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a dense per-pixel SSIM structural dissimilarity map between
    two single-channel images.

    Parameters
    ----------
    img_a, img_b : torch.Tensor, shape [1, 1, H, W], float32
        Input images. Values expected in [0, 1] range (clamp applied internally).

    Returns
    -------
    torch.Tensor, shape [1, 1, H, W], float32
        Per-pixel SSIM dissimilarity: 1 - SSIM(x, y).
        0 = identical, up to 1 = completely different.
    """
    device = img_a.device
    kernel = _GAUSS_KERNEL.to(device)

    pad = _SSIM_WINDOW_SIZE // 2

    def _gaussian_filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad)

    mu_a = _gaussian_filter(img_a)
    mu_b = _gaussian_filter(img_b)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = _gaussian_filter(img_a * img_a) - mu_a_sq
    sigma_b_sq = _gaussian_filter(img_b * img_b) - mu_b_sq
    sigma_ab = _gaussian_filter(img_a * img_b) - mu_ab

    # Clamp numerical noise: variances must be non-negative.
    sigma_a_sq = sigma_a_sq.clamp(min=0.0)
    sigma_b_sq = sigma_b_sq.clamp(min=0.0)

    numerator = (2 * mu_ab + _SSIM_C1) * (2 * sigma_ab + _SSIM_C2)
    denominator = (mu_a_sq + mu_b_sq + _SSIM_C1) * (sigma_a_sq + sigma_b_sq + _SSIM_C2)

    ssim_per_pixel = numerator / (denominator + 1e-8)
    # Return dissimilarity in [0, 1]: 0 = identical.
    return (1.0 - ssim_per_pixel).clamp(0.0, 1.0)


# ── Perceptual error computation ──────────────────────────────────────────────

def compute_perceptual_error_map(
    noisy_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a spatially-dense composite perceptual error map between
    the 1-spp noisy image and the 4096-spp ground truth.

    The map combines SSIM structural dissimilarity and L1 absolute error
    across luminance (perceptually-weighted RGB → Y conversion).
    Operating on luminance avoids SSIM being artificially inflated by
    hue shifts that carry little structural information.

    Parameters
    ----------
    noisy_rgb : torch.Tensor [3, H, W], float32
        The 1-spp render output. May contain fireflies (values >> 1).
    gt_rgb    : torch.Tensor [3, H, W], float32
        The 4096-spp reference render.

    Returns
    -------
    torch.Tensor [H, W], float32
        Composite perceptual error in [0, 1].
        Higher values → more perceptual difference → more rays needed.
    """
    assert noisy_rgb.shape == (3, HEIGHT, WIDTH), (
        f"noisy_rgb shape: {noisy_rgb.shape}"
    )
    assert gt_rgb.shape == (3, HEIGHT, WIDTH), (
        f"gt_rgb shape: {gt_rgb.shape}"
    )

    device = noisy_rgb.device

    # ── 1. Tone-map to [0, 1] before SSIM to suppress firefly influence. ─────
    # Reinhard simple tone mapping on the noisy frame; GT is already stable.
    def _reinhard(x: torch.Tensor) -> torch.Tensor:
        return x / (1.0 + x)

    noisy_tm = _reinhard(noisy_rgb.clamp(min=0.0))   # [3, H, W], [0, 1]
    gt_tm = _reinhard(gt_rgb.clamp(min=0.0))          # [3, H, W], [0, 1]

    # ── 2. Convert to perceptual luminance (BT.709 coefficients). ────────────
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device).view(3, 1, 1)
    noisy_luma = (noisy_tm * luma_weights).sum(dim=0, keepdim=True)   # [1, H, W]
    gt_luma = (gt_tm * luma_weights).sum(dim=0, keepdim=True)         # [1, H, W]

    # Expand to [1, 1, H, W] for conv2d compatibility.
    noisy_luma_4d = noisy_luma.unsqueeze(0)   # [1, 1, H, W]
    gt_luma_4d = gt_luma.unsqueeze(0)         # [1, 1, H, W]

    # ── 3. SSIM dissimilarity map on luminance. ───────────────────────────────
    ssim_err = _ssim_map(noisy_luma_4d, gt_luma_4d)  # [1, 1, H, W]
    ssim_err = ssim_err.squeeze(0).squeeze(0)          # [H, W]

    # ── 4. L1 error map on luminance (normalised to [0, 1]). ─────────────────
    l1_err = (noisy_luma - gt_luma).abs().squeeze(0)   # [H, W]
    # Normalise L1 by the 99th percentile of the GT luminance to keep
    # the error scale scene-agnostic.
    gt_luma_p99 = gt_luma.flatten().kthvalue(
        int(0.99 * gt_luma.numel())
    ).values.clamp(min=1e-4)
    l1_err = (l1_err / gt_luma_p99).clamp(0.0, 1.0)

    # ── 5. Composite weighted error. ──────────────────────────────────────────
    composite = _WEIGHT_SSIM * ssim_err + _WEIGHT_L1 * l1_err   # [H, W]
    composite = composite.clamp(0.0, 1.0)

    log.debug(
        "Perceptual error map | min=%.4f mean=%.4f max=%.4f",
        composite.min().item(),
        composite.mean().item(),
        composite.max().item(),
    )
    return composite   # [H, W]


# ── Tile pooling ──────────────────────────────────────────────────────────────

def pool_to_tiles(error_map: torch.Tensor) -> torch.Tensor:
    """
    Pool a full-resolution error map down to tile-level scalars.

    Strategy: max-pool (not average) over 16×16 blocks.
    Max-pool is used because even a single high-error pixel in a tile
    indicates that the tile contains a complex shading region requiring
    more rays. An average would dilute sparse sharp features (thin specular
    highlights, geometric silhouettes) that drive sampling demand.

    Parameters
    ----------
    error_map : torch.Tensor [H, W], float32
        Full-resolution perceptual error.

    Returns
    -------
    torch.Tensor [TILE_COLS, TILE_ROWS] = [120, 68], float32
        Max-pooled tile error scalars in [0, 1].
    """
    assert error_map.shape == (HEIGHT, WIDTH), (
        f"error_map shape: {error_map.shape}"
    )

    # ── Pad to exact tile multiple. ──────────────────────────────────────────
    # Height: 1080 → 1088 (pad 8 rows at bottom with 0).
    # Width:  1920 → 1920 (already divisible, no padding).
    pad_h = PADDED_H - HEIGHT   # 8
    pad_w = PADDED_W - WIDTH    # 0

    if pad_h > 0 or pad_w > 0:
        # F.pad expects [N, C, H, W] or [H, W]; we work on [H, W].
        # Add batch/channel dims, pad, then remove them.
        error_4d = error_map.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        # Padding order: (left, right, top, bottom)
        error_4d = F.pad(error_4d, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        error_map_padded = error_4d.squeeze(0).squeeze(0)   # [PADDED_H, PADDED_W]
    else:
        error_map_padded = error_map

    assert error_map_padded.shape == (PADDED_H, PADDED_W), (
        f"Padded error map shape: {error_map_padded.shape}"
    )

    # ── Reshape into tile grid and apply max pooling. ────────────────────────
    # Reshape [PADDED_H, PADDED_W] → [TILE_ROWS, TILE_SIZE, TILE_COLS, TILE_SIZE]
    tiled = error_map_padded.view(
        TILE_ROWS, TILE_SIZE, TILE_COLS, TILE_SIZE
    )
    # Max over both tile spatial dimensions.
    tile_max_yx = tiled.amax(dim=(1, 3))   # [TILE_ROWS, TILE_COLS] = [68, 120]
    tile_max = tile_max_yx.transpose(0, 1).contiguous()  # [TILE_COLS, TILE_ROWS]

    assert tile_max.shape == TILE_GRID_SHAPE, (
        f"tile_max shape: {tile_max.shape}"
    )
    return tile_max


# ── Error quantization ────────────────────────────────────────────────────────

def quantize_to_class_labels(
    tile_errors: torch.Tensor,
    adaptive: bool = True,
) -> torch.Tensor:
    """
    Map per-tile continuous error scalars to discrete class indices
    corresponding to the hardware ray budget options {0, 1, 2, 4, 8, 16}.

    Two quantization modes are supported:

    adaptive=True  (default, recommended for training data generation):
        Thresholds are set using percentile boundaries of the current
        frame's tile error distribution. This guarantees that all 6 classes
        are represented in most frames, producing balanced training batches.
        Percentile breakpoints: [0–10%=0, 10–30%=1, 30–50%=2, 50–70%=4,
                                  70–90%=8, 90–100%=16]

    adaptive=False (fixed thresholds, suitable for inference / evaluation):
        Thresholds are fixed in the [0, 1] range:
        [0.0, 0.05, 0.15, 0.30, 0.55, 0.80] → classes 0–5.

    Parameters
    ----------
    tile_errors : torch.Tensor [TILE_COLS, TILE_ROWS], float32
        Tile-level max-pooled perceptual error scalars.
    adaptive    : bool
        Whether to use per-frame adaptive percentile thresholds.

    Returns
    -------
    torch.Tensor [TILE_COLS, TILE_ROWS], dtype=torch.long
        Class index labels (0–5), NOT raw ray counts.
        Map: 0→0rpp, 1→1rpp, 2→2rpp, 3→4rpp, 4→8rpp, 5→16rpp.
    """
    assert tile_errors.shape == TILE_GRID_SHAPE, (
        f"tile_errors shape: {tile_errors.shape}"
    )
    assert tile_errors.dtype == torch.float32

    flat = tile_errors.flatten()    # [TILE_ROWS * TILE_COLS] = [8160]
    n = flat.numel()

    if adaptive:
        # Compute percentile thresholds from the sorted error distribution.
        # Using torch.sort rather than torch.quantile for CUDA compatibility.
        sorted_vals, _ = flat.sort()

        def _pct(p: float) -> float:
            idx = min(int(p * n), n - 1)
            return sorted_vals[idx].item()

        # 5 boundary values dividing the range into 6 buckets.
        thresholds = [
            _pct(0.10),   # below this → class 0 (0 rays)
            _pct(0.30),   # below this → class 1 (1 ray)
            _pct(0.50),   # below this → class 2 (2 rays)
            _pct(0.70),   # below this → class 3 (4 rays)
            _pct(0.90),   # below this → class 4 (8 rays)
            # above 90th percentile → class 5 (16 rays)
        ]
    else:
        # Fixed absolute thresholds.
        thresholds = [0.05, 0.15, 0.30, 0.55, 0.80]

    # ── Assign class labels via threshold comparison. ──────────────────────
    # Start with all tiles assigned to class 5 (highest budget).
    labels = torch.full(
        TILE_GRID_SHAPE,
        fill_value=NUM_CLASSES - 1,
        dtype=torch.long,
        device=tile_errors.device,
    )

    # Work from the highest threshold down; each condition overwrites
    # previously assigned labels for tiles below the threshold.
    for class_idx in range(NUM_CLASSES - 2, -1, -1):
        # class_idx counts from NUM_CLASSES-2 down to 0.
        # threshold index maps: class 0 → threshold[0], class 1 → threshold[1] …
        labels[tile_errors < thresholds[class_idx]] = class_idx

    # ── Validation: ensure all labels are valid class indices. ─────────────
    assert labels.min().item() >= 0, "Label underflow: negative class index."
    assert labels.max().item() < NUM_CLASSES, (
        f"Label overflow: class index {labels.max().item()} >= {NUM_CLASSES}"
    )

    log.debug(
        "Class label distribution: %s",
        {i: (labels == i).sum().item() for i in range(NUM_CLASSES)},
    )
    return labels   # [TILE_COLS, TILE_ROWS], torch.long


# ── Main public API ───────────────────────────────────────────────────────────

@dataclass
class TileLabelResult:
    """
    Container for all outputs from one labelling pass.

    Attributes
    ----------
    class_labels   : torch.Tensor [120, 68], torch.long
        Discrete class indices 0–5 per tile.
    tile_errors    : torch.Tensor [120, 68], float32
        Raw max-pooled error scalars before quantization.
    error_map      : torch.Tensor [1080, 1920], float32
        Full-resolution composite perceptual error map.
    ray_budget_map : torch.Tensor [120, 68], torch.int32
        Actual ray counts (0,1,2,4,8,16) for human-readable inspection.
        Not used for training (use class_labels instead).
    """
    class_labels: torch.Tensor
    tile_errors: torch.Tensor
    error_map: torch.Tensor
    ray_budget_map: torch.Tensor

    def __post_init__(self) -> None:
        assert self.class_labels.shape == TILE_GRID_SHAPE
        assert self.class_labels.dtype == torch.long
        assert self.tile_errors.shape == TILE_GRID_SHAPE
        assert self.error_map.shape == (HEIGHT, WIDTH)
        assert self.ray_budget_map.shape == TILE_GRID_SHAPE


class TileLabeller:
    """
    Converts a (noisy, ground-truth) render pair into tile-level class labels.

    Usage
    -----
    labeller = TileLabeller(adaptive=True)
    result   = labeller.label(noisy_rgb, gt_rgb)
    # result.class_labels : [120, 68] torch.long
    """

    def __init__(self, adaptive: bool = True) -> None:
        self.adaptive = adaptive
        # Build and cache lookup tensor: class index → ray count.
        self._index_to_rays = torch.tensor(
            RAY_BUDGET_OPTIONS, dtype=torch.int32
        )   # [6]
        log.info(
            "TileLabeller initialised | tile_size=%d | grid=[%d×%d] | adaptive=%s",
            TILE_SIZE, TILE_COLS, TILE_ROWS, adaptive,
        )

    def label(
        self,
        noisy_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
    ) -> TileLabelResult:
        """
        Full labelling pipeline for one frame pair.

        Parameters
        ----------
        noisy_rgb : torch.Tensor [3, 1080, 1920], float32
        gt_rgb    : torch.Tensor [3, 1080, 1920], float32

        Returns
        -------
        TileLabelResult
        """
        log.info("TileLabeller.label() started")

        # ── Step 1: Perceptual error map [H, W]. ──────────────────────────
        error_map = compute_perceptual_error_map(noisy_rgb, gt_rgb)

        # ── Step 2: Max-pool to tile grid [TILE_COLS, TILE_ROWS]. ─────────
        tile_errors = pool_to_tiles(error_map)

        # ── Step 3: Quantize to class labels [TILE_COLS, TILE_ROWS]. ──────
        class_labels = quantize_to_class_labels(
            tile_errors, adaptive=self.adaptive
        )

        # ── Step 4: Decode class labels → ray counts for inspection. ──────
        lookup = self._index_to_rays.to(class_labels.device)
        ray_budget_map = lookup[class_labels]   # [TILE_COLS, TILE_ROWS], int32

        result = TileLabelResult(
            class_labels=class_labels,
            tile_errors=tile_errors,
            error_map=error_map,
            ray_budget_map=ray_budget_map,
        )

        log.info(
            "TileLabeller complete | "
            "error [%.4f, %.4f] | "
            "label dist: {%s}",
            tile_errors.min().item(),
            tile_errors.max().item(),
            ", ".join(
                f"{RAY_BUDGET_OPTIONS[i]}rpp:{(class_labels == i).sum().item()}"
                for i in range(NUM_CLASSES)
            ),
        )
        return result


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[tile_labeller] Running smoke test with synthetic tensors...")

    torch.manual_seed(42)

    # Simulate a 1-spp noisy frame: ground truth + heavy noise + fireflies.
    gt_synthetic = torch.rand(3, HEIGHT, WIDTH).clamp(0.0, 2.0)
    noise = torch.randn(3, HEIGHT, WIDTH) * 0.6
    fireflies = (torch.rand(3, HEIGHT, WIDTH) < 0.001).float() * 20.0
    noisy_synthetic = (gt_synthetic + noise + fireflies).clamp(0.0)

    labeller = TileLabeller(adaptive=True)
    result = labeller.label(noisy_synthetic, gt_synthetic)

    print(f"  class_labels  : {tuple(result.class_labels.shape)}, dtype={result.class_labels.dtype}")
    print(f"  tile_errors   : {tuple(result.tile_errors.shape)}")
    print(f"  error_map     : {tuple(result.error_map.shape)}")
    print(f"  ray_budget_map: {tuple(result.ray_budget_map.shape)}")
    print(f"  unique labels : {result.class_labels.unique().tolist()}")
    print(
        f"  ray budget distribution: "
        + ", ".join(
            f"{RAY_BUDGET_OPTIONS[i]}rpp={( result.class_labels == i).sum().item()}"
            for i in range(NUM_CLASSES)
        )
    )
    print("[tile_labeller] Smoke test PASSED")
