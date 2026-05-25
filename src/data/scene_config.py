"""
scene_config.py
───────────────
Phase 1 – Scene Configuration & Dual-Pass Renderer
Project Alpha-Ray | Intelligent Ray Budget Allocation

Responsibilities:
  1. Load or construct a Cornell Box scene using Mitsuba 3 dict-based API.
  2. Configure an AOV integrator wrapping a base path tracer to emit:
       albedo (RGB), sh_normal (XYZ shading normals), depth (scalar).
  3. Execute two render passes:
       Pass A – 1 spp  → noisy G-Buffer input tensors.
       Pass B – 4096 spp → high-fidelity path-traced ground truth.
  4. Extract every film layer from Mitsuba Bitmap allocations into
       explicit NumPy arrays / PyTorch tensors with documented shapes.

Output tensors (all float32, CPU):
  gbuffer : torch.Tensor  [7, 1080, 1920]
              channels: [N_x, N_y, N_z, Depth, Alb_R, Alb_G, Alb_B]
  gt_rgb  : torch.Tensor  [3, 1080, 1920]
              channels: [R, G, B]  – linear, pre-tonemapping

Author : Principal Graphics & AI Engineer – Project Alpha-Ray
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ── Mitsuba 3 import guard ────────────────────────────────────────────────────
try:
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")
    import drjit as dr
except ImportError as exc:
    raise ImportError(
        "Mitsuba 3 with the 'cuda_ad_rgb' variant is required. "
        "Install via: pip install mitsuba  (CUDA build)"
    ) from exc

# ── Constants ─────────────────────────────────────────────────────────────────
WIDTH: int = 1920
HEIGHT: int = 1080
SPP_INPUT: int = 1
SPP_GROUNDTRUTH: int = 4096

# AOV channel names as emitted by Mitsuba's film layer system.
# Each entry is (aov_name_in_bitmap, expected_channel_count).
AOV_NORMALS_KEY: str = "sh_normal"   # 3-channel XYZ shading normals
AOV_ALBEDO_KEY: str = "albedo"       # 3-channel diffuse albedo
AOV_DEPTH_KEY: str = "depth"         # 1-channel linear depth

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class RenderResult:
    """
    Immutable container for one complete dual-pass render.

    Attributes
    ----------
    gbuffer : torch.Tensor, shape [7, H, W], float32 CPU
        Packed G-Buffer:
          [0]   N_x  – shading normal X
          [1]   N_y  – shading normal Y
          [2]   N_z  – shading normal Z
          [3]   Depth – linear world-space depth (metres)
          [4]   Alb_R
          [5]   Alb_G
          [6]   Alb_B
    gt_rgb : torch.Tensor, shape [3, H, W], float32 CPU
        Ground-truth path-traced radiance (linear, no tonemapping).
    noisy_rgb : torch.Tensor, shape [3, H, W], float32 CPU
        1-spp noisy radiance extracted from the low-spp pass.
        Used by tile_labeller to compute perceptual error maps.
    scene_name : str
        Identifier forwarded to exporter metadata.
    """

    gbuffer: torch.Tensor
    gt_rgb: torch.Tensor
    noisy_rgb: torch.Tensor
    scene_name: str = "default_cornell"

    def __post_init__(self) -> None:
        assert self.gbuffer.shape == (7, HEIGHT, WIDTH), (
            f"gbuffer shape mismatch: expected (7, {HEIGHT}, {WIDTH}), "
            f"got {tuple(self.gbuffer.shape)}"
        )
        assert self.gt_rgb.shape == (3, HEIGHT, WIDTH), (
            f"gt_rgb shape mismatch: expected (3, {HEIGHT}, {WIDTH}), "
            f"got {tuple(self.gt_rgb.shape)}"
        )
        assert self.noisy_rgb.shape == (3, HEIGHT, WIDTH), (
            f"noisy_rgb shape mismatch: expected (3, {HEIGHT}, {WIDTH}), "
            f"got {tuple(self.noisy_rgb.shape)}"
        )


# ── Scene dictionary builders ─────────────────────────────────────────────────

def _build_film(spp: int) -> Dict:
    """
    Construct the Mitsuba film dictionary for a given spp configuration.
    Uses an HDR film with multi-layer support to capture both beauty and AOVs.
    """
    return {
        "type": "hdrfilm",
        "width": WIDTH,
        "height": HEIGHT,
        "pixel_format": "rgb",
        "component_format": "float32",
        # Reconstruction filter: box for correctness; switch to 'tent' for
        # production anti-aliasing at the cost of slight blurring.
        "rfilter": {"type": "box"},
    }


def _build_aov_integrator() -> Dict:
    """
    Build the AOV integrator that wraps a path tracer.
    The 'aovs' field lists the named output variables to capture alongside
    the primary beauty render.

    AOV string format:  "<output_name>:<aov_type>"
    Supported types (Mitsuba 3 cuda_ad_rgb): albedo, sh_normal, depth,
    position, uv, geo_normal, prim_index, shape_index.
    """
    return {
        "type": "aov",
        # Declare the auxiliary outputs we need.
        # sh_normal → 3 floats (XYZ shading normals, world space)
        # albedo    → 3 floats (diffuse reflectance / base colour)
        # depth     → 1 float  (linear distance from camera origin)
        "aovs": f"{AOV_NORMALS_KEY}:sh_normal, "
                f"{AOV_ALBEDO_KEY}:albedo, "
                f"{AOV_DEPTH_KEY}:depth",
        # Inner integrator: path tracer for the beauty + AOV evaluation.
        "integrator": {
            "type": "path",
            "max_depth": 8,
            "rr_depth": 4,
        },
    }


def _build_cornell_box_scene(spp: int, use_aov: bool = True) -> Dict:
    """
    Construct a complete Cornell Box scene as a Mitsuba 3 scene dictionary.

    Parameters
    ----------
    spp      : samples per pixel for this render pass
    use_aov  : if True, use the AOV integrator; False uses plain path tracer
               (used for the ground truth pass where we do NOT need AOVs)

    Returns
    -------
    dict compatible with mi.load_dict()
    """
    integrator = _build_aov_integrator() if use_aov else {
        "type": "path",
        "max_depth": 8,
        "rr_depth": 4,
    }

    return {
        "type": "scene",
        # ── Integrator ──────────────────────────────────────────────────────
        "integrator": integrator,

        # ── Sensor (camera) ─────────────────────────────────────────────────
        "sensor": {
            "type": "perspective",
            "fov": 39.3077,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0.0, 1.0, 3.8],
                target=[0.0, 1.0, 0.0],
                up=[0.0, 1.0, 0.0],
            ),
            "film": _build_film(spp),
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            },
        },

        # ── Emitters ────────────────────────────────────────────────────────
        "area_light": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0.0, 1.98, 0.0])
                        @ mi.ScalarTransform4f.rotate([1, 0, 0], angle=90)
                        @ mi.ScalarTransform4f.scale([0.25, 0.25, 1.0]),
            "flip_normals": True,
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [18.4, 15.6, 8.0]},
            },
        },

        # ── Geometry ────────────────────────────────────────────────────────
        # Floor
        "floor": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 0, 0])
                        @ mi.ScalarTransform4f.rotate([1, 0, 0], angle=-90)
                        @ mi.ScalarTransform4f.scale([1.0, 1.0, 1.0]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.725, 0.71, 0.68]}},
        },
        # Ceiling
        "ceiling": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 2, 0])
                        @ mi.ScalarTransform4f.rotate([1, 0, 0], angle=90)
                        @ mi.ScalarTransform4f.scale([1.0, 1.0, 1.0]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.725, 0.71, 0.68]}},
        },
        # Back wall
        "back_wall": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([0, 1, -1])
                        @ mi.ScalarTransform4f.scale([1.0, 1.0, 1.0]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.725, 0.71, 0.68]}},
        },
        # Left wall (red)
        "left_wall": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([-1, 1, 0])
                        @ mi.ScalarTransform4f.rotate([0, 1, 0], angle=90)
                        @ mi.ScalarTransform4f.scale([1.0, 1.0, 1.0]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.63, 0.065, 0.05]}},
        },
        # Right wall (green)
        "right_wall": {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate([1, 1, 0])
                        @ mi.ScalarTransform4f.rotate([0, 1, 0], angle=-90)
                        @ mi.ScalarTransform4f.scale([1.0, 1.0, 1.0]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.14, 0.45, 0.091]}},
        },

        # ── Geometry: tall box ───────────────────────────────────────────────
        "tall_box": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate([-0.335, 0.6, -0.38])
                        @ mi.ScalarTransform4f.rotate([0, 1, 0], angle=15)
                        @ mi.ScalarTransform4f.scale([0.3, 0.6, 0.3]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.725, 0.71, 0.68]}},
        },

        # ── Geometry: short box ──────────────────────────────────────────────
        "short_box": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate([0.33, 0.3, -0.28])
                        @ mi.ScalarTransform4f.rotate([0, 1, 0], angle=-17)
                        @ mi.ScalarTransform4f.scale([0.3, 0.3, 0.3]),
            "bsdf": {
                # Use a rougher dielectric to exercise AOV material complexity
                "type": "roughplastic",
                "distribution": "ggx",
                "alpha": 0.15,
                "diffuse_reflectance": {"type": "rgb", "value": [0.725, 0.71, 0.68]},
            },
        },
    }


# ── Bitmap → NumPy extraction ─────────────────────────────────────────────────

def _bitmap_to_numpy(bitmap: "mi.Bitmap", channel_name: Optional[str] = None) -> np.ndarray:
    """
    Convert a Mitsuba Bitmap (or a named layer from a multi-layer bitmap)
    into a float32 NumPy array.

    Parameters
    ----------
    bitmap       : mi.Bitmap
        The source bitmap, possibly multi-layer.
    channel_name : str | None
        If provided, extract this named layer from a multi-channel bitmap
        via bitmap.split(). If None, treat the bitmap as a single-layer image.

    Returns
    -------
    np.ndarray, float32
        Shape depends on layer: [H, W, C] for multi-channel, [H, W] for scalar.
    """
    if channel_name is not None:
        # split() returns a list of (name, Bitmap) pairs – one per AOV layer.
        layers: Dict[str, "mi.Bitmap"] = dict(bitmap.split())
        if channel_name not in layers:
            available = list(layers.keys())
            raise KeyError(
                f"AOV layer '{channel_name}' not found in bitmap. "
                f"Available layers: {available}"
            )
        bitmap = layers[channel_name]

    # Convert to float32 before numpy extraction to guarantee dtype alignment.
    if bitmap.pixel_format() != mi.Bitmap.PixelFormat.RGB:
        # For scalar channels (depth) Mitsuba uses Y (luminance-equivalent).
        pass  # np.array() handles both RGB and Y formats correctly.

    arr: np.ndarray = np.array(bitmap, dtype=np.float32)
    # Mitsuba returns [H, W] for scalar and [H, W, C] for multi-channel.
    return arr


def _extract_aov_layers(
    render_output: "mi.Bitmap",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From the multi-layer AOV bitmap produced by one render pass, extract:
      - beauty_rgb  : [H, W, 3]  – path-traced radiance
      - normals_xyz : [H, W, 3]  – shading normals (world-space, unit vectors)
      - albedo_rgb  : [H, W, 3]  – diffuse albedo
      - depth_map   : [H, W]     – linear depth

    Parameters
    ----------
    render_output : mi.Bitmap
        The raw multi-layer bitmap returned by mi.render().

    Returns
    -------
    Tuple of (beauty_rgb, normals_xyz, albedo_rgb, depth_map), all float32.
    """
    layers: Dict[str, "mi.Bitmap"] = dict(render_output.split())
    log.debug("Available bitmap layers: %s", list(layers.keys()))

    # ── Beauty (primary radiance) ────────────────────────────────────────────
    # The AOV integrator stores the primary render under the key '<default>'
    # or the first entry; Mitsuba labels it with the film pixel_format name.
    # We probe common keys defensively.
    beauty_key = None
    for candidate in ("<default>", "image", "rgb", "beauty"):
        if candidate in layers:
            beauty_key = candidate
            break
    if beauty_key is None:
        # Fall back: the beauty layer is whichever 3-channel layer is NOT an AOV.
        aov_keys = {AOV_NORMALS_KEY, AOV_ALBEDO_KEY, AOV_DEPTH_KEY}
        for k, bmp in layers.items():
            if k not in aov_keys and bmp.channel_count() == 3:
                beauty_key = k
                break
    if beauty_key is None:
        raise RuntimeError(
            f"Cannot locate beauty layer in bitmap. Layers present: {list(layers.keys())}"
        )

    beauty_rgb: np.ndarray = _bitmap_to_numpy(layers[beauty_key])    # [H, W, 3]

    # ── Shading normals ──────────────────────────────────────────────────────
    normals_xyz: np.ndarray = _bitmap_to_numpy(layers[AOV_NORMALS_KEY])  # [H, W, 3]

    # ── Albedo ───────────────────────────────────────────────────────────────
    albedo_rgb: np.ndarray = _bitmap_to_numpy(layers[AOV_ALBEDO_KEY])    # [H, W, 3]

    # ── Depth ────────────────────────────────────────────────────────────────
    depth_raw: np.ndarray = _bitmap_to_numpy(layers[AOV_DEPTH_KEY])      # [H, W] or [H, W, 1]
    depth_map: np.ndarray = depth_raw.squeeze(-1) if depth_raw.ndim == 3 else depth_raw

    # ── Shape assertions before returning ───────────────────────────────────
    assert beauty_rgb.shape == (HEIGHT, WIDTH, 3), (
        f"beauty_rgb shape: {beauty_rgb.shape}"
    )
    assert normals_xyz.shape == (HEIGHT, WIDTH, 3), (
        f"normals_xyz shape: {normals_xyz.shape}"
    )
    assert albedo_rgb.shape == (HEIGHT, WIDTH, 3), (
        f"albedo_rgb shape: {albedo_rgb.shape}"
    )
    assert depth_map.shape == (HEIGHT, WIDTH), (
        f"depth_map shape: {depth_map.shape}"
    )

    return beauty_rgb, normals_xyz, albedo_rgb, depth_map


def _pack_gbuffer(
    normals_xyz: np.ndarray,
    depth_map: np.ndarray,
    albedo_rgb: np.ndarray,
) -> torch.Tensor:
    """
    Pack all G-Buffer arrays into a single [7, H, W] float32 CPU tensor.

    Channel layout (matches dataset.py and exporter.py conventions):
      [0]  N_x
      [1]  N_y
      [2]  N_z
      [3]  Depth
      [4]  Alb_R
      [5]  Alb_G
      [6]  Alb_B

    Parameters
    ----------
    normals_xyz : np.ndarray [H, W, 3]
    depth_map   : np.ndarray [H, W]
    albedo_rgb  : np.ndarray [H, W, 3]

    Returns
    -------
    torch.Tensor [7, H, W], float32, CPU (pinned memory NOT used here;
    pinning happens in the DataLoader for training throughput).
    """
    # Transpose HWC → CHW for all multi-channel inputs.
    normals_chw: np.ndarray = normals_xyz.transpose(2, 0, 1)   # [3, H, W]
    albedo_chw: np.ndarray = albedo_rgb.transpose(2, 0, 1)      # [3, H, W]
    depth_hw: np.ndarray = depth_map[np.newaxis, :, :]          # [1, H, W]

    # Stack along channel dimension: [3+1+3, H, W] = [7, H, W]
    packed: np.ndarray = np.concatenate(
        [normals_chw, depth_hw, albedo_chw], axis=0
    ).astype(np.float32)

    assert packed.shape == (7, HEIGHT, WIDTH), f"Packed G-Buffer shape: {packed.shape}"

    return torch.from_numpy(packed)


# ── Main renderer class ───────────────────────────────────────────────────────

class SceneRenderer:
    """
    Orchestrates dual-pass rendering for one scene.

    Usage
    -----
    renderer = SceneRenderer()
    result   = renderer.render(scene_path=None, scene_name="cornell_001")
    # result.gbuffer  : [7, 1080, 1920]
    # result.gt_rgb   : [3, 1080, 1920]
    # result.noisy_rgb: [3, 1080, 1920]
    """

    def __init__(
        self,
        spp_input: int = SPP_INPUT,
        spp_gt: int = SPP_GROUNDTRUTH,
    ) -> None:
        self.spp_input = spp_input
        self.spp_gt = spp_gt
        log.info(
            "SceneRenderer initialised | spp_input=%d | spp_gt=%d | variant=%s",
            spp_input, spp_gt, mi.variant(),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_scene(
        self,
        scene_path: Optional[Path],
        spp: int,
        use_aov: bool,
    ) -> "mi.Scene":
        """
        Load a scene from XML file or construct a default Cornell Box dict.
        The film and sampler configs are always overridden to match spp.
        """
        if scene_path is not None and scene_path.exists():
            log.info("Loading scene from XML: %s", scene_path)
            # Override film and sampler via parameter update after load.
            scene = mi.load_file(str(scene_path))
            # NOTE: Mitsuba 3 does not support easy override of spp via load_file
            # without re-specifying the sensor. For production use, the scene XML
            # should parameterise the sampler count, then use:
            #   params = mi.traverse(scene)
            #   params['sensor.sampler.sample_count'] = spp
            #   params.update()
            params = mi.traverse(scene)
            if "sensor.sampler.sample_count" in params:
                params["sensor.sampler.sample_count"] = spp
                params.update()
            return scene
        else:
            if scene_path is not None:
                log.warning(
                    "Scene file not found at '%s'. Falling back to built-in Cornell Box.",
                    scene_path,
                )
            log.info("Constructing default Cornell Box | spp=%d | aov=%s", spp, use_aov)
            scene_dict = _build_cornell_box_scene(spp=spp, use_aov=use_aov)
            return mi.load_dict(scene_dict)

    def _render_pass(
        self,
        scene: "mi.Scene",
        pass_name: str,
    ) -> "mi.Bitmap":
        """
        Execute one render pass, drain the DrJIT graph, and return the
        resulting Bitmap. Flushes the CUDA graph after rendering to free
        intermediate JIT allocations.

        Parameters
        ----------
        scene     : mi.Scene   – pre-loaded Mitsuba scene
        pass_name : str        – label for logging

        Returns
        -------
        mi.Bitmap – multi-layer bitmap ready for channel extraction
        """
        log.info("Render pass started: %s", pass_name)
        image: "mi.TensorXf" = mi.render(scene)

        # Evaluate the DrJIT lazy graph and transfer to host-side bitmap.
        dr.eval(image)
        dr.sync_thread()

        bitmap: "mi.Bitmap" = mi.Bitmap(image)
        log.info("Render pass complete: %s | bitmap size: %s", pass_name, bitmap.size())
        return bitmap

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        scene_path: Optional[Path] = None,
        scene_name: str = "default_cornell",
    ) -> RenderResult:
        """
        Execute both render passes and return a fully packed RenderResult.

        Parameters
        ----------
        scene_path : Path | None
            Path to a Mitsuba XML scene file. If None or file missing,
            the built-in Cornell Box is used.
        scene_name : str
            Metadata label forwarded to the exporter.

        Returns
        -------
        RenderResult with verified tensor shapes.
        """

        # ── Pass A: Low-SPP AOV pass (G-Buffers + noisy beauty) ──────────────
        log.info("=== PASS A: G-Buffer extraction at %d spp ===", self.spp_input)
        scene_aov = self._load_scene(scene_path, spp=self.spp_input, use_aov=True)
        bitmap_aov = self._render_pass(scene_aov, pass_name=f"aov_{self.spp_input}spp")

        beauty_1spp, normals_xyz, albedo_rgb, depth_map = _extract_aov_layers(bitmap_aov)

        # Pack into unified G-Buffer tensor.
        gbuffer = _pack_gbuffer(normals_xyz, depth_map, albedo_rgb)

        # Noisy RGB for error map computation in tile_labeller.
        noisy_rgb = torch.from_numpy(
            beauty_1spp.transpose(2, 0, 1).astype(np.float32)
        )  # [3, H, W]

        # Explicitly release Mitsuba scene and DrJIT temporaries before Pass B.
        del scene_aov, bitmap_aov, beauty_1spp, normals_xyz, albedo_rgb, depth_map
        dr.flush_malloc_cache()
        gc.collect()
        log.info("Pass A complete. G-Buffer packed. CUDA cache flushed.")

        # ── Pass B: High-SPP ground-truth pass (beauty only, no AOV overhead) ─
        log.info("=== PASS B: Ground-truth render at %d spp ===", self.spp_gt)
        scene_gt = self._load_scene(scene_path, spp=self.spp_gt, use_aov=False)
        bitmap_gt = self._render_pass(scene_gt, pass_name=f"gt_{self.spp_gt}spp")

        # Ground-truth is a single beauty layer; extract directly.
        gt_layers = dict(bitmap_gt.split())
        # Select the beauty/primary layer (same logic as _extract_aov_layers).
        gt_beauty_key = None
        for candidate in ("<default>", "image", "rgb", "beauty"):
            if candidate in gt_layers:
                gt_beauty_key = candidate
                break
        if gt_beauty_key is None:
            # Any 3-channel layer is the beauty for a non-AOV render.
            for k, bmp in gt_layers.items():
                if bmp.channel_count() == 3:
                    gt_beauty_key = k
                    break
        if gt_beauty_key is None:
            # Last resort: convert full bitmap directly.
            log.warning("Could not identify beauty key in GT bitmap layers; using full bitmap.")
            gt_np = _bitmap_to_numpy(bitmap_gt)
        else:
            gt_np = _bitmap_to_numpy(gt_layers[gt_beauty_key])   # [H, W, 3]

        gt_rgb = torch.from_numpy(
            gt_np.transpose(2, 0, 1).astype(np.float32)
        )  # [3, H, W]

        del scene_gt, bitmap_gt, gt_np, gt_layers
        dr.flush_malloc_cache()
        gc.collect()
        log.info("Pass B complete. Ground-truth packed.")

        result = RenderResult(
            gbuffer=gbuffer,
            gt_rgb=gt_rgb,
            noisy_rgb=noisy_rgb,
            scene_name=scene_name,
        )
        log.info(
            "RenderResult ready | gbuffer=%s | gt_rgb=%s | noisy_rgb=%s",
            tuple(result.gbuffer.shape),
            tuple(result.gt_rgb.shape),
            tuple(result.noisy_rgb.shape),
        )
        return result


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    scene_xml = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    renderer = SceneRenderer(spp_input=SPP_INPUT, spp_gt=SPP_GROUNDTRUTH)
    result = renderer.render(scene_path=scene_xml, scene_name="smoke_test")

    print(f"\n[scene_config] Smoke test PASSED")
    print(f"  gbuffer   : {tuple(result.gbuffer.shape)}  dtype={result.gbuffer.dtype}")
    print(f"  gt_rgb    : {tuple(result.gt_rgb.shape)}   dtype={result.gt_rgb.dtype}")
    print(f"  noisy_rgb : {tuple(result.noisy_rgb.shape)} dtype={result.noisy_rgb.dtype}")
    print(f"  gbuffer range  [{result.gbuffer.min():.4f}, {result.gbuffer.max():.4f}]")
    print(f"  gt_rgb range   [{result.gt_rgb.min():.4f}, {result.gt_rgb.max():.4f}]")
