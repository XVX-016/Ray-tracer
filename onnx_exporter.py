"""
Export the Phase 2 RayBudgetFCN PyTorch model to ONNX.

Static tensor contract:
    input_gbuffers      [1, 7, 1080, 1920] float32
    output_tile_logits  [1, 6, 120, 68]    float32

Example:
    python onnx_exporter.py ^
        --checkpoint outputs/checkpoints/best.pt ^
        --output outputs/exports/ray_budget_fcn.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from src.training.model import (
    IMG_H_IN,
    IMG_W_IN,
    IN_CHANNELS,
    NUM_CLASSES,
    TILE_X,
    TILE_Y,
    RayBudgetFCN,
)

log = logging.getLogger("onnx_exporter")

INPUT_NAME = "input_gbuffers"
OUTPUT_NAME = "output_tile_logits"
STATIC_INPUT_SHAPE = (1, IN_CHANNELS, IMG_H_IN, IMG_W_IN)
STATIC_OUTPUT_SHAPE = (1, NUM_CLASSES, TILE_X, TILE_Y)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    dynamic_batch: bool = False,
    verify: bool = True,
) -> None:
    if opset < 17:
        raise ValueError("TensorRT Phase 3 export requires ONNX opset >= 17")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = RayBudgetFCN()
    state = load_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch. missing={missing}, unexpected={unexpected}"
        )
    model.eval()

    dummy = torch.zeros(STATIC_INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        reference = model(dummy)
    if tuple(reference.shape) != STATIC_OUTPUT_SHAPE:
        raise RuntimeError(
            f"Model output shape {tuple(reference.shape)} does not match "
            f"{STATIC_OUTPUT_SHAPE}"
        )

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        }

    log.info("Exporting ONNX: %s", output_path)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
    )

    if verify:
        import onnx

        graph = onnx.load(str(output_path))
        onnx.checker.check_model(graph)
        log.info("ONNX checker passed: %s", output_path)

    log.info("Export complete")
    log.info("input : %s %s", INPUT_NAME, STATIC_INPUT_SHAPE)
    log.info("output: %s %s", OUTPUT_NAME, STATIC_OUTPUT_SHAPE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RayBudgetFCN to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export dynamic batch axis only. TensorRT builder still profiles batch=1.",
    )
    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip onnx.checker validation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        verify=not args.skip_onnx_check,
    )


if __name__ == "__main__":
    main()

