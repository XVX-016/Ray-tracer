"""
Phase 2 tile-space FCN for intelligent ray budget allocation.

The network performs exact 16x spatial reduction from 1080p input to the
Phase 1 structural grid. Height is padded once from 1080 to 1088 so four
stride-2 stages land exactly on 68 tile rows. Width is already divisible by
16 and lands on 120 tile columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

IMG_H_IN = 1080
IMG_W_IN = 1920
PADDED_H = 1088
PADDED_W = 1920
PAD_BOTTOM = PADDED_H - IMG_H_IN
PAD_RIGHT = PADDED_W - IMG_W_IN

IN_CHANNELS = 7
NUM_CLASSES = 6
TILE_X = 120
TILE_Y = 68


class ConvBNAct(nn.Module):
    """TensorRT-friendly Conv2d + BatchNorm2d + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1.0e-5, momentum=0.03),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    """Low-FLOP residual block used only after reaching tile space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1.0e-5, momentum=0.03),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x + residual)


class RayBudgetFCN(nn.Module):
    """
    Lightweight FCN for per-tile ray budget logits.

    Input:
        [B, 7, 1080, 1920]

    Output:
        [B, 6, 120, 68]
    """

    def __init__(self, width: int = 32, tile_blocks: int = 2) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if tile_blocks < 0:
            raise ValueError("tile_blocks must be non-negative")

        c1 = width
        c2 = width * 2
        c3 = width * 3
        c4 = width * 4

        self.input_pad = nn.ZeroPad2d((0, PAD_RIGHT, 0, PAD_BOTTOM))
        self.stage1 = ConvBNAct(IN_CHANNELS, c1, stride=2)
        self.stage2 = ConvBNAct(c1, c2, stride=2)
        self.stage3 = ConvBNAct(c2, c3, stride=2)
        self.stage4 = ConvBNAct(c3, c4, stride=2)
        self.tile_refine = nn.Sequential(
            *[DepthwiseSeparableBlock(c4) for _ in range(tile_blocks)]
        )
        self.head = nn.Conv2d(c4, NUM_CLASSES, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        assert x.shape == (batch, IN_CHANNELS, IMG_H_IN, IMG_W_IN), (
            f"Expected input [B,{IN_CHANNELS},{IMG_H_IN},{IMG_W_IN}], "
            f"got {tuple(x.shape)}"
        )

        x = self.input_pad(x)
        assert x.shape == (batch, IN_CHANNELS, PADDED_H, PADDED_W), (
            f"After input_pad expected {(batch, IN_CHANNELS, PADDED_H, PADDED_W)}, "
            f"got {tuple(x.shape)}"
        )

        x = self.stage1(x)
        assert x.shape[2:] == (544, 960), f"stage1 spatial mismatch: {tuple(x.shape)}"

        x = self.stage2(x)
        assert x.shape[2:] == (272, 480), f"stage2 spatial mismatch: {tuple(x.shape)}"

        x = self.stage3(x)
        assert x.shape[2:] == (136, 240), f"stage3 spatial mismatch: {tuple(x.shape)}"

        x = self.stage4(x)
        assert x.shape[2:] == (TILE_Y, TILE_X), f"stage4 spatial mismatch: {tuple(x.shape)}"

        x = self.tile_refine(x)
        assert x.shape[2:] == (TILE_Y, TILE_X), f"tile_refine mismatch: {tuple(x.shape)}"

        logits_yx = self.head(x)
        assert logits_yx.shape == (batch, NUM_CLASSES, TILE_Y, TILE_X), (
            f"Head expected [B,{NUM_CLASSES},{TILE_Y},{TILE_X}], "
            f"got {tuple(logits_yx.shape)}"
        )

        logits = logits_yx.permute(0, 1, 3, 2).contiguous()
        assert logits.shape == (batch, NUM_CLASSES, TILE_X, TILE_Y), (
            f"Output expected [B,{NUM_CLASSES},{TILE_X},{TILE_Y}], "
            f"got {tuple(logits.shape)}"
        )
        return logits


def build_model(checkpoint_path: str | Path | None = None) -> RayBudgetFCN:
    model = RayBudgetFCN()
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = RayBudgetFCN().eval()
    x = torch.zeros(1, IN_CHANNELS, IMG_H_IN, IMG_W_IN)
    with torch.no_grad():
        y = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"params={params:,}")
    print(f"input={tuple(x.shape)} output={tuple(y.shape)}")

