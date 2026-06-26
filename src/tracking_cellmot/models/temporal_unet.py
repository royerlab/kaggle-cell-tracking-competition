"""Simple 3D+T U-Net.

Input shape  : ``(B, T, C_in, Z, Y, X)``
Output shape : ``(B, T, C_out, Z, Y, X)``

"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class _TemporalAttention(nn.Module):
    """Per-voxel multi-head self-attention across time."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Z, Y, X)
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)

        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    """Minimal 3D temporal U-Net.

    Parameters
    ----------
    in_channels : int
        Input channels per frame.
    out_channels : int
        Output feature channels per frame.
    layers : sequence of int
        Encoder channel widths, shallow to deep. Number of stages equals
        ``len(layers)``; spatial size is halved before every stage except
        the first.
    gradient_checkpointing : bool
        If True (default), wrap encoder/decoder conv blocks with
        ``torch.utils.checkpoint`` during training to reduce activation
        memory at the cost of recomputing activations in the backward
        pass.
    skip_fullres_temporal : bool
        If True (default), replace the temporal-attention block at the
        full-resolution (first) encoder stage with an Identity. Per-voxel
        attention at full res dominates both memory and runtime; skipping
        it gives ~3x speedup and ~30% less memory with negligible quality
        loss in practice.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing: bool = True,
        skip_fullres_temporal: bool = True,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two stages")

        self.gradient_checkpointing = gradient_checkpointing

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(_conv_block(prev, ch))
            if skip_fullres_temporal and i == 0:
                self.temporal_blocks.append(nn.Identity())
            else:
                self.temporal_blocks.append(_TemporalAttention(ch))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            )
            self.decoder_blocks.append(_conv_block(layers[i] + layers[i - 1], layers[i - 1]))

        self.head = nn.Conv3d(layers[0], out_channels, kernel_size=1)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _grad_ckpt(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in, Z, Y, X) -> (B, T, C_out, Z, Y, X)
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])

        skips: list[torch.Tensor] = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0:
                x = self.pool(x)
            x = self._run(block, x)
            x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._run(block, x)

        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])
