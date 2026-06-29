#!/usr/bin/env python
"""
Train a temporal UNet + transformer edge predictor end-to-end.

The UNet (TemporalUNet3D) runs on each batch of consecutive frame pairs
during training.  Its output feature maps are indexed at integer node
coordinates (round + clamp), concatenated with sinusoidal positional
embeddings, and fed to SimpleNodeTransformer.  Gradients flow back through
the integer indexing into the UNet weights.

Usage:
    uv run scripts/train_unet_transformer.py --split 0 --epochs 50
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import tracksdata as td

from tracking_cellmot.io import invert_time_graph, open_dataset
from tracking_cellmot.models import SimpleNodeTransformer, TemporalUNet3D

from itertools import cycle as _cycle


def compute_gt_transition_matrix(
    gt_ids_t: np.ndarray,
    gt_ids_t1: np.ndarray,
    edge_attrs: pl.DataFrame,
) -> torch.Tensor:
    """Build GT transition matrix directly as a float32 torch tensor."""
    t_to_row = {nid: i for i, nid in enumerate(gt_ids_t)}
    t1_to_col = {nid: i for i, nid in enumerate(gt_ids_t1)}

    matrix = torch.zeros(len(gt_ids_t), len(gt_ids_t1), dtype=torch.float32)
    for source_id, target_id in zip(edge_attrs["source_id"], edge_attrs["target_id"]):
        if source_id in t_to_row and target_id in t1_to_col:
            matrix[t_to_row[source_id], t1_to_col[target_id]] = 1.0

    return matrix


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE on annotated rows and columns (sparse GT — unannotated cells ignored)."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    probs = torch.softmax(logits, dim=0)  # dim=0 intentional: divisions allowed, merges aren't
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    loss = ((1 - p_t) ** 2) * bce

    div_rows = target.sum(dim=1) > 1
    weight = torch.ones_like(loss)
    weight[div_rows] = 1.0

    return (loss * weight)[mask].mean()


def compute_batch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_t: torch.Tensor,
    mask_t1: torch.Tensor,
) -> torch.Tensor:
    """Compute loss over a batch by slicing out real (unpadded) regions."""
    B = logits.shape[0]
    losses = []
    for b in range(B):
        nt = mask_t[b].sum().item()
        nt1 = mask_t1[b].sum().item()
        losses.append(compute_loss(logits[b, :nt, :nt1], target[b, :nt, :nt1]))
    return torch.stack(losses).mean()


def _evaluate_pair(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, int, int]:
    """Per-pair evaluation. Returns (loss, correct, total)."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    if not active_rows.any():
        return 0.0, 0, 0

    loss = compute_loss(logits, target).item()
    probs = torch.softmax(logits, dim=0)
    preds = (probs > 0.5).float()

    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    correct = (preds[mask] == target[mask]).sum().item()
    total = mask.sum().item()

    return loss, correct, total

from augmentations import brightness_augment, flip_augment

DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]
from dataspec import INTERACTIVE, WEIGHTS_PATH

DEFAULT_METHOD = "unet_transformer"
_POS_EMBED_DIM = 8   # per axis; total = 4 axes × _POS_EMBED_DIM = 32


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class FrameWindowData:
    """Metadata for a window of W consecutive frames (no image data stored).

    ``t_start`` is the first frame index so the dataset can retrieve
    ``image[t_start:t_start+n_frames]`` from zarr at batch time.
    """

    t_start: int                        # first frame index in the window
    n_frames: int                       # W (window size)
    pos_feats: list[torch.Tensor]       # W tensors, each (N_i, D)
    coords: list[torch.Tensor]          # W tensors, each (N_i, 3)
    node_counts: list[int]              # GT nodes per frame
    targets: list[torch.Tensor]         # W-1 transition matrices


@dataclass(frozen=True)
class VideoMeta:
    """Lightweight per-video metadata enabling on-demand frame loading.

    Workers open the zarr file themselves and load only the two frames
    they need per batch item.  No full video tensor is kept in RAM
    during training.
    """

    zarr_path: Path
    image_shape: tuple[int, ...]  # (T, Z_ds, Y_ds, X_ds) after downsampling
    downsample: tuple[int, ...]   # spatial downsample strides (Z, Y, X)
    voxel_size: tuple[float, ...] # physical voxel size = scale * downsample
    q_low: float                  # 0.1% quantile for normalization
    q_high: float                 # 99.9% quantile for normalization


# =============================================================================
# Data preparation
# =============================================================================

def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> np.ndarray:
    """Sinusoidal positional embeddings for node coordinates (no intensity term).

    Parameters
    ----------
    coords : np.ndarray
        (N, 4) with columns [t, z, y, x].
    image_shape : tuple
        Full image shape (T, Z, Y, X) used for normalisation.
    pos_embed_dim : int
        Half-dimension per axis (sin half + cos half).

    Returns
    -------
    np.ndarray
        Shape (N, 4 * pos_embed_dim), float32.
    """
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def get_window_data(
    gt_graph: td.graph.BaseGraph,
    image_shape: tuple[int, ...],
    t_start: int,
    window_size: int = 2,
    downsample: tuple[int, ...] = (1, 1, 1),
) -> FrameWindowData | None:
    """Build a FrameWindowData for ``window_size`` consecutive frames.

    Coordinates are divided by *downsample* (Z, Y, X) so they match the
    downsampled image grid.  *image_shape* should already be the
    downsampled shape.

    Returns ``None`` if any frame in the window has zero GT nodes.
    """
    gt_attrs = gt_graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edge_attrs = gt_graph.edge_attrs(attr_keys=["source_id", "target_id"])

    ds = np.array(downsample, dtype=np.float32)  # (3,)

    per_frame_ids: list[np.ndarray] = []
    pos_feats: list[torch.Tensor] = []
    coords_list: list[torch.Tensor] = []
    node_counts: list[int] = []

    for i in range(window_size):
        t = t_start + i
        gt_t = gt_attrs.filter(pl.col("t") == t)
        if len(gt_t) == 0:
            return None

        gt_coords_t = gt_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) / ds
        gt_ids = gt_t["node_id"].to_numpy()
        n_gt = len(gt_coords_t)

        full_coords = np.column_stack([np.full(n_gt, t, dtype=np.float32), gt_coords_t])

        pos_feats.append(torch.from_numpy(extract_pos_features(full_coords, image_shape)))
        coords_list.append(torch.from_numpy(gt_coords_t))
        per_frame_ids.append(gt_ids)
        node_counts.append(n_gt)

    targets: list[torch.Tensor] = []
    for i in range(window_size - 1):
        targets.append(compute_gt_transition_matrix(
            per_frame_ids[i], per_frame_ids[i + 1], edge_attrs,
        ))

    return FrameWindowData(
        t_start=t_start,
        n_frames=window_size,
        pos_feats=pos_feats,
        coords=coords_list,
        node_counts=node_counts,
        targets=targets,
    )


def pad_window(
    window: FrameWindowData,
    max_nodes: int,
) -> dict[str, torch.Tensor]:
    """Pad GT nodes to ``max_nodes``; returns metadata only.

    Stores per-frame data as ``(W, max_nodes, ...)`` and per-pair targets as
    ``(W-1, max_nodes, max_nodes)``.
    """
    W = window.n_frames
    D = window.pos_feats[0].shape[1]
    M = max_nodes

    pos_feats = torch.zeros(W, M, D, dtype=torch.float32)
    coords = torch.zeros(W, M, 3, dtype=torch.float32)
    masks = torch.zeros(W, M, dtype=torch.bool)
    node_counts = torch.zeros(W, dtype=torch.long)

    for i in range(W):
        n = window.node_counts[i]
        pos_feats[i, :n] = window.pos_feats[i]
        coords[i, :n] = window.coords[i]
        masks[i, :n] = True
        node_counts[i] = n

    targets = torch.zeros(W - 1, M, M, dtype=torch.float32)
    for i in range(W - 1):
        nt = window.node_counts[i]
        nt1 = window.node_counts[i + 1]
        targets[i, :nt, :nt1] = window.targets[i]

    return {
        "t_start": window.t_start,
        "n_frames": W,
        "pos_feats": pos_feats,
        "coords": coords,
        "masks": masks,
        "targets": targets,
        "node_counts": node_counts,
    }


class FrameWindowDataset(Dataset):
    """Fixed-size dataset of UNet frame windows for batched training.

    Accepts a list of ``(VideoMeta, windows)`` tuples — one per source video.
    No image data is stored in RAM.  Each ``__getitem__`` call opens the zarr
    file for that video, reads W frames, and normalises them on the CPU.
    """

    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[FrameWindowData]]],
        max_nodes: int | None = None,
        augmentations: list | None = None,
    ):
        all_windows = [w for _, windows in video_data for w in windows]
        if max_nodes is None:
            max_nodes = max(max(w.node_counts) for w in all_windows)

        self.max_nodes = max_nodes
        self.augmentations = augmentations or []

        self._data: list[tuple[dict, VideoMeta]] = []
        for video_meta, windows in video_data:
            for window in windows:
                meta = pad_window(window, max_nodes)
                self._data.append((meta, video_meta))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        meta, vm = self._data[idx]
        t_start = meta["t_start"]
        W = meta["n_frames"]
        dz, dy, dx = vm.downsample

        z = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
        target_shape = list(vm.image_shape[1:])

        # Strided read from zarr — spatial downsample at I/O time: (W, Z_ds, Y_ds, X_ds)
        raw = z[t_start : t_start + W, ::dz, ::dy, ::dx].astype(np.float32)
        imgs = torch.from_numpy((raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)).clamp(0.0)

        if list(imgs.shape[1:]) != target_shape:
            imgs = F.interpolate(
                imgs[:, None], size=target_shape,
                mode="trilinear", align_corners=False,
            )[:, 0]

        if self.augmentations:
            rng = np.random.default_rng()
            c, m = meta["coords"], meta["masks"]
            for aug in self.augmentations:
                imgs, c, m = aug(imgs, c, m, rng=rng)
            meta = {**meta, "coords": c, "masks": m}

        return {
            **meta,
            "imgs": imgs.half(),  # (W, *spatial)
            "image_shape": torch.tensor(vm.image_shape, dtype=torch.long),
            "voxel_size": torch.tensor(vm.voxel_size, dtype=torch.float32),
            "downsample": torch.tensor(vm.downsample, dtype=torch.float32),
        }


def load_dataset_windows(
    ds_path: Path,
    window_size: int = 2,
    invert_time: bool = False,
    max_frames: int | None = None,
    downsample: tuple[int, ...] = (1, 1, 1),
) -> tuple[VideoMeta, list[FrameWindowData]]:
    """Load per-window metadata and video stats for one dataset.

    Uses ``open_dataset(load_image=False)`` to read zarr metadata and tracks
    without loading the full image into RAM.

    Returns
    -------
    tuple[VideoMeta, list[FrameWindowData]]
        Lightweight video metadata and per-window node data.  No image tensor.
    """
    ds = open_dataset(ds_path, normalize=False, require_tracks=True,
                      load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")

    image_shape = ds.image_shape
    tracks = ds.tracks
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))

    if invert_time:
        tracks = invert_time_graph(tracks, max_t=image_shape[0])

    if max_frames is not None:
        image_shape = (max_frames, *image_shape[1:])
        tracks = tracks.filter(td.NodeAttr("t") < max_frames).subgraph()

    video_meta = VideoMeta(
        zarr_path=ds.zarr_path,
        image_shape=image_shape,
        downsample=downsample,
        voxel_size=voxel_size,
        q_low=float(ds.quantiles["0.001"]),
        q_high=float(ds.quantiles["0.999"]),
    )

    windows: list[FrameWindowData] = []
    for t in range(image_shape[0] - window_size + 1):
        data = get_window_data(tracks, image_shape, t, window_size, downsample=downsample)
        if data is not None:
            windows.append(data)

    return video_meta, windows


# =============================================================================
# Model
# =============================================================================

class UNetNodeTransformer(nn.Module):
    """TemporalUNet3D encoder + SimpleNodeTransformer edge predictor.

    Forward pass:
      1. Stack frames t and t+1 → (B, 2, 1, *spatial) → UNet → (B, 2, C_feat, *spatial)
      2. Integer-index feature maps at node coords (round + clamp; differentiable)
      3. Concatenate with sinusoidal positional embeddings
      4. Cross-attention transformer → (B, max_nodes, max_nodes) edge logits
    """

    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def _index_features(
        self,
        feat_maps: torch.Tensor,  # (B, C, *spatial)
        coords: torch.Tensor,     # (B, max_nodes, 3)
        mask: torch.Tensor,       # (B, max_nodes) bool
    ) -> torch.Tensor:
        """Integer-index feat_maps at node positions; padded slots → zeros.

        Gradients flow through the *feature map values* but NOT through the
        coordinates (integer indexing is non-differentiable w.r.t. position).
        """
        B, C = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]

        out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
        for b in range(B):
            nt = int(mask[b].sum().item())
            if nt == 0:
                continue
            z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
            y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
            x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
            out[b, :nt] = feat_maps[b, :, z, y, x].T
        return out

    def detect(
        self,
        frame: torch.Tensor,  # (*spatial) — single pre-downsampled frame
    ) -> torch.Tensor:
        """Run UNet + detection head on a single frame.

        Returns
        -------
        torch.Tensor
            (*spatial) detection logits at the input (already downsampled) resolution.
        """
        # Duplicate the frame into a fake pair so the temporal UNet can run.
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, *spatial)
        unet_out = self.unet(pair)          # (1, 2, C_feat, *spatial)
        det = self.detect_head(unet_out[0, 0:1])  # (1, 1, *spatial)
        return det[0, 0]  # (*spatial)

    def encode(
        self,
        imgs: torch.Tensor,  # (B, W, *spatial) — already downsampled
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run UNet encoder on W pre-downsampled frames.

        Returns ``(unet_out, det_logits)`` where *unet_out* is
        ``(B, W, C_feat, *spatial)`` and *det_logits* is a list of W
        tensors each ``(B, 1, *spatial)``.
        """
        window = imgs.unsqueeze(2)  # (B, W, 1, *spatial)
        unet_out = self.unet(window)  # (B, W, C_feat, *spatial)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]
        return unet_out, det_logits

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,  # (B, N_src, C_feat) pre-indexed
        unet_feat_tgt: torch.Tensor,  # (B, N_tgt, C_feat) pre-indexed
        coords_src: torch.Tensor,     # (B, N_src, 3)
        coords_tgt: torch.Tensor,     # (B, N_tgt, 3)
        pos_feat_src: torch.Tensor,   # (B, N_src, pos_feat_dim)
        pos_feat_tgt: torch.Tensor,   # (B, N_tgt, pos_feat_dim)
        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
    ) -> torch.Tensor:
        """Run transformer edge predictor on pre-indexed UNet features."""
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)


# =============================================================================
# Detection loss
# =============================================================================


def compute_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.1,
) -> torch.Tensor:
    """BCE detection loss: GT node voxels are positive, all others lightly penalised.

    Positive and negative terms are normalised by count so that each
    contributes unit magnitude before *neg_weight* scaling.

    Parameters
    ----------
    det_logits : torch.Tensor
        (B, 1, Z, Y, X) raw logits from the detection head.
    coords : torch.Tensor
        (B, max_nodes, 3) GT node coordinates in downsampled space.
    mask : torch.Tensor
        (B, max_nodes) boolean mask for real (non-padded) nodes.
    neg_weight : float
        Weight for negative (non-GT) voxels.  Positives get weight 1.0.
    """
    B = det_logits.shape[0]
    spatial = det_logits.shape[2:]  # (Z, Y, X)
    logits = det_logits[:, 0]  # (B, Z, Y, X)
    target = torch.zeros_like(logits)

    # Mark GT voxels as positive for each sample in the batch.
    nt = mask.sum(dim=1).long()       # (B,)
    for b in range(B):
        n_gt = nt[b]
        if n_gt <= 0:
            continue
        gt_coords = coords[b, :nt[b]]
        zi = gt_coords[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt_coords[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt_coords[:, 2].long().clamp(0, spatial[2] - 1)
        n_unique = len(torch.unique(torch.stack([zi, yi, xi], dim=1), dim=0))
        if n_unique < n_gt:
            import warnings
            warnings.warn(
                f"Sample {b}: {n_gt - n_unique}/{n_gt} GT nodes collapsed to "
                f"duplicate voxels after downsampling — these are undetectable.",
                stacklevel=2,
            )
        target[b, zi, yi, xi] = 1.0

    # Per-sample normalisation: weight_pos = 1/n_pos, weight_neg = neg_weight/n_neg.
    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)          # (B,)
    n_neg = (target.numel() // B - n_pos).clamp(min=1)             # (B,)
    # Broadcast to spatial dims.
    shape = (B,) + (1,) * len(spatial)
    w_pos = (1.0 / n_pos).reshape(shape)
    w_neg = (neg_weight / n_neg).reshape(shape)
    weight = torch.where(target == 1.0, w_pos, w_neg)

    return F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, reduction="sum",
    ) / B


# =============================================================================
# Detection → matching → edge targets (used during training, GPU-vectorised)
# =============================================================================


def _pos_embed_torch(
    coords: torch.Tensor,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = _POS_EMBED_DIM,
) -> torch.Tensor:
    """Batched sinusoidal positional embeddings (pure torch, stays on device).

    Parameters
    ----------
    coords : (*, 4) with columns [t, z, y, x].
    image_shape : (T, Z, Y, X) for normalisation.

    Returns
    -------
    torch.Tensor  shape (*, 4 * pos_embed_dim).
    """
    shape_t = torch.tensor(image_shape, dtype=torch.float32, device=coords.device)
    norms = coords / shape_t.clamp(min=1)  # (*, 4)
    freqs = (2.0 ** torch.arange(pos_embed_dim // 2, device=coords.device, dtype=torch.float32)) * torch.pi
    parts = []
    for ax in range(4):
        angles = norms[..., ax].unsqueeze(-1) * freqs  # (*, D//2)
        parts.extend([angles.sin(), angles.cos()])
    return torch.cat(parts, dim=-1)


def detect_and_match(
    det_logits: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    image_shape: tuple[int, ...],
    det_threshold: float = 0.3,
    pool_kernel_um: float = 5.0,
    max_match_distance: float = 5.0,
    voxel_size: tuple[float, ...] | None = None,
    frame_index: int = 0,
    window_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect peaks in logits, match to GT, return padded edge-prediction inputs.

    All coordinates are in downsampled space.

    Parameters
    ----------
    det_logits : (B, 1, Z, Y, X)
        Raw detection logits.
    gt_coords : (B, max_nodes, 3)
        GT node coordinates (downsampled).
    mask : (B, max_nodes)
        Boolean mask for real (non-padded) nodes.
    image_shape : (T, Z, Y, X) downsampled shape for pos-embed normalisation.
    det_threshold : minimum logit to accept a peak.
    pool_kernel_um : local-max suppression distance in microns.
    max_match_distance : maximum physical distance for detection→GT matching.
    voxel_size : (Z, Y, X) physical voxel size (scale * downsample).
        When provided, distances are computed in physical units.

    Returns
    -------
    coords : (B, M, 3)  detected coordinates (downsampled).
    pos    : (B, M, 4*_POS_EMBED_DIM)  positional embeddings.
    mask   : (B, M)  bool.
    matches : list[Tensor]  per-sample match arrays.
    """
    B = det_logits.shape[0]
    device = det_logits.device
    vs = (
        torch.tensor(voxel_size, dtype=torch.float32, device=device)
        if voxel_size is not None
        else None
    )

    # Convert physical suppression distance (microns) to per-axis voxel kernel.
    if voxel_size is not None:
        pool_kernel = tuple(
            max(1, k if k % 2 == 1 else k + 1)
            for k in (max(1, round(pool_kernel_um / s)) for s in voxel_size)
        )
    else:
        k = max(1, round(pool_kernel_um))
        pool_kernel = (k if k % 2 == 1 else k + 1,) * 3

    pad = tuple(k // 2 for k in pool_kernel)

    # --- 1. Batched local-max peak detection on GPU -------------------------
    with torch.no_grad():
        pooled = F.max_pool3d(det_logits, pool_kernel, stride=1, padding=pad)
        is_peak = (det_logits == pooled) & (det_logits > det_threshold)
        # (N_total, 4): columns [b, z, y, x]
        peak_idx = torch.nonzero(is_peak[:, 0])

    batch_ids = peak_idx[:, 0]                         # (N_total,)
    peak_coords = peak_idx[:, 1:].float()              # (N_total, 3)

    # --- 2. Per-sample matching (small loop, torch.cdist on GPU) ------------
    nt_per_sample = mask.sum(dim=1).long()              # (B,)

    sample_matches: list[torch.Tensor] = []             # each (n_det,) long
    sample_coords: list[torch.Tensor] = []              # each (n_det, 3)
    max_det = 0

    for b in range(B):
        sel = batch_ids == b
        det_b = peak_coords[sel]                        # (n_det, 3)
        n_det = det_b.shape[0]
        nt = int(nt_per_sample[b].item())
        gt_b = gt_coords[b, :nt]                        # (n_gt, 3)
        n_gt = gt_b.shape[0]

        matched = torch.full((n_det,), -1, dtype=torch.long, device=device)
        if n_det > 0 and n_gt > 0:
            if vs is not None:
                dists = torch.cdist(det_b * vs, gt_b * vs)
            else:
                dists = torch.cdist(det_b, gt_b)
            min_d, min_i = dists.min(dim=1)             # (n_det,)
            order = min_d.argsort()
            gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True

        sample_matches.append(matched)
        sample_coords.append(det_b)
        if n_det > max_det:
            max_det = n_det

    max_det = max(max_det, 1)

    # --- 3. Pad coords / pos / mask (on GPU) --------------------------------
    padded_coords = torch.zeros(B, max_det, 3, device=device)
    padded_mask = torch.zeros(B, max_det, dtype=torch.bool, device=device)
    for b in range(B):
        n = sample_coords[b].shape[0]
        if n == 0:
            continue
        padded_coords[b, :n] = sample_coords[b]
        padded_mask[b, :n] = True

    # Positional embeddings (batched, on GPU).
    # Use window-relative time (0, 1, ..., W-1) normalised by W, not absolute frame index.
    t_col = torch.full((B, max_det, 1), frame_index, device=device, dtype=torch.float32)
    full_coords = torch.cat([t_col, padded_coords], dim=-1)  # (B, M, 4)
    pos_shape = (window_size,) + image_shape[1:] if window_size is not None else image_shape
    padded_pos = _pos_embed_torch(full_coords, pos_shape)     # (B, M, D)

    return padded_coords, padded_pos, padded_mask, sample_matches


def build_matched_edge_targets(
    match_t: list[torch.Tensor],
    match_t1: list[torch.Tensor],
    gt_target: torch.Tensor,
    max_det_t: int,
    max_det_t1: int,
) -> torch.Tensor:
    """Build (B, max_det_t, max_det_t1) edge targets via vectorised indexing."""
    B = gt_target.shape[0]
    device = gt_target.device
    target = torch.zeros(B, max_det_t, max_det_t1, device=device)

    for b in range(B):
        mt = match_t[b]                                 # (n_det_t,)
        mt1 = match_t1[b]                               # (n_det_t1,)
        gt_trans = gt_target[b]                          # (N_gt_t, N_gt_t1)
        n_t, n_t1 = mt.shape[0], mt1.shape[0]
        if n_t == 0 or n_t1 == 0:
            continue

        valid_t = mt >= 0
        valid_t1 = mt1 >= 0
        valid_mask = valid_t.unsqueeze(1) & valid_t1.unsqueeze(0)  # (n_t, n_t1)
        safe_t = mt.clamp(min=0)
        safe_t1 = mt1.clamp(min=0)
        block = gt_trans[safe_t][:, safe_t1] * valid_mask.float()
        target[b, :n_t, :n_t1] = block

    return target


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: UNetNodeTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    det_loss_weight: float = 0.1,
    det_neg_weight: float = 0.1,
    max_iters: int | None = None,
    pool_kernel_um: float = 5.0,
) -> tuple[float, float]:
    """Train for one epoch, return (avg edge loss, avg detection loss).

    When *max_iters* is set, the loader is cycled repeatedly until that many
    iterations have been performed, regardless of dataset size.
    """
    model.train()
    total_edge_loss = 0.0
    total_det_loss = 0.0
    n_samples = 0

    if max_iters is not None:
        batch_iter = _cycle(loader)
        pbar = tqdm(range(max_iters), desc="  iters", leave=False, disable=not INTERACTIVE)
    else:
        batch_iter = iter(loader)
        pbar = tqdm(range(len(loader)), desc="  batches", leave=False, disable=not INTERACTIVE)

    t_data, t_forward, t_backward = 0.0, 0.0, 0.0
    t0 = time.perf_counter()

    for _ in pbar:
        batch = next(batch_iter)

        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)       # (B, W, *sp)
        coords = batch["coords"].to(device, non_blocking=True)                         # (B, W, M, 3)
        pos_feats = batch["pos_feats"].to(device, non_blocking=True)                   # (B, W, M, D)
        masks = batch["masks"].to(device, non_blocking=True)                           # (B, W, M)
        targets = batch["targets"].to(device, non_blocking=True)                       # (B, W-1, M, M)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)                                   # (3,)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_data += t1 - t0

        B, W = imgs.shape[:2]

        # --- 1. Encode: UNet features + detection logits --------------------
        unet_out, det_logits = model.encode(imgs)
        # unet_out: (B, W, C, *spatial),  det_logits: list of W × (B, 1, *spatial)

        # --- 2. Detection loss over all W frames ---------------------------
        det_losses = [
            compute_detection_loss(
                det_logits[i], coords[:, i], masks[:, i],
                det_neg_weight,
            )
            for i in range(W)
        ]
        det_loss = sum(det_losses) / W

        # --- 3. Per-frame detect → match → index UNet features -------------
        frame_det: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              list[torch.Tensor], torch.Tensor]] = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i], coords[:, i], masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i, window_size=W,
            )
            unet_feat = model._index_features(
                unet_out[:, i], det_c, det_m,
            )
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

        # --- 4. Per-pair edge prediction and loss -------------------------
        block_losses = []
        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3],
                targets[:, i], ns, nt,
            )
            edge_logits = model.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1],
                frame_det[i][2], frame_det[i + 1][2],
            )
            block_losses.append(compute_batch_loss(
                edge_logits, pair_target,
                frame_det[i][2], frame_det[i + 1][2],
            ))
        edge_loss = sum(block_losses) / len(block_losses)

        # --- 5. Combined loss -----------------------------------------------
        loss = edge_loss + det_loss_weight * det_loss

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_forward += t2 - t1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_backward += t3 - t2

        total_edge_loss += edge_loss.item() * B
        total_det_loss += det_loss.item() * B
        n_samples += B

        t0 = time.perf_counter()

    t_total = t_data + t_forward + t_backward
    if t_total > 0:
        print(
            f"  [timing] data: {t_data:.1f}s ({100*t_data/t_total:.0f}%) | "
            f"forward: {t_forward:.1f}s ({100*t_forward/t_total:.0f}%) | "
            f"backward: {t_backward:.1f}s ({100*t_backward/t_total:.0f}%) | "
            f"total: {t_total:.1f}s"
        )

    return (
        total_edge_loss / max(n_samples, 1),
        total_det_loss / max(n_samples, 1),
    )


@torch.no_grad()
def evaluate(
    model: UNetNodeTransformer,
    loader: DataLoader,
    device: torch.device,
    pool_kernel_um: float = 5.0,
) -> tuple[float, float, float]:
    """Evaluate model using detect→match→predict (same path as training).

    Returns (avg_loss, accuracy, node_recall).
    """
    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        pos_feats = batch["pos_feats"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)

        B, W = imgs.shape[:2]
        unet_out, det_logits = model.encode(imgs)
        frame_det: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              list[torch.Tensor], torch.Tensor]] = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i], coords[:, i], masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i, window_size=W,
            )
            unet_feat = model._index_features(
                unet_out[:, i], det_c, det_m,
            )
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

            # Node recall: how many GT nodes were matched by a detection?
            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = (matches[b] >= 0).sum().item()
                gt_total += n_gt
                gt_matched += n_matched

        # Per-pair evaluation.
        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3],
                targets[:, i], ns, nt,
            )
            pair_logits = model.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1],
                frame_det[i][2], frame_det[i + 1][2],
            )

            for b in range(B):
                ns_b = int(frame_det[i][2][b].sum().item())
                nt_b = int(frame_det[i + 1][2][b].sum().item())
                pair_loss, pair_correct, pair_total = _evaluate_pair(
                    pair_logits[b, :ns_b, :nt_b], pair_target[b, :ns_b, :nt_b],
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                n_pairs += 1

    node_recall = gt_matched / max(gt_total, 1)
    return total_loss / max(n_pairs, 1), correct / max(total, 1), node_recall


# =============================================================================
# Main training function
# =============================================================================

def train(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    method: str = DEFAULT_METHOD,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    num_workers: int = 4,  # benchmark_preload.py: 4 workers, no pin_memory is optimal
    unet_out_channels: int = 32,
    unet_layers: list[int] | None = None,
    unet_weights: Path | None = None,
    downsample: tuple[int, ...] = (1, 4, 4),
    det_loss_weight: float = 1e1,
    det_neg_weight: float = 1e-2,
    max_iters: int | None = None,
    debug_video: Path | None = None,
    seed: int | None = None,
    max_frames: int | None = None,
    window_size: int = 2,
    augmentations: list | None = DEFAULT_AUGMENTATIONS,
    pool_kernel_um: float = 5.0,
    data_parallel: bool = True,
) -> UNetNodeTransformer:
    """Train on one fold from a pre-computed splits file.

    If *debug_video* is set the splits file is ignored and that single dataset
    is used for both train and test (quick sanity-check / overfitting run).
    """
    if unet_layers is None:
        unet_layers = [32, 64, 128]

    if debug_video is not None:
        train_files = test_files = [debug_video]
        print(f"Debug mode: using single video {debug_video.name}", flush=True)
    else:
        folds = json.loads(splits_file.read_text())
        fold_data = folds[fold]
        train_files = [data_dir / name for name in fold_data["train"]]
        test_files = [data_dir / name for name in fold_data["test"]]
        print(f"Fold {fold}: {len(train_files)} train, {len(test_files)} test", flush=True)

    output_dir = WEIGHTS_PATH / method / f"split_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arch config so predict_unet_transformer can reconstruct the model.
    model_config = {
        "unet_out_channels": unet_out_channels,
        "unet_layers": unet_layers,
        "downsample": list(downsample),
        "window_size": window_size,
        "pool_kernel_um": pool_kernel_um,
    }
    (output_dir / "config.json").write_text(json.dumps(model_config, indent=2))

    def _load(
        files: list[Path], desc: str,
    ) -> list[tuple[VideoMeta, list[FrameWindowData]]]:
        print(f"Loading {desc} ({len(files)} datasets)...", flush=True)
        data: list[tuple[VideoMeta, list[FrameWindowData]]] = []
        for f in tqdm(files, desc=desc, disable=not INTERACTIVE):
            video_meta, windows = load_dataset_windows(
                f, window_size=window_size,
                max_frames=max_frames,
                downsample=downsample,
            )
            data.append((video_meta, windows))
        n_windows = sum(len(w) for _, w in data)
        print(f"  {desc} done: {n_windows} windows total", flush=True)
        return data

    train_video_data = _load(train_files, "train")
    test_video_data = _load(test_files, "test")

    # Compute consistent max_nodes across train + test.
    all_windows = [w for _, ws in train_video_data + test_video_data for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    print(f"max_nodes={max_nodes}", flush=True)

    pos_feat_dim = 4 * _POS_EMBED_DIM

    train_ds = FrameWindowDataset(train_video_data, max_nodes=max_nodes, augmentations=augmentations)
    test_ds = FrameWindowDataset(test_video_data, max_nodes=max_nodes)
    g = None
    worker_init_fn = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

        def worker_init_fn(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, pin_memory=False,
        generator=g, worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, pin_memory=False,
        generator=g, worker_init_fn=worker_init_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=unet_out_channels,
        layers=unet_layers,
    )
    if unet_weights is not None:
        state = torch.load(unet_weights, map_location="cpu", weights_only=True)
        missing, unexpected = unet.load_state_dict(state, strict=False)
        print(f"  UNet weights: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)

    # Simple multi-GPU: split the heavy UNet pass across all visible GPUs.
    # Only the UNet is wrapped (it takes/returns plain batched tensors); the
    # detection head and transformer stay on cuda:0. Checkpoints are saved with
    # the DataParallel "module." prefix stripped so they load on a single GPU.
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        model.unet = nn.DataParallel(model.unet)
        print(
            f"DataParallel: UNet across {n_gpus} GPUs "
            f"(effective per-GPU batch {max(1, batch_size // n_gpus)})",
            flush=True,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"Starting training for {n_epochs} epochs (batch_size={batch_size})...", flush=True)

    best_score = 0.0
    save_path = output_dir / "edge_predictor_best.pth"
    pbar = tqdm(range(n_epochs), desc="Training", disable=not INTERACTIVE)
    print(f"Detection loss: weight={det_loss_weight}, neg_weight={det_neg_weight}", flush=True)

    for epoch in pbar:
        t0 = time.monotonic()
        edge_loss, det_loss = train_epoch(
            model, train_loader, optimizer, device, det_loss_weight, det_neg_weight,
            max_iters=max_iters, pool_kernel_um=pool_kernel_um,
        )
        train_time = time.monotonic() - t0

        t0 = time.monotonic()
        test_loss, test_acc, test_recall = evaluate(model, test_loader, device, pool_kernel_um=pool_kernel_um)
        test_time = time.monotonic() - t0

        score = test_acc * test_recall
        is_best = score >= best_score

        if is_best:
            best_score = score
            # Normalise any DataParallel "unet.module." prefix to "unet." so the
            # checkpoint loads on a single GPU (e.g. in the prediction script).
            torch.save(
                {k.replace("unet.module.", "unet.", 1): v for k, v in model.state_dict().items()},
                save_path,
            )

        marker = "*" if is_best else " "
        pbar.set_postfix(edge=f"{edge_loss:.4f}", det=f"{det_loss:.4f}", acc=f"{test_acc:.4f}")
        print(
            f"  Epoch {epoch:3d}/{n_epochs} | edge={edge_loss:.4f} | det={det_loss:.4f} | "
            f"test_loss={test_loss:.4f} | acc={test_acc:.4f} | recall={test_recall:.4f} | best={best_score:.4f} {marker} | "
            f"train={train_time:.1f}s test={test_time:.1f}s",
            flush=True,
        )

    print(f"\nBest score (acc*recall): {best_score:.4f}, saved to {save_path}", flush=True)
    if save_path.exists():
        state = torch.load(save_path, map_location=device, weights_only=True)
        if isinstance(model.unet, nn.DataParallel):
            state = {
                (k.replace("unet.", "unet.module.", 1) if k.startswith("unet.") else k): v
                for k, v in state.items()
            }
        model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train UNet + transformer edge predictor end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--split", type=str, default="0",
                        help="Split index (0-4) or 'all'.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Frames pairs per batch. All images in a fold must share "
                             "the same spatial shape for batch_size > 1.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker processes for parallel frame loading (default: 4).")
    parser.add_argument("--unet-out-channels", type=int, default=32)
    parser.add_argument("--unet-layers", type=str, default="32,64,128",
                        help="Comma-separated UNet channel widths, shallow→deep.")
    parser.add_argument("--unet-weights", type=str, default=None,
                        help="Path to pretrained UNet weights; loaded with strict=False.")
    parser.add_argument("--downsample", type=str, default="1,4,4",
                        help="Comma-separated spatial downsample strides Z,Y,X (default: 1,4,4).")
    parser.add_argument("--det-loss-weight", type=float, default=1e0,
                        help="Weight for detection loss relative to edge loss (default: 1e1).")
    parser.add_argument("--det-neg-weight", type=float, default=1e-2,
                        help="Per-voxel weight for non-GT (negative) voxels in detection loss (default: 1e-2).")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Max training iterations per epoch. None = full epoch.")
    parser.add_argument("--debug-video", type=str, default=None,
                        help="Path to a single dataset for quick debugging. "
                             "Ignores --fold and splits file; trains and evaluates on this video only.")
    parser.add_argument("--window-size", type=int, default=2,
                        help="Number of consecutive frames per training window (default: 2).")
    parser.add_argument("--pool-kernel-um", type=float, default=5.0,
                        help="Local-max suppression distance in microns (default: 5.0).")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true", default=True,
                        help="Split the UNet across all visible GPUs via nn.DataParallel "
                             "when more than one is available (default: on).")
    parser.add_argument("--single-gpu", dest="data_parallel", action="store_false",
                        help="Disable multi-GPU; train on cuda:0 only.")

    args = parser.parse_args()

    from dataspec import DATASET_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else data_dir / "dataset_splits.json"
    unet_layers = [int(x) for x in args.unet_layers.split(",")]
    unet_weights = Path(args.unet_weights) if args.unet_weights else None
    debug_video = Path(args.debug_video) if args.debug_video else None
    downsample = tuple(int(x) for x in args.downsample.split(","))

    folds = [0] if debug_video is not None else (
        range(5) if args.split == "all" else [int(args.split)]
    )
    for fold in folds:
        train(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            method=args.method,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            unet_out_channels=args.unet_out_channels,
            unet_layers=unet_layers,
            unet_weights=unet_weights,
            downsample=downsample,
            det_loss_weight=args.det_loss_weight,
            det_neg_weight=args.det_neg_weight,
            max_iters=args.max_iters,
            debug_video=debug_video,
            window_size=args.window_size,
            pool_kernel_um=args.pool_kernel_um,
            data_parallel=args.data_parallel,
        )


if __name__ == "__main__":
    main()
