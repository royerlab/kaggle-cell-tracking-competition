#!/usr/bin/env python
"""
Visualize predictions alongside ground truth tracks in napari.

Press the "Next Dataset" button to advance to the next dataset.

Usage:
    uv run visualize/visualize_predictions.py --method baseline --split 0
"""

import argparse
import random
import sys
from pathlib import Path

import napari
import numpy as np
import tracksdata as td

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataspec import DATASET_PATH, PREDICTIONS_PATH, USERNAME
from tracking_cellmot.io import open_dataset
from tracking_cellmot.metrics import _evaluate_matched_graph, evaluate as compute_metric

from visualize_utils import _compute_contrast_limits


VIDEOS_DIR = Path(__file__).resolve().parent.parent / "videos"


def _edges_to_tracks(
    node_coords: dict[int, tuple[float, float, float, float]],
    edges: list[tuple[int, int]],
    track_id_offset: int = 0,
) -> np.ndarray:
    """Convert edges to napari tracks array ``[track_id, t, z, y, x]``.

    Each edge becomes a 2-point track so it renders as a single line segment.
    """
    if not edges:
        return np.empty((0, 5))
    rows: list[list[float]] = []
    for i, (src, tgt) in enumerate(edges):
        tid = track_id_offset + i
        rows.append([tid, *node_coords[src]])
        rows.append([tid, *node_coords[tgt]])
    return np.array(rows)


def _build_node_coords(
    graph: td.graph.BaseGraph,
) -> dict[int, tuple[float, float, float, float]]:
    """Return ``{node_id: (t, z, y, x)}`` for every node in *graph*."""
    attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"]
    )
    return {
        int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]): (row["t"], row["z"], row["y"], row["x"])
        for row in attrs.iter_rows(named=True)
    }


def _classify_edges(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Classify edges using the same definitions as the Jaccard metric.

    Returns
    -------
    tp_edges
        TP: prediction edges with ``matched_edge_mask == True``.
    fp_edges
        FP: prediction edges with ``pred_valid == True`` and
        ``matched_edge_mask == False`` (same ``pred_valid`` as the Jaccard).
    all_pred_edges
        Every prediction edge (context layer).
    fn_edges
        FN: GT edges not covered by any TP prediction edge.
        Expressed as GT node-id pairs.
    """
    import polars as pl

    # Reuse the exact metric logic to get pred_valid and matched_edge_mask
    edge_df = _evaluate_matched_graph(pred_graph, gt_graph)

    SRC = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    TGT = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    MATCHED = td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK

    tp_edges: list[tuple[int, int]] = [
        (row[SRC], row[TGT])
        for row in edge_df.filter(pl.col(MATCHED)).iter_rows(named=True)
    ]
    fp_edges: list[tuple[int, int]] = [
        (row[SRC], row[TGT])
        for row in edge_df.filter(pl.col("pred_valid") & ~pl.col(MATCHED)).iter_rows(named=True)
    ]

    # All prediction edges (from the full graph, not deduplicated)
    all_edge_attrs = pred_graph.edge_attrs(attr_keys=[SRC, TGT])
    all_pred_edges: list[tuple[int, int]] = [
        (row[SRC], row[TGT])
        for row in all_edge_attrs.iter_rows(named=True)
    ]

    # FN: GT edges not matched by any TP prediction edge
    node_attrs = pred_graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    match_map: dict[int, int] = {
        int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]): int(row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
        for row in node_attrs.iter_rows(named=True)
    }
    matched_gt_edge_set: set[tuple[int, int]] = {
        (match_map[src], match_map[tgt]) for src, tgt in tp_edges
    }

    gt_edge_attrs = gt_graph.edge_attrs(attr_keys=[SRC, TGT])
    fn_edges: list[tuple[int, int]] = [
        (row[SRC], row[TGT])
        for row in gt_edge_attrs.iter_rows(named=True)
        if (row[SRC], row[TGT]) not in matched_gt_edge_set
    ]

    return tp_edges, fp_edges, all_pred_edges, fn_edges


def visualize_prediction(
    ds_path: Path,
    pred_geff: Path,
    *,
    max_distance: float | None = None,
    title: str = "napari",
    show_next_button: bool = False,
) -> None:
    """Visualize a single prediction alongside its ground truth."""
    ds = open_dataset(ds_path, require_tracks=True)
    scale = ds.scale

    # Load prediction graph
    result = td.graph.IndexedRXGraph.from_geff(pred_geff)
    pred_graph = result[0] if isinstance(result, tuple) else result

    # Compute matching between prediction and ground truth
    metric_kwargs: dict = dict(scale=scale)
    if max_distance is not None:
        metric_kwargs["max_distance"] = max_distance
    er = compute_metric(pred_graph, ds.tracks, **metric_kwargs)
    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    div_denom = er.division_tp + er.division_fp + er.division_fn
    div_jaccard = er.division_tp / div_denom if div_denom > 0 else float("nan")
    print(
        f"  Edge Jaccard: {edge_jaccard:.4f} "
        f"(TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}) | "
        f"Division Jaccard: {div_jaccard:.4f} "
        f"(TP={er.division_tp} FP={er.division_fp} FN={er.division_fn})"
    )

    # Classify edges using the same TP/FP/FN definitions as the Jaccard metric
    tp_edges, fp_edges, all_pred_edges, fn_edges = _classify_edges(pred_graph, ds.tracks)

    pred_coords = _build_node_coords(pred_graph)
    gt_coords = _build_node_coords(ds.tracks)

    # Image to numpy
    image = ds.image
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
    elif hasattr(image, 'compute'):
        image = image.compute()

    viewer = napari.Viewer(title=title, ndisplay=3)

    # Add image
    contrast_limits = _compute_contrast_limits(image)
    viewer.add_image(
        image,
        scale=scale,
        contrast_limits=contrast_limits,
        name="image",
        colormap="gray",
    )

    # Register single-color colormaps
    from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap

    _EDGE_COLORS = {
        "solid_gold": [[1, 0.84, 0, 1], [1, 0.84, 0, 1]],
        "solid_green": [[0, 0.8, 0, 1], [0, 0.8, 0, 1]],
        "solid_red": [[1, 0, 0, 1], [1, 0, 0, 1]],
        "solid_white": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "solid_blue": [[0.2, 0.4, 1, 1], [0.2, 0.4, 1, 1]],
    }
    for cmap_name, colors in _EDGE_COLORS.items():
        AVAILABLE_COLORMAPS[cmap_name] = Colormap(colors=colors, name=cmap_name)

    # Ground truth tracks
    gt_tracks, gt_graph_dict = ds.napari_tracks()
    if len(gt_tracks) > 0:
        viewer.add_tracks(
            gt_tracks,
            graph=gt_graph_dict,
            tail_width=5,
            tail_length=100,
            opacity=1.0,
            name="Ground truth tracks",
            scale=scale,
            blending="translucent",
            colormap="solid_gold",
            color_by="track_id",
        )

    # Edge layers: each edge is a 2-point napari track
    # White (all predictions) is added first so it sits behind the others.
    n = 0
    edge_layers: list[tuple[str, np.ndarray, str, bool]] = [
        ("All predictions (white)", _edges_to_tracks(pred_coords, all_pred_edges, n), "solid_white", False),
        ("TP edges (green)", _edges_to_tracks(pred_coords, tp_edges, n := n + len(all_pred_edges)), "solid_green", True),
        ("FP edges (red)", _edges_to_tracks(pred_coords, fp_edges, n := n + len(tp_edges)), "solid_red", True),
        ("FN edges (blue)", _edges_to_tracks(gt_coords, fn_edges, n := n + len(fp_edges)), "solid_blue", True),
    ]

    for name, tracks, cmap_name, visible in edge_layers:
        if len(tracks) > 0:
            viewer.add_tracks(
                tracks,
                tail_width=5,
                tail_length=100,
                opacity=1.0,
                name=name,
                scale=scale,
                blending="translucent",
                color_by="track_id",
                colormap=cmap_name,
                visible=visible,
            )

    from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout

    controls = QWidget()
    layout = QVBoxLayout()
    controls.setLayout(layout)

    if show_next_button:
        btn_next = QPushButton("Next Dataset ▶")
        btn_next.clicked.connect(viewer.close)
        layout.addWidget(btn_next)

    sample_name = ds_path.stem

    def _record_video() -> None:
        import imageio

        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = VIDEOS_DIR / f"{sample_name}.mp4"

        time_axis = viewer.dims.axis_labels.index("t") if "t" in viewer.dims.axis_labels else 0
        n_frames = image.shape[time_axis]

        writer = imageio.get_writer(str(out_path), fps=7, macro_block_size=1)
        for t in range(n_frames):
            viewer.dims.set_point(time_axis, t)
            frame = viewer.screenshot(canvas_only=True)
            writer.append_data(frame[:, :, :3])  # drop alpha channel
        writer.close()

        print(f"Video saved to {out_path}")

    btn_record = QPushButton("Record Video")
    btn_record.clicked.connect(_record_video)
    layout.addWidget(btn_record)

    viewer.window.add_dock_widget(controls, area="left", name="Controls")

    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize predictions alongside ground truth in napari.")
    parser.add_argument("--user", type=str, default=USERNAME, help=f"Username for predictions (default: {USERNAME}).")
    parser.add_argument("--method", type=str, required=True, help="Method name (matches predict --method).")
    parser.add_argument("--split", type=int, required=True, help="Split index (0-4).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling datasets (default: 0).")
    parser.add_argument("--slice", type=str, default=None, help="Python slice of dataset list, e.g. '1:3' or ':5'.")
    args = parser.parse_args()

    pred_dir = PREDICTIONS_PATH / args.user / args.method / f"split_{args.split}"
    if not pred_dir.exists():
        print(f"Predictions directory not found: {pred_dir}")
        return

    # Find all prediction geff files that have matching GT
    pred_geffs = sorted(pred_dir.glob("*.geff"))
    pairs: list[tuple[Path, Path]] = []
    for pred_geff in pred_geffs:
        ds_path = DATASET_PATH / pred_geff.stem
        if (DATASET_PATH / f"{pred_geff.stem}.zarr").exists():
            pairs.append((ds_path, pred_geff))

    if not pairs:
        print(f"No matching GT/prediction pairs found")
        return

    random.seed(args.seed)
    random.shuffle(pairs)

    if args.slice is not None:
        s = slice(*[int(x) if x else None for x in args.slice.split(":")])
        pairs = pairs[s]

    print(f"Found {len(pairs)} dataset(s) with predictions (seed={args.seed})")

    for i, (ds_path, pred_geff) in enumerate(pairs):
        print(f"\n[{i + 1}/{len(pairs)}] Loading: {ds_path.stem}")

        visualize_prediction(
            ds_path,
            pred_geff,
            title=f"[{i + 1}/{len(pairs)}] {ds_path.stem}",
            show_next_button=i < len(pairs) - 1,
        )


if __name__ == "__main__":
    main()
