"""End-to-end integration test: train → predict → evaluate for UNet transformer."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tracksdata as td

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_unet_transformer import train, DEFAULT_AUGMENTATIONS
from predict_unet_transformer import predict_video, build_graph, load_model, PredictConfig
from tracking_cellmot.io import open_dataset
from tracking_cellmot.metrics import evaluate, node_recall


# 5-frame clip extracted from the full dataset: frames 26–30, division at t=2.
_FIXTURE_DIR = Path(__file__).parent / "data" / "division_clip"
DS_PATH = _FIXTURE_DIR / "division_clip"

# Architecture used in the test (small for speed).
_TEST_CONFIG = {
    "unet_out_channels": 16,
    "unet_layers": [16, 32],
    "unet_stem": True,
    "downsample": [4, 4, 4],
    "pool_kernel_um": 5.0,
    "predict": {
        "threshold": 0.5,
        "det_tta": False,
        "max_parents_per_node": 1,
        "max_children_per_node": 2,
        "use_ilp": False,
        "edge_activation": "softmax",
    },
}


def _seed_everything(seed: int = 42) -> None:
    # Warm up CUDA to ensure consistent kernel selection across parametrized runs.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _ = torch.zeros(1, device="cuda")  # trigger lazy CUDA init
        torch.cuda.synchronize()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.mark.slow
@pytest.mark.parametrize("window_size", [2])
def test_unet_transformer_overfit_and_evaluate(tmp_path: Path, window_size: int) -> None:
    """Train UNet transformer on a single video, predict, and assert jaccard == 1."""
    _seed_everything()

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    model = train(
        data_dir=_FIXTURE_DIR,
        fold=0,
        splits_file=_FIXTURE_DIR / "dataset_splits.json",  # ignored when debug_video is set
        method="unet_transformer",
        n_epochs=10,
        lr=1e-3,
        batch_size=32,
        num_workers=8,
        max_iters=200,
        unet_out_channels=_TEST_CONFIG["unet_out_channels"],
        unet_layers=_TEST_CONFIG["unet_layers"],
        downsample=tuple(_TEST_CONFIG["downsample"]),
        det_loss_weight=1e0,
        det_neg_weight=5e-2,
        debug_video=DS_PATH,
        seed=42,
        window_size=window_size,
        augmentations=None,
        pool_kernel_um=_TEST_CONFIG["pool_kernel_um"],
    )

    # Save weights + config so load_model can reconstruct the architecture.
    save_path = weights_dir / "edge_predictor_best.pth"
    torch.save(model.state_dict(), save_path)
    test_config = {**_TEST_CONFIG, "window_size": window_size}
    (weights_dir / "config.json").write_text(json.dumps(test_config))

    # Reload from disk (tests the save/load round-trip).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model, window_size, downsample = load_model(save_path, device)

    # Predict on the same video.
    cfg = PredictConfig(**_TEST_CONFIG["predict"], pool_kernel_um=_TEST_CONFIG["pool_kernel_um"])
    coords, edges = predict_video(loaded_model, DS_PATH, device, cfg, window_size=window_size, downsample=downsample)
    pred_graph = build_graph(coords, edges)

    # Load GT graph for evaluation.
    ds = open_dataset(DS_PATH, require_tracks=True)
    gt_graph = ds.tracks

    er = evaluate(pred_graph, gt_graph, scale=ds.scale)
    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    div_denom = er.division_tp + er.division_fp + er.division_fn
    div_jaccard = er.division_tp / div_denom if div_denom > 0 else float("nan")
    # node_recall requires graph.match() to have been called, which evaluate()
    # skips for empty predicted graphs — guard against that here.
    if pred_graph.num_nodes() > 0 and pred_graph.num_edges() > 0:
        recall = node_recall(pred_graph, gt_graph)
    else:
        recall = 0.0
    n_pred_nodes = pred_graph.num_nodes()
    n_pred_edges = pred_graph.num_edges()
    n_gt_nodes = gt_graph.num_nodes()
    n_gt_edges = gt_graph.num_edges()
    print(
        f"Nodes: {n_pred_nodes} pred / {n_gt_nodes} GT | "
        f"Edges: {n_pred_edges} pred / {n_gt_edges} GT | "
        f"Jaccard: {jaccard:.4f} | Node recall: {recall:.4f} | Division Jaccard: {div_jaccard:.4f}"
    )
    assert jaccard >= 0.95, f"Expected jaccard >= 0.95 after overfitting, got {jaccard:.4f}"
    assert recall == 1.0, f"Expected node_recall=1.0, got {recall:.4f}"
    assert div_jaccard == 1.0 or np.isnan(div_jaccard), f"Expected division_jaccard=1.0 (or NaN if no divisions), got {div_jaccard:.4f}"
