#!/usr/bin/env python
"""Evaluate predictions against ground truth and save results.

Discovers predictions from predictions/{username}/{method}/split_{fold}/*.geff,
loads matching GT from DATASET_PATH/{name}.geff, computes edge and division
TP/FP/FN per sample via :func:`tracking_cellmot.metrics.evaluate`, and upserts
per-sample counts into a shared SQLite DB (WAL mode for concurrency).

Per-run summary prints **cumulative (micro-averaged) Jaccard** for edges and
divisions — i.e. TP/FP/FN are summed across all samples of the run and Jaccard
is computed from those totals.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --split 0
"""

import argparse
import json
from pathlib import Path

import tracksdata as td
from geff import GeffMetadata
from tqdm import tqdm

from tracking_cellmot.io import open_dataset
from tracking_cellmot.metrics import (
    evaluate as compute_metric,
    nan_metrics_row,
    node_recall,
    per_sample_metrics,
    summarise,
)

from dataspec import DATASET_PATH, INTERACTIVE, PREDICTIONS_PATH, USERNAME

PREDICTIONS_DIR = PREDICTIONS_PATH
DATA_DIR = DATASET_PATH


def _read_estimated_n_total(geff_path: Path) -> float:
    """Read ``estimated_number_of_nodes`` from a GEFF file's metadata extras.

    Returns NaN when the key is missing or the file can't be read.
    """
    try:
        meta = GeffMetadata.read(geff_path)
    except Exception:
        return float("nan")
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def discover_runs(
    predictions_dir: Path,
    username: str | None = None,
    method: str | None = None,
    fold: int | None = None,
) -> list[dict]:
    """Find all (username, method, fold) combinations with predictions.

    Layout: predictions/{username}/{method}/split_{fold}/*.geff
    """
    runs = []
    users = [username] if username else sorted(p.name for p in predictions_dir.iterdir() if p.is_dir())
    for u in users:
        u_dir = predictions_dir / u
        if not u_dir.is_dir():
            continue
        methods = [method] if method else sorted(p.name for p in u_dir.iterdir() if p.is_dir())
        for m in methods:
            m_dir = u_dir / m
            if not m_dir.is_dir():
                continue
            splits = [f"split_{fold}"] if fold is not None else sorted(p.name for p in m_dir.iterdir() if p.is_dir() and p.name.startswith("split_"))
            for s in splits:
                s_dir = m_dir / s
                if not s_dir.is_dir():
                    continue
                geffs = sorted(s_dir.glob("*.geff"))
                if geffs:
                    runs.append({"username": u, "method": m, "split": s, "dir": s_dir, "geffs": geffs})
    return runs


def _nan_row(username: str, method: str, split: str, dataset: str) -> dict:
    return {
        "username": username, "method": method,
        "split": split, "dataset": dataset,
        **nan_metrics_row(),
    }


def evaluate_run(run: dict, max_distance: float | None = None) -> list[dict]:
    """Evaluate all predictions in a single run, return per-sample results."""
    username = run["username"]
    method = run["method"]
    split = run["split"]
    results: list[dict] = []

    # Check for missing predictions against the splits file
    splits_file = DATA_DIR / "dataset_splits.json"
    if splits_file.exists():
        fold_idx = int(split.split("_")[1])
        folds = json.loads(splits_file.read_text())
        expected = set(folds[fold_idx]["test"])
        found = {p.stem for p in run["geffs"]}
        missing = sorted(expected - found)
        if missing:
            print(f"  WARNING: {len(missing)} missing predictions for {username}/{method}/{split}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            for name in missing:
                results.append(_nan_row(username, method, split, name))

    desc = f"  {username}/{method}/{split}"
    for pred_path in tqdm(run["geffs"], desc=desc, leave=False, disable=not INTERACTIVE):
        name = pred_path.stem
        gt_path = DATA_DIR / f"{name}.geff"

        if not gt_path.exists():
            print(f"  WARNING: GT not found for {name}, skipping")
            continue

        try:
            ds = open_dataset(DATA_DIR / name, require_tracks=True)
            pred_result = td.graph.IndexedRXGraph.from_geff(pred_path)
            pred_graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result

            kwargs: dict = dict(scale=ds.scale)
            if max_distance is not None:
                kwargs["max_distance"] = max_distance
            er = compute_metric(pred_graph, ds.tracks, **kwargs)

            # node_recall needs the pred graph to have been matched, which
            # evaluate() only does when pred has edges + nodes.
            if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0:
                recall = node_recall(pred_graph, ds.tracks)
            else:
                recall = 0.0

            n_total = _read_estimated_n_total(gt_path)
            row = {
                "username": username, "method": method,
                "split": split, "dataset": name,
                **per_sample_metrics(er, n_total, recall),
            }
        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")
            row = _nan_row(username, method, split, name)

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking predictions.")
    parser.add_argument("--method", type=str, default=None, help="Evaluate only this method. Default: all discovered.")
    parser.add_argument("--split", type=str, default=None, help="Split index (0-4) or 'all'. Default: all discovered.")
    parser.add_argument("--max-distance", type=float, default=None)
    args = parser.parse_args()

    fold = None if args.split is None or args.split == "all" else int(args.split)

    runs = discover_runs(PREDICTIONS_DIR, username=USERNAME, method=args.method, fold=fold)
    if not runs:
        print("No predictions found.")
        return

    print(f"Found {len(runs)} run(s)")

    for run in runs:
        results = evaluate_run(run, max_distance=args.max_distance)

        s = summarise(results)
        print(
            f"  {run['username']}/{run['method']}/{run['split']}: "
            f"score={s['score']:.4f}  "
            f"edge_jaccard={s['edge_jaccard']:.4f}  "
            f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f} (n_adj={s['n_adj']})  "
            f"division_jaccard={s['division_jaccard']:.4f} "
            f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
            f"node_recall={s['node_recall']:.4f}  (n={s['n']})"
        )


if __name__ == "__main__":
    main()
