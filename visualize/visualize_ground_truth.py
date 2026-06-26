#!/usr/bin/env python
"""
Visualize datasets with ground truth tracks in napari.

Press the "Next Dataset" button to advance to the next dataset.

Usage:
    uv run visualize/visualize_ground_truth.py
"""

import argparse
import random
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataspec import DATASET_PATH
from tracking_cellmot.io import list_datasets, open_dataset

from visualize_utils import show_with_napari


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ground truth tracks in napari.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory. Default: DATASET_PATH.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling datasets (default: 0).")
    parser.add_argument("--dataset", type=str, default=None, help="Load only the dataset with this stem name (skips shuffling).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATASET_PATH
    datasets = list_datasets(data_dir)

    if not datasets:
        print(f"No valid datasets found in {data_dir}")
        return

    if args.dataset is not None:
        matches = [p for p in datasets if p.stem == args.dataset]
        if not matches:
            available = ", ".join(sorted(p.stem for p in datasets))
            print(f"Dataset '{args.dataset}' not found in {data_dir}.\nAvailable: {available}")
            return
        datasets = matches
        print(f"Loading dataset: {args.dataset}")
    else:
        random.seed(args.seed)
        random.shuffle(datasets)
        print(f"Found {len(datasets)} dataset(s) (seed={args.seed})")

    for i, ds_path in enumerate(datasets):
        print(f"\n[{i + 1}/{len(datasets)}] Loading: {ds_path.stem}")

        ds = open_dataset(ds_path, require_tracks=True)
        print(f"  Shape: {ds.image.shape}")
        print(f"  Scale: {ds.scale}")
        if ds.tracks is not None:
            print(f"  Tracks: {ds.tracks.num_nodes()} nodes, {ds.tracks.num_edges()} edges")
        else:
            print("  Tracks: None")

        show_with_napari(
            ds.image,
            graph=ds.napari_tracks(),
            scale=ds.scale,
            names=["image"],
            title=f"[{i + 1}/{len(datasets)}] {ds_path.stem}",
            show_next_button=i < len(datasets) - 1,
        )


if __name__ == "__main__":
    main()
