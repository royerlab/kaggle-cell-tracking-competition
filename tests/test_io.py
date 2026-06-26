"""Tests for I/O utilities using real data from DATASET_PATH."""

import numpy as np
import pytest
from pathlib import Path

from tracking_cellmot.io import Dataset, list_datasets, open_dataset


import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from dataspec import DATASET_PATH

DATA_DIR = DATASET_PATH
# A dataset with a loadable geff
DS_NAME = "2024_03_22_dorado_0001_0190_1651_0467"
DS_PATH = DATA_DIR / DS_NAME


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

def test_list_datasets_with_geff():
    datasets = list_datasets(DATA_DIR, require_geff=True)
    assert len(datasets) > 0
    for ds in datasets:
        assert (DATA_DIR / f"{ds.stem}.zarr").exists()
        assert (DATA_DIR / f"{ds.stem}.geff").exists()


def test_list_datasets_without_geff_is_superset():
    with_geff = list_datasets(DATA_DIR, require_geff=True)
    without_geff = list_datasets(DATA_DIR, require_geff=False)
    assert len(without_geff) >= len(with_geff)


def test_list_datasets_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        list_datasets(DATA_DIR / "does_not_exist")


# ---------------------------------------------------------------------------
# open_dataset — basic shape / type checks
# ---------------------------------------------------------------------------

def test_open_dataset_shape():
    """Image should be 4D (T, Z, Y, X)."""
    ds = open_dataset(DS_PATH, normalize=False)
    assert ds.image.ndim == 4


def test_open_dataset_no_normalize_preserves_dtype():
    """Without normalization the raw uint16 dtype is kept."""
    ds = open_dataset(DS_PATH, normalize=False)
    assert ds.image.dtype == np.uint16


def test_open_dataset_normalize_returns_float32():
    import torch
    ds = open_dataset(DS_PATH, normalize=True)
    assert ds.image.dtype == torch.float32


def test_open_dataset_scale_is_3_tuple():
    ds = open_dataset(DS_PATH, normalize=False)
    assert len(ds.scale) == 3
    assert all(isinstance(s, float) for s in ds.scale)


def test_open_dataset_returns_dataset_instance():
    ds = open_dataset(DS_PATH, normalize=False)
    assert isinstance(ds, Dataset)
    assert ds.path == DS_PATH


# ---------------------------------------------------------------------------
# open_dataset — tracks loading
# ---------------------------------------------------------------------------

def test_open_dataset_no_tracks_by_default():
    ds = open_dataset(DS_PATH, normalize=False)
    assert ds.tracks is None


def test_open_dataset_loads_tracks():
    ds = open_dataset(DS_PATH, normalize=False, require_tracks=True)
    assert ds.tracks is not None
    assert ds.tracks.num_nodes() > 0
    assert ds.tracks.num_edges() > 0


# ---------------------------------------------------------------------------
# open_dataset — error handling
# ---------------------------------------------------------------------------

def test_open_dataset_missing_zarr_raises():
    with pytest.raises(FileNotFoundError):
        open_dataset(DATA_DIR / "nonexistent_dataset", normalize=False)


def test_open_dataset_zarr_extension_stripped():
    """Passing the path with .zarr extension should work fine."""
    ds = open_dataset(DS_PATH.with_suffix(".zarr"), normalize=False)
    assert ds.image.ndim == 4
