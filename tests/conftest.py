"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tracksdata as td
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_FIXTURE_DIR = Path(__file__).parent / "data" / "division_clip"
_FIXTURE_NAME = "division_clip"

# Frames [17, 22) from the source video — division at original t=19 → fixture t=2.
_SRC_START = 17
_SRC_END = 22


def _build_division_clip_fixture() -> None:
    """Generate tests/data/division_clip from the source dataset."""
    from dataspec import DATASET_PATH

    src_dir = DATASET_PATH / "2024_03_22_dorado_0002_0198_0184_0605"
    src_zarr = src_dir.parent / f"{src_dir.name}.zarr"
    src_geff = src_dir.parent / f"{src_dir.name}.geff"

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- zarr clip ---
    src_group = zarr.open_group(str(src_zarr), mode="r")
    clip = np.array(src_group["0"][_SRC_START:_SRC_END])
    dst = zarr.open_group(str(_FIXTURE_DIR / f"{_FIXTURE_NAME}.zarr"), mode="w")
    dst.create_array("0", data=clip, chunks=(1, *clip.shape[1:]))
    # Copy multiscales from source; recompute quantiles for the clip.
    src_attrs = dict(src_group.attrs)
    flat = clip.ravel()[::50].astype(float)
    src_attrs["image_statistics"] = {
        "quantiles": {str(q): float(np.quantile(flat, float(q)))
                      for q in ["0.0", "0.001", "0.01", "0.1", "0.9", "0.99", "0.999", "1.0"]}
    }
    dst.attrs.update(src_attrs)

    # --- geff clip: filter frames, remap t to 0-based ---
    g, _ = td.graph.IndexedRXGraph.from_geff(str(src_geff))
    sliced = (
        g.filter(td.NodeAttr("t") >= _SRC_START, td.NodeAttr("t") < _SRC_END)
        .subgraph()
        .detach()
    )
    na = sliced.node_attrs()
    sliced.update_node_attrs(
        attrs={"t": (na["t"] - _SRC_START).to_list()},
        node_ids=na["node_id"].to_list(),
    )
    sliced.to_geff(str(_FIXTURE_DIR / f"{_FIXTURE_NAME}.geff"), overwrite=True)


@pytest.fixture(scope="session", autouse=True)
def division_clip_fixture() -> None:
    """Ensure the division_clip test fixture exists, building it if necessary."""
    zarr_path = _FIXTURE_DIR / f"{_FIXTURE_NAME}.zarr"
    geff_ok = (_FIXTURE_DIR / f"{_FIXTURE_NAME}.geff").exists()
    # Rebuild if zarr is missing, geff is missing, or zarr lacks attrs (old fixture).
    needs_rebuild = not zarr_path.exists() or not geff_ok
    if not needs_rebuild:
        attrs = dict(zarr.open_group(str(zarr_path), mode="r").attrs)
        needs_rebuild = "multiscales" not in attrs or "image_statistics" not in attrs
    if needs_rebuild:
        _build_division_clip_fixture()
