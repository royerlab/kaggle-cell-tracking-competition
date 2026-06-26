"""Freeze tests for edge + division metrics on hand-crafted sandbox examples.

Each JSON file under ``visualize/sandbox_examples/`` was built interactively
with ``visualize/division_metric_sandbox.py`` and its counts were manually
checked to be correct.  These tests pin that behavior so changes to
:func:`evaluate_divisions` or the edge-matching pipeline don't silently
regress it.

The computation here mirrors the sandbox's ``_recompute`` method exactly:

* Edge metrics come from ``_evaluate`` + ``_evaluate_matched_graph`` on a
  fresh copy of the pred graph (matching mutates state).
* Division metrics come from ``evaluate_divisions`` on the *original*
  (unmatched) pred graph — the sandbox does this intentionally, and the
  result for some examples (e.g. ``successive_div``) differs from calling
  ``evaluate_divisions`` on an already-matched graph.

To add a new frozen case: save an example from the sandbox, then add an
entry to ``EXPECTED`` with the (manually-verified) counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import polars as pl
import pytest
import tracksdata as td

from tracking_cellmot.division_metrics import evaluate_divisions
from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph, _jaccard, evaluate

SANDBOX_DIR = Path(__file__).resolve().parents[1] / "visualize" / "sandbox_examples"


class SandboxMetrics(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    div_tp: int
    div_fp: int
    div_fn: int


EXPECTED: dict[str, SandboxMetrics] = {
    "3_division":          SandboxMetrics(edge_tp=2, edge_fp=2, edge_fn=0, div_tp=0, div_fp=1, div_fn=0),
    "bug":                 SandboxMetrics(edge_tp=0, edge_fp=5, edge_fn=8, div_tp=1, div_fp=0, div_fn=1),
    "complex":             SandboxMetrics(edge_tp=2, edge_fp=2, edge_fn=8, div_tp=1, div_fp=0, div_fn=0),
    "complex2":            SandboxMetrics(edge_tp=4, edge_fp=4, edge_fn=6, div_tp=1, div_fp=0, div_fn=0),
    "division":            SandboxMetrics(edge_tp=4, edge_fp=1, edge_fn=3, div_tp=1, div_fp=1, div_fn=1),
    "division_at_end":     SandboxMetrics(edge_tp=5, edge_fp=0, edge_fn=1, div_tp=1, div_fp=0, div_fn=0),
    "division_node_stage": SandboxMetrics(edge_tp=3, edge_fp=3, edge_fn=1, div_tp=1, div_fp=1, div_fn=0),
    "division_test":       SandboxMetrics(edge_tp=4, edge_fp=2, edge_fn=2, div_tp=1, div_fp=0, div_fn=0),
    "duplicated_division": SandboxMetrics(edge_tp=0, edge_fp=8, edge_fn=5, div_tp=1, div_fp=0, div_fn=0),
    "late_division":       SandboxMetrics(edge_tp=2, edge_fp=3, edge_fn=4, div_tp=1, div_fp=0, div_fn=0),
    "merge_delay":         SandboxMetrics(edge_tp=3, edge_fp=2, edge_fn=2, div_tp=1, div_fp=0, div_fn=0),
    "node_conflict":       SandboxMetrics(edge_tp=3, edge_fp=2, edge_fn=4, div_tp=1, div_fp=0, div_fn=0),
    "simle":               SandboxMetrics(edge_tp=3, edge_fp=2, edge_fn=3, div_tp=0, div_fp=0, div_fn=0),
    "simple":              SandboxMetrics(edge_tp=2, edge_fp=2, edge_fn=4, div_tp=0, div_fp=0, div_fn=0),
    "successive_div":      SandboxMetrics(edge_tp=3, edge_fp=2, edge_fn=3, div_tp=1, div_fp=0, div_fn=1),
}


def _graph_from_dict(data: dict) -> td.graph.InMemoryGraph:
    """Build an InMemoryGraph from a sandbox JSON ``{"nodes": ..., "edges": ...}``.

    Mirrors ``GraphModel.to_tracksdata`` in the sandbox (y-only layout, z=x=0).
    """
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    uid_to_td: dict[int, int] = {}
    for node in data["nodes"]:
        uid_to_td[node["uid"]] = g.add_node(
            {"t": node["t"], "z": 0.0, "y": node["y"], "x": 0.0}
        )
    for edge in data["edges"]:
        g.add_edge(uid_to_td[edge["source"]], uid_to_td[edge["target"]], {})
    return g


def _compute_sandbox_metrics(
    pred: td.graph.BaseGraph,
    gt: td.graph.BaseGraph,
    max_distance: float,
) -> tuple[SandboxMetrics, float]:
    """Replicate the sandbox ``_recompute`` math. Returns (counts, edge_jaccard)."""
    pred_copy = pred.copy()
    _evaluate(pred_copy, gt, "jaccard", None, max_distance)
    edge_attrs = _evaluate_matched_graph(pred_copy, gt)
    edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    edge_valid_pred = int(edge_attrs["pred_valid"].sum())
    edge_fp = edge_valid_pred - edge_tp
    edge_fn = gt.num_edges() - edge_tp

    div = evaluate_divisions(pred, gt, max_distance=max_distance)

    counts = SandboxMetrics(
        edge_tp=edge_tp, edge_fp=edge_fp, edge_fn=edge_fn,
        div_tp=div.tp, div_fp=div.fp, div_fn=div.fn,
    )
    return counts, _jaccard(edge_tp, edge_fp, edge_fn)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_sandbox_example_matches_frozen_counts(name: str) -> None:
    path = SANDBOX_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    gt = _graph_from_dict(data["gt"])
    pred = _graph_from_dict(data["pred"])
    max_distance = data.get("max_distance", 7.0)

    counts, edge_jaccard = _compute_sandbox_metrics(pred, gt, max_distance)
    expected = EXPECTED[name]

    assert counts == expected, f"{name}: {counts} != expected {expected}"

    # Edge Jaccard is derived from the counts, but freeze it too so the
    # assertion message is informative if a count drifts.
    expected_jaccard = _jaccard(expected.edge_tp, expected.edge_fp, expected.edge_fn)
    assert edge_jaccard == pytest.approx(expected_jaccard)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_evaluate_agrees_with_sandbox(name: str) -> None:
    """The high-level ``evaluate()`` must produce the same counts as the
    sandbox pipeline. This catches the prior regression where ``_evaluate``
    matched the graph in place and then ``evaluate_divisions`` saw stale
    ``MATCHED_NODE_ID`` attrs on the re-matched copy, inflating division TP.
    """
    path = SANDBOX_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    gt = _graph_from_dict(data["gt"])
    pred = _graph_from_dict(data["pred"])
    max_distance = data.get("max_distance", 7.0)

    r = evaluate(pred, gt, max_distance=max_distance)
    expected = EXPECTED[name]

    assert (r.edge_tp, r.edge_fp, r.edge_fn) == (
        expected.edge_tp, expected.edge_fp, expected.edge_fn,
    )
    assert (r.division_tp, r.division_fp, r.division_fn) == (
        expected.div_tp, expected.div_fp, expected.div_fn,
    )


def test_all_sandbox_examples_are_covered() -> None:
    """Every saved sandbox JSON must have a frozen expectation."""
    on_disk = {p.stem for p in SANDBOX_DIR.glob("*.json")}
    missing = on_disk - set(EXPECTED)
    assert not missing, (
        f"Sandbox examples missing from EXPECTED: {sorted(missing)}. "
        "Add entries with manually-verified counts."
    )
