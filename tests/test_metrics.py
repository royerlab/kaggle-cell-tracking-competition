import sys
from pathlib import Path

import pytest
import tracksdata as td
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from dataspec import DATASET_PATH

from tracking_cellmot.metrics import (
    _compute_score,
    _evaluate_matched_graph,
    evaluate,
)


DATA_DIR = DATASET_PATH
GEFF_PATH = DATA_DIR / "2024_03_22_dorado_0001_0190_1651_0467.geff"


def _load_geff():
    result = td.graph.IndexedRXGraph.from_geff(GEFF_PATH)
    return result[0] if isinstance(result, tuple) else result


def _jaccard_of(pred, gt, **kwargs) -> float:
    """Run :func:`evaluate` and return the edge Jaccard as a float.

    Returns 0.0 when TP+FP+FN is zero, matching the pre-existing test
    expectations (``_jaccard`` itself returns NaN in that case, but empty
    predictions vs non-empty GT always have FN>0 so this only trips on
    truly-empty pairs).
    """
    er = evaluate(pred, gt, **kwargs)
    denom = er.edge_tp + er.edge_fp + er.edge_fn
    return er.edge_tp / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    r"""
    Prediction graph:

        0 - 1 - 2 - 4
              \   \
                3   5

    GT graph:

        0 - 1 - 2
              \
                3

    Node correspondence (pred_id -> gt_id): 1->0, 2->1, 5->3
    Matched edges: (1,2) and (2,5)
    """
    graph = td.graph.InMemoryGraph()
    for t in [0, 1, 2, 2, 3, 3]:
        graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: t})

    graph.add_edge(0, 1, {})
    edge_1_2 = graph.add_edge(1, 2, {})
    graph.add_edge(1, 3, {})
    graph.add_edge(2, 4, {})
    edge_2_5 = graph.add_edge(2, 5, {})

    graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, pl.Int64, -1)
    graph.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK, pl.Boolean, False)

    # Explicitly initialise all edges to False first — edges added before add_edge_attr_key
    # get null rather than the default in newer tracksdata, causing Object dtype in polars.
    all_edge_ids = graph.edge_ids()
    graph.update_edge_attrs(
        edge_ids=all_edge_ids,
        attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [False] * len(all_edge_ids)},
    )

    graph.update_node_attrs(
        node_ids=[1, 2, 5],
        attrs={td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: [0, 1, 3]},
    )
    graph.update_edge_attrs(
        edge_ids=[edge_1_2, edge_2_5],
        attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [True, True]},
    )

    gt_graph = td.graph.InMemoryGraph()
    for t in [1, 2, 3, 3]:
        gt_graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: t})

    gt_graph.add_edge(0, 1, {})
    gt_graph.add_edge(1, 2, {})
    gt_graph.add_edge(1, 3, {})

    return graph, gt_graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_evaluate_matched_graph_overlap():
    """Matched edges should correspond to the manually set correspondence."""
    graph, gt_graph = _make_graph()
    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    matched_edges = (
        edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)
        .select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .to_numpy()
        .tolist()
    )
    assert sorted(matched_edges) == sorted([[1, 2], [2, 5]])


def test_evaluate_matched_graph_pred_valid():
    """pred_valid should mark all edges where at least one endpoint is matched."""
    graph, gt_graph = _make_graph()
    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    valid_edges = (
        edge_attrs.filter("pred_valid")
        .select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .to_numpy()
        .tolist()
    )
    assert sorted(valid_edges) == sorted([[1, 2], [1, 3], [2, 4], [2, 5]])


def test_compute_score_jaccard():
    graph, gt_graph = _make_graph()
    edge_attrs = _evaluate_matched_graph(graph, gt_graph)
    # intersection=2, gt=3, valid_pred=4  ->  2 / (3 + 4 - 2) = 2/5
    score = _compute_score(edge_attrs, gt_num_edges=3, metric="jaccard")
    assert score == pytest.approx(2 / 5)


def test_compute_score_dice():
    graph, gt_graph = _make_graph()
    edge_attrs = _evaluate_matched_graph(graph, gt_graph)
    # 2*2 / (3 + 4) = 4/7
    score = _compute_score(edge_attrs, gt_num_edges=3, metric="dice")
    assert score == pytest.approx(4 / 7)


def test_compute_score_empty_returns_nan():
    """Zero-denominator should return NaN, not raise ZeroDivisionError."""
    df = pl.DataFrame(
        {td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [], "pred_valid": []},
        schema={
            td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: pl.Boolean,
            "pred_valid": pl.Boolean,
        },
    )
    import math
    assert math.isnan(_compute_score(df, gt_num_edges=0, metric="jaccard"))
    assert math.isnan(_compute_score(df, gt_num_edges=0, metric="dice"))


def test_evaluate_perfect_prediction():
    """A prediction that exactly matches GT should score 1.0."""
    gt_graph = td.graph.InMemoryGraph()
    gt_graph.add_node_attr_key("z", pl.Float64, 0.0)
    gt_graph.add_node_attr_key("y", pl.Float64, 0.0)
    gt_graph.add_node_attr_key("x", pl.Float64, 0.0)
    gt_graph.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    gt_graph.add_node(attrs={"t": 1, "z": 0.0, "y": 0.0, "x": 0.0})
    gt_graph.add_edge(0, 1, {})

    # Identical prediction
    pred_graph = td.graph.InMemoryGraph()
    pred_graph.add_node_attr_key("z", pl.Float64, 0.0)
    pred_graph.add_node_attr_key("y", pl.Float64, 0.0)
    pred_graph.add_node_attr_key("x", pl.Float64, 0.0)
    pred_graph.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    pred_graph.add_node(attrs={"t": 1, "z": 0.0, "y": 0.0, "x": 0.0})
    pred_graph.add_edge(0, 1, {})

    score = _jaccard_of(pred_graph, gt_graph, max_distance=1.0)
    assert score == pytest.approx(1.0)


def test_evaluate_no_matching_prediction():
    """A prediction with no overlap with GT should score 0.0."""
    gt_graph = td.graph.InMemoryGraph()
    gt_graph.add_node_attr_key("z", pl.Float64, 0.0)
    gt_graph.add_node_attr_key("y", pl.Float64, 0.0)
    gt_graph.add_node_attr_key("x", pl.Float64, 0.0)
    gt_graph.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    gt_graph.add_node(attrs={"t": 1, "z": 0.0, "y": 0.0, "x": 0.0})
    gt_graph.add_edge(0, 1, {})

    # Prediction nodes are far from GT — no matches within max_distance=1.0
    pred_graph = td.graph.InMemoryGraph()
    pred_graph.add_node_attr_key("z", pl.Float64, 0.0)
    pred_graph.add_node_attr_key("y", pl.Float64, 0.0)
    pred_graph.add_node_attr_key("x", pl.Float64, 0.0)
    pred_graph.add_node(attrs={"t": 0, "z": 100.0, "y": 100.0, "x": 100.0})
    pred_graph.add_node(attrs={"t": 1, "z": 100.0, "y": 100.0, "x": 100.0})
    pred_graph.add_edge(0, 1, {})

    score = _jaccard_of(pred_graph, gt_graph, max_distance=1.0)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Real-data tests using a geff from DATASET_PATH
# ---------------------------------------------------------------------------

def test_distance_threshold():
    """Nodes within max_distance=15 should be matched in all spatial directions."""
    def _simple_graph(z=0.0, y=0.0, x=0.0):
        g = td.graph.InMemoryGraph()
        g.add_node_attr_key("z", pl.Float64, 0.0)
        g.add_node_attr_key("y", pl.Float64, 0.0)
        g.add_node_attr_key("x", pl.Float64, 0.0)
        g.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
        g.add_node(attrs={"t": 1, "z": z, "y": y, "x": x})
        g.add_edge(0, 1, {})
        return g

    gt = _simple_graph()

    for axis in ["z", "y", "x"]:
        kwargs_near = {axis: 14.0}
        kwargs_far  = {axis: 16.0}

        score_near = _jaccard_of(_simple_graph(**kwargs_near), gt, max_distance=15.0)
        assert score_near == pytest.approx(1.0), f"Expected match within threshold along {axis}"

        score_far = _jaccard_of(_simple_graph(**kwargs_far), gt, max_distance=15.0)
        assert score_far == pytest.approx(0.0), f"Expected no match outside threshold along {axis}"


def test_scale_parameter():
    """scale should multiply voxel distances before comparing to max_distance."""
    def _simple_graph(z=0.0, y=0.0, x=0.0):
        g = td.graph.InMemoryGraph()
        g.add_node_attr_key("z", pl.Float64, 0.0)
        g.add_node_attr_key("y", pl.Float64, 0.0)
        g.add_node_attr_key("x", pl.Float64, 0.0)
        g.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
        g.add_node(attrs={"t": 1, "z": z, "y": y, "x": x})
        g.add_edge(0, 1, {})
        return g

    gt = _simple_graph()

    # 4 voxels away in z — isotropic distance = 4, well within max_distance=15
    pred = _simple_graph(z=4.0)
    assert _jaccard_of(pred, gt, max_distance=15.0) == pytest.approx(1.0)

    # Same displacement but with scale=(4, 1, 1): physical distance = 4*4 = 16 > 15 — no match
    assert _jaccard_of(pred, gt, max_distance=15.0, scale=(4.0, 1.0, 1.0)) == pytest.approx(0.0)

    # Halving scale should bring it back in range: 2*4 = 8 < 15 — matches again
    assert _jaccard_of(pred, gt, max_distance=15.0, scale=(2.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_extra_edge_at_track_end_not_penalized():
    """Predicting one extra edge beyond a track end should not lower the jaccard score.

    GT:   1 → 2 → 3
    Pred: 1 → 2 → 3 → 4  (node 4 is unmatched)
    Expected jaccard: 1.0
    """
    gt = td.graph.InMemoryGraph()
    gt.add_node_attr_key("z", pl.Float64, 0.0)
    gt.add_node_attr_key("y", pl.Float64, 0.0)
    gt.add_node_attr_key("x", pl.Float64, 0.0)
    for t in range(3):
        gt.add_node(attrs={"t": t, "z": 0.0, "y": 0.0, "x": float(t)})
    gt.add_edge(0, 1, {})
    gt.add_edge(1, 2, {})

    pred = td.graph.InMemoryGraph()
    pred.add_node_attr_key("z", pl.Float64, 0.0)
    pred.add_node_attr_key("y", pl.Float64, 0.0)
    pred.add_node_attr_key("x", pl.Float64, 0.0)
    for t in range(4):
        pred.add_node(attrs={"t": t, "z": 0.0, "y": 0.0, "x": float(t)})
    pred.add_edge(0, 1, {})
    pred.add_edge(1, 2, {})
    pred.add_edge(2, 3, {})  # spurious: node 3 at t=3 has no GT match

    score = _jaccard_of(pred, gt, max_distance=0.5)
    assert score == pytest.approx(1.0), f"Expected 1.0 but got {score:.4f}"


def test_extra_edge_at_track_start_not_penalized():
    """Predicting one extra edge before a track start should not lower the jaccard score.

    GT:   2 → 3 → 4
    Pred: 1 → 2 → 3 → 4  (node 1 is unmatched)
    Expected jaccard: 1.0
    """
    gt = td.graph.InMemoryGraph()
    gt.add_node_attr_key("z", pl.Float64, 0.0)
    gt.add_node_attr_key("y", pl.Float64, 0.0)
    gt.add_node_attr_key("x", pl.Float64, 0.0)
    for t in range(1, 4):
        gt.add_node(attrs={"t": t, "z": 0.0, "y": 0.0, "x": float(t)})
    gt.add_edge(0, 1, {})
    gt.add_edge(1, 2, {})

    pred = td.graph.InMemoryGraph()
    pred.add_node_attr_key("z", pl.Float64, 0.0)
    pred.add_node_attr_key("y", pl.Float64, 0.0)
    pred.add_node_attr_key("x", pl.Float64, 0.0)
    for t in range(4):
        pred.add_node(attrs={"t": t, "z": 0.0, "y": 0.0, "x": float(t)})
    pred.add_edge(0, 1, {})  # spurious: node 0 at t=0 has no GT match
    pred.add_edge(1, 2, {})
    pred.add_edge(2, 3, {})

    score = _jaccard_of(pred, gt, max_distance=0.5)
    assert score == pytest.approx(1.0), f"Expected 1.0 but got {score:.4f}"


def test_spurious_edge_to_gt_interior_is_penalized():
    """Spurious edges involving GT nodes with known degree are penalized; others are not.

    GT graph (3 nodes, 2 edges):

        t=0     t=1     t=2
         A ────► B ────► C

    A: out_deg=1, in_deg=0  (track start)
    B: out_deg=1, in_deg=1  (interior)
    C: out_deg=0, in_deg=1  (track end)

    D is an unmatched background node far from all GT nodes.

    An edge is penalized (pred_valid=True) when we have GT evidence to judge it:
      - source matched to GT node with out_deg>0: we know its true targets
      - target matched to GT node with in_deg>0: we know its true parents

    Incoming to GT nodes (D→?):
      D→A: NOT penalized — A has in_deg=0, so we don't know A's parents
      D→B: penalized     — B has in_deg=1, so D→B is a known false positive
      D→C: penalized     — C has in_deg=1, so D→C is a known false positive

    Outgoing from GT nodes (?→D):
      A→D: penalized     — A has out_deg=1, so A→D is a known false positive
      B→D: penalized     — B has out_deg=1, so B→D is a known false positive
      C→D: NOT penalized — C has out_deg=0, so we don't know C's successors
    """
    import copy

    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    D_node = {"t": 1, "z": 100.0, "y": 100.0, "x": 100.0}
    PERFECT = 2 / 3  # Jaccard = 2 / (2 + 3 - 2) when 1 FP is added

    def _build(nodes, edges):
        g = td.graph.InMemoryGraph()
        g.add_node_attr_key("z", pl.Float64, 0.0)
        g.add_node_attr_key("y", pl.Float64, 0.0)
        g.add_node_attr_key("x", pl.Float64, 0.0)
        ids = {}
        for name, attrs in nodes.items():
            ids[name] = g.add_node(attrs=attrs)
        for src, tgt in edges:
            g.add_edge(ids[src], ids[tgt], {})
        return g

    def fresh_gt():
        return _build(copy.deepcopy(gt_nodes), [("A", "B"), ("B", "C")])

    def pred_with_extra(extra_edge):
        nodes = {**copy.deepcopy(gt_nodes), "D": copy.deepcopy(D_node)}
        return _build(nodes, [("A", "B"), ("B", "C"), extra_edge])

    # D→A: A has in_deg=0 → NOT penalized
    score = _jaccard_of(pred_with_extra(("D", "A")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(1.0), f"D→A should NOT be penalized, got {score:.4f}"

    # D→B: B has in_deg=1 → penalized
    score = _jaccard_of(pred_with_extra(("D", "B")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(PERFECT), f"D→B SHOULD be penalized, got {score:.4f}"

    # D→C: C has in_deg=1 → penalized
    score = _jaccard_of(pred_with_extra(("D", "C")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(PERFECT), f"D→C SHOULD be penalized, got {score:.4f}"

    # A→D: A has out_deg=1 → penalized
    score = _jaccard_of(pred_with_extra(("A", "D")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(PERFECT), f"A→D SHOULD be penalized, got {score:.4f}"

    # B→D: B has out_deg=1 → penalized
    score = _jaccard_of(pred_with_extra(("B", "D")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(PERFECT), f"B→D SHOULD be penalized, got {score:.4f}"

    # C→D: C has out_deg=0 → NOT penalized
    score = _jaccard_of(pred_with_extra(("C", "D")), fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(1.0), f"C→D should NOT be penalized, got {score:.4f}"


def test_division_spurious_child_and_missing_child_penalized():
    """Division edges are fully annotated — extra or missing children are penalized.

    GT graph (4 nodes, 3 edges, B divides):

        t=0     t=1     t=2
         A ────► B ────► C
                  ╲
                   ►  D

    B: out_deg=2, in_deg=1
    C: out_deg=0, in_deg=1
    D: out_deg=0, in_deg=1

    E is an unmatched background node far from all GT nodes.

    B→E: penalized — B has out_deg=2, so all B's children are known; E is not one of them.
    B→C missing (pred = A→B, B→D only): penalized — B→C is a known GT edge that's absent.
    """
    import copy

    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 10.0, "x": 0.0},
        "D": {"t": 2, "z": 0.0, "y": -10.0, "x": 0.0},
    }
    gt_edges = [("A", "B"), ("B", "C"), ("B", "D")]

    def _build(nodes, edges):
        g = td.graph.InMemoryGraph()
        g.add_node_attr_key("z", pl.Float64, 0.0)
        g.add_node_attr_key("y", pl.Float64, 0.0)
        g.add_node_attr_key("x", pl.Float64, 0.0)
        ids = {}
        for name, attrs in nodes.items():
            ids[name] = g.add_node(attrs=attrs)
        for src, tgt in edges:
            g.add_edge(ids[src], ids[tgt], {})
        return g

    def fresh_gt():
        return _build(copy.deepcopy(gt_nodes), gt_edges)

    # Perfect prediction
    pred = _build(copy.deepcopy(gt_nodes), gt_edges)
    score = _jaccard_of(pred, fresh_gt(), max_distance=5.0)
    assert score == pytest.approx(1.0), f"Perfect division should be 1.0, got {score:.4f}"

    # B→E: spurious extra child — penalized (B has out_deg=2, so B's children are known)
    pred_nodes = {
        **copy.deepcopy(gt_nodes),
        "E": {"t": 2, "z": 100.0, "y": 100.0, "x": 100.0},
    }
    pred = _build(pred_nodes, [("A", "B"), ("B", "C"), ("B", "D"), ("B", "E")])
    score = _jaccard_of(pred, fresh_gt(), max_distance=5.0)
    # intersection=3, gt_edges=3, valid_pred=4 → 3/(3+4-3) = 3/4
    assert score == pytest.approx(3 / 4), f"B→E should be penalized, got {score:.4f}"

    # B→C missing: predict A→B and B→D only — penalized (B→C is a known GT edge)
    pred = _build(copy.deepcopy(gt_nodes), [("A", "B"), ("B", "D")])
    score = _jaccard_of(pred, fresh_gt(), max_distance=5.0)
    # intersection=2, gt_edges=3, valid_pred=2 → 2/(3+2-2) = 2/3
    assert score == pytest.approx(2 / 3), f"Missing B→C should be penalized, got {score:.4f}"


def test_node_matching_prefers_closer_node():
    """When a spurious node A' competes with the true node A for matching to A_gt,
    the closer one should win. The loser becomes unmatched, making its edge ignorable.

    GT:   A_gt(0,0,0) ──► B_gt(0,0,0) ──► C_gt(0,0,0)

    Pred: A(0,9,0) ──► B(0,0,0) ──► C(0,0,0)     (correct track)
          A'(0,10,0) ──► D(0,100,0)                (spurious track)

    A is at distance 9 from A_gt, A' is at distance 10.
    Matching at max_distance=15 should prefer A over A'.
    A matches A_gt → A→B is evaluable and correct.
    A' is unmatched → A'→D has out_valid=False, D is unmatched → in_valid=False → ignored.
    Jaccard = 1.0.

    Swapped: A at distance 10, A' at distance 9.
    A' matches A_gt → A'→D has out_valid=True (A_gt has out_deg=1) → false positive.
    A is unmatched → A→B has out_valid=False, B matches B_gt with in_deg=1 → in_valid=True → false positive.
    Jaccard < 1.0.
    """
    import copy

    gt_nodes = {
        "A_gt": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B_gt": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C_gt": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt_edges = [("A_gt", "B_gt"), ("B_gt", "C_gt")]

    def _build(nodes, edges):
        g = td.graph.InMemoryGraph()
        g.add_node_attr_key("z", pl.Float64, 0.0)
        g.add_node_attr_key("y", pl.Float64, 0.0)
        g.add_node_attr_key("x", pl.Float64, 0.0)
        ids = {}
        for name, attrs in nodes.items():
            ids[name] = g.add_node(attrs=attrs)
        for src, tgt in edges:
            g.add_edge(ids[src], ids[tgt], {})
        return g

    def fresh_gt():
        return _build(copy.deepcopy(gt_nodes), gt_edges)

    # Case 1: A closer (9) than A' (10) → A matches A_gt, A' unmatched → Jaccard 1.0
    pred1_nodes = {
        "A":  {"t": 0, "z": 0.0, "y": 9.0,   "x": 0.0},   # distance 9 from A_gt
        "Ap": {"t": 0, "z": 0.0, "y": 10.0,  "x": 0.0},   # distance 10 from A_gt
        "B":  {"t": 1, "z": 0.0, "y": 0.0,   "x": 0.0},
        "C":  {"t": 2, "z": 0.0, "y": 0.0,   "x": 0.0},
        "D":  {"t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
    }
    pred1 = _build(pred1_nodes, [("A", "B"), ("B", "C"), ("Ap", "D")])
    score1 = _jaccard_of(pred1, fresh_gt(), max_distance=15.0)
    assert score1 == pytest.approx(1.0), f"A closer than A': expected 1.0, got {score1:.4f}"

    # Case 2: A' closer (9) than A (10) → A' matches A_gt, A unmatched → Jaccard < 1.0
    pred2_nodes = {
        "A":  {"t": 0, "z": 0.0, "y": 10.0,  "x": 0.0},   # distance 10 from A_gt
        "Ap": {"t": 0, "z": 0.0, "y": 9.0,   "x": 0.0},   # distance 9 from A_gt
        "B":  {"t": 1, "z": 0.0, "y": 0.0,   "x": 0.0},
        "C":  {"t": 2, "z": 0.0, "y": 0.0,   "x": 0.0},
        "D":  {"t": 1, "z": 0.0, "y": 100.0, "x": 0.0},
    }
    pred2 = _build(pred2_nodes, [("A", "B"), ("B", "C"), ("Ap", "D")])
    score2 = _jaccard_of(pred2, fresh_gt(), max_distance=15.0)
    assert score2 < 1.0, f"A' closer than A: expected < 1.0, got {score2:.4f}"


def test_unmatched_gt_node_does_not_corrupt_nearby_match():
    """Regression: tracksdata _fill_empty bug destroys real matches.

    When optimal bipartite matching fails (no full matching), _fill_empty fills
    empty rows/cols with -1. If empty cols are checked AFTER filling rows, a column
    with one real weight of 1.0 and one filled row of -1.0 sums to 0.0, gets falsely
    flagged as empty, and the real match is overwritten.

    GT (frame 0: 1 node, frame 1: 2 nodes, division):

        t=0         t=1
         A ────► B   (near pred)
          ╲
           ► C       (far from any pred)

    Pred (frame 0: 1 node, frame 1: 10 background + 1 matching B):
        p0 (matches A)
        bg0..bg9 (far from everything)
        p1 (exact match for B)

    C has no pred within max_distance → empty row in the weight matrix.
    After filling that row with -1, B's column sums to 1.0 + (-1.0) = 0.0.
    Without the fix, B's column is falsely flagged empty and the A→B match is lost.
    """
    gt = td.graph.InMemoryGraph()
    gt.add_node_attr_key("z", pl.Float64, 0.0)
    gt.add_node_attr_key("y", pl.Float64, 0.0)
    gt.add_node_attr_key("x", pl.Float64, 0.0)

    g0 = gt.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    g1 = gt.add_node(attrs={"t": 1, "z": 29.0, "y": 5.0, "x": 56.0})
    g2 = gt.add_node(attrs={"t": 1, "z": 36.0, "y": 232.0, "x": 74.0})
    gt.bulk_add_edges([
        {"source_id": g0, "target_id": g1},
        {"source_id": g0, "target_id": g2},
    ])

    pred = td.graph.InMemoryGraph()
    pred.add_node_attr_key("z", pl.Float64, 0.0)
    pred.add_node_attr_key("y", pl.Float64, 0.0)
    pred.add_node_attr_key("x", pl.Float64, 0.0)

    p0 = pred.add_node(attrs={"t": 0, "z": 0.0, "y": 0.0, "x": 0.0})
    for i in range(10):
        pred.add_node(attrs={"t": 1, "z": 150.0 + i, "y": 150.0, "x": 150.0})
    p1 = pred.add_node(attrs={"t": 1, "z": 36.0, "y": 232.0, "x": 74.0})
    pred.bulk_add_edges([{"source_id": p0, "target_id": p1}])

    # Without the fix this raises:
    # ValueError: Attribute 'match_score' has wrong size. Expected 1, got 2
    score = _jaccard_of(pred, gt)
    assert score >= 0.0


def test_evaluate_geff_against_itself_is_perfect():
    """Evaluating a graph against itself should give score 1.0."""
    gt = _load_geff()
    pred = _load_geff()
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(1.0)


def test_evaluate_geff_score_decreases_when_edges_removed():
    """Removing edges from the prediction should strictly lower the score."""
    gt = _load_geff()

    scores = []
    for n_remove in [0,1,2,3,10, 50, 200]:
        pred = _load_geff()
        edge_ids = pred.edge_ids()
        if n_remove > 0:
            for eid in edge_ids[:n_remove]:
                pred.remove_edge(edge_id=eid)
        scores.append(_jaccard_of(pred, gt, max_distance=1.0))

    # Each successive removal should lower (or at worst not raise) the score
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]
    assert scores[2] < scores[1]
    assert scores[3] < scores[2]


def test_evaluate_geff_score_decreases_when_nodes_removed():
    """Removing nodes (and their incident edges) should lower the score."""
    gt = _load_geff()

    scores = []
    for n_remove in [0, 10, 50, 150]:
        pred = _load_geff()
        node_ids = pred.node_ids()
        if n_remove > 0:
            for nid in node_ids[:n_remove]:
                pred.remove_node(node_id=nid)
        scores.append(_jaccard_of(pred, gt, max_distance=1.0))

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < scores[0]
    assert scores[2] < scores[1]
    assert scores[3] < scores[2]


# ---------------------------------------------------------------------------
# Edge-case / exploit tests
# ---------------------------------------------------------------------------

def _build_graph(nodes, edges):
    """Helper: build InMemoryGraph from {name: attrs} dict and [(src, tgt)] list."""
    import copy as _copy
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    ids = {}
    for name, attrs in (nodes.items() if isinstance(nodes, dict) else nodes):
        ids[name] = g.add_node(attrs=_copy.deepcopy(attrs))
    for src, tgt in edges:
        g.add_edge(ids[src], ids[tgt], {})
    return g


def test_dice_geq_jaccard():
    """Dice >= Jaccard should always hold (mathematically: dice = 2J/(1+J))."""
    graph, gt_graph = _make_graph()
    edge_attrs = _evaluate_matched_graph(graph, gt_graph)
    gt_num_edges = gt_graph.num_edges()

    j = _compute_score(edge_attrs, gt_num_edges, "jaccard")
    d = _compute_score(edge_attrs, gt_num_edges, "dice")
    assert d >= j - 1e-10, f"Dice ({d}) should be >= Jaccard ({j})"


def test_score_bounds_unit():
    """_compute_score should always return values in [0, 1]."""
    import polars as pl

    # Perfect: all edges matched
    df = pl.DataFrame({
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [True, True, True],
        "pred_valid": [True, True, True],
    })
    assert _compute_score(df, gt_num_edges=3, metric="jaccard") == pytest.approx(1.0)
    assert _compute_score(df, gt_num_edges=3, metric="dice") == pytest.approx(1.0)

    # Some FP edges
    df = pl.DataFrame({
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [True, False, False],
        "pred_valid": [True, True, True],
    })
    j = _compute_score(df, gt_num_edges=2, metric="jaccard")
    d = _compute_score(df, gt_num_edges=2, metric="dice")
    assert 0.0 <= j <= 1.0
    assert 0.0 <= d <= 1.0
    assert d >= j

    # No matches at all
    df = pl.DataFrame({
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [False, False],
        "pred_valid": [True, True],
    })
    assert _compute_score(df, gt_num_edges=3, metric="jaccard") == pytest.approx(0.0)
    assert _compute_score(df, gt_num_edges=3, metric="dice") == pytest.approx(0.0)


def test_cross_track_interior_edge_penalized():
    """An edge between interior nodes of DIFFERENT tracks is correctly penalized.

    GT:
        Track 1: A → B → C  (B interior: out_deg=1, in_deg=1)
        Track 2: D → E → F  (E interior: out_deg=1, in_deg=1)

    Pred: all correct edges + B→E (cross-track FP)

    B→E: source B has out_deg=1 → out_valid=True → penalized ✓
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
        "D": {"t": 0, "z": 0.0, "y": 50.0, "x": 0.0},
        "E": {"t": 1, "z": 0.0, "y": 50.0, "x": 0.0},
        "F": {"t": 2, "z": 0.0, "y": 50.0, "x": 0.0},
    }
    gt_edges = [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")]

    gt = _build_graph(nodes, gt_edges)
    pred = _build_graph(nodes, gt_edges + [("B", "E")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    # intersection=4, gt=4, valid_pred=5 → 4/(4+5-4)=4/5
    assert score == pytest.approx(4 / 5), f"Cross-track interior edge should be penalized, got {score:.4f}"


def test_reparenting_node_penalized():
    """Connecting a GT-interior node to the wrong parent is penalized.

    GT: A → B → C      (B has in_degree=1, parent is A)
    Pred: D→B, B→C     (D is unmatched, replaces A)

    D→B: D unmatched → out_valid=False; B in_degree=1 → in_valid=True → penalized ✓
    Missing A→B: still in gt_num_edges denominator → score drops ✓
    """
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(gt_nodes, [("A", "B"), ("B", "C")])

    pred_nodes = {
        "D": {"t": 0, "z": 100.0, "y": 100.0, "x": 100.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    pred = _build_graph(pred_nodes, [("D", "B"), ("B", "C")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    # B→C matched, D→B not matched but penalized (B.in_degree=1)
    # intersection=1, gt=2, valid_pred=2 → 1/(2+2-1)=1/3
    assert score == pytest.approx(1 / 3), f"Re-parenting should be penalized, got {score:.4f}"


def test_missing_division_child_penalized():
    """Missing one child of a division is penalized (known from GT).

    GT: A → B, A → C  (A divides, out_degree=2)
    Pred: A → B only (missing A → C)
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 10.0, "x": 0.0},
        "C": {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "B"), ("A", "C")])
    pred = _build_graph(nodes, [("A", "B")])

    score = _jaccard_of(pred, gt, max_distance=1.0)
    # intersection=1, gt=2, valid_pred=1 → 1/(2+1-1)=1/2
    assert score == pytest.approx(1 / 2), f"Missing division child should be penalized, got {score:.4f}"


def test_duplicate_edges_cannot_inflate_score():
    """Duplicate edges (same source→target) must not inflate the score above 1.0.

    tracksdata allows multigraph edges. Its match() inner-join marks ALL duplicates
    as matched, making intersection > gt_num_edges → Jaccard > 1.0.

    GT:   A → B → C  (2 edges)
    Pred: A → B (×2), B → C  (3 edge objects, but only 2 unique pairs)

    Without dedup: intersection=3, gt=2, valid_pred=3 → Jaccard = 3/(2+3-3) = 1.5
    With dedup:    intersection=2, gt=2, valid_pred=2 → Jaccard = 1.0
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "B"), ("B", "C")])
    pred = _build_graph(nodes, [("A", "B"), ("A", "B"), ("B", "C")])

    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score <= 1.0, f"Duplicate edges inflated score to {score:.4f}"
    assert score == pytest.approx(1.0)


def test_duplicate_matched_edge_kept_after_dedup():
    """Dedup must prefer the matched row when (src, tgt) duplicates disagree on the mask.

    Before the fix, ``unique(keep="first")`` could drop the matched copy if the
    unmatched copy sorted first, destroying a real TP. Using
    ``sort(MATCHED_EDGE_MASK, descending=True)`` before the dedup guarantees the
    matched row wins.
    """
    df = pl.DataFrame({
        td.DEFAULT_ATTR_KEYS.EDGE_SOURCE: [1, 1],
        td.DEFAULT_ATTR_KEYS.EDGE_TARGET: [2, 2],
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [False, True],
    })
    # Reproduce the dedup path from _evaluate_matched_graph directly.
    deduped = df.sort(
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK, descending=True,
    ).unique(
        subset=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET],
        keep="first",
    )
    assert deduped.height == 1
    assert deduped[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].to_list() == [True]


def test_pred_no_edges_scores_zero():
    """Pred with matched nodes but no edges → score 0.0 with warning."""
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(gt_nodes, [("A", "B")])
    pred = _build_graph(gt_nodes, [])  # nodes match but no edges

    with pytest.warns(UserWarning, match="no edges or no nodes"):
        score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(0.0)


def test_empty_pred_no_nodes_scores_zero():
    """Completely empty pred (0 nodes, 0 edges) → score 0.0 with warning."""
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(gt_nodes, [("A", "B")])

    pred = td.graph.InMemoryGraph()
    pred.add_node_attr_key("z", pl.Float64, 0.0)
    pred.add_node_attr_key("y", pl.Float64, 0.0)
    pred.add_node_attr_key("x", pl.Float64, 0.0)

    with pytest.warns(UserWarning, match="no edges or no nodes"):
        score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(0.0)


def test_compute_score_invalid_metric():
    """_compute_score should raise ValueError for unknown metric."""
    import polars as pl

    df = pl.DataFrame({
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: [True],
        "pred_valid": [True],
    })
    with pytest.raises(ValueError, match="Invalid metric"):
        _compute_score(df, gt_num_edges=1, metric="f1")


def test_colocated_gt_nodes_ambiguous_matching():
    """Two GT nodes at the exact same position cause nondeterministic matching.

    GT:  A(t=0, 0,0,0) → C(t=1, 0,0,0)
         B(t=0, 0,0,0) → D(t=1, 0,50,0)

    Pred: identical copy.

    Bipartite matching at t=0 may swap A↔B (both at distance 0). If swapped:
      pred A → GT B, pred B → GT A
    Then edge A→C becomes (matched-to-B)→C which expects GT edge B→C — doesn't exist.
    A perfect prediction scores 0.0.
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},  # same pos as A
        "C": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "D": {"t": 1, "z": 0.0, "y": 50.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "C"), ("B", "D")])
    pred = _build_graph(nodes, [("A", "C"), ("B", "D")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    # Score is 0.0 or 1.0 depending on arbitrary tie-breaking in bipartite matching.
    # This documents that co-located GT nodes make the metric unreliable.
    assert score in (pytest.approx(0.0), pytest.approx(1.0))


def test_spoiler_node_steals_match():
    """A closer spoiler node steals the GT match, making the correct edge unmatched.

    GT:   A(0,0,0) → B(0,0,0)
    Pred: A(0,5,0) → B(0,0,0)  + spoiler(0,1,0) with no edges

    spoiler is closer to GT A (dist=1) than pred A (dist=5).
    spoiler steals the match → pred A becomes unmatched → edge A→B is not matched.
    But A→B is still penalized (B.in_degree=1 → in_valid=True).
    intersection=0, valid_pred=1, gt=1 → 0/2 = 0.0
    """
    gt = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    pred = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 5.0, "x": 0.0},
        "spoiler": {"t": 0, "z": 0.0, "y": 1.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    score = _jaccard_of(pred, gt)
    assert score == pytest.approx(0.0), f"Spoiler should steal match, got {score:.4f}"


def test_self_loops_on_interior_nodes_penalized():
    """Self-loops on GT-interior nodes are penalized (source has out_deg>0).

    GT: A → B → C → D → E  (4 edges)
    Pred: correct track + self-loops on B, C, D

    Each self-loop on interior node X: X.out_deg>0 → out_valid=True → penalized.
    intersection=4, valid_pred=7, gt=4 → 4/(4+7-4)=4/7
    """
    nodes = {f"N{i}": {"t": i, "z": 0.0, "y": 0.0, "x": float(i)} for i in range(5)}
    gt_edges = [(f"N{i}", f"N{i+1}") for i in range(4)]
    gt = _build_graph(nodes, gt_edges)
    pred = _build_graph(nodes, gt_edges + [("N1", "N1"), ("N2", "N2"), ("N3", "N3")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(4 / 7), f"Self-loops should be penalized, got {score:.4f}"


def test_reverse_edge_at_boundary_invisible():
    """Reverse edge B→A (end→start) is invisible — both endpoints are boundary nodes.

    GT: A → B  (A: start in_deg=0, B: end out_deg=0)
    Pred: A→B + B→A

    B→A: B.out_deg=0 → out_valid=False, A.in_deg=0 → in_valid=False → invisible.
    Score = 1.0 despite the false reverse edge.
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "B")])
    pred = _build_graph(nodes, [("A", "B"), ("B", "A")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(1.0), f"Reverse boundary edge should be invisible, got {score:.4f}"


def test_dense_bipartite_cross_edges_penalized():
    """Full bipartite graph between timeframes: cross-edges are correctly penalized.

    GT: A0→B0, A1→B1, A2→B2  (3 independent tracks)
    Pred: full 3×3 bipartite = 9 edges (3 correct + 6 FP)

    Each Ai has out_deg=1 in GT → all edges from Ai have out_valid=True → penalized.
    intersection=3, valid_pred=9, gt=3 → 3/(3+9-3)=3/9=1/3
    """
    nodes = {}
    for i in range(3):
        nodes[f"A{i}"] = {"t": 0, "z": 0.0, "y": float(i * 50), "x": 0.0}
        nodes[f"B{i}"] = {"t": 1, "z": 0.0, "y": float(i * 50), "x": 0.0}
    gt = _build_graph(nodes, [(f"A{i}", f"B{i}") for i in range(3)])
    pred = _build_graph(nodes, [(f"A{i}", f"B{j}") for i in range(3) for j in range(3)])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(1 / 3), f"Dense bipartite should score 1/3, got {score:.4f}"


def test_skip_connection_scores_zero():
    """Skip edge (A→C instead of A→B, B→C) — not a GT edge, scores 0.

    GT: A → B → C
    Pred: A → C only

    A→C is not matched. A.out_deg=1 → out_valid=True, C.in_deg=1 → in_valid=True.
    intersection=0, valid_pred=1, gt=2 → 0/(2+1-0)=0
    """
    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "B"), ("B", "C")])
    pred = _build_graph(nodes, [("A", "C")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(0.0)


def test_distance_matching_respects_timeframes():
    """Pred nodes at different timeframes from GT should not match, even if co-located.

    GT: A(t=0) → B(t=1)
    Pred: X(t=5, same pos as A) → Y(t=6, same pos as B)
    """
    gt = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    pred = _build_graph({
        "X": {"t": 5, "z": 0.0, "y": 0.0, "x": 0.0},
        "Y": {"t": 6, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("X", "Y")])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(0.0)


def test_correct_track_with_many_unmatched_noise_edges():
    """Correct track + 100 edges between far-away unmatched nodes → score 1.0.

    All noise edges have unmatched endpoints → out_valid=False, in_valid=False → invisible.
    """
    import copy

    nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(nodes, [("A", "B"), ("B", "C")])

    pred_nodes = copy.deepcopy(nodes)
    noise_edges = []
    for i in range(100):
        pred_nodes[f"noise_{i}"] = {"t": i % 3, "z": 500.0 + i, "y": 500.0 + i, "x": 500.0 + i}
    for i in range(99):
        noise_edges.append((f"noise_{i}", f"noise_{i+1}"))
    pred = _build_graph(pred_nodes, [("A", "B"), ("B", "C")] + noise_edges)
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(1.0), f"Unmatched noise should be invisible, got {score:.4f}"


def test_hub_with_extra_children_penalized():
    """GT hub with out_deg=5; pred adds 5 extra children → penalized.

    All hub's outgoing edges have out_valid=True (hub.out_deg=5).
    intersection=5, valid_pred=10, gt=5 → 5/(5+10-5)=0.5
    """
    import copy

    nodes = {"hub": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0}}
    gt_e = []
    for i in range(5):
        nodes[f"C{i}"] = {"t": 1, "z": 0.0, "y": float(i * 20), "x": 0.0}
        gt_e.append(("hub", f"C{i}"))
    gt = _build_graph(nodes, gt_e)

    pred_nodes = copy.deepcopy(nodes)
    for i in range(5):
        pred_nodes[f"F{i}"] = {"t": 1, "z": 100.0 + i, "y": 100.0, "x": 100.0}
    pred = _build_graph(pred_nodes, gt_e + [("hub", f"F{i}") for i in range(5)])
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(0.5), f"Extra hub children should be penalized, got {score:.4f}"


def test_nan_coordinates_no_match():
    """NaN coordinates produce NaN distances → no match, score 0."""
    gt = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    pred = _build_graph({
        "A": {"t": 0, "z": float("nan"), "y": float("nan"), "x": float("nan")},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    score = _jaccard_of(pred, gt)
    assert score == pytest.approx(0.0)


def test_inf_coordinates_no_match():
    """Inf coordinates produce infinite distances → no match, score 0."""
    gt = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    pred = _build_graph({
        "A": {"t": 0, "z": float("inf"), "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    score = _jaccard_of(pred, gt)
    assert score == pytest.approx(0.0)


def test_same_graph_object_as_pred_and_gt():
    """Using the same graph object for both pred and gt should give 1.0.

    This works because match() reads from 'other' before writing to 'self',
    and here self IS other. Fragile but currently works.
    """
    g = _build_graph({
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }, [("A", "B")])
    score = _jaccard_of(g, g, max_distance=1.0)
    assert score == pytest.approx(1.0)


def test_score_asymmetric_big_pred_vs_small_gt():
    """Score is asymmetric: extra pred edges beyond GT boundary are free.

    GT:   A → B          (1 edge)
    Pred: A → B → C      (2 edges)

    B→C: B is track-end in GT (out_deg=0) → out_valid=False; C unmatched → in_valid=False.
    pred_valid=False → invisible. Score = 1.0.

    But swap: GT has A→B→C (2 edges), pred has A→B only → score = 0.5.
    """
    small_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    big_nodes = {
        **small_nodes,
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }

    # Big pred, small GT → extra edge invisible → 1.0
    score = _jaccard_of(
        _build_graph(big_nodes, [("A", "B"), ("B", "C")]),
        _build_graph(small_nodes, [("A", "B")]),
        max_distance=1.0,
    )
    assert score == pytest.approx(1.0)

    # Small pred, big GT → missing edge → 0.5
    score = _jaccard_of(
        _build_graph(small_nodes, [("A", "B")]),
        _build_graph(big_nodes, [("A", "B"), ("B", "C")]),
        max_distance=1.0,
    )
    assert score == pytest.approx(0.5)


def test_false_merge_into_interior_penalized():
    """False merge into GT-interior node is penalized (target has in_deg>0).

    GT: A → B → C  (B has in_degree=1)
    Pred: A→B, B→C, D→B  (D is unmatched, false merge into B)

    D→B: D unmatched → out_valid=False; B.in_deg=1 → in_valid=True → penalized.
    """
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(gt_nodes, [("A", "B"), ("B", "C")])
    pred = _build_graph(
        {**gt_nodes, "D": {"t": 0, "z": 100.0, "y": 100.0, "x": 100.0}},
        [("A", "B"), ("B", "C"), ("D", "B")],
    )
    score = _jaccard_of(pred, gt, max_distance=1.0)
    # intersection=2, valid_pred=3, gt=2 → 2/(2+3-2) = 2/3
    assert score == pytest.approx(2 / 3), f"False merge into interior should be penalized, got {score:.4f}"


def test_false_merge_into_track_start_invisible():
    """False merge into track-start is invisible (target has in_deg=0).

    GT: A → B → C  (A has in_degree=0, track start)
    Pred: A→B, B→C, D→A  (D is unmatched, false merge into start)

    D→A: D unmatched → out_valid=False; A.in_deg=0 → in_valid=False → invisible.
    """
    gt_nodes = {
        "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
        "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
        "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
    }
    gt = _build_graph(gt_nodes, [("A", "B"), ("B", "C")])
    pred = _build_graph(
        {**gt_nodes, "D": {"t": 0, "z": 100.0, "y": 100.0, "x": 100.0}},
        [("A", "B"), ("B", "C"), ("D", "A")],
    )
    score = _jaccard_of(pred, gt, max_distance=1.0)
    assert score == pytest.approx(1.0), f"False merge into track-start should be invisible, got {score:.4f}"


def test_summarise_no_divisions_warns_and_drops_term():
    """When no sample contains any division, ``summarise`` should warn and
    drop the division term instead of returning NaN for ``score``."""
    import math
    from tracking_cellmot.metrics import summarise, per_sample_metrics, EvaluationResult

    # Two samples, no divisions anywhere, edges all correct.
    rows = [
        per_sample_metrics(
            EvaluationResult(
                edge_tp=5, edge_fp=0, edge_fn=0,
                division_tp=0, division_fp=0, division_fn=0,
                num_pred_nodes=10,
            ),
            n_total=10,
            node_recall=1.0,
        ),
        per_sample_metrics(
            EvaluationResult(
                edge_tp=3, edge_fp=1, edge_fn=2,
                division_tp=0, division_fp=0, division_fn=0,
                num_pred_nodes=8,
            ),
            n_total=8,
            node_recall=0.8,
        ),
    ]
    with pytest.warns(UserWarning, match="No divisions"):
        s = summarise(rows)
    # division_jaccard stays NaN (there was nothing to score) but score must be
    # finite — it's just the adjusted edge Jaccard.
    assert math.isnan(s["division_jaccard"])
    assert math.isfinite(s["score"])
    assert s["score"] == pytest.approx(s["adj_edge_jaccard"])
