import copy

import polars as pl
import tracksdata as td

from tracking_cellmot.division_metrics import (
    DivisionCounts,
    count_matched_pred_divisions,
    evaluate_divisions,
    extract_divisions,
    score_divisions,
)


def _build_graph(nodes: dict, edges: list) -> tuple[td.graph.InMemoryGraph, dict[str, int]]:
    """Build an InMemoryGraph, returning (graph, name→node_id mapping)."""
    g = td.graph.InMemoryGraph()
    g.add_node_attr_key("z", pl.Float64, 0.0)
    g.add_node_attr_key("y", pl.Float64, 0.0)
    g.add_node_attr_key("x", pl.Float64, 0.0)
    ids = {}
    for name, attrs in nodes.items():
        ids[name] = g.add_node(attrs=copy.deepcopy(attrs))
    for src, tgt in edges:
        g.add_edge(ids[src], ids[tgt], {})
    return g, ids


class TestExtractDivisions:
    def test_no_divisions(self):
        """Linear track with no divisions returns an empty dict."""
        g, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C")],
        )
        assert extract_divisions(g) == {}

    def test_single_division(self):
        """One division: parent + divider + children + grandchild."""
        #  A → B → C1 → E
        #       ↘ C2 → D
        g, ids = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "D":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
                "E":  {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C1"), ("B", "C2"), ("C2", "D"), ("C1", "E")],
        )
        divs = extract_divisions(g)
        assert list(divs.keys()) == [ids["B"]]
        sub = divs[ids["B"]]
        # A (parent), B (divider), C1, C2 (children), D, E (grandchildren)
        assert sub.num_nodes() == 6
        # A→B, B→C1, B→C2, C1→E, C2→D
        assert sub.num_edges() == 5

    def test_single_division_no_parent(self):
        """Divider is a root node — no parent included."""
        #  B → C1 → E
        #   ↘ C2 → D
        g, ids = _build_graph(
            {
                "B":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 1, "z": 0.0, "y": 5.0, "x": 0.0},
                "C2": {"t": 1, "z": 0.0, "y": -5.0, "x": 0.0},
                "D":  {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "E":  {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("B", "C1"), ("B", "C2"), ("C2", "D"), ("C1", "E")],
        )
        divs = extract_divisions(g)
        sub = divs[ids["B"]]
        # B (divider), C1, C2 (children), D, E (grandchildren)
        assert sub.num_nodes() == 5
        # B→C1, B→C2, C1→E, C2→D
        assert sub.num_edges() == 4

    def test_single_division_leaf_children(self):
        """Children are leaf nodes — no grandchildren to include."""
        #  A → B → C1
        #       ↘ C2
        g, ids = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C1"), ("B", "C2")],
        )
        divs = extract_divisions(g)
        sub = divs[ids["B"]]
        # A (parent), B (divider), C1, C2 (children), no grandchildren
        assert sub.num_nodes() == 4
        # A→B, B→C1, B→C2
        assert sub.num_edges() == 3

    def test_two_independent_divisions(self):
        """Two divisions in separate lineages: two separate entries."""
        g, ids = _build_graph(
            {
                "P1":   {"t": 0, "z": 0.0, "y": 10.0, "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 10.0, "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "C1a2": {"t": 3, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b2": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
                "P2":   {"t": 0, "z": 0.0, "y": -10.0, "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "C2b":  {"t": 2, "z": 0.0, "y": -15.0, "x": 0.0},
                "C2a2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
                "C2b2": {"t": 3, "z": 0.0, "y": -15.0, "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "C1a2"), ("C1b", "C1b2"),
                ("P2", "D2"), ("D2", "C2a"), ("D2", "C2b"),
                ("C2a", "C2a2"), ("C2b", "C2b2"),
            ],
        )
        divs = extract_divisions(g)
        assert set(divs.keys()) == {ids["D1"], ids["D2"]}
        for sub in divs.values():
            # parent + divider + 2 children + 2 grandchildren
            assert sub.num_nodes() == 6
            # parent→divider + divider→child1 + divider→child2 + child1→gc1 + child2→gc2
            assert sub.num_edges() == 5

    def test_chained_divisions_are_separate(self):
        """Two divisions separated by one edge produce two entries."""
        #  A → B → C1 → E → F1
        #       ↘ C2       ↘ F2
        #            ↘ G
        g, ids = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "E":  {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
                "G":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
                "F1": {"t": 4, "z": 0.0, "y": 8.0, "x": 0.0},
                "F2": {"t": 4, "z": 0.0, "y": 2.0, "x": 0.0},
            },
            [
                ("A", "B"), ("B", "C1"), ("B", "C2"),
                ("C1", "E"), ("C2", "G"),
                ("E", "F1"), ("E", "F2"),
            ],
        )
        divs = extract_divisions(g)
        assert set(divs.keys()) == {ids["B"], ids["E"]}

        # B's division: A (parent), B (divider), C1, C2 (children), E, G (grandchildren)
        sub_b = divs[ids["B"]]
        assert sub_b.num_nodes() == 6
        assert sub_b.num_edges() == 5

        # E's division: C1 (parent), E (divider), F1, F2 (children), no grandchildren
        sub_e = divs[ids["E"]]
        assert sub_e.num_nodes() == 4
        assert sub_e.num_edges() == 3

    def test_empty_graph(self):
        """An empty graph returns an empty dict."""
        g = td.graph.InMemoryGraph()
        assert extract_divisions(g) == {}

    def test_preserves_node_attributes(self):
        """Extracted nodes retain their spatial and temporal attributes."""
        g, ids = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 1.0, "y": 2.0, "x": 3.0},
                "C1": {"t": 2, "z": 4.0, "y": 5.0, "x": 6.0},
                "C2": {"t": 2, "z": 7.0, "y": 8.0, "x": 9.0},
            },
            [("A", "B"), ("B", "C1"), ("B", "C2")],
        )
        divs = extract_divisions(g)
        attrs = divs[ids["B"]].node_attrs().sort(td.DEFAULT_ATTR_KEYS.NODE_ID)
        # A (parent), B (divider), C1, C2 (children)
        assert attrs["y"].to_list() == [0.0, 2.0, 5.0, 8.0]
        assert attrs["z"].to_list() == [0.0, 1.0, 4.0, 7.0]
        assert attrs["x"].to_list() == [0.0, 3.0, 6.0, 9.0]


# ---------------------------------------------------------------------------
# GT used by most score_divisions tests:
#   P(t=0) → D(t=1) → C1(t=2,y=+5) → G1(t=3,y=+5)
#                     ↘ C2(t=2,y=-5) → G2(t=3,y=-5)
# ---------------------------------------------------------------------------
_GT_NODES = {
    "P":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
    "D":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
    "C1": {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
    "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
    "G1": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
    "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
}
_GT_EDGES = [("P", "D"), ("D", "C1"), ("D", "C2"), ("C1", "G1"), ("C2", "G2")]


def _make_gt():
    return _build_graph(_GT_NODES, _GT_EDGES)


class TestScoreDivisions:
    def test_perfect_prediction(self):
        """Exact copy of GT → 1."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 1 for v in scores.values())

    def test_disconnected_child(self):
        """One daughter not connected to the rest → 0."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            _GT_NODES,
            [("P", "D"), ("D", "C1"), ("C1", "G1"), ("C2", "G2")],  # no D→C2
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 0 for v in scores.values())

    def test_linear_no_fork(self):
        """Pred tracks the lineage but never splits → 0 (no fork)."""
        gt, _ = _make_gt()
        # Only one daughter present in pred
        pred, _ = _build_graph(
            {
                "P":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "G1": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("P", "D"), ("D", "C1"), ("C1", "G1")],
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 0 for v in scores.values())

    def test_no_matched_nodes(self):
        """Pred has no nodes near the GT division → 0."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {
                "X": {"t": 0, "z": 0.0, "y": 100.0, "x": 100.0},
                "Y": {"t": 1, "z": 0.0, "y": 100.0, "x": 100.0},
            },
            [("X", "Y")],
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 0 for v in scores.values())

    def test_fork_but_wrong_topology(self):
        """Pred has two nodes at t=2 but they belong to different tracks → 0."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {
                "P":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                # C2 is spatially correct but starts a separate track
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "G1": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
                "G2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
            },
            [("P", "D"), ("D", "C1"), ("C1", "G1"), ("C2", "G2")],  # C2 disconnected
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 0 for v in scores.values())

    def test_two_divisions_mixed_scores(self):
        """Two GT divisions: pred gets one right (1) and one wrong (0)."""
        gt, gt_ids = _build_graph(
            {
                # Division 1
                "P1":   {"t": 0, "z": 0.0, "y": 10.0, "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 10.0, "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C1a2": {"t": 3, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b2": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                # Division 2
                "P2":   {"t": 0, "z": 0.0, "y": -10.0, "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2b":  {"t": 2, "z": 0.0, "y": -15.0, "x": 0.0},
                "C2a2": {"t": 3, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2b2": {"t": 3, "z": 0.0, "y": -15.0, "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "C1a2"), ("C1b", "C1b2"),
                ("P2", "D2"), ("D2", "C2a"), ("D2", "C2b"),
                ("C2a", "C2a2"), ("C2b", "C2b2"),
            ],
        )
        pred, _ = _build_graph(
            {
                # Division 1: correct
                "P1":   {"t": 0, "z": 0.0, "y": 10.0, "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 10.0, "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C1a2": {"t": 3, "z": 0.0, "y": 15.0, "x": 0.0},
                "C1b2": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                # Division 2: linear (no split)
                "P2":   {"t": 0, "z": 0.0, "y": -10.0, "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2a2": {"t": 3, "z": 0.0, "y": -5.0,  "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "C1a2"), ("C1b", "C1b2"),
                ("P2", "D2"), ("D2", "C2a"), ("C2a", "C2a2"),
            ],
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert scores[gt_ids["D1"]] == 1
        assert scores[gt_ids["D2"]] == 0

    def test_no_gt_divisions(self):
        """GT with no divisions → empty dict."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B")],
        )
        pred, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B")],
        )
        assert score_divisions(pred, gt, max_distance=1.0) == {}

    def test_connected_via_intermediate_nodes(self):
        """Matched nodes are connected through unmatched intermediate nodes → 1."""
        gt, _ = _make_gt()
        # Pred has extra unmatched node M between D and C1/C2
        pred, _ = _build_graph(
            {
                "P":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "M":  {"t": 2, "z": 0.0, "y": 0.0,  "x": 0.0},  # no GT match
                "C1": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},  # matches GT G1
                "C2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},  # matches GT G2
            },
            [("P", "D"), ("D", "M"), ("M", "C1"), ("M", "C2")],
        )
        scores = score_divisions(pred, gt, max_distance=1.0)
        assert all(v == 1 for v in scores.values())

    def test_matched_parent_from_different_track(self):
        """A nearby node from another track steals the GT parent match,
        but the correct dividing component still covers both stages → 1.

        GT:  A(t=0) → B(t=1) → D(t=2,y=0) → C1(t=3,y=5)
                                              → C2(t=3,y=-5)

        Division subgraph: B(t=1), D(t=2), C1(t=3), C2(t=3)
          one-node stage: t=1 (B), t=2 (D)
          two-node stage: t=3 (C1, C2)

        Pred has two independent tracks:
          Track 1 (correct division):
            P1(t=0,y=-20) → P2(t=1,y=-15) → P3(t=2,y=0.5) → P4(t=3,y=5)
                                                               → P5(t=3,y=-5)
          Track 2 (steals GT parent B):
            Q1(t=0,y=0.1) → Q2(t=1,y=0.1) → Q3(t=2,y=50)

        Q2 matches B (parent), P3 matches D (divider), P4/P5 match C1/C2.
        Component {P3,P4,P5} covers t=2 (one-node) and t=3 (two-node) → 1.
        """
        gt, _ = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D":  {"t": 2, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C1": {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C2": {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
            },
            [("A", "B"), ("B", "D"), ("D", "C1"), ("D", "C2")],
        )
        pred, _ = _build_graph(
            {
                # Track 1: correct division
                "P1": {"t": 0, "z": 0.0, "y": -20.0, "x": 0.0},
                "P2": {"t": 1, "z": 0.0, "y": -15.0, "x": 0.0},
                "P3": {"t": 2, "z": 0.0, "y": 0.5,   "x": 0.0},
                "P4": {"t": 3, "z": 0.0, "y": 5.0,   "x": 0.0},
                "P5": {"t": 3, "z": 0.0, "y": -5.0,  "x": 0.0},
                # Track 2: steals GT parent B at t=1
                "Q1": {"t": 0, "z": 0.0, "y": 0.1,   "x": 0.0},
                "Q2": {"t": 1, "z": 0.0, "y": 0.1,   "x": 0.0},
                "Q3": {"t": 2, "z": 0.0, "y": 50.0,  "x": 0.0},
            },
            [
                ("P1", "P2"), ("P2", "P3"), ("P3", "P4"), ("P3", "P5"),
                ("Q1", "Q2"), ("Q2", "Q3"),
            ],
        )
        scores = score_divisions(pred, gt, max_distance=5.0)
        assert all(v == 1 for v in scores.values())


class TestCountMatchedPredDivisions:
    def test_perfect_prediction(self):
        """Exact copy of GT → 1 matched pred division."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        assert count_matched_pred_divisions(pred, gt, max_distance=1.0) == 1

    def test_spurious_division_on_linear_gt(self):
        """GT is linear, pred adds a split → 1 matched pred division."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C")],
        )
        pred, _ = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C1"), ("B", "C2")],
        )
        assert count_matched_pred_divisions(pred, gt, max_distance=1.0) == 1

    def test_two_divisions(self):
        """Two pred divisions matched to GT nodes → 2."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D": {"t": 0, "z": 0.0, "y": 20.0, "x": 0.0},
                "E": {"t": 1, "z": 0.0, "y": 20.0, "x": 0.0},
                "F": {"t": 2, "z": 0.0, "y": 20.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")],
        )
        pred, _ = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "D":  {"t": 0, "z": 0.0, "y": 20.0, "x": 0.0},
                "E":  {"t": 1, "z": 0.0, "y": 20.0, "x": 0.0},
                "F1": {"t": 2, "z": 0.0, "y": 20.0, "x": 0.0},
                "F2": {"t": 2, "z": 0.0, "y": 25.0, "x": 0.0},
            },
            [
                ("A", "B"), ("B", "C1"), ("B", "C2"),
                ("D", "E"), ("E", "F1"), ("E", "F2"),
            ],
        )
        assert count_matched_pred_divisions(pred, gt, max_distance=1.0) == 2

    def test_unmatched_division_not_counted(self):
        """Pred division far from any GT node → 0."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B")],
        )
        pred, _ = _build_graph(
            {
                "X": {"t": 0, "z": 0.0, "y": 100.0, "x": 0.0},
                "Y": {"t": 1, "z": 0.0, "y": 105.0, "x": 0.0},
                "Z": {"t": 1, "z": 0.0, "y": 95.0,  "x": 0.0},
            },
            [("X", "Y"), ("X", "Z")],
        )
        assert count_matched_pred_divisions(pred, gt, max_distance=1.0) == 0

    def test_mixed_real_and_spurious(self):
        """One real GT division + one spurious pred division → 2 matched."""
        gt, _ = _build_graph(
            {
                # Real division
                "P":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                # Linear track
                "A":  {"t": 0, "z": 0.0, "y": 20.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 20.0, "x": 0.0},
                "C":  {"t": 2, "z": 0.0, "y": 20.0, "x": 0.0},
            },
            [("P", "D"), ("D", "C1"), ("D", "C2"), ("A", "B"), ("B", "C")],
        )
        pred, _ = _build_graph(
            {
                # Correct division
                "P":  {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                # Spurious division
                "A":  {"t": 0, "z": 0.0, "y": 20.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 20.0, "x": 0.0},
                "E1": {"t": 2, "z": 0.0, "y": 20.0, "x": 0.0},
                "E2": {"t": 2, "z": 0.0, "y": 25.0, "x": 0.0},
            },
            [
                ("P", "D"), ("D", "C1"), ("D", "C2"),
                ("A", "B"), ("B", "E1"), ("B", "E2"),
            ],
        )
        assert count_matched_pred_divisions(pred, gt, max_distance=1.0) == 2


class TestEvaluateDivisions:
    def test_perfect_prediction(self):
        """Exact copy of GT → all TP, no FN or FP."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=1, fn=0, fp=0)

    def test_missed_division(self):
        """Pred is linear through the division → FN."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(
            {
                "P":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "D":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
                "G1": {"t": 3, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("P", "D"), ("D", "C1"), ("C1", "G1")],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=0, fn=1, fp=0)

    def test_spurious_division(self):
        """GT is linear, pred adds a split → FP."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C")],
        )
        pred, _ = _build_graph(
            {
                "A":  {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B":  {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
                "C1": {"t": 2, "z": 0.0, "y": 0.0, "x": 0.0},
                "C2": {"t": 2, "z": 0.0, "y": 5.0, "x": 0.0},
            },
            [("A", "B"), ("B", "C1"), ("B", "C2")],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=0, fn=0, fp=1)

    def test_mixed_tp_fn_fp(self):
        """Two GT divisions: one correct, one missed, plus a spurious one."""
        gt, _ = _build_graph(
            {
                # Division 1
                "P1":   {"t": 0, "z": 0.0, "y": 10.0,  "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 10.0,  "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 15.0,  "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": 5.0,   "x": 0.0},
                "C1a2": {"t": 3, "z": 0.0, "y": 15.0,  "x": 0.0},
                "C1b2": {"t": 3, "z": 0.0, "y": 5.0,   "x": 0.0},
                # Division 2
                "P2":   {"t": 0, "z": 0.0, "y": -10.0, "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2b":  {"t": 2, "z": 0.0, "y": -15.0, "x": 0.0},
                "C2a2": {"t": 3, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2b2": {"t": 3, "z": 0.0, "y": -15.0, "x": 0.0},
                # Linear track
                "A":    {"t": 0, "z": 0.0, "y": 30.0,  "x": 0.0},
                "B":    {"t": 1, "z": 0.0, "y": 30.0,  "x": 0.0},
                "C":    {"t": 2, "z": 0.0, "y": 30.0,  "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "C1a2"), ("C1b", "C1b2"),
                ("P2", "D2"), ("D2", "C2a"), ("D2", "C2b"),
                ("C2a", "C2a2"), ("C2b", "C2b2"),
                ("A", "B"), ("B", "C"),
            ],
        )
        pred, _ = _build_graph(
            {
                # Division 1: correct (TP)
                "P1":   {"t": 0, "z": 0.0, "y": 10.0,  "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 10.0,  "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 15.0,  "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": 5.0,   "x": 0.0},
                "C1a2": {"t": 3, "z": 0.0, "y": 15.0,  "x": 0.0},
                "C1b2": {"t": 3, "z": 0.0, "y": 5.0,   "x": 0.0},
                # Division 2: missed — linear only (FN)
                "P2":   {"t": 0, "z": 0.0, "y": -10.0, "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": -10.0, "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": -5.0,  "x": 0.0},
                "C2a2": {"t": 3, "z": 0.0, "y": -5.0,  "x": 0.0},
                # Linear track: spurious split (FP)
                "A":    {"t": 0, "z": 0.0, "y": 30.0,  "x": 0.0},
                "B":    {"t": 1, "z": 0.0, "y": 30.0,  "x": 0.0},
                "E1":   {"t": 2, "z": 0.0, "y": 30.0,  "x": 0.0},
                "E2":   {"t": 2, "z": 0.0, "y": 35.0,  "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "C1a2"), ("C1b", "C1b2"),
                ("P2", "D2"), ("D2", "C2a"), ("C2a", "C2a2"),
                ("A", "B"), ("B", "E1"), ("B", "E2"),
            ],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=1, fn=1, fp=1)

    def test_no_divisions_in_either(self):
        """No divisions anywhere → all zeros."""
        gt, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B")],
        )
        pred, _ = _build_graph(
            {
                "A": {"t": 0, "z": 0.0, "y": 0.0, "x": 0.0},
                "B": {"t": 1, "z": 0.0, "y": 0.0, "x": 0.0},
            },
            [("A", "B")],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert counts == DivisionCounts(tp=0, fn=0, fp=0)

    def test_returns_named_tuple(self):
        """Result is a DivisionCounts with named fields."""
        gt, _ = _make_gt()
        pred, _ = _build_graph(_GT_NODES, _GT_EDGES)
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        assert isinstance(counts, DivisionCounts)
        assert counts.tp == 1
        assert counts.fn == 0
        assert counts.fp == 0

    def test_duplicate_pred_divisions_no_tp_inflation(self):
        """Duplicating a pred division structure should not inflate TP.

        Pred has 3 identical copies of the GT division.  Only 1 GT division
        exists, so TP must stay at 1.
        """
        gt, _ = _make_gt()
        # Three copies of the same division at the same positions
        pred, _ = _build_graph(
            {
                # Copy 1
                "P1":   {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D1":   {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C1a":  {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C1b":  {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "G1a":  {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                "G1b":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
                # Copy 2 (identical positions)
                "P2":   {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D2":   {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C2a":  {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C2b":  {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "G2a":  {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                "G2b":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
                # Copy 3 (identical positions)
                "P3":   {"t": 0, "z": 0.0, "y": 0.0,  "x": 0.0},
                "D3":   {"t": 1, "z": 0.0, "y": 0.0,  "x": 0.0},
                "C3a":  {"t": 2, "z": 0.0, "y": 5.0,  "x": 0.0},
                "C3b":  {"t": 2, "z": 0.0, "y": -5.0, "x": 0.0},
                "G3a":  {"t": 3, "z": 0.0, "y": 5.0,  "x": 0.0},
                "G3b":  {"t": 3, "z": 0.0, "y": -5.0, "x": 0.0},
            },
            [
                ("P1", "D1"), ("D1", "C1a"), ("D1", "C1b"),
                ("C1a", "G1a"), ("C1b", "G1b"),
                ("P2", "D2"), ("D2", "C2a"), ("D2", "C2b"),
                ("C2a", "G2a"), ("C2b", "G2b"),
                ("P3", "D3"), ("D3", "C3a"), ("D3", "C3b"),
                ("C3a", "G3a"), ("C3b", "G3b"),
            ],
        )
        counts = evaluate_divisions(pred, gt, max_distance=1.0)
        # Bipartite matching is one-to-one: only 1 of the 3 duplicate pred
        # divisions gets matched, so TP stays at 1 and FP stays at 0.
        assert counts == DivisionCounts(tp=1, fn=0, fp=0)
