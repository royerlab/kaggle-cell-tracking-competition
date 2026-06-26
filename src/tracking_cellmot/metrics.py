import warnings
from typing import Literal, NamedTuple

import polars as pl
import tracksdata as td


class EvaluationResult(NamedTuple):
    """Counts returned by :func:`evaluate`."""

    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int


class DatasetsResult(NamedTuple):
    """Cumulative (micro-averaged) Jaccards plus the combined score."""

    edge_jaccard: float
    division_jaccard: float
    score: float


# Penalty coefficient for the adjusted edge Jaccard:
#   J_adj = max(0, J · (1 - ADJUSTMENT_ALPHA · total_node_ratio))
ADJUSTMENT_ALPHA: float = 0.1

# Weight of the division Jaccard in the combined run-level score:
#   score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT · division_jaccard
SCORE_DIVISION_WEIGHT: float = 0.1

COUNT_COLUMNS: tuple[str, ...] = (
    "edge_tp", "edge_fp", "edge_fn",
    "division_tp", "division_fp", "division_fn",
    "num_pred_nodes",
)
METRIC_COLUMNS: tuple[str, ...] = COUNT_COLUMNS + (
    "node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard",
)


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


# function is split for easier testing
def _evaluate_matched_graph(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> pl.DataFrame:
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK])
    # Guard against duplicate edges (same source→target pair appearing multiple times).
    # tracksdata's match() inner-join marks all duplicates as matched, which inflates
    # the intersection count and can push scores above 1.0. Sort matched rows first
    # so the dedup keeps the matched copy when duplicates disagree on the mask.
    edge_attrs = edge_attrs.sort(
        td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK, descending=True,
    ).unique(
        subset=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET],
        keep="first",
    )
    node_attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])

    # I'm assuming valid ground-truth edges are always 100% correct if they have an edge.
    # Therefore, we don't have cases where the cell divided, but not in the ground truth.
    gt_node_ids = gt_graph.node_ids()
    gt_node_attrs = pl.DataFrame(
        {
            td.DEFAULT_ATTR_KEYS.NODE_ID: gt_node_ids,
            "out_degree": gt_graph.out_degree(gt_node_ids),
            "in_degree": gt_graph.in_degree(gt_node_ids),
        }
    ).with_columns(
        (pl.col("out_degree") > 0).alias("out_valid"),
        (pl.col("in_degree") > 0).alias("in_valid"),
    )

    # merging ground truth graph into the predicted graph
    node_attrs = node_attrs.join(
        gt_node_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).with_columns(
        pl.col("out_valid").fill_null(False),
        pl.col("in_valid").fill_null(False),
    )

    # merge out valid into source and in valid into target
    edge_attrs = edge_attrs.join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "out_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "in_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )

    edge_attrs = edge_attrs.with_columns(
        (pl.col("out_valid") | pl.col("in_valid")).alias("pred_valid"),
    )

    # sanity check that `pred_valid` is a superset of all matched edges
    assert edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)["pred_valid"].all()

    return edge_attrs


def _compute_score(
    edge_attrs: pl.DataFrame,
    gt_num_edges: int,
    metric: Literal["jaccard", "dice"],
) -> float:
    intersection = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
    n_valid_pred_edges = int(edge_attrs["pred_valid"].sum())

    if metric == "jaccard":
        num = intersection
        denom = gt_num_edges + n_valid_pred_edges - intersection
    elif metric == "dice":
        num = 2 * intersection
        denom = gt_num_edges + n_valid_pred_edges
    else:
        raise ValueError(f"Invalid metric: {metric}")

    return num / denom if denom > 0 else float("nan")


def _evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    metric: Literal["jaccard", "dice"],
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> float:
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in graph.node_attr_keys():
        warnings.warn("Graph already matched, overwriting previous matching.")
        # Reset matching attributes to defaults before re-matching
        all_node_ids = graph.node_ids()
        graph.update_node_attrs(
            node_ids=all_node_ids,
            attrs={
                td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1,
                td.DEFAULT_ATTR_KEYS.MATCH_SCORE: 0.0,
            },
        )
        all_edge_ids = graph.edge_ids()
        if len(all_edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=all_edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )

    from tracksdata.metrics import DistanceMatching
    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    if graph.num_edges() == 0 or graph.num_nodes() == 0:
        warnings.warn("Predicted graph has no edges or no nodes, returning score 0.0.")
        return 0.0

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning
            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            graph.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    return _compute_score(edge_attrs, gt_graph.num_edges(), metric)


def evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> EvaluationResult:
    """
    Evaluate a predicted graph against a ground-truth graph using
    centroid-distance node matching.

    Computes edge TP/FP/FN, division TP/FP/FN (via
    :func:`tracking_cellmot.division_metrics.evaluate_divisions`), and the
    total number of predicted nodes (irrespective of matching).

    Parameters
    ----------
    graph : tracksdata.graph.BaseGraph
        The predicted graph. Matching attributes are written onto *graph*
        as a side effect.
    gt_graph : tracksdata.graph.BaseGraph
        The ground truth graph.
    scale : tuple[float, ...] | None, optional
        Physical scale for each spatial dimension (e.g., (z, y, x)) to
        account for anisotropy. If None, assumes isotropic data.
    max_distance : float, optional
        Maximum distance between centroids to be considered as a match.

    Returns
    -------
    EvaluationResult
    """
    from .division_metrics import evaluate_divisions

    # Match graph against gt_graph (in place); discard the returned score.
    _evaluate(graph, gt_graph, "jaccard", scale, max_distance)

    if graph.num_edges() == 0:
        edge_tp = 0
        edge_fp = 0
        edge_fn = gt_graph.num_edges()
    else:
        edge_attrs = _evaluate_matched_graph(graph, gt_graph)
        edge_tp = int(edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum())
        edge_valid_pred = int(edge_attrs["pred_valid"].sum())
        edge_fp = edge_valid_pred - edge_tp
        edge_fn = gt_graph.num_edges() - edge_tp

    div = evaluate_divisions(
        graph, gt_graph, scale=scale, max_distance=max_distance,
    )

    return EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div.tp,
        division_fp=div.fp,
        division_fn=div.fn,
        num_pred_nodes=graph.num_nodes(),
    )


def evaluate_datasets(
    graph_pairs: list[tuple[td.graph.BaseGraph, td.graph.BaseGraph]],
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DatasetsResult:
    """Run :func:`evaluate` on each (pred, gt) pair and return cumulative
    (micro-averaged) edge and division Jaccard.

    Per-pair TP/FP/FN counts are summed across the whole list before the
    Jaccard is computed, so larger datasets dominate the score naturally.

    Parameters
    ----------
    graph_pairs : list of (pred_graph, gt_graph)
        Predicted / ground-truth graph pairs. Each *pred_graph* is mutated
        in place by matching (same side effect as :func:`evaluate`).
    scale : tuple[float, ...] | None, optional
        Physical voxel scale used for centroid-distance matching.
    max_distance : float, optional
        Maximum centroid distance for a match.

    Returns
    -------
    DatasetsResult
        Named tuple with ``edge_jaccard``, ``division_jaccard``, and the
        combined ``score = edge_jaccard + SCORE_DIVISION_WEIGHT *
        division_jaccard``. If no divisions exist anywhere in the input
        the division term is dropped and ``score = edge_jaccard``.
    """
    edge_tp = edge_fp = edge_fn = 0
    div_tp = div_fp = div_fn = 0
    for pred, gt in graph_pairs:
        r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        edge_tp += r.edge_tp
        edge_fp += r.edge_fp
        edge_fn += r.edge_fn
        div_tp += r.division_tp
        div_fp += r.division_fp
        div_fn += r.division_fn

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    has_divisions = (div_tp + div_fp + div_fn) > 0
    division_jaccard = _jaccard(div_tp, div_fp, div_fn) if has_divisions else float("nan")
    score = edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard if has_divisions else edge_jaccard

    return DatasetsResult(
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        score=score,
    )


def _matched_node_ids(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Return a DataFrame with NODE_ID and MATCHED_NODE_ID (as Int64) for *graph*."""
    node_attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    )
    return node_attrs


def node_recall(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> float:
    """Fraction of GT nodes that were matched by a predicted node.

    The predicted graph must already be matched (e.g. via :func:`evaluate` or
    ``graph.match``).
    """
    node_attrs = _matched_node_ids(graph)
    matched = node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )
    n_matched_gt = matched[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].n_unique()
    return n_matched_gt / gt_graph.num_nodes()


def per_sample_metrics(
    er: EvaluationResult,
    n_total: float,
    node_recall: float,
) -> dict:
    """Derive per-sample metric columns from an :class:`EvaluationResult`.

    Computes ``edge_jaccard``, ``total_node_ratio`` (``(N_pred − N_total) / N_total``),
    and the adjusted edge Jaccard ``J_adj = max(0, J · (1 − α · total_node_ratio))``
    with α = :data:`ADJUSTMENT_ALPHA`.

    Parameters
    ----------
    er
        Counts for one (pred, gt) pair — see :func:`evaluate`.
    n_total
        Target node count (e.g. from the GEFF ``estimated_number_of_nodes``
        metadata extra). Pass ``float("nan")`` when unavailable; that makes
        ``total_node_ratio`` and ``adj_edge_jaccard`` also NaN.
    node_recall
        Fraction of GT nodes matched by a predicted node.

    Returns
    -------
    dict
        One entry per key in :data:`METRIC_COLUMNS`.
    """
    if n_total > 0:
        total_node_ratio = (er.num_pred_nodes - n_total) / n_total
    else:
        total_node_ratio = float("nan")

    edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
    edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else float("nan")
    if edge_jaccard == edge_jaccard and total_node_ratio == total_node_ratio:
        adj_edge_jaccard = max(
            0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio),
        )
    else:
        adj_edge_jaccard = float("nan")

    return {
        "edge_tp": er.edge_tp, "edge_fp": er.edge_fp, "edge_fn": er.edge_fn,
        "division_tp": er.division_tp,
        "division_fp": er.division_fp,
        "division_fn": er.division_fn,
        "num_pred_nodes": er.num_pred_nodes,
        "node_recall": node_recall,
        "total_node_ratio": total_node_ratio,
        "edge_jaccard": edge_jaccard,
        "adj_edge_jaccard": adj_edge_jaccard,
    }


def nan_metrics_row() -> dict:
    """Return a dict with every :data:`METRIC_COLUMNS` key set to NaN."""
    return {col: float("nan") for col in METRIC_COLUMNS}


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-sample metric rows into a run-level summary.

    - ``edge_jaccard`` / ``division_jaccard``: micro-averaged across valid rows
      (TP/FP/FN summed, then Jaccard).
    - ``adj_edge_jaccard``: per-sample adjusted Jaccard weight-averaged by
      sample size ``w_i = TP_i + FP_i + FN_i``; rows with NaN are skipped.
    - ``score``: ``adj_edge_jaccard + SCORE_DIVISION_WEIGHT · division_jaccard``.

    Parameters
    ----------
    rows
        Per-sample dicts as produced by :func:`per_sample_metrics`. Rows with
        NaN ``edge_tp`` are treated as failed evaluations and skipped.
    """
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {
            "n": 0, "edge_jaccard": float("nan"),
            "division_jaccard": float("nan"),
            "division_tp": 0, "division_fp": 0, "division_fn": 0,
            "node_recall": float("nan"),
            "adj_edge_jaccard": float("nan"), "n_adj": 0,
            "score": float("nan"),
        }
    totals = {c: sum(r[c] for r in valid) for c in COUNT_COLUMNS}

    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    weights = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    total_w = sum(weights)
    if total_w > 0:
        adj_edge_jaccard = sum(
            w * r["adj_edge_jaccard"] for w, r in zip(weights, adj_rows)
        ) / total_w
    else:
        adj_edge_jaccard = float("nan")

    division_total = (
        totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    )
    if division_total == 0:
        warnings.warn(
            "No divisions present across any sample in this split; "
            "dropping division term from the combined score."
        )
        division_jaccard = float("nan")
        score = adj_edge_jaccard
    else:
        division_jaccard = _jaccard(
            totals["division_tp"], totals["division_fp"], totals["division_fn"],
        )
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    return {
        "n": len(valid),
        "edge_jaccard": _jaccard(
            totals["edge_tp"], totals["edge_fp"], totals["edge_fn"],
        ),
        "division_jaccard": division_jaccard,
        "division_tp": totals["division_tp"],
        "division_fp": totals["division_fp"],
        "division_fn": totals["division_fn"],
        "node_recall": sum(r["node_recall"] for r in valid) / len(valid),
        "adj_edge_jaccard": adj_edge_jaccard,
        "n_adj": len(adj_rows),
        "score": score,
    }
