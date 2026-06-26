import warnings
from collections import deque
from typing import NamedTuple

import polars as pl
import tracksdata as td


class DivisionCounts(NamedTuple):
    """Counts for division event evaluation."""

    tp: int
    fn: int
    fp: int


def _reset_matching_attrs(graph: td.graph.BaseGraph) -> None:
    """Reset any pre-existing match attrs in place so a fresh ``.match()`` isn't
    contaminated by stale values carried in from a previous matching pass."""
    node_keys = graph.node_attr_keys()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in node_keys:
        node_ids = graph.node_ids()
        if len(node_ids) > 0:
            reset: dict = {td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1}
            if td.DEFAULT_ATTR_KEYS.MATCH_SCORE in node_keys:
                reset[td.DEFAULT_ATTR_KEYS.MATCH_SCORE] = 0.0
            graph.update_node_attrs(node_ids=node_ids, attrs=reset)
    if td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK in graph.edge_attr_keys():
        edge_ids = graph.edge_ids()
        if len(edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )


def extract_divisions(
    graph: td.graph.BaseGraph,
) -> dict[int, td.graph.BaseGraph]:
    """Extract individual division events as separate subgraphs.

    Each division event includes the parent of the dividing node, the
    dividing node, its children, and the grandchildren::

        parent → divider → child1 → grandchild1
                         → child2 → grandchild2

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The input tracking graph.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from dividing node ID to a subgraph containing the
        parent, divider, children, and grandchildren.
    """
    divisions: dict[int, td.graph.BaseGraph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        keep = [*parents, div_node, *children, *grandchildren]
        divisions[div_node] = graph.filter(node_ids=keep).subgraph()
    return divisions


def match_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, td.graph.BaseGraph]:
    """Match the predicted graph against each GT division subgraph.

    Extracts division events from *gt_graph* via :func:`extract_divisions`,
    then runs ``pred_graph.match(gt_div, ...)`` for each one independently.
    A fresh copy of *pred_graph* is used per division so matchings don't
    interfere.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    dict[int, td.graph.BaseGraph]
        Mapping from GT dividing-node ID to the matched copy of
        *pred_graph* for that division.
    """
    from tracksdata.metrics import DistanceMatching
    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    gt_divisions = extract_divisions(gt_graph)
    matched: dict[int, td.graph.BaseGraph] = {}

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for div_node, gt_div in gt_divisions.items():
            pred_copy = pred_graph.copy()
            _reset_matching_attrs(pred_copy)
            with warnings.catch_warnings():
                from scipy.sparse import SparseEfficiencyWarning
                warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
                pred_copy.match(gt_div, matching=matching)
            matched[div_node] = pred_copy
    finally:
        set_options(show_progress=prev_show_progress)

    return matched


def _match_full(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> td.graph.BaseGraph:
    """Match the full pred graph against the full GT graph, return the matched copy."""
    from tracksdata.metrics import DistanceMatching
    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    pred_copy = pred_graph.copy()
    _reset_matching_attrs(pred_copy)

    from tracksdata.options import get_options, set_options

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning
            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            pred_copy.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    return pred_copy


def _matched_node_attrs(graph: td.graph.BaseGraph) -> pl.DataFrame:
    """Return node attrs (node_id, matched_node_id, t) for matched pred nodes."""
    node_attrs = graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
            "t",
        ],
    )
    return node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )


def _has_stage_coverage(
    matched_attrs: pl.DataFrame,
    gt_div: td.graph.BaseGraph,
    divider_id: int,
) -> bool:
    """Check that matches cover both stages of a GT division.

    The GT division subgraph has a *one-node stage* (timepoints with a
    single GT node — parent and divider, pre-split) and two or more
    *daughter lineages* (each child of *divider_id* plus its descendants
    within the subgraph). A valid match requires:

    * ≥1 matched prediction node whose timepoint falls in the one-node
      stage, AND
    * matched prediction nodes whose matched GT nodes cover ≥2 distinct
      daughter lineages. Lineage hits may occur at different timepoints;
      a single daughter matched only at t=divider+2 still counts.

    When the subgraph contains a secondary divider (e.g. successive
    divisions), *divider_id* disambiguates which split we're scoring.
    """
    if matched_attrs.is_empty():
        return False

    gt_time_counts = (
        gt_div.node_attrs(attr_keys=["t"])
        .group_by("t")
        .agg(pl.len().alias("n"))
    )
    one_node_times = set(gt_time_counts.filter(pl.col("n") == 1)["t"].to_list())
    if not one_node_times:
        return False

    children = gt_div.successors(divider_id)
    if len(children) < 2:
        return False

    def _descendants(seed: int) -> set[int]:
        out: set[int] = {seed}
        stack = [seed]
        while stack:
            for nxt in gt_div.successors(stack.pop()):
                if nxt not in out:
                    out.add(nxt)
                    stack.append(nxt)
        return out

    lineages = [_descendants(c) for c in children]

    matched_time_counts = matched_attrs.group_by("t").agg(pl.len().alias("n"))
    has_one = matched_time_counts.filter(pl.col("t").is_in(one_node_times)).height > 0
    if not has_one:
        return False

    matched_gt_ids = set(matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list())
    lineages_covered = sum(1 for lin in lineages if lin & matched_gt_ids)
    return lineages_covered >= 2


def _weakly_connected_components(
    graph: td.graph.BaseGraph,
    node_ids: list[int],
) -> list[tuple[set[int], set[int]]]:
    """Partition *node_ids* into weakly-connected components of *graph*.

    Returns one ``(matched_subset, visited)`` pair per component:
    *matched_subset* is the component restricted to *node_ids*; *visited*
    is every graph node reachable from the component (including unmatched
    intermediaries). The visited set lets callers locate structural
    features -- in particular pred dividing nodes -- that may sit on
    unmatched nodes between matched ones.
    """
    remaining = set(node_ids)
    components: list[tuple[set[int], set[int]]] = []
    while remaining:
        seed = next(iter(remaining))
        visited: set[int] = {seed}
        queue: deque[int] = deque([seed])
        component: set[int] = {seed}
        while queue:
            current = queue.popleft()
            neighbors = graph.successors(current) + graph.predecessors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if neighbor in remaining:
                        component.add(neighbor)
        components.append((component, visited))
        remaining -= component
    return components


def _bipartite_max_matching(
    left: list[int],
    edges: dict[int, set[int]],
) -> dict[int, int]:
    """Maximum-cardinality bipartite matching via DFS augmenting paths.

    *edges* maps each left-side vertex to the set of adjacent right-side
    vertices. Returns only the matched pairs as a ``left → right`` dict.
    """
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())

    return match_l


def score_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, int]:
    """Score each GT division: 1 if the prediction recovers it, 0 otherwise.

    For each GT division, the predicted graph is matched against the
    division subgraph and checked for a spanning component satisfying:

    1. At least one matched prediction node in the GT's one-node stage
       (pre-division timepoints).
    2. At least two matched prediction nodes at the same timepoint in the
       GT's two-node stage (post-division timepoints).
    3. All matched prediction nodes in a single weakly-connected
       component of the prediction graph.

    Each such spanning component is associated with the *pred dividing
    nodes* (out-degree ≥ 2) it contains. A maximum-cardinality bipartite
    matching is then computed so each pred dividing node serves at most
    one GT division, and each GT division is paired with at most one
    pred dividing node. A GT division scores 1 only if it is paired in
    that matching -- this prevents a single pred fork from being
    credited to multiple GT divisions.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    dict[int, int]
        Mapping from GT dividing-node ID to 1 (paired) or 0 (not).
    """
    matched = match_divisions(
        pred_graph, gt_graph, scale, max_distance,
    )
    gt_divisions = extract_divisions(gt_graph)
    pred_div_nodes = set(pred_graph.dividing_nodes())

    candidates: dict[int, set[int]] = {}
    for div_node, matched_pred in matched.items():
        matched_attrs = _matched_node_attrs(matched_pred)
        node_ids = matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
        components = _weakly_connected_components(matched_pred, node_ids)
        gt_div = gt_divisions[div_node]
        div_candidates: set[int] = set()
        for matched_subset, visited in components:
            comp_attrs = matched_attrs.filter(
                pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).is_in(list(matched_subset))
            )
            if _has_stage_coverage(comp_attrs, gt_div, div_node):
                div_candidates |= visited & pred_div_nodes
        candidates[div_node] = div_candidates

    pairing = _bipartite_max_matching(list(candidates), candidates)
    return {div: int(div in pairing) for div in candidates}


def count_matched_pred_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> int:
    """Count predicted division nodes whose matched GT node is annotated.

    Matches the full predicted graph against the full GT graph.  Among
    predicted nodes that were matched to a GT node, counts how many are
    dividing (out-degree >= 2) in the prediction *and* whose matched GT
    node has at least one child.  A matched GT node with no children marks
    the end of the annotation — we can't tell whether the cell actually
    divided there, so such predicted divisions are excluded from the count
    (and therefore from the FP tally).

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    int
        Number of matched predicted division nodes.
    """
    matched_pred = _match_full(
        pred_graph, gt_graph, scale, max_distance,
    )

    node_attrs = matched_pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID],
    )
    matched_nodes = node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )

    count = 0
    for row in matched_nodes.iter_rows(named=True):
        pred_node = row[td.DEFAULT_ATTR_KEYS.NODE_ID]
        gt_node = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        if (
            matched_pred.out_degree(pred_node) >= 2
            and gt_graph.out_degree(gt_node) >= 1
        ):
            count += 1
    return count


def evaluate_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionCounts:
    """Compute TP, FN, and FP counts for division events.

    - **TP**: GT divisions correctly recovered in the prediction
      (matched nodes connected and forking).
    - **FN**: GT divisions not recovered.
    - **FP**: Predicted divisions whose matched GT node is not dividing.

    Parameters
    ----------
    pred_graph : td.graph.BaseGraph
        The predicted tracking graph.
    gt_graph : td.graph.BaseGraph
        The ground-truth tracking graph.
    scale : tuple[float, ...] | None
        Physical voxel scale used for centroid-distance matching.
    max_distance : float
        Maximum centroid distance for a match.

    Returns
    -------
    DivisionCounts
        Named tuple with ``tp``, ``fn``, and ``fp`` fields.
    """
    scores = score_divisions(
        pred_graph, gt_graph, scale, max_distance,
    )
    tp = sum(scores.values())
    fn = len(scores) - tp
    matched_pred_divs = count_matched_pred_divisions(
        pred_graph, gt_graph, scale, max_distance,
    )
    fp = max(0, matched_pred_divs - tp)
    return DivisionCounts(tp=tp, fn=fn, fp=fp)
