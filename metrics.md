# Evaluating tracking predictions

A tracking graph has nodes (cell detections with a timepoint and centroid)
and directed edges linking a cell at one timepoint to the same cell — or its
daughters — at the next. A **cell division** is a node with exactly two
outgoing edges.

## Edge Jaccard

Our ground-truth annotations are **sparse**: we haven't annotated every
cell in the videos. The ground truth contains a subset of the true nodes
and the edges between those nodes.

The metric proceeds as follows:

1. **Node matching** pairs predicted nodes with ground-truth nodes by
   centroid distance, up to a maximum distance of **7 µm**. Matching
   uses an optimal bipartite assignment, so each predicted node pairs
   with at most one ground-truth node.
2. **Edge matching** counts a predicted edge as a **true positive (TP)** when
   both endpoints are matched to ground-truth nodes connected by a
   ground-truth edge. Every ground-truth edge without such a match is a
   **false negative (FN)**. A predicted edge that is not a TP is counted
   as a **false positive (FP)** in either of these cases:

   - A predicted edge with a target node that matches a GT node that is connected to another source node.
   - A predicted edge with a source node that matches a GT node that is connected to another target node.

   All other predicted edges are ignored by our metric.

The edge Jaccard is then `TP / (TP + FP + FN)`.

Because the ground truth is sparse, a correct prediction will inevitably include nodes and edges the ground truth doesn't cover. Predicted nodes that do not match a ground-truth node are not counted as false positives.
![Edge Jaccard on the `simple` example](assets/figure.svg)
<!-- *Example showing how predicted edges are labelled TP, FP, and FN against a sparse ground truth to compute the edge Jaccard.* -->

### Adjusted edge Jaccard

To penalise false-positive node predictions, the edge Jaccard is scaled by
a penalty on the total number of predicted nodes:

```
adjusted_jaccard = max(0, jaccard · (1 − a · (T_pred − T_true) / T_true))
```

where `T_pred` is the total number of predicted nodes, `T_true` is a provided
coarse estimate of the total number of true nodes (including those the ground
truth doesn't annotate), and `a = 0.1` is the weighting coefficient.

## Division Jaccard

The exact timepoint at which a cell visibly splits is somewhat
subjective, so we score divisions with a tolerance of ±1 timepoint on
either side of the ground-truth split.

For each ground-truth division (a GT node that splits into two children,
with its parent and grandchildren included for context), we ask whether
the prediction contains a corresponding split. A GT division is a
**true positive (TP)** when a predicted fork can be paired with it
under *all* of the following criteria:

- **One-node-stage coverage.** The prediction has at least one matched
  node at a pre-split timepoint of the GT division (parent or divider
  era), anchoring the pre-division track.
- **Both daughter lineages touched.** The prediction has matched nodes
  in each of the two daughter lineages of the GT divider, where a
  lineage is one child plus its descendants inside the division
  subgraph. Hits in the two lineages may occur at *different*
  timepoints — the two daughters don't have to be predicted
  simultaneously, which absorbs ±1-timepoint offsets in when the split
  becomes visible.
- **Single connected component.** All the matched predicted nodes
  above lie in one connected component of the predicted graph.
- **Contains a predicted fork.** The component includes at least one
  predicted dividing node (a predicted cell with two outgoing edges).

A GT division that fails any of these is a **false negative (FN)**.

A predicted dividing cell whose match lands on a GT cell with outgoing
edges (so the region is annotated) but that is not paired to any GT
division by the bipartite matching is a **false positive (FP)**.
Predicted divisions in unannotated regions are ignored, mirroring the
edge rule.

The division Jaccard is then `TP / (TP + FP + FN)`.

![Division Jaccard on the `simple` example](assets/division.svg)


![Division Jaccard on the `simple` example](assets/late_division.svg)
*A predicted fork that occurs one timepoint after the ground-truth split, it is still counted as a TP in our division metrics.*

## Final score

We aggregate the results from each video into a single score by **micro-averaging**:
per-sample TP, FP, and FN counts are summed across the whole split
*before* the Jaccard is computed, so larger samples contribute
proportionally more than small ones and a sample with zero events
doesn't skew the average.

Concretely:

- **Adjusted edge Jaccard** is the per-sample adjusted Jaccard
  weight-averaged by sample size `w_i = TP_i + FP_i + FN_i`.

- **Division Jaccard** uses the summed division TP/FP/FN across all videos.

The final combined score is
```
score = adjusted_edge_jaccard + w · division_jaccard
```

with a small weight `w = 0.1` on the division term.