# tracking-cellmot

A baseline for the [**Biohub – Cell Tracking During Development**](https://kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition: cell tracking in 3D+time microscopy with **sparse ground truth**.

<p align="center">
  <img src="assets/demo.gif" alt="Cell tracking demo" width="480">
</p>

## The challenge

The goal is to track cells across frames in 3D microscopy videos. Ground truth annotations are sparse — only a subset of cells are annotated in each video. Annotations are stored in [tracksdata](https://github.com/royerlab/tracksdata) `.geff` format: nodes represent approximate cell centers `(t, z, y, x)`, edges link cells across time, and divisions appear as one node at `t` linked to two nodes at `t+1`.

## Baseline method

End-to-end detection + linking, trained jointly:

1. **Detection**: a 3D U-Net with temporal attention (`TemporalUNet3D`) produces per-voxel features and a single-channel detection map; cell centres are recovered with local-max suppression.
2. **Linking**: per-node features from the U-Net are pooled at the detected centres and fed to a cross-attention transformer (`SimpleNodeTransformer`), which scores every (t, t+1) node pair
3. **Sparse supervision**: only edges with ground truth are used for backpropagation — background detections and unannotated cells are ignored during training

## Metrics

Predictions are scored on edge and division detection (TP/FP/FN counts, micro-averaged Jaccard, and a combined final score). The metrics are defined in detail in [`metrics.md`](metrics.md).

## Installation

```bash
uv sync
```

## Running scripts

### Training

```bash
uv run python scripts/train_unet_transformer.py --method baseline --split 0 --epochs 1
```

### Prediction

```bash
uv run python scripts/predict_unet_transformer.py --method baseline --split 0
```

### Evaluation

```bash
uv run python scripts/evaluate.py --method baseline --split 0
```

Each run prints the evaluation metrics to the terminal.

### Visualization

Visualization uses [napari](https://napari.org); install the optional `viz`
extra first:

```bash
uv sync --extra viz
```

Browse ground-truth tracks for every dataset under `DATASET_PATH`:

```bash
uv run python visualize/visualize_ground_truth.py
```

Browse predictions side-by-side with ground truth (TP / FP / FN edges colour-coded):

```bash
uv run python visualize/visualize_predictions.py --method baseline --split 0
```

## Python API

### Loading a dataset

```python
from tracking_cellmot.io import open_dataset
from scripts.dataspec import DATASET_PATH

ds = open_dataset(
    DATASET_PATH / "sample_1",
    normalize=True,       # quantile normalise
    require_tracks=True,  # load paired .geff ground truth
)

print(ds.image.shape)  # (T, Z, Y, X)
print(ds.scale)        # voxel scale in microns
```

### Building a prediction graph

A prediction graph is just a `tracksdata` `InMemoryGraph` of cell detections
linked across time. Tiny example with two nodes and one edge:

```python
import polars as pl
import tracksdata as td

predicted_graph = td.graph.InMemoryGraph()
for key in ("z", "y", "x"):
    predicted_graph.add_node_attr_key(key, pl.Float64, 0.0)

n0 = predicted_graph.add_node({"t": 0, "z": 0.0, "y": 10.0, "x": 20.0})
n1 = predicted_graph.add_node({"t": 1, "z": 0.0, "y": 11.0, "x": 21.0})
predicted_graph.add_edge(n0, n1, {})
```

A division is encoded as one source node with two outgoing edges
(`add_edge(parent, child_a, {})` and `add_edge(parent, child_b, {})`).

### Evaluating predictions

For a more detailed description of the metrics, refer to [`metrics.md`](metrics.md).

```python
from tracking_cellmot.metrics import evaluate
from tracking_cellmot.io import open_dataset, save_graph

result = evaluate(
    graph=predicted_graph,
    gt_graph=ds.tracks,
    scale=ds.scale,
)

# evaluate() returns raw counts; aggregate across datasets with evaluate_datasets.
print(f"Edges     TP/FP/FN: {result.edge_tp}/{result.edge_fp}/{result.edge_fn}")
print(f"Divisions TP/FP/FN: {result.division_tp}/{result.division_fp}/{result.division_fn}")
```

### Evaluating across multiple datasets

```python
from tracking_cellmot.metrics import evaluate_datasets

result = evaluate_datasets(
    graph_pairs=[(pred_graph_1, gt_graph_1), (pred_graph_2, gt_graph_2), ...],
    scale=ds.scale,
)
# Counts are summed across all pairs before the Jaccards are computed
# (micro-averaged).
print(f"Edge Jaccard:     {result.edge_jaccard:.4f}")
print(f"Division Jaccard: {result.division_jaccard:.4f}")
print(f"Final Score:            {result.score:.4f}")
```

## Data format

- **Images**: OME-Zarr with dimensions `(T, Z, Y, X)`. Scale of the spatial axes `Z, Y, X` is `1.625, 0.40625, 0.40625`, in microns per pixel.
- **Tracks**: GEFF files (tracksdata) — sparse spatial graphs with nodes `(t, z, y, x)` and temporal edges. Only a subset of cells are annotated. Divisions are encoded as one source node with two target edges.

### Layout

Each dataset is a `{name}.zarr` image with a paired `{name}.geff` track graph in the same directory. On Kaggle these sit under the competition mount:

```
/kaggle/input/competitions/biohub-cell-tracking-during-development/
├── train/   {name}.zarr + {name}.geff   (ground truth provided)
└── test/    {name}.zarr                  (no ground truth — for submission)
```

`scripts/dataspec.py` resolves the dataset directory automatically: it uses `$CELLMOT_DATA_DIR` if set, otherwise the Kaggle `train/` mount when running on Kaggle, otherwise local `./data/dense_channel`. Every script also accepts `--data-dir` (and `--splits`) to point at any location per run — e.g. `--data-dir <competition>/test` for prediction.

## Dependencies

- PyTorch
- [tracksdata](https://github.com/royerlab/tracksdata) — graph format and matching
- Zarr, Polars, SciPy

For visualization (optional):

- Napari
