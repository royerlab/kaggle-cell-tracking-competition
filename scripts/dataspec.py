"""Central path configuration for CellMot datasets."""

import os
import sys
from pathlib import Path

INTERACTIVE = sys.stderr.isatty()

USERNAME = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

_BASE = Path(__file__).resolve().parent.parent
DATASET_PATH = _BASE / "data/dense_channel"
PREDICTIONS_PATH = _BASE / "predictions"
RESULTS_PATH = _BASE / "results"
WEIGHTS_PATH = _BASE / "weights"