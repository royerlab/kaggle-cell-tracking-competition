"""Central path configuration for CellMot datasets."""

import os
import sys
from pathlib import Path

INTERACTIVE = sys.stderr.isatty()

USERNAME = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

_BASE = Path(__file__).resolve().parent.parent

# Kaggle competition mount (present when running in a Kaggle notebook).
_KAGGLE_TRAIN = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
)


def _default_dataset_path() -> Path:
    """Resolve the dataset directory.

    Priority:
      1. ``$CELLMOT_DATA_DIR`` — explicit override (any environment).
      2. the Kaggle competition ``train`` mount, when running on Kaggle.
      3. local ``./data/dense_channel`` — default for local development.

    Scripts also accept ``--data-dir`` to override this per run (e.g. to point
    at the competition ``test`` directory for prediction).
    """
    env = os.environ.get("CELLMOT_DATA_DIR")
    if env:
        return Path(env)
    if _KAGGLE_TRAIN.exists():
        return _KAGGLE_TRAIN
    return _BASE / "data/dense_channel"


DATASET_PATH = _default_dataset_path()
PREDICTIONS_PATH = _BASE / "predictions"
RESULTS_PATH = _BASE / "results"
WEIGHTS_PATH = _BASE / "weights"