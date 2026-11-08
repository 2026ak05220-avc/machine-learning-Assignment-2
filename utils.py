"""Shared helpers for the Dry Bean classifier project.

Imported by both the training script (train_models.py) and the Streamlit
front-end (app.py) so that the metric definitions used to build the README
comparison table are byte-for-byte the same ones the web app reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"

TARGET_COLUMN = "Class"

# Feature order the pipelines were fitted on. Kept here as the single source of
# truth so the app can validate an uploaded CSV before touching a model.
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# Registry: display name -> artefact file name inside model/
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def multiclass_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """One-vs-rest macro AUC that tolerates a subset of classes in y_true.

    A user-uploaded test slice need not contain every bean variety. When a class
    is absent its one-vs-rest curve is undefined, so the remaining probability
    columns are re-normalised and scored among themselves instead of failing.
    """
    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")

    try:
        if present.size == n_classes:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))

        sub = proba[:, present].astype(float)
        row_sums = sub.sum(axis=1, keepdims=True)
        # A row can put all of its mass on classes that are absent here (KNN
        # with distance weights does this). Such a row carries no information
        # about the present classes, so it becomes a uniform distribution --
        # leaving it at zero would break the "rows sum to 1" contract that
        # roc_auc_score enforces for multiclass input.
        empty = (row_sums == 0).ravel()
        sub[empty, :] = 1.0 / present.size
        row_sums[empty] = 1.0
        sub = sub / row_sums
        if present.size == 2:
            return float(roc_auc_score(y_true, sub[:, 1], labels=present))
        return float(
            roc_auc_score(y_true, sub, multi_class="ovr", average="macro", labels=present)
        )
    except ValueError:
        return float("nan")


def score_model(model, X: pd.DataFrame, y_true: np.ndarray, n_classes: int,
                average: str = "macro") -> dict:
    """Return the six assignment metrics for one fitted pipeline.

    Precision/recall/F1 are averaged over the classes that actually occur in
    y_true. On the full hold-out split that is all seven varieties; on a
    partial upload it stops absent classes from contributing a spurious zero
    to the macro mean. Predictions that fall outside the present labels still
    cost the true class its recall, so errors are never hidden.
    """
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    labels = np.unique(y_true)
    kw = {"labels": labels, "average": average, "zero_division": 0}

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": multiclass_auc(y_true, proba, n_classes),
        "Precision": float(precision_score(y_true, y_pred, **kw)),
        "Recall": float(recall_score(y_true, y_pred, **kw)),
        "F1": float(f1_score(y_true, y_pred, **kw)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame over every known class."""
    idx = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    labels = [f"{name}" for name in class_names]
    return pd.DataFrame(cm, index=[f"true: {l}" for l in labels],
                        columns=[f"pred: {l}" for l in labels])


def validate_columns(df: pd.DataFrame) -> tuple[list[str], bool]:
    """Missing feature columns, plus whether the target column is present."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    return missing, TARGET_COLUMN in df.columns


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
