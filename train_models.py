"""Train and persist five classifiers on the UCI Dry Bean Dataset.

Run once (offline) to produce everything the Streamlit app consumes:

    python train_models.py

Outputs
-------
data/dry_bean.csv        full dataset snapshot (so the repo is reproducible)
test_data.csv            held-out 20% stratified split, features + Class
model/<name>.joblib      one fitted Pipeline per classifier
model/target_encoder.joblib
model/metrics.json       metrics on the held-out split, for the README table

Nothing in app.py ever re-fits a model; it only loads these artefacts.
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (
    DATA_DIR,
    FEATURE_COLUMNS,
    METRIC_ORDER,
    MODEL_DIR,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    TARGET_COLUMN,
    dump_json,
    score_model,
)

RANDOM_STATE = 17
TEST_SIZE = 0.20
UCI_DRY_BEAN_ID = 602


def build_models() -> dict:
    """The five required classifiers, each wrapped in a scaling pipeline.

    Every estimator sits behind StandardScaler so that a single uploaded CSV of
    raw bean measurements can be fed to any of them through one code path. The
    tree is scale-invariant and is only standardised for that uniformity.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                criterion="entropy", max_depth=14, min_samples_leaf=6,
                min_samples_split=12, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=11, weights="distance", metric="minkowski", p=2)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-8)),
        ]),
        # SVC has no native probabilities. Rather than SVC(probability=True)
        # -- deprecated in scikit-learn 1.9 -- the decision function is
        # Platt-scaled explicitly, which is what the AUC column needs.
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE),
                method="sigmoid", cv=5, ensemble=False)),
        ]),
    }


def load_dataset(refresh: bool) -> pd.DataFrame:
    """Read the cached CSV snapshot, downloading from UCI the first time."""
    snapshot = DATA_DIR / "dry_bean.csv"
    if snapshot.exists() and not refresh:
        print(f"[data] reading cached snapshot {snapshot}")
        return pd.read_csv(snapshot)

    print("[data] downloading Dry Bean Dataset from UCI ...")
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=UCI_DRY_BEAN_ID)
    df = repo.data.features.copy()
    df[TARGET_COLUMN] = repo.data.targets.iloc[:, 0].values

    # Guard against upstream column renames (the raw xlsx ships "AspectRation").
    df = df.rename(columns={"AspectRation": "AspectRatio"})
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot, index=False)
    print(f"[data] cached to {snapshot}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download the dataset from UCI instead of using the cache")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.refresh)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] {df.shape[0]} instances x {len(FEATURE_COLUMNS)} features "
          f"after dropping exact duplicates")
    print(f"[data] class balance:\n{df[TARGET_COLUMN].value_counts().to_string()}")

    X = df[FEATURE_COLUMNS]
    encoder = LabelEncoder().fit(df[TARGET_COLUMN])
    y = encoder.transform(df[TARGET_COLUMN])
    n_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[split] train={X_train.shape[0]}  test={X_test.shape[0]}")

    # The exact held-out split is shipped as test_data.csv, so the marks the app
    # shows are reproducible from the repo alone.
    test_csv = X_test.copy()
    test_csv[TARGET_COLUMN] = encoder.inverse_transform(y_test)
    test_csv.to_csv(PROJECT_ROOT / "test_data.csv", index=False)
    print(f"[split] wrote test_data.csv ({test_csv.shape[0]} rows)")

    joblib.dump(
        {"encoder": encoder, "feature_columns": FEATURE_COLUMNS},
        MODEL_DIR / "target_encoder.joblib",
    )

    metrics = {}
    for name, pipeline in build_models().items():
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        scores = score_model(pipeline, X_test, y_test, n_classes, average="macro")
        scores["fit_seconds"] = round(fit_seconds, 2)
        metrics[name] = scores

        joblib.dump(pipeline, MODEL_DIR / MODEL_REGISTRY[name])
        row = "  ".join(f"{m}={scores[m]:.4f}" for m in METRIC_ORDER)
        print(f"[fit] {name:<24} {row}  ({fit_seconds:.1f}s)")

    best = max(metrics, key=lambda k: metrics[k]["F1"])
    print(f"\n[result] highest macro-F1: {best}")

    print("\n[report] per-class detail for the winning model")
    winner = joblib.load(MODEL_DIR / MODEL_REGISTRY[best])
    print(classification_report(
        y_test, winner.predict(X_test), target_names=encoder.classes_, digits=4))

    dump_json(
        {
            "dataset": "UCI Dry Bean Dataset (id=602)",
            "n_instances": int(df.shape[0]),
            "n_features": len(FEATURE_COLUMNS),
            "classes": encoder.classes_.tolist(),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "average": "macro",
            "random_state": RANDOM_STATE,
            "best_model_by_f1": best,
            "metrics": metrics,
        },
        MODEL_DIR / "metrics.json",
    )
    print(f"[done] artefacts in {MODEL_DIR}")


if __name__ == "__main__":
    main()
