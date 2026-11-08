# Dry Bean Variety Classification — ML Assignment 2

Five classical classifiers trained on the UCI Dry Bean Dataset, served through an
interactive Streamlit app that scores an uploaded test CSV against all five saved
models without retraining any of them.

---

## a. Problem statement

Dry bean seeds of different registered varieties are visually similar but command
different market prices, so sacks are routinely adulterated or mislabelled. Manual
grading by inspectors is slow, subjective and inconsistent between graders.

The task is therefore a **supervised multi-class classification** problem: given 16
geometric measurements extracted automatically from a high-resolution photograph of
a single bean, predict which of **7 registered varieties** the seed belongs to.

* **Input:** 16 continuous features — 6 raw dimension measurements (`Area`,
  `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `ConvexArea`, `EquivDiameter`),
  6 derived shape descriptors (`AspectRatio`, `Eccentricity`, `Extent`, `Solidity`,
  `Roundness`, `Compactness`) and 4 engineered shape factors (`ShapeFactor1`–`4`).
* **Output:** one of `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`.
* **Why it is non-trivial:** the classes are imbalanced (26.2% `DERMASON` vs 3.9%
  `BOMBAY`) and two varieties — `SIRA` and `DERMASON` — overlap heavily on almost
  every shape factor, which is where every model below loses most of its accuracy.
* **Business framing:** a single model that reaches >93% accuracy on unseen seeds can
  replace a manual grading step on the sorting line, so the metric that matters is
  balanced performance across *all* varieties, not just the frequent ones. Macro
  averaging and MCC are reported for exactly that reason.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository, dataset id **602** — <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> |
| Original study | Koklu, M. & Ozkan, I. A. (2020), *Multiclass classification of dry beans using computer vision and machine learning techniques*, **Computers and Electronics in Agriculture** 174, 105507 |
| Instances (raw) | **13,611** — well above the 500 minimum |
| Instances (used) | **13,543** after removing 68 exact duplicate rows |
| Features | **16** numeric, all continuous — above the 12 minimum |
| Target | `Class` — 7 bean varieties (multi-class) |
| Missing values | none |
| Acquisition | 13,611 grains photographed with a high-resolution camera; a computer-vision pipeline segmented each grain and extracted the 16 descriptors |

**Feature list**

| # | Feature | Meaning |
|---|---|---|
| 1 | `Area` | Pixel count inside the bean boundary |
| 2 | `Perimeter` | Boundary length |
| 3 | `MajorAxisLength` | Longest line through the bean |
| 4 | `MinorAxisLength` | Longest line perpendicular to the major axis |
| 5 | `AspectRatio` | MajorAxisLength / MinorAxisLength |
| 6 | `Eccentricity` | Eccentricity of the equivalent ellipse |
| 7 | `ConvexArea` | Area of the smallest enclosing convex polygon |
| 8 | `EquivDiameter` | Diameter of a circle of the same area |
| 9 | `Extent` | Area / bounding-box area |
| 10 | `Solidity` | Area / ConvexArea (surface roughness) |
| 11 | `Roundness` | 4·π·Area / Perimeter² |
| 12 | `Compactness` | EquivDiameter / MajorAxisLength |
| 13 | `ShapeFactor1` | MajorAxisLength / Area |
| 14 | `ShapeFactor2` | Engineered dimensionless shape factor |
| 15 | `ShapeFactor3` | Engineered dimensionless shape factor |
| 16 | `ShapeFactor4` | Engineered dimensionless shape factor |

**Class distribution (13,543 de-duplicated rows)**

| Variety | Rows | Share |
|---|---:|---:|
| DERMASON | 3,546 | 26.2% |
| SIRA | 2,636 | 19.5% |
| SEKER | 2,027 | 15.0% |
| HOROZ | 1,860 | 13.7% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.8% |
| BOMBAY | 522 | 3.9% |

**Preprocessing** — identical for every model, so the comparison is fair:

1. Drop exact duplicate rows (13,611 → 13,543).
2. Stratified 80/20 train/test split, `random_state=17` → **10,834 train / 2,709 test**.
   The test half is committed as `test_data.csv`.
3. `LabelEncoder` on the target, persisted to `model/target_encoder.joblib`.
4. `StandardScaler` **fitted on the training split only**, carried inside each
   `Pipeline` so no test statistics ever leak into training.

## c. GitHub repository link

> <https://github.com/2026ak05220-avc/machine-learning-Assignment-2>

Repository contents:

```
project-folder/
├── app.py                      # Streamlit application (inference only)
├── train_models.py             # trains + persists all five models
├── utils.py                    # shared metric / validation helpers
├── requirements.txt            # pinned dependencies
├── README.md                   # this file
├── test_data.csv               # 2,709-row stratified hold-out split
├── .streamlit/config.toml      # app theme
├── data/
│   └── dry_bean.csv            # full dataset snapshot (13,543 rows)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── target_encoder.joblib   # LabelEncoder + feature order
    └── metrics.json            # metrics recorded at training time
```

## d. Models used

All five classifiers were trained on the **same** 10,834-row training split and
evaluated on the **same** 2,709-row hold-out split. Each is wrapped in an
`sklearn.pipeline.Pipeline` with `StandardScaler` so a single raw CSV can be routed
through any of them identically.

| Model | Configuration |
|---|---|
| Logistic Regression | multinomial, `lbfgs`, `C=1.0`, `max_iter=3000` |
| Decision Tree | `criterion='entropy'`, `max_depth=14`, `min_samples_leaf=6`, `min_samples_split=12` |
| KNN | `n_neighbors=11`, `weights='distance'`, Euclidean (`p=2`) |
| Naive Bayes | `GaussianNB`, `var_smoothing=1e-8` |
| Support Vector Machine | RBF kernel, `C=10`, `gamma='scale'`, Platt-scaled via `CalibratedClassifierCV(cv=5)` for probabilities |

### Comparison table

Metrics on the 2,709-row hold-out split. Precision, Recall and F1 are
**macro-averaged** (every variety weighted equally, which is the honest choice given
the 26% / 4% class imbalance). AUC is one-vs-rest macro ROC-AUC from
`predict_proba`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9228 | **0.9946** | 0.9359 | 0.9349 | 0.9353 | 0.9067 |
| Decision Tree | 0.9062 | 0.9733 | 0.9204 | 0.9193 | 0.9194 | 0.8867 |
| KNN | 0.9276 | 0.9901 | 0.9407 | 0.9375 | 0.9390 | 0.9125 |
| Naive Bayes | 0.8944 | 0.9911 | 0.9021 | 0.9054 | 0.9036 | 0.8726 |
| **Support Vector Machine** | **0.9332** | 0.9904 | **0.9451** | **0.9424** | **0.9436** | **0.9192** |

Training cost, as printed by `train_models.py` and stored in `model/metrics.json`:

| Model | Fit time |
|---|---:|
| Logistic Regression | 0.21 s |
| Decision Tree | 0.21 s |
| KNN | 0.01 s |
| Naive Bayes | 0.01 s |
| Support Vector Machine | 2.23 s |

Per-class F1 on the hold-out split — this is where the models actually differ.
Each column can be reproduced in the live app by selecting that model and reading
the *Classification report* panel:

| Variety | Log. Reg. | Dec. Tree | KNN | Naive Bayes | SVM |
|---|---:|---:|---:|---:|---:|
| BARBUNYA | 0.939 | 0.899 | 0.938 | 0.830 | 0.947 |
| BOMBAY | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CALI | 0.932 | 0.916 | 0.938 | 0.871 | 0.935 |
| DERMASON | 0.918 | 0.900 | 0.921 | 0.902 | 0.927 |
| HOROZ | 0.948 | 0.944 | 0.949 | 0.939 | 0.952 |
| SEKER | 0.947 | 0.927 | 0.946 | 0.938 | 0.956 |
| **SIRA** | **0.862** | **0.850** | **0.881** | **0.845** | **0.888** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Accuracy 0.9228, about one point below the RBF SVM, which is high for a linear model. This suggests the 16 shape features already carry enough discriminatory information for a linear decision rule to work well on this data. It records the best AUC of the five (0.9946), so its ranking of class probabilities is the most reliable even though its final `argmax` predictions are not the most accurate. A high AUC alongside lower accuracy points to the decision thresholds rather than the learned ordering being the limiting factor, though I did not run a threshold-tuning experiment to confirm this. Its weakest class is `SIRA` (F1 0.862), the one that overlaps most with `DERMASON`. Training takes 0.21 s. |
| **Decision Tree** | Weakest on five of the six metrics (accuracy 0.9062, AUC 0.9733, MCC 0.8867), even with depth capped at 14 and a minimum of 6 samples per leaf to control overfitting. Its AUC is clearly the lowest (0.9733 against roughly 0.99 for the others), which is expected: a tree derives probabilities from leaf frequencies, so many test rows share the same few probability values and the ROC curve has little to separate. Its worst class is `BARBUNYA` (F1 0.899), one of the smaller classes, where axis-aligned splits have less data to work with. Its advantage is interpretability — the fitted tree can be printed as readable if/else rules on the shape factors, which none of the other four models offer. |
| **KNN** | Second best overall (accuracy 0.9276, F1 0.9390, MCC 0.9125) and the best of the non-kernel models. It handles the hardest class better than the linear and tree models (`SIRA` F1 0.881), which is consistent with local neighbourhoods being able to follow an irregular boundary. One caveat on interpretation: with `weights='distance'` the training score is not meaningful, because every training point is its own nearest neighbour at zero distance, so only the held-out numbers are informative. Training is effectively free (0.01 s) since the model only stores the data, but prediction is costly — `knn.joblib` is 1.5 MB and holds all 10,834 training rows, and each prediction requires distance computations against them. |
| **Naive Bayes** | Weakest overall (accuracy 0.8944, F1 0.9036, MCC 0.8726). The likely reason is that its conditional independence assumption does not hold on this dataset: `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` all measure size and are strongly correlated (pairwise Pearson r between 0.967 and 0.9999, measured on the de-duplicated data). Treating correlated features as independent evidence can over-sharpen the posteriors. Its worst class is `BARBUNYA` (F1 0.830, 0.117 below the SVM), which it most often confuses with `CALI`. Two points in its favour: an AUC of 0.9911, higher than both the tree and the SVM, so its ranking of classes is sound even when the final prediction is wrong; and a 0.01 s fit from closed-form statistics with no iterative optimisation. |
| **Support Vector Machine** | Best on accuracy, precision, recall, F1 and MCC (0.9332 / 0.9451 / 0.9424 / 0.9436 / 0.9192). The RBF kernel allows a non-linear boundary, which is a plausible explanation for it handling the `SIRA`/`DERMASON` overlap better than the others — `SIRA` F1 reaches 0.888, the highest of the five, and it leads or ties on 6 of the 7 varieties. This is an interpretation of the per-class results rather than a decision-boundary analysis. Two caveats. Its AUC (0.9904) is below both Logistic Regression and Naive Bayes, because SVC has no native probability output and the Platt sigmoid fitted over 5 folds only approximates one, so for ranking tasks the logistic model is the better choice. It is also the slowest to train (2.23 s, roughly 10x the others), and kernel SVM training cost grows steeply as the dataset gets larger. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF).** It wins 5 of the 6 required metrics and performs best on the hardest class (`SIRA`, F1 0.888). Its one loss is AUC, where Logistic Regression's 0.9946 reflects better probability ranking. The practical conclusion is to use the SVM when a hard variety label is what the sorting line needs, and Logistic Regression when downstream logic needs probability estimates or a tunable threshold — it also trains roughly 10x faster and its saved model is far smaller. It is worth noting that the spread across all five models is only about 3.9 accuracy points (0.8944 to 0.9332), which suggests the 16 shape features, rather than the choice of classifier, account for most of the performance here. |

## The Streamlit application

> **Live app:** <https://dry-bean-classifier-2026ak05220.streamlit.app/>

The app never calls `.fit()`. It loads the five persisted pipelines and the label
encoder, then only predicts — so the metrics on screen are a true read of saved-model
behaviour on unseen rows.

Required features, and where each lives:

| Requirement | Where it is in the app |
|---|---|
| **Dataset upload option (CSV)** | Sidebar → *1 · Test data* → `Upload a test CSV`. Only test data is uploaded, per the free-tier guidance; a 20 MB cap is set in `.streamlit/config.toml`. The bundled `test_data.csv` is scored by default and can be downloaded from the sidebar as an upload template. |
| **Model selection dropdown** | Sidebar → *2 · Model* → `Classifier to inspect`, listing all five models. |
| **Display of evaluation metrics** | *Selected model* tab shows all six metrics as cards, each with its delta against the training-time hold-out figure. *Compare all five* tab shows the full 5 × 6 table with the best value per column highlighted, plus a sortable bar chart for any chosen metric. |
| **Confusion matrix / classification report** | *Selected model* tab shows a 7 × 7 confusion-matrix heatmap (raw counts, or row-normalised recall view) side by side with a per-class precision/recall/F1/support report, and calls out the model's best and worst class. |

Beyond the requirements: macro vs weighted averaging toggle, row-level predictions
with a model-confidence column and a misclassifications-only filter, CSV export of
both predictions and the comparison table, class-distribution and feature-summary
views, and a row cap to keep the free tier responsive.

**Input contract.** The uploaded CSV must contain all 16 feature columns plus the
`Class` target. Column order does not matter and extra columns are ignored. The app
reports a clear error for a missing feature column, a missing target, or an
unparseable file, and drops rows with missing values or unrecognised labels with a
visible warning.

## Reproducing the results

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

python train_models.py      # retrains and rewrites model/ + test_data.csv
streamlit run app.py        # serves the app at http://localhost:8501
```

`train_models.py` reads `data/dry_bean.csv` if present and otherwise downloads the
dataset from UCI via `ucimlrepo` (`--refresh` forces a re-download). Every random
seed is fixed at 17, so the table above reproduces exactly.

Dependencies are **version-pinned** in `requirements.txt` because `model/*.joblib`
artefacts must be unpickled by the same scikit-learn build that wrote them — a
mismatch is the most common cause of a working repo failing on Streamlit Cloud.

## Notes on metric choices

* **Macro averaging** is the headline because the classes are imbalanced 26% to 4%;
  weighted averaging would let `DERMASON` performance mask `BOMBAY` and `BARBUNYA`.
  The app exposes both.
* **AUC** is one-vs-rest macro over `predict_proba`. If an uploaded CSV happens to
  contain only a subset of the seven varieties, the probability columns for the
  present classes are re-normalised and scored among themselves; with a single class
  present AUC is undefined and the app displays `n/a` rather than a misleading number.
* **MCC** is reported because it is the metric least fooled by class imbalance — it
  only approaches 1.0 when every variety is predicted well.
