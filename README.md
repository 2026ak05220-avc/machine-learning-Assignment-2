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

* **Input:** 16 continuous features — 4 raw dimension measurements (area, perimeter,
  axis lengths), 6 derived shape descriptors (aspect ratio, eccentricity, extent,
  solidity, roundness, compactness) and 4 engineered shape factors.
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

Supporting diagnostics referenced in the observations below:

| Model | Train acc. | Test acc. | Gap | 5-fold CV macro-F1 (train) | Fit time |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9262 | 0.9228 | +0.0033 | 0.9357 ± 0.0062 | 0.21 s |
| Decision Tree | 0.9555 | 0.9062 | +0.0493 | 0.9163 ± 0.0056 | 0.21 s |
| KNN | 1.0000 | 0.9276 | +0.0724 | 0.9376 ± 0.0055 | 0.01 s |
| Naive Bayes | 0.8967 | 0.8944 | +0.0023 | 0.9057 ± 0.0080 | 0.01 s |
| Support Vector Machine | 0.9382 | 0.9332 | +0.0050 | 0.9420 ± 0.0042 | 2.23 s |

Per-class F1 on the hold-out split — this is where the models actually differ:

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
| **Logistic Regression** | Far stronger than a linear model has any right to be here (0.9228 accuracy, within 1 point of the RBF SVM) — the 16 engineered shape factors are already close to linearly separable after standardisation. It posts the **best AUC of all five (0.9946)**, meaning its *ranking* of class probabilities is the most reliable even though its hard `argmax` decisions are not the most accurate; with a tuned per-class threshold it would close most of the gap to the SVM. Train/test gap of just +0.0033 shows essentially zero overfitting, and the 5-fold CV spread (±0.0062) is tight. Its weakness is structural: a single hyperplane per class cannot bend around the curved `SIRA`/`DERMASON` boundary, so `SIRA` F1 drops to 0.862. Trains in 0.21 s. |
| **Decision Tree** | The **weakest model on five of the six metrics** (accuracy 0.9062, AUC 0.9733, MCC 0.8867) and the clearest case of variance, not bias: it fits the training split to 0.9555 but generalises to 0.9062, a +0.0493 gap even after capping depth at 14 and requiring 6 samples per leaf. Its AUC is dramatically the worst (0.9733 vs ~0.99 for everyone else) because a tree's probabilities are coarse leaf-frequency estimates — many test rows land on an identical confident value, which flattens the ROC curve and leaves no useful ranking to threshold. It degrades most on the minority `BARBUNYA` class (F1 0.899), where axis-aligned splits run short of data. Its redeeming quality is interpretability: the fitted tree can be printed as readable `if ShapeFactor1 < …` rules for a grading line, which none of the other four offer. |
| **KNN** | The best non-kernel model and the **runner-up overall** (accuracy 0.9276, F1 0.9390, MCC 0.9125), and notably the best of the five at the hardest class (`SIRA` F1 0.881) — local neighbourhoods adapt to the irregular `SIRA`/`DERMASON` boundary that the linear model cannot curve around. Training accuracy is exactly **1.0000**, which is an artefact of `weights='distance'` rather than genuine overfitting (each training point is its own nearest neighbour at distance 0); the honest reading is the +0.0724 train/test gap, the largest here. Fitting is instant (0.01 s) because it only stores the data, but it pays at inference: the 1.5 MB `knn.joblib` carries all 10,834 training rows and every prediction is a distance scan, making it the slowest and heaviest of the five to *serve* — the opposite trade-off to the SVM. |
| **Naive Bayes** | The **weakest overall (accuracy 0.8944, F1 0.9036, MCC 0.8726)**, and the reason is visible in the data: its conditional-independence assumption is badly violated, since `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` are near-perfectly collinear measures of the same physical size. Treating them as independent evidence multiplies the same signal four times and over-sharpens the posteriors, which is why it collapses specifically on `BARBUNYA` (F1 0.830, a full 0.117 below the SVM) where it confuses the mottled bean with `CALI`. Two things still recommend it: a **0.9911 AUC that beats both tree-based and kernel ranking**, so the class ordering is sound even when the `argmax` is wrong, and a +0.0023 train/test gap — the best-calibrated-to-its-own-capacity model here. It fits in 0.01 s from closed-form statistics with no iteration at all, making it the right pick when training budget is the binding constraint. |
| **Support Vector Machine** | **The winner on accuracy, Precision, Recall, F1 and MCC** (0.9332 / 0.9451 / 0.9424 / 0.9436 / 0.9192). The RBF kernel is what the dataset needs: it bends a non-linear boundary through the overlapping `SIRA`/`DERMASON` region, taking `SIRA` F1 to 0.888 — the best of any model on the hardest class — and it leads or ties on 6 of the 7 varieties. Generalisation is excellent (+0.0050 train/test gap) and it has the **tightest CV spread of all five (±0.0042)**, so the win is stable rather than a lucky split. Two honest caveats: its AUC (0.9904) is *below* Logistic Regression's and Naive Bayes's because SVC has no native probabilities and the Platt sigmoid fitted over 5 folds only approximates them — so for ranking or threshold tuning the logistic model is the better instrument. And it is the most expensive to train (2.23 s, ~10× the others, and that gap widens quadratically with more data). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF).** It takes 5 of the 6 required metrics, wins the hardest class (`SIRA`, F1 0.888), has the most stable cross-validation of the five (±0.0042) and shows almost no overfitting (+0.0050). Its only loss is AUC, where Logistic Regression's 0.9946 reflects better-calibrated probability *ranking* — so the practical recommendation is SVM when a hard variety label is what the sorting line consumes, and Logistic Regression if downstream logic needs trustworthy probabilities or a tunable threshold (at 1/10th the training cost and 1/150th the model size). The spread across all five models is only ~3.9 accuracy points (0.8944 → 0.9332), which says the 16 engineered shape features — not the choice of classifier — are doing most of the work on this dataset. |

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
