"""Dry Bean Classifier Bench -- a Streamlit front-end for five saved models.

The app is strictly an *inference* surface: it loads the pipelines persisted by
train_models.py and scores whatever test CSV the user supplies. No estimator is
ever re-fitted here, so the numbers on screen are an honest read of how the
shipped models behave on unseen rows.
"""

from __future__ import annotations

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

from utils import (
    FEATURE_COLUMNS,
    METRIC_ORDER,
    MODEL_DIR,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    TARGET_COLUMN,
    confusion_frame,
    load_json,
    score_model,
    validate_columns,
)

BEAN_NOTES = {
    "BARBUNYA": "Large mottled cranberry bean",
    "BOMBAY": "Biggest seed in the set -- separates almost perfectly on area",
    "CALI": "Elongated white kidney type",
    "DERMASON": "Most frequent variety (~26% of rows)",
    "HOROZ": "Slender, high aspect ratio",
    "SEKER": "Round and compact",
    "SIRA": "Hardest class -- overlaps Dermason on most shape factors",
}

st.set_page_config(
    page_title="Dry Bean Classifier Bench",
    page_icon="\N{BEANS}",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.1rem; max-width: 1180px;}
      .bean-hero {
        border-radius: 14px; padding: 1.35rem 1.6rem; margin-bottom: 1.4rem;
        background: linear-gradient(110deg, #5b4636 0%, #96704a 55%, #c9a227 100%);
        color: #fdf8ef;
      }
      .bean-hero h1 {margin: 0 0 .35rem 0; font-size: 1.72rem; letter-spacing: -.02em;}
      .bean-hero p {margin: 0; opacity: .93; font-size: .94rem;}
      .bean-chip {
        display: inline-block; margin: .35rem .4rem 0 0; padding: .17rem .62rem;
        border-radius: 999px; background: rgba(253,248,239,.17);
        font-size: .74rem; letter-spacing: .03em;
      }
      div[data-testid="stMetric"] {
        background: #faf7f2; border: 1px solid #e6ddcf;
        border-radius: 11px; padding: .7rem .85rem;
      }
      div[data-testid="stMetricLabel"] p {
        font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; color: #7a6a55;
      }
      .verdict {
        border-left: 4px solid #c9a227; background: #fcf9f1;
        padding: .8rem 1rem; border-radius: 0 9px 9px 0; font-size: .93rem;
      }
      [data-testid="stSidebar"] {background: #f7f3ec;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artefacts():
    """Load the label encoder and all five pipelines once per session."""
    bundle = joblib.load(MODEL_DIR / "target_encoder.joblib")
    models = {
        name: joblib.load(MODEL_DIR / fname)
        for name, fname in MODEL_REGISTRY.items()
        if (MODEL_DIR / fname).exists()
    }
    return bundle["encoder"], models


@st.cache_data(show_spinner=False)
def load_training_metrics():
    path = MODEL_DIR / "metrics.json"
    return load_json(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_uploaded_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


@st.cache_data(show_spinner=False)
def read_bundled_test_csv():
    path = PROJECT_ROOT / "test_data.csv"
    return pd.read_csv(path) if path.exists() else None


try:
    encoder, MODELS = load_artefacts()
except FileNotFoundError:
    st.error(
        "Model artefacts are missing. Run `python train_models.py` to generate "
        "the `model/` folder, then reload this page."
    )
    st.stop()

CLASS_NAMES = list(encoder.classes_)
N_CLASSES = len(CLASS_NAMES)
train_meta = load_training_metrics()

st.markdown(
    f"""
    <div class="bean-hero">
      <h1>Dry Bean Classifier Bench</h1>
      <p>Five saved scikit-learn pipelines scored side by side on your own test CSV.
         Inference only &mdash; nothing is retrained in the browser.</p>
      <div>
        <span class="bean-chip">UCI Dry Bean Dataset</span>
        <span class="bean-chip">{len(FEATURE_COLUMNS)} features</span>
        <span class="bean-chip">{N_CLASSES} classes</span>
        <span class="bean-chip">{len(MODELS)} models loaded</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar: data source, model choice, averaging
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("1 - Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type="csv",
        key="upload_csv",
        help=f"Needs the {len(FEATURE_COLUMNS)} bean measurement columns plus "
             f"the '{TARGET_COLUMN}' target column.",
    )
    # An uploaded file takes precedence automatically. The chooser only appears
    # once something is uploaded, so a stored widget value can never silently
    # shadow a freshly uploaded CSV.
    if uploaded is not None:
        source_mode = st.radio(
            "Score which data?",
            ["Uploaded CSV", "Repository test_data.csv"],
            index=0,
            key="source_mode",
            help="The repository copy is the 20% stratified hold-out split "
                 "produced by train_models.py -- handy as a reference.",
        )
    else:
        source_mode = "Repository test_data.csv"
        st.caption("No upload yet - scoring the repository hold-out split.")

    st.subheader("2 - Model")
    model_choice = st.selectbox(
        "Classifier to inspect",
        list(MODELS.keys()),
        index=len(MODELS) - 1,
        key="model_choice",
    )

    st.subheader("3 - Metric options")
    average = st.radio(
        "Averaging for Precision / Recall / F1",
        ["macro", "weighted"],
        horizontal=True,
        key="average",
        help="Macro treats all seven varieties equally; weighted follows class "
             "frequency. The README table reports macro.",
    )
    row_cap = st.slider(
        "Max rows to score", 200, 5000, 3000, step=200, key="row_cap",
        help="Keeps the free Streamlit tier responsive on large uploads.",
    )

    bundled = read_bundled_test_csv()
    if bundled is not None:
        st.download_button(
            "Download sample test_data.csv",
            data=bundled.to_csv(index=False).encode("utf-8"),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Resolve the active dataframe
# --------------------------------------------------------------------------- #
source_label = None
raw_df = None

if uploaded is not None and source_mode == "Uploaded CSV":
    try:
        raw_df = read_uploaded_csv(uploaded.getvalue())
        source_label = f"uploaded file - {uploaded.name}"
    except Exception as exc:  # malformed CSV, wrong delimiter, ...
        st.error(f"Could not parse that CSV: {exc}")
        st.stop()
elif bundled is not None:
    raw_df = bundled
    source_label = "repository hold-out split - test_data.csv"

if raw_df is None:
    st.info(
        "Upload a test CSV in the sidebar to begin. It needs the "
        f"{len(FEATURE_COLUMNS)} bean measurement columns plus `{TARGET_COLUMN}`."
    )
    st.stop()

missing, has_target = validate_columns(raw_df)
if missing:
    st.error(
        "The CSV is missing required feature column(s): "
        + ", ".join(f"`{c}`" for c in missing)
    )
    st.caption("Expected feature order: " + ", ".join(FEATURE_COLUMNS))
    st.stop()
if not has_target:
    st.error(
        f"The CSV has every feature but no `{TARGET_COLUMN}` column. "
        "Evaluation metrics need ground-truth labels."
    )
    st.stop()

work = raw_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
dropped_na = len(raw_df) - len(work)

unknown = sorted(set(work[TARGET_COLUMN].astype(str)) - set(CLASS_NAMES))
if unknown:
    st.warning(
        "Dropping rows whose label is not one of the seven trained varieties: "
        + ", ".join(unknown)
    )
    work = work[work[TARGET_COLUMN].astype(str).isin(CLASS_NAMES)]

if work.empty:
    st.error("No scorable rows remain after cleaning.")
    st.stop()

truncated = len(work) > row_cap
if truncated:
    work = work.sample(n=row_cap, random_state=0).sort_index()

X = work[FEATURE_COLUMNS].astype(float)
y_true = encoder.transform(work[TARGET_COLUMN].astype(str))

note = f"**{len(X):,} rows** scored from {source_label}"
if dropped_na:
    note += f" - {dropped_na} row(s) dropped for missing values"
if truncated:
    note += f" - sampled down to the {row_cap:,}-row cap"
st.caption(note)


# --------------------------------------------------------------------------- #
# Scoring. Cached on a fingerprint of the active data so that flipping between
# tabs or models does not re-score all five pipelines.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Scoring models ...")
def score_all(fingerprint: str, frame: pd.DataFrame, labels: np.ndarray,
              avg: str) -> pd.DataFrame:
    rows = {name: score_model(model, frame, labels, N_CLASSES, average=avg)
            for name, model in MODELS.items()}
    return pd.DataFrame(rows).T[METRIC_ORDER]


fingerprint = (
    f"{pd.util.hash_pandas_object(X, index=True).sum()}"
    f"-{int(y_true.sum())}-{len(X)}-{average}"
)
results = score_all(fingerprint, X, y_true, average)

tab_model, tab_compare, tab_data, tab_about = st.tabs(
    ["Selected model", "Compare all five", "Data & classes", "About"]
)


# --------------------------------------------------------------------------- #
# Tab 1 -- one model in depth
# --------------------------------------------------------------------------- #
with tab_model:
    model = MODELS[model_choice]
    scores = results.loc[model_choice]
    y_pred = model.predict(X)

    st.markdown(f"#### {model_choice}")
    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_ORDER):
        value = float(scores[metric])
        delta = None
        if train_meta and model_choice in train_meta["metrics"] and average == "macro":
            ref = train_meta["metrics"][model_choice].get(metric)
            # Only worth showing when it would read as a real difference. Scoring
            # the bundled split reproduces the training-time figure exactly, and
            # six cards all reading "+0.0000" is noise, not information.
            if ref is not None and not np.isnan(value) and abs(value - ref) >= 5e-5:
                delta = f"{value - ref:+.4f} vs hold-out"
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}", delta,
                   delta_color="off")

    st.divider()
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_frame(y_true, y_pred, CLASS_NAMES)
        normalise = st.checkbox("Show row-normalised (recall) view", value=False,
                                key="normalise_cm")

        plot_df = cm.copy()
        fmt = "d"
        if normalise:
            denom = plot_df.to_numpy().sum(axis=1, keepdims=True).astype(float)
            denom[denom == 0] = 1.0
            plot_df = pd.DataFrame(plot_df.to_numpy() / denom,
                                   index=cm.index, columns=cm.columns)
            fmt = ".2f"

        fig, ax = plt.subplots(figsize=(6.4, 5.1))
        sns.heatmap(plot_df, annot=True, fmt=fmt, cmap="YlOrBr", cbar=False,
                    linewidths=.5, linecolor="#fdfbf7", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:
        st.markdown("##### Classification report")
        report = classification_report(
            y_true, y_pred, labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        per_class = pd.DataFrame(report).T.loc[CLASS_NAMES]
        per_class["support"] = per_class["support"].astype(int)
        st.dataframe(
            per_class.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}",
                 "f1-score": "{:.3f}", "support": "{:d}"}
            ).background_gradient(cmap="YlOrBr", subset=["f1-score"], vmin=0.5, vmax=1.0),
            width="stretch",
        )

        worst = per_class["f1-score"].idxmin()
        best = per_class["f1-score"].idxmax()
        st.markdown(
            f"<div class='verdict'><b>{model_choice}</b> handles <b>{best}</b> best "
            f"(F1 {per_class.loc[best, 'f1-score']:.3f}) and struggles most with "
            f"<b>{worst}</b> (F1 {per_class.loc[worst, 'f1-score']:.3f}).</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Row-level predictions"):
        detail = work[[TARGET_COLUMN]].copy()
        detail.insert(0, "predicted", encoder.inverse_transform(y_pred))
        detail["confidence"] = model.predict_proba(X).max(axis=1).round(3)
        detail["correct"] = detail["predicted"] == detail[TARGET_COLUMN].astype(str)
        only_errors = st.checkbox("Show misclassifications only", value=False,
                                  key="errors_only")
        view = detail[~detail["correct"]] if only_errors else detail
        st.dataframe(view, width="stretch", height=300)
        st.download_button(
            "Download predictions as CSV",
            data=detail.to_csv(index=True).encode("utf-8"),
            file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Tab 2 -- all five models
# --------------------------------------------------------------------------- #
with tab_compare:
    st.markdown("#### Evaluation metrics on the active test data")
    st.dataframe(
        results.style.format("{:.4f}").highlight_max(axis=0, color="#f6e7b4"),
        width="stretch",
    )

    champion = results["F1"].idxmax()
    st.markdown(
        f"<div class='verdict'>Best {average} F1 on these {len(X):,} rows: "
        f"<b>{champion}</b> ({results.loc[champion, 'F1']:.4f}), "
        f"accuracy {results.loc[champion, 'Accuracy']:.4f}, "
        f"MCC {results.loc[champion, 'MCC']:.4f}.</div>",
        unsafe_allow_html=True,
    )

    metric_to_plot = st.selectbox("Chart a metric", METRIC_ORDER, index=0,
                                  key="metric_to_plot")
    chart_df = results[[metric_to_plot]].sort_values(metric_to_plot)
    top = chart_df[metric_to_plot].idxmax()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    colours = ["#c9a227" if name == top else "#a98f6f" for name in chart_df.index]
    ax.barh(chart_df.index, chart_df[metric_to_plot], color=colours, height=.62)
    for name, value in chart_df[metric_to_plot].items():
        ax.text(value + .004, name, f"{value:.3f}", va="center", fontsize=9)
    lo = max(0.0, float(np.nanmin(chart_df[metric_to_plot])) - 0.06)
    ax.set_xlim(lo, 1.015)
    ax.set_xlabel(metric_to_plot)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.download_button(
        "Download this comparison table",
        data=results.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    if train_meta:
        with st.expander("Reference: metrics recorded at training time"):
            ref_table = pd.DataFrame(train_meta["metrics"]).T[METRIC_ORDER]
            st.caption(
                f"Macro-averaged on the {train_meta['test_rows']:,}-row stratified "
                f"hold-out split (random_state={train_meta['random_state']})."
            )
            st.dataframe(ref_table.style.format("{:.4f}"), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 3 -- the data itself
# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("#### Active test data")
    st.dataframe(work.head(200), width="stretch", height=280)

    counts = work[TARGET_COLUMN].astype(str).value_counts().reindex(CLASS_NAMES).fillna(0)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("##### Class distribution")
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.bar(counts.index, counts.to_numpy(), color="#96704a")
        ax.set_ylabel("rows")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with c2:
        st.markdown("##### The seven varieties")
        st.dataframe(
            pd.DataFrame(
                {"variety": CLASS_NAMES,
                 "rows here": [int(counts.get(c, 0)) for c in CLASS_NAMES],
                 "note": [BEAN_NOTES.get(c, "") for c in CLASS_NAMES]}
            ).set_index("variety"),
            width="stretch",
        )

    with st.expander("Feature summary statistics"):
        st.dataframe(X.describe().T.style.format("{:.3f}"), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 4 -- provenance
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        f"""
#### What this app does

1. Loads five pre-fitted `sklearn.pipeline.Pipeline` objects from `model/`
   (`StandardScaler` then classifier), together with the `LabelEncoder` used
   for the target at training time.
2. Validates that the supplied CSV carries all {len(FEATURE_COLUMNS)} bean
   measurements plus the `{TARGET_COLUMN}` column.
3. Computes Accuracy, one-vs-rest macro AUC, Precision, Recall, F1 and MCC,
   then renders a confusion matrix and per-class classification report.

**No model is refitted in the app.** `train_models.py` is the only place that
calls `.fit()`; this page only calls `.predict()` and `.predict_proba()`.

#### Dataset

Koklu, M. and Ozkan, I. A. (2020), *Multiclass classification of dry beans
using computer vision and machine learning techniques*, **Computers and
Electronics in Agriculture** 174, 105507. Distributed by the UCI Machine
Learning Repository (dataset id 602): 13,611 grain images reduced to 16 shape
and dimension features across 7 registered varieties.

#### Required CSV columns

```
{", ".join(FEATURE_COLUMNS)}, {TARGET_COLUMN}
```
"""
    )
    if train_meta:
        st.caption(
            f"Artefacts trained on {train_meta['train_rows']:,} rows - "
            f"best model by macro F1 at training time: {train_meta['best_model_by_f1']}."
        )
