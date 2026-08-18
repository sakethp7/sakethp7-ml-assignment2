"""
===============================================================================
 Term Deposit Radar  --  Streamlit inference & analysis dashboard
 BITS Pilani WILP  |  M.Tech (AIML/DSE)  |  Machine Learning  |  Assignment 2
===============================================================================

 This app performs INFERENCE ONLY. Nothing is trained here -- the five
 scikit-learn Pipelines were fitted offline by `train.py` and are loaded from
 `model/*.joblib`. Keeping training out of the app is what lets it boot in a
 couple of seconds inside Streamlit Community Cloud's small free container.

 Features:
   a. CSV upload of test data              -> sidebar uploader, with a bundled
                                              `test_data.csv` fallback
   b. Model selection dropdown             -> sidebar selectbox over 5 models
   c. Display of evaluation metrics        -> Accuracy / AUC / Precision /
                                              Recall / F1 / MCC
   d. Confusion matrix + classification report

 Run locally:  streamlit run app.py
===============================================================================
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
APP_ROOT = Path(__file__).parent
MODEL_DIR = APP_ROOT / "model"
DEFAULT_TEST_CSV = APP_ROOT / "test_data.csv"
REPORT_DIR = APP_ROOT / "reports"

st.set_page_config(
    page_title="Term Deposit Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the header card and metric styling.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      .td-hero {
          background: linear-gradient(120deg, #123f5e 0%, #2b7a9b 55%, #43a49b 100%);
          padding: 1.5rem 1.8rem; border-radius: 14px; color: #ffffff;
          margin-bottom: 1.4rem;
      }
      .td-hero h1 { margin: 0 0 .35rem 0; font-size: 1.85rem; letter-spacing:.3px; }
      .td-hero p  { margin: 0; opacity: .92; font-size: .95rem; }
      .td-note {
          border-left: 4px solid #2b7a9b; background: rgba(43,122,155,.09);
          padding: .7rem 1rem; border-radius: 0 8px 8px 0; font-size: .9rem;
      }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      section[data-testid="stSidebar"] { min-width: 21rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="td-hero">
      <h1>📡 Term Deposit Radar</h1>
      <p>Comparing five classifiers on the UCI Bank Marketing campaign &mdash;
         which contacted clients actually subscribe to a term deposit?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid")
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# -----------------------------------------------------------------------------
# Loading (cached -- these run once per container, not once per widget click)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metadata() -> dict | None:
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading trained models…")
def load_models(file_map: dict) -> tuple[dict, list]:
    """Deserialise each saved Pipeline. Missing/incompatible files are reported,
    not fatal -- the app still works with whichever models did load."""
    models, problems = {}, []
    for label, filename in file_map.items():
        path = MODEL_DIR / filename
        if not path.exists():
            problems.append(f"`{filename}` is missing from `model/`")
            continue
        try:
            models[label] = joblib.load(path)
        except Exception as exc:
            problems.append(f"`{filename}` failed to load — {type(exc).__name__}: {exc}")
    return models, problems


@st.cache_data(show_spinner=False)
def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Uploaded files may use ';' (the raw UCI format) or ',' (our export)."""
    for sep in (",", ";"):
        try:
            frame = pd.read_csv(io.BytesIO(payload), sep=sep)
        except Exception:
            continue
        if frame.shape[1] > 3:
            return normalise_columns(frame)
    raise ValueError("Could not parse the file as CSV with ',' or ';' separator.")


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the column naming train.py used: lowercase, dots -> underscores."""
    frame = frame.copy()
    frame.columns = [str(c).strip().replace(".", "_").lower() for c in frame.columns]
    return frame


@st.cache_data(show_spinner=False)
def load_default_test() -> pd.DataFrame | None:
    if not DEFAULT_TEST_CSV.exists():
        return None
    return normalise_columns(pd.read_csv(DEFAULT_TEST_CSV))


# -----------------------------------------------------------------------------
# Feature alignment + scoring helpers
# -----------------------------------------------------------------------------
def prepare_features(frame: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, list]:
    """
    Reshape an arbitrary uploaded CSV into exactly the columns the pipelines were
    fitted on. Re-applies the same `pdays` recoding train.py used, drops the leaky
    `duration` column if the user uploaded a raw UCI file, and fills any genuinely
    absent column with NaN so the pipeline's imputer can handle it.
    """
    expected = meta["feature_columns"]
    work = frame.copy()
    notes = []

    if "duration" in work.columns and "duration" in meta.get("dropped_columns", []):
        work = work.drop(columns=["duration"])
        notes.append("Dropped `duration` (post-call leakage, excluded during training).")

    if "was_contacted_before" in expected and "was_contacted_before" not in work.columns:
        if "pdays" in work.columns:
            never = work["pdays"] == 999
            work["was_contacted_before"] = (~never).astype(int)
            work.loc[never, "pdays"] = 0
            notes.append("Rebuilt `was_contacted_before` from the `pdays` 999 sentinel.")

    missing = [c for c in expected if c not in work.columns]
    for col in missing:
        work[col] = np.nan
    if missing:
        notes.append(
            f"{len(missing)} expected column(s) absent, imputed by the pipeline: "
            + ", ".join(f"`{c}`" for c in missing[:8])
            + (" …" if len(missing) > 8 else "")
        )

    extra = [c for c in work.columns if c not in expected and c != meta["target_column"]]
    if extra:
        notes.append(
            f"{len(extra)} unused column(s) ignored: "
            + ", ".join(f"`{c}`" for c in extra[:8])
            + (" …" if len(extra) > 8 else "")
        )

    return work[expected], notes


def extract_truth(frame: pd.DataFrame, meta: dict) -> np.ndarray | None:
    """Return 0/1 ground truth if the upload carries a label column, else None."""
    target = meta["target_column"]
    if target not in frame.columns:
        return None
    series = frame[target]
    if series.dtype.kind in "biufc":
        return series.astype(int).to_numpy()
    return (
        series.astype(str).str.strip().str.lower() == meta["positive_label"]
    ).astype(int).to_numpy()


def positive_proba(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class, robust to class-order surprises."""
    proba = pipeline.predict_proba(features)
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    index = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, index]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# -----------------------------------------------------------------------------
# Boot
# -----------------------------------------------------------------------------
meta = load_metadata()
if meta is None:
    st.error("### No trained models found")
    st.markdown(
        """
Run the training script first, then copy its artefacts next to `app.py`:

```bash
python train.py          # or upload train.py to Google Colab and run it there
```

It produces `model/` (five `.joblib` pipelines + `metadata.json`) and
`test_data.csv`. The repository layout the app expects is:

```
project-folder/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```
"""
    )
    st.stop()

models, load_problems = load_models(meta["model_files"])
for problem in load_problems:
    st.sidebar.warning(problem)

if not models:
    st.error(
        "None of the model files could be loaded. This is almost always a "
        "scikit-learn version mismatch between the machine that ran `train.py` "
        f"(scikit-learn {meta['library_versions']['scikit-learn']}) and this one. "
        "Pin that exact version in `requirements.txt` and redeploy."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("1 · Test data")
    upload = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help="Comma- or semicolon-separated. Include the `y` column to see "
             "evaluation metrics; without it the app runs in prediction-only mode.",
    )

    source_frame, source_label = None, ""
    if upload is not None:
        try:
            source_frame = read_csv_bytes(upload.getvalue())
            source_label = f"uploaded · {upload.name}"
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")

    if source_frame is None:
        source_frame = load_default_test()
        source_label = "bundled · test_data.csv"

    if source_frame is None:
        st.error("No data available. Upload a CSV to continue.")
        st.stop()

    st.caption(f"**Active dataset:** {source_label}  \n"
               f"{source_frame.shape[0]:,} rows × {source_frame.shape[1]} columns")

    if DEFAULT_TEST_CSV.exists():
        st.download_button(
            "⬇️ Download sample test_data.csv",
            data=DEFAULT_TEST_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("2 · Model")
    available = [m for m in meta["model_order"] if m in models]
    chosen_model = st.selectbox("Classifier", available, index=len(available) - 1)

    st.divider()
    st.subheader("3 · Decision threshold")
    threshold = st.slider(
        "Classify as “yes” when P(subscribe) ≥",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="Lower it to catch more subscribers (higher recall, more wasted "
             "calls); raise it to only call the most confident leads.",
    )

    st.divider()
    st.caption(
        f"Trained on scikit-learn {meta['library_versions']['scikit-learn']} · "
        f"{meta['n_rows_total']:,} rows · {meta['n_features']} features"
    )


# -----------------------------------------------------------------------------
# Score every model once, so all tabs share the same computation
# -----------------------------------------------------------------------------
features, prep_notes = prepare_features(source_frame, meta)
y_true = extract_truth(source_frame, meta)

scored = {}
for name, pipeline in models.items():
    proba = positive_proba(pipeline, features)
    scored[name] = {
        "proba": proba,
        "pred": (proba >= threshold).astype(int),
    }

live_table = None
if y_true is not None:
    live_table = pd.DataFrame(
        {
            name: compute_metrics(y_true, res["pred"], res["proba"])
            for name, res in scored.items()
        }
    ).T.loc[[m for m in meta["model_order"] if m in scored]][METRIC_ORDER]
    live_table.index.name = "Model"


tab_data, tab_eval, tab_compare, tab_single, tab_about = st.tabs(
    ["📊 Data", "🎯 Evaluation", "🏁 Compare models", "🧍 Single client", "ℹ️ About"]
)


# -----------------------------------------------------------------------------
# Tab 1 — Data
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Active test set")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{source_frame.shape[0]:,}")
    c2.metric("Columns supplied", source_frame.shape[1])
    c3.metric("Features used by models", len(meta["feature_columns"]))
    c4.metric(
        "Labelled?",
        "Yes" if y_true is not None else "No",
        help="Metrics need the ground-truth `y` column.",
    )

    for note in prep_notes:
        st.markdown(f'<div class="td-note">{note}</div>', unsafe_allow_html=True)
    if prep_notes:
        st.write("")

    st.dataframe(source_frame.head(200), use_container_width=True, height=340)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Class balance in this file**")
        if y_true is not None:
            counts = pd.Series(y_true).map({0: "no", 1: "yes"}).value_counts()
            fig, ax = plt.subplots(figsize=(5, 3.4))
            sns.barplot(x=counts.index, y=counts.values, ax=ax,
                        hue=counts.index, legend=False,
                        palette=["#5b8c9e", "#e07a5f"])
            for i, v in enumerate(counts.values):
                ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})",
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Clients")
            ax.set_xlabel("Subscribed to term deposit")
            ax.set_ylim(0, counts.max() * 1.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                f"Only {np.mean(y_true):.1%} of contacted clients subscribed. "
                "Accuracy alone would look impressive for a model that never "
                "predicts “yes” — which is why MCC and AUC lead the comparison."
            )
        else:
            st.info("Upload a file containing the `y` column to see the balance.")

    with right:
        st.markdown("**Numeric feature distribution**")
        numeric_here = [
            c for c in meta["numeric_columns"] if c in source_frame.columns
        ]
        if numeric_here:
            pick = st.selectbox("Feature", numeric_here, key="dist_pick")
            fig, ax = plt.subplots(figsize=(5, 3.4))
            if y_true is not None:
                plot_frame = pd.DataFrame(
                    {
                        pick: pd.to_numeric(source_frame[pick], errors="coerce"),
                        "Subscribed": pd.Series(y_true).map({0: "no", 1: "yes"}),
                    }
                ).dropna()
                sns.kdeplot(data=plot_frame, x=pick, hue="Subscribed",
                            fill=True, common_norm=False, alpha=.45, ax=ax)
            else:
                sns.histplot(pd.to_numeric(source_frame[pick], errors="coerce")
                             .dropna(), bins=30, ax=ax, color="#2b7a9b")
            ax.set_title(f"{pick}", fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No recognised numeric feature columns in this file.")


# -----------------------------------------------------------------------------
# Tab 2 — Evaluation of the selected model
# -----------------------------------------------------------------------------
with tab_eval:
    st.subheader(f"{chosen_model} — evaluation at threshold {threshold:.2f}")

    result = scored[chosen_model]
    preds, proba = result["pred"], result["proba"]

    if y_true is None:
        st.warning(
            "The active file has no `y` column, so metrics cannot be computed. "
            "Predictions are shown below and can be downloaded."
        )
    else:
        live = compute_metrics(y_true, preds, proba)
        baseline = meta["metrics"].get(chosen_model, {})

        cols = st.columns(6)
        for col, metric in zip(cols, METRIC_ORDER):
            reference = baseline.get(metric)
            delta = (
                f"{live[metric] - reference:+.3f} vs training run"
                if reference is not None and not np.isnan(live[metric])
                else None
            )
            col.metric(
                metric,
                "—" if np.isnan(live[metric]) else f"{live[metric]:.4f}",
                delta=delta,
                delta_color="normal",
            )

        st.caption(
            "“vs training run” compares against the held-out score recorded by "
            "`train.py` at threshold 0.50. Moving the slider changes every metric "
            "except AUC, which is threshold-independent by construction."
        )

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Confusion matrix")
            matrix = confusion_matrix(y_true, preds, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            sns.heatmap(
                matrix, annot=True, fmt=",d", cmap="YlGnBu", cbar=False,
                square=True, xticklabels=["no", "yes"],
                yticklabels=["no", "yes"], annot_kws={"size": 13}, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            tn, fp, fn, tp = matrix.ravel()
            st.markdown(
                f"""
- **{tp:,}** subscribers correctly identified (true positives)
- **{fn:,}** subscribers missed (false negatives — lost revenue)
- **{fp:,}** wasted calls (false positives — wasted agent time)
- **{tn:,}** correctly left alone (true negatives)
"""
            )

        with right:
            st.markdown("#### Classification report")
            report = classification_report(
                y_true, preds,
                target_names=["no (0)", "yes (1)"],
                digits=4, zero_division=0, output_dict=True,
            )
            st.dataframe(
                pd.DataFrame(report).T.round(4),
                use_container_width=True,
            )

            st.markdown("#### ROC & Precision–Recall")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
            fpr, tpr, _ = roc_curve(y_true, proba)
            ax1.plot(fpr, tpr, color="#2b7a9b", lw=2,
                     label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
            ax1.plot([0, 1], [0, 1], "k--", lw=1)
            ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
            ax1.set_title("ROC", fontsize=10); ax1.legend(fontsize=8)

            prec, rec, _ = precision_recall_curve(y_true, proba)
            ax2.plot(rec, prec, color="#e07a5f", lw=2)
            ax2.axhline(np.mean(y_true), ls="--", c="k", lw=1,
                        label=f"Base rate = {np.mean(y_true):.3f}")
            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
            ax2.set_title("Precision–Recall", fontsize=10); ax2.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()
    st.markdown("#### Row-level predictions")
    output = source_frame.copy()
    output["predicted_label"] = np.where(preds == 1, "yes", "no")
    output["probability_yes"] = proba.round(4)
    if y_true is not None:
        output["correct"] = np.where(preds == y_true, "✓", "✗")

    st.dataframe(output.head(300), use_container_width=True, height=330)
    st.download_button(
        f"⬇️ Download all {len(output):,} predictions ({chosen_model})",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen_model.split(' ')[0].lower()}.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Tab 3 — Compare every model
# -----------------------------------------------------------------------------
with tab_compare:
    st.subheader("All five models on the active test set")

    if live_table is None:
        st.warning(
            "Metrics need ground-truth labels. Showing the scores recorded during "
            "training instead."
        )
        st.dataframe(
            pd.DataFrame(meta["metrics"]).T.loc[meta["model_order"]][METRIC_ORDER]
            .round(4),
            use_container_width=True,
        )
    else:
        st.dataframe(
            live_table.round(4).style.highlight_max(axis=0, color="#c8e6c9")
            .format("{:.4f}"),
            use_container_width=True,
        )
        st.caption(f"Computed live at threshold {threshold:.2f}. "
                   "Green marks the best model per metric.")

        winners = ", ".join(
            f"**{m}** → {live_table[m].idxmax()}" for m in ["AUC", "F1", "MCC"]
        )
        st.markdown(f'<div class="td-note">Leaders — {winners}</div>',
                    unsafe_allow_html=True)
        st.write("")

        left, right = st.columns([1.15, 1])

        with left:
            st.markdown("#### Metric-by-metric")
            tidy = live_table.reset_index().melt(
                id_vars="Model", var_name="Metric", value_name="Score"
            )
            fig, ax = plt.subplots(figsize=(9, 4.6))
            sns.barplot(data=tidy, x="Metric", y="Score", hue="Model", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("#### ROC curves, all models")
            fig, ax = plt.subplots(figsize=(6, 4.6))
            for name in live_table.index:
                fpr, tpr, _ = roc_curve(y_true, scored[name]["proba"])
                ax.plot(fpr, tpr, lw=2,
                        label=f"{name} ({live_table.loc[name, 'AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()
        st.markdown("#### Live scores vs. the original training run")
        recorded = pd.DataFrame(meta["metrics"]).T.loc[live_table.index][METRIC_ORDER]
        drift = (live_table - recorded).round(4)
        st.dataframe(
            drift.style.background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.1)
            .format("{:+.4f}"),
            use_container_width=True,
        )
        st.caption(
            "Near-zero everywhere means this file matches the held-out split "
            "`train.py` evaluated on. Large gaps mean either a different dataset "
            "or a threshold other than 0.50."
        )

    if REPORT_DIR.exists():
        images = sorted(REPORT_DIR.glob("*.png"))
        if images:
            st.divider()
            st.markdown("#### Figures from the training run")
            for image in images:
                with st.expander(image.stem.replace("_", " ").title()):
                    st.image(str(image), use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 4 — Single client what-if
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("Score one client")
    st.caption(
        "Fill in a profile and every model scores it at once — handy for sanity-"
        "checking where the classifiers disagree. Defaults come from the medians "
        "and most common categories of the training data."
    )

    numeric_cols = meta["numeric_columns"]
    categorical_cols = meta["categorical_columns"]
    levels = meta.get("categorical_levels", {})
    summary = meta.get("numeric_summary", {})

    with st.form("single_client"):
        grid = st.columns(3)
        record = {}

        for i, col in enumerate(numeric_cols):
            stats = summary.get(col, {})
            low = float(stats.get("min", 0.0))
            high = float(stats.get("max", 100.0))
            mid = float(stats.get("median", (low + high) / 2))
            step = 1.0 if float(high - low) > 25 else 0.01
            record[col] = grid[i % 3].number_input(
                col, min_value=low, max_value=high, value=mid, step=step
            )

        offset = len(numeric_cols)
        for i, col in enumerate(categorical_cols):
            options = levels.get(col, ["unknown"])
            record[col] = grid[(offset + i) % 3].selectbox(col, options)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        one_row = pd.DataFrame([record])[meta["feature_columns"]]
        verdicts = []
        for name in meta["model_order"]:
            if name not in models:
                continue
            p = float(positive_proba(models[name], one_row)[0])
            verdicts.append(
                {
                    "Model": name,
                    "P(subscribe)": round(p, 4),
                    "Decision": "yes" if p >= threshold else "no",
                }
            )

        verdict_frame = pd.DataFrame(verdicts)
        votes = int((verdict_frame["Decision"] == "yes").sum())

        a, b = st.columns([1, 1.6])
        with a:
            st.metric("Models voting “yes”", f"{votes} / {len(verdict_frame)}")
            st.metric("Mean probability",
                      f"{verdict_frame['P(subscribe)'].mean():.3f}")
            if votes > len(verdict_frame) / 2:
                st.success("Majority say this client is worth calling.")
            else:
                st.info("Majority say this client is unlikely to subscribe.")
        with b:
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            sns.barplot(data=verdict_frame, y="Model", x="P(subscribe)",
                        ax=ax, color="#2b7a9b")
            ax.axvline(threshold, ls="--", c="#e07a5f", lw=2,
                       label=f"threshold {threshold:.2f}")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.dataframe(verdict_frame, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab 5 — About
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("About this app")

    st.markdown(
        f"""
**Problem.** A Portuguese retail bank ran direct-marketing phone campaigns.
Each call costs agent time, and only about **11%** of contacted clients end up
subscribing to a term deposit. Predicting who will subscribe *before* dialling
lets the bank spend the same budget on a far better-targeted list.

**Dataset.** [{meta['dataset_name']}]({meta['dataset_url']}) —
**{meta['n_rows_total']:,} instances × {meta['n_features']} input features**,
comfortably past the assignment minimums of 500 rows and 12 features. Features
span client demographics (age, job, marital status, education), the current and
previous campaign (contact channel, month, number of contacts, previous
outcome), and the macroeconomic context at call time (euribor 3-month rate,
employment variation rate, consumer price and confidence indices).

**Two decisions worth calling out.**

1. `duration` — the length of the call — is **excluded**. It is only known once
   the call has ended, so it leaks the outcome; the UCI authors recommend
   dropping it for a realistic predictive model. Keeping it would inflate AUC
   to roughly 0.94.
2. `pdays = 999` is a *sentinel* for "never previously contacted", not a real
   duration. Left raw it distorts every distance and every split, so it is
   split into a binary `was_contacted_before` flag with the magnitude zeroed.

**Models.** Five classifiers, each wrapped in a single scikit-learn `Pipeline`
so that imputation, scaling and one-hot encoding are fitted on training data
only and travel with the model — no leakage across the split, and no chance of
the app preprocessing differently from training.
"""
    )

    st.markdown("**Metrics recorded during training** (held-out split, threshold 0.50)")
    st.dataframe(
        pd.DataFrame(meta["metrics"]).T.loc[meta["model_order"]][METRIC_ORDER].round(4),
        use_container_width=True,
    )

    versions = meta["library_versions"]
    st.markdown(
        f"""
**Reproducibility.** `random_state = {meta['random_state']}`, stratified
{int((1 - meta['test_size']) * 100)}/{int(meta['test_size'] * 100)} split.
Trained on Python {versions['python']} with scikit-learn
{versions['scikit-learn']}, pandas {versions['pandas']}, numpy
{versions['numpy']}. Those versions are pinned in `requirements.txt` —
unpickling a Pipeline under a different scikit-learn build is the usual reason
a Streamlit Cloud deployment fails to boot.

**Retraining.** `train.py` is standalone and Colab-ready: run it, download
`assignment2_artifacts.zip`, and drop `model/` plus `test_data.csv` back into
this repository.
"""
    )
