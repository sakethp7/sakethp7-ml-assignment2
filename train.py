"""
===============================================================================
 Term Deposit Subscription Prediction  --  Model Training Pipeline
 BITS Pilani WILP  |  M.Tech (AIML/DSE)  |  Machine Learning  |  Assignment 2
===============================================================================

 Dataset : Bank Marketing (UCI ML Repository, id=222)
           https://archive.ics.uci.edu/dataset/222/bank+marketing
           bank-additional-full.csv -> 41,188 instances x 20 input features
           (comfortably above the required minimum of 500 rows / 12 features)

 Task    : Binary classification -- will a contacted client subscribe to a
           term deposit? (target column `y`, values 'yes' / 'no')

 Models trained:
     1. Logistic Regression
     2. Decision Tree Classifier
     3. K-Nearest Neighbours Classifier
     4. Naive Bayes (Gaussian)
     5. Random Forest Classifier  (ensemble)

 Metrics reported for every model:
     Accuracy, ROC-AUC, Precision, Recall, F1, Matthews Corr. Coefficient

 HOW TO RUN
 ----------
   Google Colab :  upload this file, then in a cell ->  !python train.py
                   (or simply paste the whole file into one cell and run)
   Local        :  python train.py

 WHAT IT PRODUCES  (everything app.py needs)
 -------------------------------------------
   model/<model_name>.joblib   one fitted end-to-end sklearn Pipeline each
   model/metadata.json         feature list, metrics, library versions
   model/metrics.csv           the comparison table as raw CSV
   test_data.csv               held-out test split -> upload this to Streamlit
   reports/*.png               confusion matrices, ROC curves, metric bars
   requirements.txt            pinned to the versions actually used here
   assignment2_artifacts.zip   all of the above, zipped for easy download
===============================================================================
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe: works in Colab and on a plain server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
class CFG:
    """Experiment configuration."""

    SEED = 42
    TEST_SIZE = 0.20

    DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    PREFERRED_CSV = "bank-additional-full.csv"   # the 20-feature variant
    TARGET = "y"
    POSITIVE_LABEL = "yes"

    # `duration` is the call length in seconds. It is only known AFTER the call
    # has happened, so it leaks the outcome; the dataset authors recommend
    # dropping it for a realistic predictive model.
    DROP_LEAKY_DURATION = True

    OUT_MODEL_DIR = Path("model")
    OUT_REPORT_DIR = Path("reports")
    OUT_TEST_CSV = Path("test_data.csv")
    OUT_REQUIREMENTS = Path("requirements.txt")
    OUT_ZIP = Path("assignment2_artifacts.zip")


# Fixed display order for tables and plots.
MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
]

# Short slugs used for the .joblib filenames.
MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest (Ensemble)": "random_forest",
}

sns.set_theme(style="whitegrid", palette="deep")
np.random.seed(CFG.SEED)


def banner(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


# =============================================================================
# 2. DATA ACQUISITION
# =============================================================================
def _read_nested_zip(raw_bytes: bytes, wanted: str) -> pd.DataFrame | None:
    """
    The UCI archive for this dataset is a zip-inside-a-zip:

        bank+marketing.zip
          |-- bank.zip             -> bank-full.csv, bank.csv
          |-- bank-additional.zip  -> bank-additional-full.csv, bank-additional.csv

    Walk both levels and return the first CSV whose name matches `wanted`.
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as outer:
        names = outer.namelist()

        # Level 1: is the CSV sitting directly in the outer archive?
        for name in names:
            if name.endswith(wanted):
                with outer.open(name) as fh:
                    return pd.read_csv(fh, sep=";", quotechar='"')

        # Level 2: descend into every nested .zip we can find.
        for name in names:
            if not name.lower().endswith(".zip"):
                continue
            inner_bytes = outer.read(name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                for inner_name in inner.namelist():
                    if inner_name.endswith(wanted):
                        with inner.open(inner_name) as fh:
                            return pd.read_csv(fh, sep=";", quotechar='"')
    return None


def load_bank_marketing() -> pd.DataFrame:
    """
    Fetch the dataset, trying the sources in order of preference:

        1. the official UCI zip  (gives us the richer 20-feature variant)
        2. the `ucimlrepo` package, if it happens to be installed
        3. any *.csv already sitting next to this script

    Whichever path wins, we return a DataFrame whose column names have been
    normalised (dots -> underscores) so the rest of the script is source-agnostic.
    """
    banner("STEP 1  |  LOADING THE BANK MARKETING DATASET")
    frame = None

    # ---- Source 1: UCI archive -------------------------------------------
    try:
        print(f"Downloading {CFG.DATA_URL} ...")
        request = Request(CFG.DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        print(f"  received {len(payload) / 1024:.0f} KB")

        frame = _read_nested_zip(payload, CFG.PREFERRED_CSV)
        if frame is None:
            print(f"  '{CFG.PREFERRED_CSV}' not found, falling back to bank-full.csv")
            frame = _read_nested_zip(payload, "bank-full.csv")
        if frame is not None:
            print("  source: UCI ML Repository archive")
    except Exception as exc:  # network blocked, TLS issue, archive moved, ...
        print(f"  download failed ({type(exc).__name__}: {exc})")

    # ---- Source 2: ucimlrepo package -------------------------------------
    if frame is None:
        try:
            from ucimlrepo import fetch_ucirepo

            print("Trying the `ucimlrepo` package instead ...")
            bundle = fetch_ucirepo(id=222)
            frame = pd.concat([bundle.data.features, bundle.data.targets], axis=1)
            print("  source: ucimlrepo (id=222)")
        except Exception as exc:
            print(f"  ucimlrepo unavailable ({type(exc).__name__}: {exc})")

    # ---- Source 3: a local copy ------------------------------------------
    if frame is None:
        for candidate in sorted(Path(".").glob("*.csv")):
            if candidate.name == CFG.OUT_TEST_CSV.name:
                continue
            print(f"Trying local file {candidate} ...")
            for sep in (";", ","):
                try:
                    trial = pd.read_csv(candidate, sep=sep)
                except Exception:
                    continue
                if trial.shape[1] > 5:
                    frame = trial
                    print(f"  source: {candidate} (sep='{sep}')")
                    break
            if frame is not None:
                break

    if frame is None:
        raise RuntimeError(
            "Could not obtain the dataset.\n"
            "Fix: download bank+marketing.zip from\n"
            "  https://archive.ics.uci.edu/dataset/222/bank+marketing\n"
            "extract bank-additional-full.csv next to this script, and re-run.\n"
            "In Colab you can also run:  !pip install ucimlrepo"
        )

    # Normalise: 'emp.var.rate' -> 'emp_var_rate'. Dots in column names cause
    # endless friction later (attribute access, some plotting libs, SQL export).
    frame.columns = [str(c).strip().replace(".", "_").lower() for c in frame.columns]
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


# =============================================================================
# 3. EXPLORATION + PREPARATION
# =============================================================================
def explore(frame: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    banner("STEP 2  |  EXPLORATORY SUMMARY")
    print(f"Shape                : {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"Input features       : {frame.shape[1] - 1}  (assignment minimum: 12)")
    print(f"Instances            : {frame.shape[0]:,}  (assignment minimum: 500)")
    print(f"Duplicate rows       : {int(frame.duplicated().sum())}")
    print(f"Missing cells        : {int(frame.isna().sum().sum())}")

    print("\nColumn dtypes:")
    dtype_summary = (
        frame.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .to_frame("n_columns")
    )
    print(dtype_summary.to_string())

    print(f"\nTarget distribution ('{CFG.TARGET}'):")
    counts = frame[CFG.TARGET].value_counts()
    for label, n in counts.items():
        print(f"  {str(label):<6} {n:>7,}   ({n / len(frame):6.2%})")
    minority = counts.min() / counts.sum()
    print(f"\nMinority class share : {minority:.2%}")
    print(
        "  -> The classes are imbalanced, so plain accuracy is a misleading\n"
        "     headline number. AUC, F1 and MCC are the metrics to compare on."
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Two dataset-specific cleanups, both documented in the UCI data description.

    * `pdays == 999` is a sentinel meaning "this client was never contacted in a
      previous campaign". Left as-is, a model reads 999 as an enormous number of
      days and the scale wrecks distance-based learners like kNN. We split the
      sentinel into its own boolean flag and neutralise the magnitude.

    * `duration` is the call duration in seconds and is unknown until the call
      ends, i.e. it leaks the label. The dataset authors say it "should be
      discarded if the intention is to have a realistic predictive model".
    """
    banner("STEP 3  |  FEATURE ENGINEERING")
    out = frame.copy()

    if "pdays" in out.columns:
        sentinel = 999 if (out["pdays"] == 999).any() else -1
        never_contacted = out["pdays"] == sentinel
        out["was_contacted_before"] = (~never_contacted).astype(int)
        out.loc[never_contacted, "pdays"] = 0
        print(
            f"  pdays  : sentinel {sentinel} found in {never_contacted.sum():,} rows "
            "-> new flag `was_contacted_before`, sentinel zeroed"
        )

    if CFG.DROP_LEAKY_DURATION and "duration" in out.columns:
        out = out.drop(columns=["duration"])
        print("  duration: dropped (post-call leakage -- see docstring)")

    print(f"  final feature count: {out.shape[1] - 1}")
    return out


def build_preprocessor(
    x_frame: pd.DataFrame, verbose: bool = True
) -> tuple[ColumnTransformer, list, list]:
    """
    One ColumnTransformer shared by all five models, so the comparison is fair:
    every algorithm sees exactly the same feature matrix.

      numeric      -> median impute -> standardise
      categorical  -> most-frequent impute -> one-hot (unknown categories ignored)

    Standardising matters for Logistic Regression (convergence) and kNN
    (Euclidean distance). It is harmless for the tree-based models.
    """
    numeric_cols = x_frame.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in x_frame.columns if c not in numeric_cols]

    # `sparse_output` replaced `sparse` in scikit-learn 1.2; support both so the
    # script runs on Colab today and on an older pinned environment.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_branch, numeric_cols),
            ("cat", categorical_branch, categorical_cols),
        ],
        remainder="drop",
    )

    if verbose:
        print(f"  numeric columns     ({len(numeric_cols):>2}): {numeric_cols}")
        print(f"  categorical columns ({len(categorical_cols):>2}): {categorical_cols}")
    return preprocessor, numeric_cols, categorical_cols


# =============================================================================
# 4. MODELS
# =============================================================================
def build_models() -> dict:
    """
    The five classifiers.

    `class_weight="balanced"` is used wherever the estimator supports it: with
    an ~11% positive rate an unweighted model can score ~89% accuracy by
    predicting "no" for everybody. kNN and GaussianNB have no such parameter.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=CFG.SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,               # unpruned trees memorise this dataset
            min_samples_leaf=25,
            criterion="gini",
            class_weight="balanced",
            random_state=CFG.SEED,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=25,            # odd-ish k, large enough to smooth noise
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8,
        ),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=CFG.SEED,
        ),
    }


def score_model(y_true, y_pred, y_proba) -> dict:
    """The six metrics the assignment asks for, in the order it lists them."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# =============================================================================
# 5. VISUALS
# =============================================================================
def plot_confusion_matrices(store: dict, y_test) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, name in zip(axes, MODEL_ORDER):
        matrix = confusion_matrix(y_test, store[name]["y_pred"])
        sns.heatmap(
            matrix,
            annot=True,
            fmt=",d",
            cmap="YlGnBu",
            cbar=False,
            square=True,
            xticklabels=["no", "yes"],
            yticklabels=["no", "yes"],
            ax=ax,
        )
        ax.set_title(f"{name}\nF1 = {store[name]['metrics']['F1']:.3f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    axes[-1].axis("off")
    fig.suptitle("Confusion Matrices on the Held-Out Test Set", fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(CFG.OUT_REPORT_DIR / "confusion_matrices.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(store: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_test, store[name]["y_proba"])
        auc = store[name]["metrics"]["AUC"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Term Deposit Subscription")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CFG.OUT_REPORT_DIR / "roc_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(results: pd.DataFrame) -> None:
    tidy = results.reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=tidy, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Across All Six Evaluation Metrics")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(CFG.OUT_REPORT_DIR / "metric_comparison.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(pipeline, top_n: int = 20) -> None:
    """Random Forest impurity importances, mapped back to readable column names."""
    try:
        names = pipeline.named_steps["prep"].get_feature_names_out()
        importances = pipeline.named_steps["clf"].feature_importances_
    except Exception as exc:
        print(f"  (skipped feature importance plot: {exc})")
        return

    ranked = (
        pd.Series(importances, index=[n.split("__", 1)[-1] for n in names])
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(x=ranked.values, y=ranked.index, ax=ax, color="#2b7a9b")
    ax.set_title(f"Random Forest -- Top {top_n} Feature Importances")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(CFG.OUT_REPORT_DIR / "feature_importance.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 6. REQUIREMENTS
# =============================================================================
def write_requirements() -> None:
    """
    Pin to the versions this run actually used. Loading a pickled sklearn
    Pipeline under a different scikit-learn version is the single most common
    way a Streamlit Cloud deployment breaks, so we freeze what worked.
    """
    import matplotlib as mpl

    try:
        import streamlit

        streamlit_pin = f"streamlit=={streamlit.__version__}"
    except Exception:
        # Colab has no Streamlit; a recent floor is the safe choice.
        streamlit_pin = "streamlit>=1.36.0"

    lines = [
        streamlit_pin,
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pd.__version__}",
        f"numpy=={np.__version__}",
        f"matplotlib=={mpl.__version__}",
        f"seaborn=={sns.__version__}",
        f"joblib=={joblib.__version__}",
        "",
    ]
    CFG.OUT_REQUIREMENTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {CFG.OUT_REQUIREMENTS} (pinned to this run's versions)")


# =============================================================================
# 7. MAIN
# =============================================================================
def main() -> None:
    started = time.time()
    CFG.OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CFG.OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"python       {sys.version.split()[0]}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"pandas       {pd.__version__}")
    print(f"numpy        {np.__version__}")

    # ---- data ------------------------------------------------------------
    raw = load_bank_marketing()
    if CFG.TARGET not in raw.columns:
        raise KeyError(
            f"Expected a target column named '{CFG.TARGET}'. Got: {list(raw.columns)}"
        )
    explore(raw)

    frame = engineer_features(raw)
    x_all = frame.drop(columns=[CFG.TARGET])
    y_all = (
        frame[CFG.TARGET].astype(str).str.strip().str.lower()
        == CFG.POSITIVE_LABEL
    ).astype(int)

    # ---- split -----------------------------------------------------------
    banner("STEP 4  |  TRAIN / TEST SPLIT")
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.SEED,
        stratify=y_all,  # keep the ~11% positive rate identical in both halves
    )
    print(f"  train: {x_train.shape[0]:,} rows   positives {y_train.mean():.2%}")
    print(f"  test : {x_test.shape[0]:,} rows   positives {y_test.mean():.2%}")


    test_export = x_test.copy()
    test_export[CFG.TARGET] = frame.loc[x_test.index, CFG.TARGET].values
    test_export.to_csv(CFG.OUT_TEST_CSV, index=False)
    print(f"  wrote {CFG.OUT_TEST_CSV}  ({test_export.shape[0]:,} rows) "
          "-> upload this in the Streamlit app")

    reloaded = pd.read_csv(CFG.OUT_TEST_CSV)
    x_test = reloaded[x_all.columns]
    y_test = (
        reloaded[CFG.TARGET].astype(str).str.strip().str.lower()
        == CFG.POSITIVE_LABEL
    ).astype(int)

    banner("STEP 5  |  PREPROCESSING PLAN")
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)

    # ---- train + evaluate ------------------------------------------------
    banner("STEP 6  |  TRAINING AND EVALUATING 5 MODELS")
    store, rows = {}, {}

    for name, estimator in build_models().items():
        print(f"\n--- {name} ---")
        pipeline = Pipeline(
            [
                # A fresh, unfitted preprocessor per model: each pipeline must own
                # its own transformer state so the saved artefacts are standalone.
                ("prep", build_preprocessor(x_train, verbose=False)[0]),
                ("clf", estimator),
            ]
        )

        fit_start = time.time()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.time() - fit_start

        predict_start = time.time()
        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test)[:, 1]
        predict_seconds = time.time() - predict_start

        metrics = score_model(y_test, y_pred, y_proba)
        metrics_with_timing = dict(metrics)
        metrics_with_timing["FitSeconds"] = round(fit_seconds, 2)
        metrics_with_timing["PredictSeconds"] = round(predict_seconds, 2)

        store[name] = {"y_pred": y_pred, "y_proba": y_proba, "metrics": metrics}
        rows[name] = metrics_with_timing

        print(f"  fit {fit_seconds:6.2f}s   predict {predict_seconds:5.2f}s")
        print("  " + "   ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        path = CFG.OUT_MODEL_DIR / f"{MODEL_SLUGS[name]}.joblib"
        joblib.dump(pipeline, path, compress=3)
        print(f"  saved -> {path} ({path.stat().st_size / 1024:.0f} KB)")

    results = pd.DataFrame(rows).T.loc[MODEL_ORDER]
    results.index.name = "Model"
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    # ---- report ----------------------------------------------------------
    banner("STEP 7  |  COMPARISON TABLE")
    print(results[metric_cols].round(4).to_string())
    print("\nTiming (seconds):")
    print(results[["FitSeconds", "PredictSeconds"]].to_string())

    print("\nPer-model classification reports:")
    for name in MODEL_ORDER:
        print(f"\n>>> {name}")
        print(
            classification_report(
                y_test,
                store[name]["y_pred"],
                target_names=["no (0)", "yes (1)"],
                digits=4,
                zero_division=0,
            )
        )

    results.to_csv(CFG.OUT_MODEL_DIR / "metrics.csv")

    banner("STEP 8  |  PLOTS")
    plot_confusion_matrices(store, y_test)
    plot_roc_curves(store, y_test)
    plot_metric_bars(results[metric_cols])
    plot_feature_importance(
        joblib.load(CFG.OUT_MODEL_DIR / f"{MODEL_SLUGS['Random Forest (Ensemble)']}.joblib")
    )
    for png in sorted(CFG.OUT_REPORT_DIR.glob("*.png")):
        print(f"  {png}")

    # ---- metadata for the Streamlit app ----------------------------------
    banner("STEP 9  |  ARTIFACTS FOR THE STREAMLIT APP")
    metadata = {
        "dataset_name": "Bank Marketing (UCI ML Repository, id=222)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "task": "Binary classification -- term deposit subscription",
        "n_rows_total": int(frame.shape[0]),
        "n_features": int(x_all.shape[1]),
        "target_column": CFG.TARGET,
        "positive_label": CFG.POSITIVE_LABEL,
        "class_balance": {
            str(k): int(v) for k, v in raw[CFG.TARGET].value_counts().items()
        },
        "feature_columns": x_all.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "categorical_levels": {
            c: sorted(x_all[c].astype(str).dropna().unique().tolist())
            for c in categorical_cols
        },
        "numeric_summary": {
            c: {
                "min": float(x_all[c].min()),
                "max": float(x_all[c].max()),
                "median": float(x_all[c].median()),
            }
            for c in numeric_cols
        },
        "dropped_columns": (["duration"] if CFG.DROP_LEAKY_DURATION else []),
        "test_size": CFG.TEST_SIZE,
        "random_state": CFG.SEED,
        "model_files": {name: f"{MODEL_SLUGS[name]}.joblib" for name in MODEL_ORDER},
        "model_order": MODEL_ORDER,
        "metrics": {
            name: {k: float(v) for k, v in results.loc[name].items()}
            for name in MODEL_ORDER
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    meta_path = CFG.OUT_MODEL_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  wrote {meta_path}")

    winner = results["MCC"].idxmax()
    write_requirements()

    # ---- bundle ----------------------------------------------------------
    with zipfile.ZipFile(CFG.OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(CFG.OUT_MODEL_DIR.rglob("*")):
            bundle.write(path, path.as_posix())
        for path in sorted(CFG.OUT_REPORT_DIR.rglob("*")):
            bundle.write(path, path.as_posix())
        for path in (CFG.OUT_TEST_CSV, CFG.OUT_REQUIREMENTS):
            if path.exists():
                bundle.write(path, path.name)
    print(f"  wrote {CFG.OUT_ZIP} ({CFG.OUT_ZIP.stat().st_size / 1024:.0f} KB)")

    banner(f"DONE in {time.time() - started:.1f}s  --  winner: {winner}")
 

    # Colab convenience: trigger the browser download automatically.
    if "google.colab" in sys.modules:
        try:
            from google.colab import files

            files.download(str(CFG.OUT_ZIP))
        except Exception as exc:
            print(f"(automatic download unavailable: {exc})")


if __name__ == "__main__":
    main()
