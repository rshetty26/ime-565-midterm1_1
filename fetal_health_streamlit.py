import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fetal Health Classification", layout="wide")

DATASET_PATH = "fetal_health.csv"
USER_SAMPLE_PATH = "fetal_health_user.csv"
IMG_PATH = "fetal_health_image.gif"

RF_PKL = "random_forest.pickle"
DT_PKL = "decision_tree.pickle"
ADA_PKL = "adaboost.pickle"
SOFT_PKL = "soft_vote.pickle"

TARGET_COL = "fetal_health"
CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathological"}
CLASS_COLORS = {"Normal": "lime", "Suspect": "yellow", "Pathological": "orange"}

PRECOMP_ASSETS = {
    "Random Forest": {
        "cm_img": "rf_confusion_mat.svg",
        "fi_img": "rf_feature_imp.svg",
        "report_csv": "rf_class_report.csv"
    },
    "Decision Tree": {
        "cm_img": "dt_confusion_mat.svg",
        "fi_img": "dt_feature_imp.svg",
        "report_csv": "dt_class_report.csv"
    },
    "AdaBoost": {
        "cm_img": "ab_confusion_mat.svg",
        "fi_img": "ab_feature_imp.svg",
        "report_csv": "ab_class_report.csv"
    },
    "Soft Voting Classifier": {
        "cm_img": "sv_confusion_mat.png",
        "fi_img": "sv_feature_imp.png",
        "report_csv": "sv_class_report.csv"
    },
}

def _file_exists(p: str) -> bool:
    return p and os.path.exists(p)

@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return load_csv(path)
    except Exception:
        return None

def build_soft_voter_from_pickles(rf, dt, ada, df_all: pd.DataFrame) -> VotingClassifier:
    from sklearn.calibration import CalibratedClassifierCV

    df = df_all.copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' not found in dataset for soft vote weight computation.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Make trainable clones ---
    rf_c  = clone(rf)
    dt_c  = clone(dt)
    ada_c = clone(ada)

    # --- Fix AdaBoost algorithm if environment disallows 'SAMME.R' ---
    try:
        ada_params = ada_c.get_params()
        if ada_params.get("algorithm", None) == "SAMME.R":
            ada_c.set_params(algorithm="SAMME")  # compatible with your sklearn
    except Exception:
        pass

    # --- Fit clones for validation F1 (weights) ---
    rf_val_pred  = rf_c.fit(X_train, y_train).predict(X_val)
    dt_val_pred  = dt_c.fit(X_train, y_train).predict(X_val)
    ada_val_pred = ada_c.fit(X_train, y_train).predict(X_val)

    f1_rf  = f1_score(y_val, rf_val_pred,  average="macro")
    f1_dt  = f1_score(y_val, dt_val_pred,  average="macro")
    f1_ada = f1_score(y_val, ada_val_pred, average="macro")
    weights = np.array([f1_rf, f1_dt, f1_ada], dtype=float)
    weights = weights / weights.sum()

    # --- Ensure every estimator has predict_proba for soft voting ---
    rf_for_vote = clone(rf)
    dt_for_vote = clone(dt)
    ada_for_vote = clone(ada)
    try:
        ada_params2 = ada_for_vote.get_params()
        if ada_params2.get("algorithm", None) == "SAMME.R":
            ada_for_vote.set_params(algorithm="SAMME")
    except Exception:
        pass

    # If AdaBoost still lacks predict_proba, calibrate it
    try:
        _ = ada_for_vote.fit(X_train, y_train)
        has_proba = hasattr(ada_for_vote, "predict_proba")
    except Exception:
        has_proba = False

    if not has_proba:
        ada_for_vote = CalibratedClassifierCV(base_estimator=clone(ada_for_vote), cv=3, method="sigmoid")
        ada_for_vote.fit(X_train, y_train)

    voter = VotingClassifier(
        estimators=[("rf", rf_for_vote), ("dt", dt_for_vote), ("ada", ada_for_vote)],
        voting="soft",
        weights=weights.tolist()
    )
    voter.fit(X_train, y_train)
    return voter


def get_or_make_soft_voter(df_all: Optional[pd.DataFrame]) -> Tuple[Optional[VotingClassifier], Optional[str]]:
    if Path(SOFT_PKL).exists():
        try:
            return load_pickle(SOFT_PKL), f"Loaded soft voting model from `{SOFT_PKL}`."
        except Exception as e:
            return None, f"Could not load `{SOFT_PKL}`: {e}"
    if not all(Path(p).exists() for p in [RF_PKL, DT_PKL, ADA_PKL]):
        return None, "Soft Voting requires either `soft_vote.pickle` or all three base pickles present."
    if df_all is None:
        return None, "Need dataset to compute soft-vote weights. Place `fetal_health.csv` in the app folder."
    try:
        rf = load_pickle(RF_PKL)
        dt = load_pickle(DT_PKL)
        ada = load_pickle(ADA_PKL)
        voter = build_soft_voter_from_pickles(rf, dt, ada, df_all)
        # Optional: persist for reuse
        try:
            with open(SOFT_PKL, "wb") as f:
                pickle.dump(voter, f)
        except Exception:
            pass
        return voter, "Built Soft Voting model from base pickles using F1-macro normalized weights."
    except Exception as e:
        return None, f"Failed to build Soft Voting model: {e}"

def align_columns_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return X
    return X.reindex(columns=list(names), fill_value=0)

def predict_with_proba(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    yhat = model.predict(X)
    proba = model.predict_proba(X)
    return yhat, proba

def add_predictions(df_in: pd.DataFrame, yhat: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    out = df_in.copy()
    pred_ids = yhat.astype(int)
    pred_names = [CLASS_NAMES.get(i, str(i)) for i in pred_ids]
    out["Predicted Class"] = pred_names
    out["Prediction Probability"] = np.max(proba, axis=1)
    return out

def style_pred_column(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def color_row(row):
        color = CLASS_COLORS.get(row["Predicted Class"], "")
        return ["" for _ in row.index[:-2]] + [f"background-color: {color}", ""]
    styler = df.style.format({"Prediction Probability": "{:.3f}"}).apply(color_row, axis=1)
    return styler

def feature_importance_for_model(model) -> Optional[pd.Series]:
    if hasattr(model, "feature_importances_"):
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            return pd.Series(model.feature_importances_)
        return pd.Series(model.feature_importances_, index=list(names))
    if isinstance(model, VotingClassifier):
        weights = np.array(getattr(model, "weights", None))
        if weights is None:
            weights = np.ones(len(model.estimators_)) / len(model.estimators_)
        importances = []
        names = None
        for key, est in model.named_estimators_.items():
            if not hasattr(est, "feature_importances_"):
                return None
            if names is None:
                names = getattr(est, "feature_names_in_", None)
            importances.append(est.feature_importances_)
        if not importances:
            return None
        W = np.vstack(importances)
        weighted = (weights[:, None] * W).sum(axis=0)
        if names is None:
            return pd.Series(weighted)
        return pd.Series(weighted, index=list(names))
    return None

def plot_feature_importance(fi: pd.Series, title: str) -> plt.Figure:
    fi_sorted = fi.sort_values(ascending=True)
    fig = plt.figure(figsize=(8, max(4, 0.25 * len(fi_sorted))))
    plt.barh(fi_sorted.index, fi_sorted.values)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    return fig

def compute_report_and_cm(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, np.ndarray, pd.DataFrame]:
    rep_text = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3])
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score", "support"]
    class_keys = [k for k in rep_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    try:
        class_keys = sorted(class_keys, key=lambda x: float(x))
    except:
        class_keys = sorted(class_keys)
    rows = {m: [] for m in metrics}
    for ck in class_keys:
        rows["precision"].append(rep_dict[ck]["precision"])
        rows["recall"].append(rep_dict[ck]["recall"])
        rows["f1-score"].append(rep_dict[ck]["f1-score"])
        rows["support"].append(float(rep_dict[ck]["support"]))
    acc = accuracy_score(y_true, y_pred)
    macro = rep_dict.get("macro avg", {"precision": None, "recall": None, "f1-score": None})
    weighted = rep_dict.get("weighted avg", {"precision": None, "recall": None, "f1-score": None})
    total_support = float(sum(rows["support"]))
    rows["precision"].extend([acc, macro["precision"], weighted["precision"]])
    rows["recall"].extend([acc, macro["recall"], weighted["recall"]])
    rows["f1-score"].extend([acc, macro["f1-score"], weighted["f1-score"]])
    rows["support"].extend([total_support, total_support, total_support])
    class_cols = [f"{float(k):.1f}" for k in class_keys]
    cols = class_cols + ["accuracy", "macro avg", "weighted avg"]
    rep_csv_like_rf = pd.DataFrame(rows, index=cols).T
    return rep_text, cm, rep_csv_like_rf

def plot_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
    labels = ["Normal","Suspect","Pathological"]
    fig = plt.figure(figsize=(5.5,4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig

st.title("Fetal Health Classification")
st.caption("Predict fetal health (Normal, Suspect, Pathological) from CTG feature values.")

if Path(IMG_PATH).exists():
    st.image(IMG_PATH, use_container_width=True)

df_all = safe_load_csv(DATASET_PATH)

with st.sidebar:
    st.header("Data & Model")
    uploaded = st.file_uploader(
        "Upload CTG CSV (optionally include `fetal_health` for evaluation)",
        type=["csv"]
    )

    st.subheader("Model Choice")
    model_name = st.radio(
        "Choose model",
        ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting Classifier"],
        index=0
    )
    st.info("Soft Voting uses normalized F1-macro weights across RF, DT, and AdaBoost.")

    with st.expander("Sample Data Format", expanded=False):
        if df_all is not None:
            st.write("Preview from `fetal_health.csv` (expected features):")
            st.dataframe(df_all.drop(columns=[TARGET_COL]) if (TARGET_COL in df_all.columns) else df_all, use_container_width=True)
        user_sample = safe_load_csv(USER_SAMPLE_PATH)
        if user_sample is not None:
            st.write("Example `fetal_health_user.csv` preview:")
            st.dataframe(user_sample, use_container_width=True)
        st.caption("Label column (if included) must be `fetal_health`: 1=Normal, 2=Suspect, 3=Pathological.")

loaded_msg = []
rf = dt = ada = soft = None
if model_name != "Soft Voting Classifier":
    try:
        if model_name == "Random Forest":
            rf = load_pickle(RF_PKL)
            model = rf
            loaded_msg.append(f"Loaded `{RF_PKL}`.")
        elif model_name == "Decision Tree":
            dt = load_pickle(DT_PKL)
            model = dt
            loaded_msg.append(f"Loaded `{DT_PKL}`.")
        elif model_name == "AdaBoost":
            ada = load_pickle(ADA_PKL)
            model = ada
            loaded_msg.append(f"Loaded `{ADA_PKL}`.")
    except Exception as e:
        st.error(f"Could not load selected model: {e}")
        st.stop()
else:
    soft, soft_status = get_or_make_soft_voter(df_all)
    if soft is None:
        st.error(soft_status)
        st.stop()
    model = soft
    loaded_msg.append(soft_status)

st.caption(" ".join(loaded_msg))

if uploaded is not None:
    try:
        df_user = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

    if df_user.empty:
        st.warning("Uploaded CSV is empty.")
        st.stop()

    y_true = None
    if TARGET_COL in df_user.columns:
        y_true = df_user[TARGET_COL].astype(int).values
        X_user = df_user.drop(columns=[TARGET_COL])
    else:
        X_user = df_user.copy()

    try:
        X_enc = align_columns_to_model(X_user, model)
    except Exception as e:
        st.error(f"Column alignment failed: {e}")
        st.stop()

    try:
        yhat, proba = predict_with_proba(model, X_enc)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Predictions")
    st.markdown("**Legend:** ⬛ Normal (lime) ⬛ Suspect (yellow) ⬛ Pathological (orange)")
    results_df = add_predictions(df_user, yhat, proba)
    st.dataframe(style_pred_column(results_df), use_container_width=True)

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions as CSV",
        data=csv_bytes,
        file_name="fetal_health_predictions.csv",
        mime="text/csv"
    )

    st.subheader("Feature Importance")
    fi = feature_importance_for_model(model)
    if fi is not None:
        fig_fi = plot_feature_importance(fi, f"Feature Importance — {model_name}")
        st.pyplot(fig_fi)
        st.dataframe(fi.sort_values(ascending=False).to_frame("importance"), use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

    if y_true is not None:
        st.subheader("Evaluation (using provided ground truth)")
        rep_text, cm, rep_csv_like_rf = compute_report_and_cm(y_true, yhat)

        c1, c2 = st.columns([1,1])
        with c1:
            st.text("Classification Report")
            st.text(rep_text)
            rep_csv_bytes = rep_csv_like_rf.to_csv(index=True).encode("utf-8")
            st.download_button(
                "Download Classification Report (CSV)",
                data=rep_csv_bytes,
                file_name="classification_report.csv",
                mime="text/csv"
            )
        with c2:
            fig_cm = plot_confusion_matrix(cm, "Confusion Matrix")
            st.pyplot(fig_cm)
    else:
        assets = PRECOMP_ASSETS.get(model_name, {})
        any_asset = False
        st.subheader("Reference Metrics (from training/validation)")

        c1, c2 = st.columns([1,1])

        with c1:
            rep_csv = assets.get("report_csv")
            if rep_csv and os.path.exists(rep_csv):
                st.markdown("**Classification Report (reference)**")
                rep_df = pd.read_csv(rep_csv)
                st.dataframe(rep_df, use_container_width=True)
                st.download_button(
                    "Download Classification Report (CSV)",
                    data=rep_df.to_csv(index=False).encode("utf-8"),
                    file_name=os.path.basename(rep_csv),
                    mime="text/csv"
                )
                any_asset = True

            fi_img = assets.get("fi_img")
            if fi_img and os.path.exists(fi_img):
                st.markdown("**Feature Importance (reference)**")
                st.image(fi_img, use_container_width=True)
                with open(fi_img, "rb") as f:
                    st.download_button(
                        "Download Feature Importance",
                        data=f,
                        file_name=os.path.basename(fi_img)
                    )
                any_asset = True

        with c2:
            cm_img = assets.get("cm_img")
            if cm_img and os.path.exists(cm_img):
                st.markdown("**Confusion Matrix (reference)**")
                st.image(cm_img, use_container_width=True)
                with open(cm_img, "rb") as f:
                    st.download_button(
                        "Download Confusion Matrix",
                        data=f,
                        file_name=os.path.basename(cm_img)
                    )
                any_asset = True

        if not any_asset:
            st.info("No reference artifacts found. Provide a CSV with a `fetal_health` column to see live evaluation.")
else:
    st.info("Upload a CSV to generate predictions.")

st.markdown("---")
st.caption("Notes: Soft Voting weights are normalized F1-macro scores across RF, DT, and AdaBoost. Weighted feature importance for Soft Voting is computed as the weighted sum of base models’ importances using those weights.")
