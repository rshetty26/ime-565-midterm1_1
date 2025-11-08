import numpy as np
import pandas as pd
import pickle, json, os, sys, warnings
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
TARGET_COL = "fetal_health"

DATA_PATH = "fetal_health.csv"
RF_PATH   = "random_forest.pickle"
DT_PATH   = "decision_tree.pickle"
ADA_PATH  = "adaboost.pickle"

OUT_SOFT_VOTER  = "soft_voter.pickle"
OUT_WEIGHTS     = "model_weights.json"
OUT_FEATURE_IMP = "weighted_feature_importance.csv"
OUT_REPORT      = "val_class_report.txt"
OUT_REPORT_CSV  = "sv_class_report.csv"
OUT_CM_PNG      = "sv_confusion_mat.png"
OUT_FI_PNG      = "sv_feature_imp.png"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE  = 0.15

def is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return hasattr(model, "classes_")

def load_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)

def make_splits(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_fraction, random_state=RANDOM_SEED, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def eval_f1_macro(model, X_val, y_val, X_train, y_train):
    m = model
    if not is_fitted(model):
        m = clone(model).fit(X_train, y_train)
    preds = m.predict(X_val)
    return f1_score(y_val, preds, average="macro")

if not os.path.exists(DATA_PATH):
    print(f("[ERROR] Could not find {DATA_PATH}"))
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
if TARGET_COL not in df.columns:
    print(f"[ERROR] Column '{TARGET_COL}' missing from dataset.")
    sys.exit(1)

X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df)

rf  = load_model(RF_PATH)
dt  = load_model(DT_PATH)
ada = load_model(ADA_PATH)

f1_rf  = eval_f1_macro(rf,  X_val, y_val, X_train, y_train)
f1_dt  = eval_f1_macro(dt,  X_val, y_val, X_train, y_train)
f1_ada = eval_f1_macro(ada, X_val, y_val, X_train, y_train)

f1s = np.array([f1_rf, f1_dt, f1_ada])
weights = f1s / f1s.sum()

print("\nF1 scores (rf, dt, ada):", f1s)
print("Normalized weights:", weights)

rf_fresh  = clone(rf)
dt_fresh  = clone(dt)
ada_fresh = clone(ada)

voter = VotingClassifier(
    estimators=[("rf", rf_fresh), ("dt", dt_fresh), ("ada", ada_fresh)],
    voting="soft",
    weights=weights.tolist()
)
voter.fit(X_train, y_train)

with open(OUT_SOFT_VOTER, "wb") as f:
    pickle.dump(voter, f)
print(f"\nSaved soft voter -> {OUT_SOFT_VOTER}")

y_val_pred = voter.predict(X_val)
report = classification_report(y_val, y_val_pred, digits=4)
cm = confusion_matrix(y_val, y_val_pred, labels=[1,2,3])
with open(OUT_REPORT, "w") as f:
    f.write("Validation Classification Report (Soft Voting Classifier)\n")
    f.write(report + "\n\nConfusion Matrix (rows=true, cols=pred):\n")
    f.write(np.array2string(cm))
print(f"Saved validation report -> {OUT_REPORT}")

acc = accuracy_score(y_val, y_val_pred)
rep = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
metrics = ["precision", "recall", "f1-score", "support"]
class_keys = [k for k in rep.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
try:
    class_keys = sorted(class_keys, key=lambda x: float(x))
except:
    class_keys = sorted(class_keys)
rows = {m: [] for m in metrics}
for ck in class_keys:
    rows["precision"].append(rep[ck]["precision"])
    rows["recall"].append(rep[ck]["recall"])
    rows["f1-score"].append(rep[ck]["f1-score"])
    rows["support"].append(float(rep[ck]["support"]))
macro = rep.get("macro avg", {"precision": None, "recall": None, "f1-score": None})
weighted = rep.get("weighted avg", {"precision": None, "recall": None, "f1-score": None})
total_support = float(sum(rows["support"]))
rows["precision"].extend([acc, macro["precision"], weighted["precision"]])
rows["recall"].extend([acc, macro["recall"], weighted["recall"]])
rows["f1-score"].extend([acc, macro["f1-score"], weighted["f1-score"]])
rows["support"].extend([acc, total_support, total_support])
class_cols = [f"{float(k):.1f}" for k in class_keys]
cols = class_cols + ["accuracy", "macro avg", "weighted avg"]
report_like_rf = pd.DataFrame(rows, index=cols).T
report_like_rf.to_csv(OUT_REPORT_CSV, index=True)
print(f"Saved classification report CSV -> {OUT_REPORT_CSV}")

weights_info = {
    "rf_f1": float(f1_rf),
    "dt_f1": float(f1_dt),
    "ada_f1": float(f1_ada),
    "normalized_weights": weights.tolist()
}
with open(OUT_WEIGHTS, "w") as f:
    json.dump(weights_info, f, indent=2)
print(f"Saved weights -> {OUT_WEIGHTS}")

feature_names = X_train.columns
importances = np.vstack([
    voter.named_estimators_["rf"].feature_importances_,
    voter.named_estimators_["dt"].feature_importances_,
    voter.named_estimators_["ada"].feature_importances_
])
weighted_importance = (weights[:, None] * importances).sum(axis=0)
fi_df = pd.DataFrame({
    "feature": feature_names,
    "weighted_importance": weighted_importance
}).sort_values("weighted_importance", ascending=False)
fi_df.to_csv(OUT_FEATURE_IMP, index=False)
print(f"Saved weighted feature importance -> {OUT_FEATURE_IMP}")

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Soft Voting)")
plt.colorbar()
tick_labels = ["Normal","Suspect","Pathological"]
tick_marks = np.arange(len(tick_labels))
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(OUT_CM_PNG, dpi=200)
plt.close()
print(f"Saved confusion matrix plot -> {OUT_CM_PNG}")

topk = len(fi_df)
plt.figure(figsize=(8, max(5, 0.25*topk)))
plt.barh(fi_df["feature"], fi_df["weighted_importance"])
plt.gca().invert_yaxis()
plt.title("Weighted Feature Importance (Soft Voting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT_FI_PNG, dpi=200)
plt.close()
print(f"Saved feature importance plot -> {OUT_FI_PNG}")
