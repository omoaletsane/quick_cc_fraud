import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Config (fast, simple)
# ---------------------------
CSV_PATH = os.environ.get("CREDITCARD_CSV", "creditcard.csv")
OUT_DIR = Path("outputs_fast"); OUT_DIR.mkdir(exist_ok=True)

# Downsample majority class to this many rows (string env for Cloud Shell)
SAMPLE_NONFRAUD = int(os.environ.get("SAMPLE_NONFRAUD", "15000"))  # keep this many non-fraud
CV_FOLDS = int(os.environ.get("CV_FOLDS", "3"))
ULTRA_FAST = os.environ.get("ULTRA_FAST", "0") == "1"  # if true: no CV; single fit per model

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------
# Load
# ---------------------------
t0 = time.time()
print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Expect columns: Time, V1..V28, Amount, Class
required_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------------------------
# Downsample majority (fast)
# ---------------------------
fraud = df[df["Class"] == 1]
nonfraud = df[df["Class"] == 0]

# keep all fraud; downsample non-fraud
nonfraud_sampled = nonfraud.sample(n=min(SAMPLE_NONFRAUD, len(nonfraud)),
                                   random_state=RANDOM_STATE)
df_small = pd.concat([fraud, nonfraud_sampled], axis=0).sample(frac=1, random_state=RANDOM_STATE)
print(f"Data size (downsampled): {df_small.shape}, fraud={fraud.shape[0]}, nonfraud_kept={nonfraud_sampled.shape[0]}")

# ---------------------------
# Split
# ---------------------------
X = df_small.drop(columns=["Class"])
y = df_small["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------
# Preprocess (scale Time, Amount)
# V1..V28 are PCA-like already
# ---------------------------
num_cols_to_scale = ["Time", "Amount"]
other_cols = [c for c in X.columns if c not in num_cols_to_scale]

preprocess = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), num_cols_to_scale),
        ("keep", "passthrough", other_cols),
    ]
)

# ---------------------------
# Models (fast)
# ---------------------------
logreg = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
dtree  = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)

pipe_lr = Pipeline([("prep", preprocess), ("clf", logreg)])
pipe_dt = Pipeline([("prep", preprocess), ("clf", dtree)])

if ULTRA_FAST:
    # no CV, single fit
    models = {
        "LogReg": pipe_lr.set_params(clf__C=1.0),
        "DecisionTree": pipe_dt.set_params(clf__max_depth=5, clf__min_samples_split=20),
    }
else:
    # tiny CV grids (fast)
    grid_lr = {
        "clf__C": [0.3, 1.0, 3.0]
    }
    grid_dt = {
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_split": [10, 20]
    }

# ---------------------------
# Train / Tune
# ---------------------------
results = []

def fit_and_eval(model_name, estimator):
    # Fit
    print(f"=== {model_name} ===")
    estimator.fit(X_train, y_train)
    # Predict
    y_pred = estimator.predict(X_test)
    # Report
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    row = {
        "Model": model_name,
        "Precision_1": rep["1"]["precision"],
        "Recall_1":    rep["1"]["recall"],
        "F1_1":        rep["1"]["f1-score"],
        "Precision_0": rep["0"]["precision"],
        "Recall_0":    rep["0"]["recall"],
        "F1_0":        rep["0"]["f1-score"],
        "Accuracy":    rep["accuracy"],
    }
    results.append(row)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{model_name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

if ULTRA_FAST:
    for name, est in models.items():
        fit_and_eval(name, est)
else:
    # Small CV (fast)
    print(f"GridSearchCV with cv={CV_FOLDS}")
    gs_lr = GridSearchCV(pipe_lr, grid_lr, scoring="f1", cv=CV_FOLDS, n_jobs=-1)
    gs_dt = GridSearchCV(pipe_dt, grid_dt, scoring="f1", cv=CV_FOLDS, n_jobs=-1)

    for name, gs in [("LogReg", gs_lr), ("DecisionTree", gs_dt)]:
        print(f"Tuning {name} …")
        gs.fit(X_train, y_train)
        print(f"Best params for {name}: {gs.best_params_}")
        # Evaluate best
        best = gs.best_estimator_
        fit_and_eval(name, best)

# ---------------------------
# Save table
# ---------------------------
df_res = pd.DataFrame(results)
df_res = df_res[["Model", "Precision_1", "Recall_1", "F1_1", "Precision_0", "Recall_0", "F1_0", "Accuracy"]]
df_res.to_csv(OUT_DIR / "results_fast.csv", index=False)
print("\n=== Results ===")
print(df_res.to_string(index=False))

elapsed = time.time() - t0
with open(OUT_DIR / "run_info.txt", "w") as f:
    f.write(f"Elapsed_seconds: {elapsed:.1f}\n")
    f.write(f"SAMPLE_NONFRAUD: {SAMPLE_NONFRAUD}\n")
    f.write(f"CV_FOLDS: {CV_FOLDS}\n")
    f.write(f"ULTRA_FAST: {ULTRA_FAST}\n")

print(f"\nSaved to: {OUT_DIR.resolve()}")
