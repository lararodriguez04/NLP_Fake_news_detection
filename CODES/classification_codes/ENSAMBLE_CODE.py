import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# =============================================================================
# CONFIG
# =============================================================================

# MODEL_SCRIPTS = {
#     "roberta_head_tail": "/home/lnlpG08/nlp/SLURMS/run_roberta_headtail.sh",
#     "roberta_swap_ht": "/home/lnlpG08/nlp/SLURMS/run_roberta_headtail_augment3.sh",
#     "bert_rand_ht":    "/home/lnlpG08/nlp/SLURMS/run_bert_headtail_augment3.sh",
#     "roberta_3augs_ht": "/home/lnlpG08/nlp/SLURMS/run_roberta_3aug_headtail.sh",}

MODEL_OUTPUTS = {
    "roberta_head_tail":
        "/home/lnlpG08/nlp/RESULTS/predictions_roberta_headtail.csv",

    "roberta_swap_ht":
        "/home/lnlpG08/nlp/RESULTS/predictions_roberta_headtail_augment3.csv",
    "roberta_3augs_ht":
        "/home/lnlpG08/nlp/RESULTS/predictions_roberta_3aug_headtail.csv",
}

LABEL_COLS = [
    "prob_true",
    "prob_false",
    "prob_mixture",
    "prob_unproven"
]

CLASS_NAMES = [
    "true",
    "false",
    "mixture",
    "unproven"
]

# =============================================================================
# 1. RUN MODELS
# =============================================================================

# print("=" * 80)
# print("RUNNING MODELS")
# print("=" * 80)

# processes = {}

# for model_name, script_path in MODEL_SCRIPTS.items():

#     print(f"\nLaunching {model_name}")

#     p = subprocess.Popen(
#         ["bash", script_path]
#     )

#     processes[model_name] = p

# # =============================================================================
# # 2. WAIT FOR ALL
# # =============================================================================

# for model_name, p in processes.items():

#     return_code = p.wait()

#     print(f"{model_name} finished with code {return_code}")

#     if return_code != 0:
#         raise RuntimeError(
#             f"{model_name} failed"
#         )

# print("\nAll models finished.\n")

# =============================================================================
# 3. LOAD PREDICTIONS
# =============================================================================

dfs = {
    name: pd.read_csv(path)
    for name, path in MODEL_OUTPUTS.items()
}

labels_true = dfs[list(dfs.keys())[0]]["label"].values

# =============================================================================
# 4. BUILD PROB MATRIX
# =============================================================================

# =============================================================================
# PROB MATRIX
# =============================================================================

prob_matrix = np.stack(
    [df[LABEL_COLS].values for df in dfs.values()],
    axis=0
)

# =============================================================================
# 1. SOFT VOTING
# =============================================================================

avg_probs = prob_matrix.mean(axis=0)
preds_soft = np.argmax(avg_probs, axis=-1)

# =============================================================================
# 2. WEIGHTED SOFT VOTING
# =============================================================================

WEIGHTS = {
    "roberta_head_tail": [0.90, 0.79, 0.55, 0.43],
    "roberta_swap_ht": [0.88, 0.75, 0.50, 0.49],
    "roberta_3augs_ht":   [0.88, 0.79, 0.53, 0.44],
}

weighted_probs = np.zeros_like(avg_probs)

for i, name in enumerate(dfs.keys()):
    if name in WEIGHTS:
        weighted_probs += prob_matrix[i] * np.array(WEIGHTS[name])

weighted_probs /= np.sum(
    [WEIGHTS[n] for n in dfs.keys() if n in WEIGHTS],
    axis=0
)

preds_weighted = np.argmax(weighted_probs, axis=-1)

# =============================================================================
# 3. HARD VOTING
# =============================================================================

hard_preds = np.stack(
    [np.argmax(df[LABEL_COLS].values, axis=-1) for df in dfs.values()],
    axis=0
)

def majority_vote(votes):
    return np.bincount(votes, minlength=4).argmax()

preds_hard = np.apply_along_axis(majority_vote, 0, hard_preds)

# =============================================================================
# 4. 🧠 CLASS-WEIGHTED CALIBRATION
# =============================================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1,2,3]),
    y=labels_true
)

class_weights = class_weights / class_weights.sum()

calibrated_probs = avg_probs * class_weights
preds_calibrated = np.argmax(calibrated_probs, axis=-1)

# =============================================================================
# 5. 🧩 STACKING MODEL
# =============================================================================

X = prob_matrix.transpose(1, 0, 2).reshape(len(labels_true), -1)
y = labels_true

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

meta_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    multi_class="multinomial"
)

meta_model.fit(X_train, y_train)

preds_stack = meta_model.predict(X_val)

# =============================================================================
# 6. 🎯 PER-CLASS ENSEMBLE SELECTION
# =============================================================================

model_names = list(dfs.keys())
best_model_per_class = []

for cls in range(4):

    best_f1 = 0
    best_model = 0

    for i, name in enumerate(model_names):

        preds = np.argmax(prob_matrix[i], axis=1)

        f1 = f1_score(labels_true, preds, labels=[cls], average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_model = i

    best_model_per_class.append(best_model)

preds_per_class = np.zeros(len(labels_true), dtype=int)

for i in range(len(labels_true)):

    scores = []

    for cls in range(4):
        m = best_model_per_class[cls]
        scores.append(prob_matrix[m, i, cls])

    preds_per_class[i] = np.argmax(scores)

# =============================================================================
# METRICS
# =============================================================================

def metrics(name, y_true, y_pred):

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2,3], zero_division=0
    )

    return {
        "method": name,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        **{f"{c}_f1": f1[i] for i, c in enumerate(CLASS_NAMES)}
    }

results = [
    metrics("soft_voting", labels_true, preds_soft),
    metrics("weighted_soft_voting", labels_true, preds_weighted),
    metrics("calibrated", labels_true, preds_calibrated),
    metrics("hard_voting", labels_true, preds_hard),
    metrics("stacking_model", y_val, preds_stack),
    metrics("per_class_ensemble", labels_true, preds_per_class),
]

results_df = pd.DataFrame(results)

# =============================================================================
# OUTPUT
# =============================================================================

print("\n=== FINAL RESULTS ===")
print(results_df.round(4))

Path("outputs").mkdir(exist_ok=True)
results_df.to_csv("outputs/ensemble_advanced_results.csv", index=False)

print("\nSaved -> outputs/ensemble_advanced_results.csv")