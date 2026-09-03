"""
Task 4: Classification with Logistic Regression
AI & ML Internship - Elevate Labs

Objective: Build a binary classifier using logistic regression.
Dataset: Breast Cancer Wisconsin (Diagnostic) dataset (data/breast_cancer.csv)
  Target: 0 = malignant, 1 = benign

Steps covered:
1. Choose a binary classification dataset
2. Train/test split and standardize features
3. Fit a Logistic Regression model
4. Evaluate with confusion matrix, precision, recall, ROC-AUC
5. Tune threshold and explain the sigmoid function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# STEP 1: Load the binary classification dataset
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOAD DATASET")
print("=" * 60)

df = pd.read_csv("data/breast_cancer.csv")
print("\nShape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nClass balance (0=malignant, 1=benign):\n", df["target"].value_counts())

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------------------------------
# STEP 2: Train/test split and standardize features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: TRAIN/TEST SPLIT & STANDARDIZATION")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Standardization matters a lot for logistic regression since it's a
# distance/gradient-based linear model - features on very different
# scales (e.g. 'mean area' ~ hundreds vs 'mean smoothness' ~ 0.1) would
# otherwise dominate the loss and slow/bias convergence.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized to mean=0, std=1 (fit on train, applied to test).")

# ---------------------------------------------------------------
# STEP 3: Fit a Logistic Regression model
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: FIT LOGISTIC REGRESSION")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_default = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 (benign)

print("\nModel fitted. Number of features:", X_train.shape[1])

# ---------------------------------------------------------------
# STEP 4: Evaluate - confusion matrix, precision, recall, ROC-AUC
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: EVALUATION (default threshold = 0.5)")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred_default)
acc = accuracy_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default)
rec = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nConfusion Matrix:\n", cm)
print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred_default,
      target_names=["malignant", "benign"]))

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=120)
plt.close()
print("\nSaved images/confusion_matrix.png")

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_curve.png", dpi=120)
plt.close()
print("Saved images/roc_curve.png")

# Precision-Recall curve (useful complement to ROC, especially if imbalanced)
prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(rec_curve, prec_curve, color="seagreen", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("images/precision_recall_curve.png", dpi=120)
plt.close()
print("Saved images/precision_recall_curve.png")

# ---------------------------------------------------------------
# STEP 5: Explain and plot the sigmoid function
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5a: SIGMOID FUNCTION")
print("=" * 60)

print(
    "\nLogistic regression computes z = w.x + b (a linear combination of features), "
    "then passes z through the sigmoid function: sigmoid(z) = 1 / (1 + e^-z), "
    "which squashes any real number into a probability between 0 and 1. "
    "A threshold (default 0.5) is then applied to that probability to get the final "
    "class label."
)

z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.figure(figsize=(7, 5))
plt.plot(z, sigmoid, color="purple", linewidth=2)
plt.axhline(0.5, color="gray", linestyle="--", label="Default threshold = 0.5")
plt.axvline(0, color="gray", linestyle=":")
plt.xlabel("z (linear combination: w.x + b)")
plt.ylabel("sigmoid(z) = predicted probability")
plt.title("The Sigmoid Function")
plt.legend()
plt.tight_layout()
plt.savefig("images/sigmoid_function.png", dpi=120)
plt.close()
print("Saved images/sigmoid_function.png")

# ---------------------------------------------------------------
# STEP 5b: Tune the classification threshold
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5b: THRESHOLD TUNING")
print("=" * 60)

thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for t in thresholds_to_try:
    y_pred_t = (y_proba >= t).astype(int)
    results.append({
        "Threshold": t,
        "Accuracy": accuracy_score(y_test, y_pred_t),
        "Precision": precision_score(y_test, y_pred_t),
        "Recall": recall_score(y_test, y_pred_t),
        "F1": f1_score(y_test, y_pred_t),
    })

results_df = pd.DataFrame(results)
print("\nMetric comparison across thresholds:\n", results_df.to_string(index=False))
results_df.to_csv("threshold_comparison.csv", index=False)
print("\nSaved threshold_comparison.csv")

print(
    "\nInterpretation: lowering the threshold below 0.5 predicts 'benign' less readily, "
    "which increases recall for the malignant class (fewer missed malignant cases) at "
    "the cost of precision (more benign cases wrongly flagged as malignant). In a medical "
    "screening context like this one, missing a malignant tumor (false negative) is usually "
    "far more costly than a false alarm, so a lower threshold favoring recall is often preferred."
)

plt.figure(figsize=(8, 5))
plt.plot(results_df["Threshold"], results_df["Precision"], marker="o", label="Precision")
plt.plot(results_df["Threshold"], results_df["Recall"], marker="o", label="Recall")
plt.plot(results_df["Threshold"], results_df["F1"], marker="o", label="F1 Score")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Classification Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("images/threshold_tuning.png", dpi=120)
plt.close()
print("Saved images/threshold_tuning.png")
