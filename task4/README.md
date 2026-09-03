# Task 4: Classification with Logistic Regression
**AI & ML Internship — Elevate Labs**

## Objective
Build a binary classifier using logistic regression.

## Tools Used
Python, Scikit-learn, Pandas, Matplotlib, Seaborn

## Dataset
**Breast Cancer Wisconsin (Diagnostic) dataset** — the real dataset, loaded
directly from `sklearn.datasets.load_breast_cancer` and exported to
`data/breast_cancer.csv`. 569 samples, 30 numeric features, binary target
(`0 = malignant`, `1 = benign`).

## Project Structure
```
├── data/
│   └── breast_cancer.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── sigmoid_function.png
│   └── threshold_tuning.png
├── export_dataset.py
├── logistic_regression.py
├── threshold_comparison.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python export_dataset.py       # only needed if data/breast_cancer.csv doesn't exist
python logistic_regression.py
```

## What Was Done

1. **Dataset:** Breast Cancer Wisconsin — a genuinely binary, real-world
   diagnostic dataset, well suited for logistic regression.
2. **Train/test split (80/20, stratified)** and **standardization** —
   features were scaled with `StandardScaler` (fit on train, applied to
   test) since logistic regression is a linear, gradient-based model and
   features here span very different scales (e.g. "mean area" in the
   hundreds vs. "mean smoothness" around 0.1).
3. **Fit** `sklearn.linear_model.LogisticRegression`.
4. **Evaluated** with a confusion matrix, precision, recall, F1, and
   ROC-AUC, plus the full classification report. Also plotted the
   Precision-Recall curve as a complementary view.
5. **Explained and plotted the sigmoid function**, and **tuned the
   decision threshold** across several values to see the precision/recall
   trade-off directly.

## Results (default threshold = 0.5)

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.9825 |
| Precision   | 0.9861 |
| Recall      | 0.9861 |
| F1 Score    | 0.9861 |
| ROC-AUC     | 0.9954 |

Confusion matrix: only 1 malignant case misclassified as benign, and 1
benign case misclassified as malignant, out of 114 test samples.

**Threshold tuning:** lowering the threshold to 0.3 pushed recall to
**1.000** (catching every malignant case in the test set) at a small
precision cost. In a medical screening context, missing a malignant tumor
(false negative) is typically far more costly than a false alarm, so a
lower threshold favoring recall is often the more sensible operating
point — not necessarily the default 0.5. See `threshold_comparison.csv`
and `images/threshold_tuning.png` for the full trade-off curve.

---

## Interview Questions & Answers

**1. How does logistic regression differ from linear regression?**
Linear regression predicts a **continuous** numeric value and fits a
straight line to the data. Logistic regression is used for
**classification** — it applies the sigmoid function to a linear
combination of the inputs so the output is a probability between 0 and 1,
which is then converted to a class label via a threshold. Their loss
functions differ too: linear regression uses squared error, logistic
regression uses log loss (cross-entropy), which is better suited to
probability outputs.

**2. What is the sigmoid function?**
The sigmoid function is `σ(z) = 1 / (1 + e^-z)`. It takes any real number
`z` (here, the linear combination `w·x + b` of the input features) and
squashes it into a value between 0 and 1, which is interpreted as the
predicted probability of the positive class. It's S-shaped, roughly
linear near `z = 0`, and flattens out toward 0 and 1 at the extremes.

**3. What is precision vs recall?**
**Precision** = TP / (TP + FP) — of everything the model *predicted* as
positive, what fraction was actually positive. It matters when false
positives are costly. **Recall** (sensitivity) = TP / (TP + FN) — of
everything that *actually was* positive, what fraction did the model
catch. It matters when false negatives are costly (like missing a
malignant tumor). The two typically trade off against each other as the
classification threshold changes.

**4. What is the ROC-AUC curve?**
The ROC (Receiver Operating Characteristic) curve plots the True Positive
Rate against the False Positive Rate at every possible classification
threshold. AUC (Area Under the Curve) summarizes that curve into a single
number between 0 and 1 — 0.5 means no better than random guessing, 1.0
means perfect separation between classes. It's useful because it
evaluates the model's ranking ability across *all* thresholds at once,
independent of which single threshold you eventually pick.

**5. What is the confusion matrix?**
A table that breaks down a classifier's predictions into four counts:
**True Positives** (correctly predicted positive), **True Negatives**
(correctly predicted negative), **False Positives** (predicted positive,
actually negative), and **False Negatives** (predicted negative, actually
positive). Every other classification metric (accuracy, precision,
recall, F1) is derived from these four numbers.

**6. What happens if classes are imbalanced?**
Accuracy becomes a misleading metric — a model that always predicts the
majority class can still score high accuracy while being useless for the
minority class. Precision, recall, F1, and ROC-AUC (or better,
Precision-Recall AUC for severe imbalance) give a much more honest
picture. Fixes include resampling (oversampling the minority class,
undersampling the majority, or SMOTE), using `class_weight='balanced'` in
the model, or choosing a lower/higher decision threshold to compensate.

**7. How do you choose the threshold?**
The default is 0.5, but the right threshold depends on the relative cost
of false positives vs. false negatives in your specific problem. You can
plot precision/recall (or F1) against threshold, as done in this task, and
pick the point that best matches your priorities — e.g. a lower threshold
to maximize recall in a medical screening context, or a higher threshold
to maximize precision when false alarms are expensive. The ROC or
Precision-Recall curve is often used to visualize this trade-off before
deciding.

**8. Can logistic regression be used for multi-class problems?**
Yes. The most common extensions are **One-vs-Rest (OvR)**, where a
separate binary classifier is trained per class against all others, and
**Multinomial (Softmax) logistic regression**, which directly generalizes
the sigmoid to multiple classes using the softmax function so all class
probabilities sum to 1. Scikit-learn's `LogisticRegression` supports both
via the `multi_class` parameter.
