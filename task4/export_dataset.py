"""
export_dataset.py
------------------
Exports the real Breast Cancer Wisconsin (Diagnostic) dataset, which
ships built into scikit-learn (`sklearn.datasets.load_breast_cancer`),
to a CSV file so it's included in this repo as data/breast_cancer.csv.

Target encoding (as defined by sklearn):
  0 = malignant
  1 = benign
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

df.to_csv("data/breast_cancer.csv", index=False)
print("Saved data/breast_cancer.csv with shape:", df.shape)
print("Target classes:", dict(enumerate(data.target_names)))
print("Class balance:\n", df["target"].value_counts())
