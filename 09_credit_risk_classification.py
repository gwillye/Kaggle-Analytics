"""09 — Credit-risk classification: Statlog German Credit (UCI, downloaded, no key).

Predicts good/bad credit — a finance/risk problem (the domain of the Neoway work).
Kaggle-ready: German Credit is also on Kaggle — point the loader at the Kaggle CSV (see README).
"""
import os
import io
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"

raw = requests.get(URL, timeout=30).text
df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None)
X = df.iloc[:, :-1].to_numpy(dtype=float)
y = (df.iloc[:, -1].to_numpy() == 2).astype(int)  # class 2 == bad credit → positive
print(f"loaded {len(df)} rows, {X.shape[1]} features, bad-rate {y.mean():.1%}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
clf = RandomForestClassifier(n_estimators=400, random_state=SEED, class_weight="balanced").fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, proba)
print(classification_report(yte, clf.predict(Xte), digits=3))

imp = clf.feature_importances_
idx = np.argsort(imp)[-10:]
plt.figure(figsize=(7, 5))
plt.barh([f"f{j}" for j in idx], imp[idx], color="#8c564b")
plt.title(f"German Credit — top risk features (ROC-AUC {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_credit_features.png"), dpi=110)
plt.close()

print(f"ROC-AUC = {auc:.3f}")
print("self-check:", "OK" if auc > 0.70 else "FAIL")
