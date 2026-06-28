"""07 — Recommendation: MovieLens 100k (real dataset, downloaded from GroupLens, no key).

Item-item collaborative filtering; RMSE on a held-out split vs an item-mean baseline.
Kaggle-ready: MovieLens is also on Kaggle — point the loader at the Kaggle CSV (see README).
"""
import os
import io
import zipfile
import numpy as np
import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=90).content))
ratings = pd.read_csv(z.open("ml-100k/u.data"), sep="\t",
                      names=["user", "item", "rating", "ts"])
print(f"ratings={len(ratings)} users={ratings.user.nunique()} items={ratings.item.nunique()}")

rng = np.random.default_rng(42)
test_mask = rng.random(len(ratings)) < 0.2
train, test = ratings[~test_mask], ratings[test_mask]

n_u = ratings.user.max() + 1
n_i = ratings.item.max() + 1
M = np.full((n_u, n_i), np.nan)
M[train.user, train.item] = train.rating

item_mean = np.nanmean(M, axis=0)
g = np.nanmean(M)
item_mean = np.where(np.isnan(item_mean), g, item_mean)
centered = np.where(np.isnan(M), 0.0, M - item_mean)
norms = np.linalg.norm(centered, axis=0)
norms[norms == 0] = 1.0
sim = (centered.T @ centered) / np.outer(norms, norms)
np.fill_diagonal(sim, 0.0)

# keep top-30 neighbors per item for speed/quality
K = 30
preds, truth, base = [], [], []
M_user_rated = ~np.isnan(M)
for u, i, r in zip(test.user.to_numpy(), test.item.to_numpy(), test.rating.to_numpy()):
    rated = np.where(M_user_rated[u])[0]
    if rated.size == 0:
        p = item_mean[i]
    else:
        s = sim[i, rated]
        top = np.argsort(s)[-K:]
        s_top = s[top]
        denom = np.abs(s_top).sum()
        p = item_mean[i] + (s_top @ (M[u, rated[top]] - item_mean[rated[top]])) / denom if denom > 0 else item_mean[i]
    preds.append(np.clip(p, 1, 5)); truth.append(r); base.append(item_mean[i])

preds, truth, base = np.array(preds), np.array(truth), np.array(base)
rmse_cf = np.sqrt(np.mean((preds - truth) ** 2))
rmse_base = np.sqrt(np.mean((base - truth) ** 2))

with open(os.path.join(OUT, "07_movielens_results.txt"), "w", encoding="utf-8") as f:
    f.write(f"MovieLens 100k — item-item CF (top-{K} neighbors)\n")
    f.write(f"RMSE CF       = {rmse_cf:.3f}\nRMSE baseline = {rmse_base:.3f}\n")
    f.write(f"improvement   = {(rmse_base - rmse_cf) / rmse_base:.1%}\n")

print(f"RMSE_cf={rmse_cf:.3f} RMSE_base={rmse_base:.3f} improve={(rmse_base-rmse_cf)/rmse_base:.1%}")
print("self-check:", "OK" if rmse_cf < rmse_base else "FAIL")
