"""06 — Time series: monthly airline passengers (classic dataset, downloaded, no key).

Log-transform + trend & seasonal (Fourier) features → forecast a held-out tail; report MAPE.
Kaggle-ready: point the loader at a Kaggle time-series CSV (date, value) — see README.
"""
import os
import io
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(io.StringIO(requests.get(URL, timeout=30).text))
y = df["Passengers"].to_numpy(dtype=float)
t = np.arange(len(y))
ly = np.log(y)  # multiplicative seasonality → additive in log space


def feats(t):
    return np.column_stack([
        t,
        np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12),
        np.sin(2 * np.pi * t / 6), np.cos(2 * np.pi * t / 6),
    ])


split = int(len(y) * 0.8)
model = LinearRegression().fit(feats(t[:split]), ly[:split])
pred = np.exp(model.predict(feats(t[split:])))
mape = mean_absolute_percentage_error(y[split:], pred)

plt.figure(figsize=(10, 4))
plt.plot(t[:split], y[:split], color="#4c78a8", label="train")
plt.plot(t[split:], y[split:], color="#222", label="actual")
plt.plot(t[split:], pred, "--", color="#e45756", label="forecast")
plt.axvline(split, color="gray", ls=":")
plt.title(f"Airline passengers — forecast MAPE {mape:.1%}")
plt.xlabel("month index")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_airline_forecast.png"), dpi=110)
plt.close()

print(f"n={len(y)} MAPE={mape:.1%}")
print("self-check:", "OK" if mape < 0.12 else "FAIL")
