"""08 — EDA on open government/economic data: Brazil indicators via the World Bank API.

No key required. Pulls a few socioeconomic indicators for Brazil, plots trends and
computes growth (CAGR). On-brand for the government/public-sector data work.
Kaggle-ready: swap for an IBGE/gov.br or Kaggle CSV (see README).
"""
import os
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

INDICATORS = {
    "GDP per capita (US$)": "NY.GDP.PCAP.CD",
    "Life expectancy (yrs)": "SP.DYN.LE00.IN",
    "Internet users (%)": "IT.NET.USER.ZS",
}


def fetch(code):
    url = f"https://api.worldbank.org/v2/country/BRA/indicator/{code}?format=json&per_page=300"
    j = requests.get(url, timeout=30).json()
    rows = [(int(d["date"]), d["value"]) for d in j[1] if d["value"] is not None]
    rows.sort()
    years = np.array([r[0] for r in rows])
    vals = np.array([r[1] for r in rows], dtype=float)
    return years, vals


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
summary = []
for ax, (label, code) in zip(axes, INDICATORS.items()):
    yrs, vals = fetch(code)
    ax.plot(yrs, vals, marker=".", color="#117733")
    ax.set_title(label, fontsize=10)
    cagr = (vals[-1] / vals[0]) ** (1 / (yrs[-1] - yrs[0])) - 1 if vals[0] > 0 else float("nan")
    summary.append(f"{label}: {vals[0]:.1f} ({yrs[0]}) -> {vals[-1]:.1f} ({yrs[-1]}), CAGR {cagr:.1%}")
plt.suptitle("Brazil — World Bank indicators")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_brazil_indicators.png"), dpi=110)
plt.close()

with open(os.path.join(OUT, "08_brazil_results.txt"), "w", encoding="utf-8") as f:
    f.write("Brazil socioeconomic indicators (World Bank)\n\n" + "\n".join(summary) + "\n")

print("\n".join(summary))
print("self-check:", "OK" if len(summary) == 3 else "FAIL")
