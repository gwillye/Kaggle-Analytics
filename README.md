# ml_mini_analyses

A portfolio of small, end-to-end data-science analyses. (This repo used to be called Kaggle-Analytics. It was renamed because none of the analyses actually need Kaggle, they all run on free public datasets that need no API key.) The idea was to cover the breadth of applied DS, classification, regression, clustering, NLP, time series, recommendation and open-data EDA, with one compact analysis per problem type. Each one runs on a real public dataset that needs no Kaggle API key, and each is structured so you can drop a Kaggle dataset in later.

Two things mattered to me here. First, reproducibility: every script is self-contained and verified by actually running it. It prints a metric plus a self-check and saves a plot to `outputs/`. Second, honesty: all the metrics in this repo come from real runs. Nothing is invented or rounded up to look better.

Each analysis also has a full write-up in [`docs/`](docs/), a mini-article that covers the problem, the approach (and why I picked it), the real results with interpretation, and how to run it. The write-up links are in the table below.

## Analyses

| # | Problem type | Dataset | Source | Headline result (real run) | Write-up |
|---|---|---|---|---|---|
| 01 | Classification | Breast Cancer Wisconsin | scikit-learn (bundled) | RF ROC-AUC 0.995, acc 0.958 | [docs/01](docs/01_breast_cancer_classification.md) |
| 02 | Regression | Diabetes progression | scikit-learn (bundled) | Ridge R² 0.438 (beats GBoost) | [docs/02](docs/02_diabetes_regression.md) |
| 03 | Clustering (unsupervised) | Wine cultivars | scikit-learn (bundled) | KMeans ARI 0.897, silhouette 0.285 | [docs/03](docs/03_wine_clustering.md) |
| 04 | Regression + Classification | Wine Quality (red) | UCI (HTTPS download) | reg R² 0.497; good-wine AUC 0.947 | [docs/04](docs/04_wine_quality.md) |
| 05 | NLP text classification | 20 Newsgroups (4 topics) | sklearn fetch (cached) | TF-IDF+LogReg acc 0.888 | [docs/05](docs/05_newsgroups_text_classification.md) |
| 06 | Time series forecasting | Airline passengers | HTTPS download | log+Fourier MAPE 11.6% | [docs/06](docs/06_airline_timeseries.md) |
| 07 | Recommendation | MovieLens 100k | GroupLens (HTTPS download) | item-item CF RMSE 0.918 vs 1.021 base (-10.1%) | [docs/07](docs/07_movielens_recommender.md) |
| 08 | EDA / open gov data | Brazil indicators | World Bank API | GDP/cap CAGR 6.1% (235 to 10,310 US$) | [docs/08](docs/08_worldbank_brazil_eda.md) |
| 09 | Credit risk (classification) | German Credit (Statlog) | UCI (HTTPS download) | RF ROC-AUC 0.804 (30% bad-rate) | [docs/09](docs/09_credit_risk_classification.md) |
| 10 | Multiclass + viz | Iris | scikit-learn (bundled) | LogReg 5-fold CV acc 0.960 | [docs/10](docs/10_iris_multiclass.md) |

Each write-up references its plot in [`outputs/`](outputs/). Note that 07 writes a text result instead of a PNG.

## Why it's built this way

The analyses are intentionally small. They were chosen to cover the breadth of applied data science on trustworthy, key-free public data, and to show good practice rather than flashy numbers. That means stratified or chronological splits where they matter, the right metric for each problem (ROC-AUC under imbalance, MAPE for forecasts, ARI for clustering, a baseline comparison for the recommender), and honest interpretation of the limitations. The actual analysis of the results lives in the `docs/` mini-articles, so that's the place to dig in.

## Swapping in a Kaggle dataset

Every script runs on a public dataset with no key required, but each one is set up so you can point it at a Kaggle dataset instead:

1. Run `pip install kaggle` and place `kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`).
2. Uncomment or add the `kaggle datasets download -d <slug>` line at the top of the script.
3. Point the loader at the downloaded CSV. Each script keeps its data-loading step isolated, so this is a one-line change.

## Running it

```bash
pip install -r requirements.txt
py 01_breast_cancer_classification.py   # then 02_..., 03_..., etc.  (use 'python' if 'py' is unavailable)
```

Outputs (plots and result files) are written to `outputs/`. Each analysis is also provided as a Colab-ready notebook (`NN_*.ipynb`), so you can open it in Jupyter or Google Colab and run it top to bottom.

## Stack

Python, scikit-learn, pandas, NumPy, matplotlib, requests.
