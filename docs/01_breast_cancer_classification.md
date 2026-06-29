# 01 - Breast Cancer Classification (Random Forest)

Headline result: ROC-AUC = 0.995, accuracy = 0.958 on a held-out 25% test set.

## Problem

The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 fine-needle-aspirate samples, each described by 30 numeric features computed from digitized cell-nucleus images (radius, texture, perimeter, area, concavity, and so on). Each sample is labelled malignant or benign, and the dataset is mildly imbalanced, roughly 37% malignant and 63% benign.

The question is a textbook but high-stakes one: can we separate malignant from benign tumors from cell-morphology measurements alone, and which measurements drive that decision? This matters because a model that is both accurate and interpretable can act as a triage or second-opinion aid, and because the feature-importance ranking is itself clinically informative.

## Approach

The model is a Random Forest with 300 trees and `random_state=42`. It was chosen deliberately over a single decision tree or a linear model for a few reasons. First, it is robust to the strong multicollinearity in this dataset, since many features are different summaries of the same geometric quantity (mean, standard-error, and "worst" of radius, perimeter, and area). Second, it needs no feature scaling. Third, it gives an out-of-the-box, ranked feature-importance view that we use for interpretation.

No preprocessing is required. Trees are scale-invariant and there are no missing values, so we feed the 30 raw features directly.

Validation is a single stratified 75/25 train-test split. Stratification preserves the class ratio in both halves so the test ROC-AUC is not distorted by an unlucky draw. We report ROC-AUC (threshold-independent ranking quality, the right metric for an imbalanced screening problem) plus the full precision/recall/F1 report.

## Results and analysis

Running `py 01_breast_cancer_classification.py` produces:

```
              precision    recall  f1-score   support
           0      0.961     0.925     0.942        53   (malignant)
           1      0.957     0.978     0.967        90   (benign)
    accuracy                          0.958       143
ROC-AUC = 0.995
```

Top features by importance are plotted in [`outputs/01_breast_cancer_features.png`](../outputs/01_breast_cancer_features.png). The ranking is dominated by the "worst" (largest-value) size and concavity measurements: worst perimeter, worst area, worst radius, worst concave points, and mean concave points.

A ROC-AUC of 0.995 means that if you pick a random malignant and a random benign case, the model ranks the malignant one as higher-risk 99.5% of the time, which is near-perfect separability. At the default 0.5 threshold, accuracy is 0.958, but the more relevant numbers for a screening tool are the per-class errors. Recall on the malignant class is 0.925, meaning about 1 in 13 malignant tumors in the test fold is missed at this threshold. Because ROC-AUC is so high, that miss rate can be driven down simply by lowering the decision threshold (trading a few benign false-positives for fewer dangerous false-negatives), since the model has the ranking power to support that trade-off.

What worked: the "worst-case" geometric features carry almost all the signal, which is medically sensible. Malignant tumors tend to have the most irregular, largest nuclei, and it is the extreme cells in a sample that betray malignancy, not the average ones.

A few limitations. First, a single split gives a point estimate, and with only 143 test cases the AUC has a non-trivial confidence interval, so k-fold cross-validation would give a more honest spread. Second, the features are heavily redundant, and a similar AUC is achievable with around 5 features, which would be worth pursuing for a deployable, auditable model. Third, these are curated research measurements, and real-world image-extracted features would be noisier.

Takeaway: cell-nucleus size and shape irregularity at the extremes is an almost perfectly separating signal for malignancy here. The modelling is easy, and the real work in a production setting would be threshold calibration toward high malignant-recall.

## How to run

```bash
pip install -r requirements.txt
py 01_breast_cancer_classification.py
```

The dataset ships inside scikit-learn, so there is no download and no Kaggle key. To swap in the Kaggle mirror (`uciml/breast-cancer-wisconsin-data`), install the `kaggle` CLI, place `kaggle.json`, uncomment the download line at the top of the script, and point the loader at the CSV.
