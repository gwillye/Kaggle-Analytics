# 05 - Text Classification: 20 Newsgroups (TF-IDF + Logistic Regression)

Headline result: accuracy = 0.888 across 4 topics on 1,557 held-out test documents.

## Problem

The 20 Newsgroups corpus is roughly 18k Usenet posts across 20 topics. We use a 4-topic subset, `rec.autos`, `sci.med`, `comp.graphics`, and `talk.politics.mideast`, deliberately chosen to span distant domains (cars, medicine, computer graphics, geopolitics). The split is the standard train/test split, 2,336 train and 1,557 test documents.

The question: can a classic bag-of-words pipeline tell these topics apart from the raw text alone, after we strip away the easy give-aways? This is the canonical NLP baseline that any modern transformer model has to beat to justify its cost, and establishing that baseline honestly is the point.

## Approach

This is hard mode by design. We remove `headers`, `footers`, and `quotes`. Newsgroup headers and signatures leak the answer (for example, an `Organization:` line naming a hospital). Stripping them forces the model to learn from actual topical content, so the reported accuracy reflects genuine language understanding rather than metadata leakage. This is the single most important methodological choice here.

For features we use TF-IDF with English stop-words, `min_df=2` (drop terms that appear in only one document, which are usually noise or typos), and `ngram_range=(1,2)` (unigrams plus bigrams, so "talk politics" or "graphics card" survive as units).

The model is Logistic Regression (`C=10`, `max_iter=2000`). Linear models are the proven strong baseline on high-dimensional sparse TF-IDF, and `C=10` loosens regularization to exploit the many informative rare terms. It is wrapped in a scikit-learn `Pipeline` so the vectorizer and classifier are fit and applied together with no leakage.

Validation uses the official train/test split (not a random one), which is the comparable, reproducible benchmark for this corpus. The metric is accuracy (the classes are near-balanced, roughly 376 to 396 test docs each, so accuracy is fair) plus per-class precision/recall and a confusion matrix.

## Results and analysis

Running `py 05_newsgroups_text_classification.py`:

```
train=2336 test=1557 classes=4
              precision    recall  f1-score   support
       autos      0.928     0.889     0.908       389
         med      0.808     0.944     0.871       396
    graphics      0.905     0.818     0.859       396
     mideast      0.934     0.902     0.917       376
    accuracy                          0.888      1557
```

The confusion matrix is in [`outputs/05_newsgroups_confusion.png`](../outputs/05_newsgroups_confusion.png).

88.8% accuracy with content-only features is a strong result. Chance is 25%, so the model is doing real work, and it does so with a transparent, millisecond-fast linear model and no GPU. The per-class breakdown is where the insight lives.

`mideast` and `autos` are the cleanest (F1 0.92 and 0.91): geopolitics and cars have distinctive, low-overlap vocabularies. `med` shows high recall (0.944) but lower precision (0.808), making it the catch-all attractor: documents from other topics that mention bodies, problems, or symptoms get pulled into "medicine." Conversely, `graphics` has the lowest recall (0.818), since its posts about files, formats, and rendering get misrouted, partly into `med`.

So the errors are semantically sensible, not random. The confusion concentrates on topic-vocabulary overlap, which is exactly where a bag-of-words model is blind to context.

There is a single point that is both a strength and a limitation: removing headers, footers, and quotes is why this number is trustworthy, and it is also why it is not 99%. With leakage left in, accuracy would inflate to near-perfect but would measure nothing useful. The honest ceiling for bag-of-words on stripped text is right around here, and closing the `graphics` versus `med` confusion needs word-order and context (embeddings or a transformer), which is the natural next step.

Takeaway: a TF-IDF plus Logistic-Regression baseline hits about 89% on clean, content-only text. Set this bar first, and only adopt a heavier model if it clearly clears it.

## How to run

```bash
pip install -r requirements.txt
py 05_newsgroups_text_classification.py   # scikit-learn downloads and caches the corpus once
```

To use a Kaggle text CSV instead, point the pipeline at columns `text` and `label` in place of `fetch_20newsgroups`. See the top-level README.
