---
type: concept
domain: data-preparation
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Imbalanced Datasets

An imbalanced dataset is one where one class is much more common than another.

> [!INFO] Core idea
> In imbalanced classification, the rare class is often the one you care about most. That is why the default metric and training strategy usually need to change.

## Why It Matters
Fraud, anomaly detection, rare disease screening, and churn prediction often have extreme class imbalance. A model can look strong numerically while missing the cases that matter.

> [!FAILURE] Accuracy paradox
> A model that predicts the majority class everywhere can achieve high accuracy and still be useless in practice.

## 1. What Changes Under Imbalance?
| Area | What Changes | Practical Consequence |
| :--- | :--- | :--- |
| Evaluation | accuracy becomes weak as a primary signal | use minority-focused metrics |
| Training | the model sees many more majority examples | minority signal may be ignored |
| Thresholding | default cutoff may be poorly aligned | threshold tuning becomes more important |

## 2. Metrics To Prioritize
- [[Classification Metrics|Precision]]
- [[Classification Metrics|Recall]]
- [[Classification Metrics|F1-score]]
- [[Classification Metrics|PR-AUC]]

> [!IMPORTANT] PR-AUC matters here
> When the positive class is rare, PR-AUC is usually more informative than ROC-AUC because it focuses on positive-class retrieval quality.

## 3. Strategy Cheat Sheet
| Strategy | Best When | Main Risk |
| :--- | :--- | :--- |
| Undersampling | dataset is large and redundant | throws away information |
| Oversampling | minority signal is too weak | can overfit duplicates |
| [[SMOTE]] | need synthetic minority support | must stay training-only |
| Class weights | want a production-friendly default | needs threshold review |
| Threshold tuning | model is fixed but decision policy is wrong | can overfit validation behavior |

Balanced ensembles and boosted trees such as [[Random Forest]] variants and [[XGBoost]] are also common practical baselines once the evaluation setup is under control.

> [!WARNING] Resampling leakage
> Apply oversampling or SMOTE only on the training split. Never let synthetic examples leak into validation or test data.

## 4. When To Use / When Not To Use
### When To Use
- Use class weights as a simple strong baseline.
- Use SMOTE or oversampling when minority coverage is too weak.
- Tune the threshold when business costs are asymmetric.

### When Not To Use
- Do not rely on accuracy as the main score.
- Do not compare models without checking minority-class metrics.
- Do not oversample before the train-test split.

> [!TIP] Practical default
> Start with class weights plus Precision, Recall, and PR-AUC. Then test whether resampling materially improves minority performance.

## Related Notes
- Prerequisites: [[Classification Metrics]]
- Related: [[SMOTE]], [[Bias in Machine Learning]], [[Tree-based Models]]

## Example
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)
```

## Sources
- He and Garcia, *Learning from Imbalanced Data*.
- Branco, Torgo, and Ribeiro, *A Survey of Predictive Modeling on Imbalanced Domains*.

## Last Reviewed
- 2026-04-10
