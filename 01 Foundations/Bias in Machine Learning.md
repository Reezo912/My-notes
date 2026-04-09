---
type: concept
domain: foundations
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Bias in Machine Learning

Bias in machine learning is systematic error that makes model behavior unfair, distorted, or unreliable for some groups or conditions.

> [!INFO] Core idea
> Bias is not only a model issue. It can enter through data collection, labeling, optimization, evaluation, or deployment.

## Why It Matters
Bias creates real operational risk: unfair outcomes, misleading evaluation, poor generalization, and loss of trust in the system.

## Bias Across The Lifecycle
| Stage | Typical Source | Example |
| :--- | :--- | :--- |
| Data collection | Underrepresentation or skewed sampling | Training mostly on one demographic |
| Labeling or measurement | Proxy labels or noisy annotation | Arrest rate used as crime rate |
| Optimization | Global objective hides subgroup failure | High average accuracy, poor minority recall |
| Evaluation and deployment | Biased benchmark or feedback loop | Model decisions shape future data |

> [!WARNING] Bias is often upstream
> If the sample or labels are distorted, a technically correct model can still produce harmful outcomes.

## Core Concepts
### Data Bias
- historical bias in the source data
- measurement bias in features or labels
- sampling problems such as [[Selection Bias]]

### Algorithmic And Objective Bias
- a single model may fit some subgroups much worse than others
- the loss function may optimize average performance while hiding uneven error distribution

### Evaluation And Deployment Bias
- benchmark coverage may be poor
- real-world use can create feedback loops that reinforce prior errors

> [!IMPORTANT] Fairness is metric-dependent
> The right fairness check depends on the cost of mistakes. A system that optimizes for equal outcomes is not the same as one that optimizes for equal error rates.

## Common Fairness Checks
| Metric | What It Checks | Useful When |
| :--- | :--- | :--- |
| Disparate impact | difference in positive outcome rates | outcome parity matters |
| Equal opportunity | difference in true positive rates | missed positives are costly |
| Demographic parity | overall positive rate parity | group-level outcome comparison |

> [!CAUTION] One fairness metric rarely settles the issue
> Different fairness criteria can conflict. The correct choice depends on product goals and policy constraints.

## Mitigation Strategies
- improve sampling and labeling quality
- measure subgroup performance explicitly
- reweight or rebalance training data when justified
- introduce fairness-aware constraints or post-processing

## Related Notes
- Prerequisites: [[Classification Metrics]], [[Imbalanced Datasets]]
- Related: [[Selection Bias]], [[Types of Missing Data]]

## Example
```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score

frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test["sex"],
)

print(frame.by_group)
```

## Sources
- Barocas, Hardt, and Narayanan, *Fairness and Machine Learning*.
- Hardt, Price, and Srebro, *Equality of Opportunity in Supervised Learning*.

## Last Reviewed
- 2026-04-10
