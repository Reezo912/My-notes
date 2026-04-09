---
type: concept
domain: foundations
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Outliers

Outliers are observations that lie far from the typical range of the rest of the data.

> [!INFO] Core idea
> Outliers matter because a small number of extreme values can distort averages, losses, thresholds, and scaling choices far more than their count suggests.

## Why It Matters
Outliers can change the story told by a dataset. They can signal bad data, rare but important events, or real business exceptions. The right response depends on which of those is true.

## Impact Map
| Area | Typical Effect | Why It Matters |
| :--- | :--- | :--- |
| Mean-based summaries | strong distortion | average no longer represents the typical case well |
| Squared-error metrics | amplified penalty | RMSE reacts more strongly than MAE |
| Distance-based methods | neighborhood distortion | KNN-like methods can behave erratically |
| Scaling | unstable range or variance | feature magnitudes become misleading |

> [!IMPORTANT] An outlier is not automatically an error
> Some outliers are bad records. Others are exactly the rare cases the model needs to understand.

## Core Mechanics
### Common Sources
- measurement or logging errors
- rare but legitimate events
- mixed populations with different behavior
- long-tailed or skewed distributions

### Typical Detection Heuristics
- z-score style distance when data is roughly normal
- IQR-based rules when robustness matters more
- model residual inspection in supervised settings

> [!WARNING] Deleting outliers can create false cleanliness
> If extreme values are real, dropping them can improve a benchmark while making the model worse in production.

## Tradeoffs And Decision Rules
### When To Investigate Aggressively
- the value is impossible or inconsistent with domain rules
- a data pipeline or sensor issue is plausible
- one observation changes the summary too much

### When To Keep Them
- they reflect rare but real business cases
- tail performance matters operationally
- the model should learn how bad or unusual cases behave

> [!CAUTION] Model family matters
> [[Linear Models]] and squared-error objectives are usually more sensitive to outliers than many [[Tree-based Models]].

> [!TIP] Practical default
> First decide whether the outlier is bad data, rare-but-real data, or evidence of a different subgroup. Then choose deletion, capping, robust preprocessing, or explicit modeling.

## Related Notes
- Related: [[Normal Distribution]], [[Skewness]], [[Regression Metrics]], [[Data Standardization]], [[Tree-based Models]]

## Sources
- Tukey, *Exploratory Data Analysis*.
- Iglewicz and Hoaglin, *How to Detect and Handle Outliers*.

## Last Reviewed
- 2026-04-10
