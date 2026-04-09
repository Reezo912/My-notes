---
type: concept
domain: foundations
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Normal Distribution

The normal distribution is a symmetric bell-shaped distribution described by a center and a spread, usually the mean and standard deviation.

> [!INFO] Core idea
> A distribution close to normal is predictable in a useful way: values cluster around the center, extreme values become less common, and distance from the mean can be interpreted consistently.

## Why It Matters
The normal distribution appears constantly in statistics, preprocessing heuristics, residual analysis, and baseline modeling assumptions. Even when the data is not perfectly normal, this shape is often the reference point for deciding what looks unusual.

## Visual Map
| Region | Approximate Share Of Values | Practical Interpretation |
| :--- | :--- | :--- |
| within `1` standard deviation | about `68%` | typical range |
| within `2` standard deviations | about `95%` | broad but still expected range |
| within `3` standard deviations | about `99.7%` | extreme values become rare |

> [!IMPORTANT] Approximate normality is often enough
> Many practical heuristics do not need perfect Gaussian data. They only need the distribution to be roughly symmetric and not dominated by heavy tails or strong skew.

## Core Mechanics
### Main Parameters
- the **mean** sets the center
- the **standard deviation** sets the spread
- symmetry means the left and right tails mirror each other

### What This Usually Helps With
- reasoning about z-scores
- deciding whether mean-based summaries are sensible
- interpreting residuals in simple statistical models
- choosing between mean and median style preprocessing heuristics

> [!WARNING] Bell-shaped does not mean harmless
> Real data can look roughly normal in the middle and still have problematic tails, hidden subgroups, or influential outliers.

## Practical Implications
### When It Helps
- use it as a reference for whether mean and variance are stable summaries
- use it when thinking about standardization and z-score style distance
- use it to interpret why some metrics and models react strongly to far-away values

### When It Misleads
- do not force real-world data into a normal assumption just because it is mathematically convenient
- do not assume skewed or heavy-tailed data behaves like a normal distribution
- do not use mean-centered reasoning blindly when [[Outliers]] or [[Skewness]] dominate

> [!CAUTION] Normal-looking data can still be multimodal
> A smooth overall histogram may hide multiple subpopulations. In that case, one mean and one standard deviation can be too crude.

> [!TIP] Practical default
> Use the normal distribution as a baseline reference. Then check whether skew, outliers, or subgroup structure make that baseline unreliable.

## Related Notes
- Related: [[Skewness]], [[Outliers]], [[Data Imputation]], [[Data Standardization]], [[Regression Metrics]]

## Sources
- DeGroot and Schervish, *Probability and Statistics*.
- Montgomery and Runger, *Applied Statistics and Probability for Engineers*.

## Last Reviewed
- 2026-04-10
