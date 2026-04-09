---
type: concept
domain: foundations
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Skewness

Skewness describes asymmetry in a distribution.

> [!INFO] Core idea
> Skew matters because it changes how trustworthy the mean is, how easy a distribution is to summarize, and how stable some preprocessing choices feel.

## Why It Matters
Many modeling and preprocessing defaults assume data is not strongly asymmetric. Once skew becomes large, means, variances, and standard scaling can stop reflecting the typical case well.

## Distribution Cheat Sheet
| Shape | Typical Pattern | Practical Consequence |
| :--- | :--- | :--- |
| roughly symmetric | balanced tails | mean and median are often similar |
| right-skewed | long tail to larger values | mean is pulled upward |
| left-skewed | long tail to smaller values | mean is pulled downward |

> [!IMPORTANT] Skew changes which summary is robust
> When a variable is strongly skewed, the median often tells the central story better than the mean.

## Core Mechanics
### What Skew Usually Signals
- long-tail behavior
- bounded variables with asymmetric spread
- mixtures of populations
- rare but high-impact values

### Why It Matters In Practice
- imputation may favor median over mean
- transformations may improve stability
- standardization may feel less intuitive than robust alternatives
- metric interpretation changes when error distributions are asymmetric

> [!WARNING] Strong skew can hide in “clean” data
> A dataset can have no missing values and no obvious pipeline errors while still being hard to model because a few large values dominate the scale.

## Tradeoffs And Decision Rules
### When To Adapt Your Preprocessing
- the mean and median are meaningfully different
- the tail contains business-critical large values
- scaling based on mean and variance produces unstable interpretation

### When Not To Overreact
- mild skew alone is not automatically a problem
- tree-based models often tolerate skew better than distance-based methods
- transforming a variable can improve a model while reducing interpretability

> [!CAUTION] Skew and outliers reinforce each other
> Strong skew often comes with influential tail values, which is why [[Outliers]] and skewness should be inspected together.

> [!TIP] Practical default
> If a feature is clearly skewed, compare mean vs median intuition first. Then decide whether robust scaling, transformation, or tree-based modeling makes more sense.

## Related Notes
- Related: [[Normal Distribution]], [[Outliers]], [[Data Imputation]], [[Data Standardization]], [[Tree-based Models]]

## Sources
- Joanes and Gill, *Comparing Measures of Sample Skewness and Kurtosis*.
- Kim, *Assessing Normal Distribution Using Skewness and Kurtosis*.

## Last Reviewed
- 2026-04-10
