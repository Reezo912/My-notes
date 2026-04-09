---
type: concept
domain: foundations
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Selection Bias

Selection bias occurs when the sample used to train or evaluate a model is not representative of the population where the model will be used.

> [!INFO] Core idea
> A model can look excellent on biased data and still fail in production because it learned from the wrong slice of reality.

## Why It Matters
Selection bias undermines generalization. Even strong metrics become misleading if the sample is systematically distorted.

## Where It Appears
| Stage | Example | Consequence |
| :--- | :--- | :--- |
| Data collection | non-random sampling | the model learns the wrong population |
| User behavior | only extreme users respond | the silent majority disappears |
| Data cleaning | dropping rows under bad assumptions | the cleaned sample becomes distorted |

> [!IMPORTANT] Cleaning can create bias
> A technically valid preprocessing step can still make the training sample less representative.

## Classic Example
Survivorship bias is the classic case: if you only study the planes that returned, you miss the evidence from the planes that crashed.

## Common ML Failure Mode
Using `dropna()` assumes the missing data can be ignored safely. If the data is actually [[Types of Missing Data|MNAR]], dropping rows can systematically remove one group from the training sample.

> [!WARNING] Silent sample distortion
> Selection bias is dangerous because the dataset can still look clean, coherent, and statistically plausible after the damage is done.

## Mitigation
- improve sampling design upstream
- compare raw and cleaned distributions
- use reweighting when the sampling problem is understood
- document underrepresented subpopulations explicitly

> [!TIP] Practical check
> Whenever cleaning removes a noticeable fraction of the data, compare important feature distributions before and after the filtering step.

## Related Notes
- Prerequisites: [[Bias in Machine Learning]], [[Data Imputation]]
- Related: [[Types of Missing Data]], [[Imbalanced Datasets]]

## Example
```python
import scipy.stats as stats

statistic, p_value = stats.ks_2samp(raw_age.dropna(), clean_age)
print("possible selection bias" if p_value < 0.05 else "no major shift detected")
```

## Sources
- Heckman, *Sample Selection Bias as a Specification Error*.
- Hernan and Robins, *Causal Inference: What If*.

## Last Reviewed
- 2026-04-10
