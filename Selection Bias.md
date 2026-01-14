## 1. What is it?
**Definition:** A systematic error that occurs when the sample obtained is not representative of the population intended to be analyzed.
*   **Statistical Reality:** The probability of being selected for the dataset is not random.
*   **The Result:** Your model learns from a distorted reality. Even if the model has 99% accuracy on the test set, it will fail in production because the real world looks different.

> [!FAILURE] Parent Concept
> Selection Bias is a specific *subset* of **[[Bias in Machine Learning]]**. While "Bias" can be algorithmic or historical, Selection Bias is purely an error in **Data Collection** or **Data Preparation**.

---

## 2. Classic Example: Survivorship Bias
The most famous form of Selection Bias.
During WWII, the military looked at planes coming back from battle to see where they were shot. They decided to reinforce the areas with the most bullet holes.
*   **The Error:** They ignored the planes that *didn't come back*. The planes with holes in the engine crashed.
*   **The Lesson:** You cannot learn from data you don't have.

---

## 3. Sources of Selection Bias in ML Pipelines

### A. During Data Collection (The Source)
*   **Sampling Bias:** Using a non-random method (e.g., conducting an online poll about internet usage).
*   **Self-Selection:** Only strongly motivated users (very happy or very angry) leave reviews. The "Silent Majority" is missing.

### B. During Data Preparation (The Engineering Trap)
This is where **[[Data Imputation]]** becomes dangerous.

> [!WARNING] The `dropna()` Trap
> When you use **Listwise Deletion** (Dropping rows with nulls), you assume the data is **[[Types of Missing Data|MCAR]]** (Missing Completely At Random).
>
> If the data is **[[Types of Missing Data|MNAR]]** (Missing Not At Random), dropping these rows creates Selection Bias.
> *   *Example:* Dropping rows where `Income` is null. If rich people tend to hide their income, your dataset now represents a poorer population than reality.

---

## 4. Impact on Model Performance
1.  **Poor Generalization:** The model overfits to the biased sample.
2.  **False Correlation:** It might learn relationships that only exist in the sample (e.g., "Only people with long bios get hired", because short-bio applications were filtered out by HR before reaching the DB).

---

## 5. Mitigation Strategies

### A. Improve Data Collection
The best fix is to stop it at the source. Ensure **Random Sampling** where every member of the population has a non-zero probability of being selected.

### B. Inverse Probability Weighting (IPW)
If you know your sample is biased (e.g., you have too many young people), you can weight the rows during training.
*   **Concept:** Give higher weight to the under-represented group.
*   **Link:** This uses the same mechanism as `class_weight` in **[[Imbalanced Datasets]]**.

### C. Heckman Correction (Advanced)
A statistical method to correct selection bias from non-randomly selected samples. It involves a two-step process:
1.  Model the probability of being selected (Probit model).
2.  Use that probability to correct the final regression.

---

## 6. Python Example: Detecting Distribution Shift
A simple way to check for selection bias is to compare the distribution of a feature in your "Cleaned" data vs. the "Raw" data.

```python
import pandas as pd
import scipy.stats as stats

# 1. Raw Data (Population) vs Cleaned Data (Sample after dropna)
raw_age = df_raw['age']
clean_age = df_raw.dropna()['age'] # Potential Selection Bias here

# 2. Visual Check
print(f"Raw Mean: {raw_age.mean()} | Clean Mean: {clean_age.mean()}")

# 3. Statistical Check (Kolmogorov-Smirnov Test)
# Null Hypothesis: The two samples come from the same distribution.
statistic, p_value = stats.ks_2samp(raw_age.dropna(), clean_age)

if p_value < 0.05:
print("WARNING: Significant Selection Bias detected. The distributions are different.")
else:
print("Pass: The sample looks representative.")
```