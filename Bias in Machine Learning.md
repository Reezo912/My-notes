## 1. What is Bias?
**Definition:** Systematic error introduced into sampling or testing by selecting or encouraging one outcome or answer over others. In ML, it means the model behaves unfairly or inaccurately for certain groups of data.

> [!FAILURE] The Consequence
> A biased model does not just "fail"; it **discriminates**.
> *   **Allocative Harm:** Denying opportunities (loans, jobs) to a group.
> *   **Representational Harm:** Reinforcing stereotypes.

---

## 2. Where does it come from? (The Lifecycle)
Bias can enter the pipeline at three distinct stages.

### 2.1. Pre-Processing (Data Bias)
The most common source. The data itself reflects historical prejudices or collection errors.

*   **[[Selection Bias]]:** The training data is not representative of the real population.
*   *Example:* Training a FaceID system mostly on white men.
*   **Historical Bias:** The data is correctly sampled, but the world was biased when data was generated.
*   *Example:* Hiring data from the 1980s showing few women in tech.
*   **Measurement Bias:** Errors in how features are labeled or measured.
*   *Example:* Using "Arrest Rate" as a proxy for "Crime Rate" (heavily biased by policing tactics).

### 2.2. Processing (Algorithmic Bias)
The algorithm itself amplifies the bias.

*   **Aggregation Bias:** Using a "one-size-fits-all" model for distinct populations.
*   **Optimization Bias:** The loss function minimizes global error but ignores the error distribution across subgroups.

### 2.3. Post-Processing (Deployment Bias)
*   **Evaluation Bias:** Testing the model on a benchmark dataset that is also biased.
*   **Feedback Loop:** The model's predictions influence future data collection (e.g., predictive policing).

---

## 3. How to Measure it? (Fairness Metrics)
You cannot fix what you cannot measure. We use **Fairness Metrics** (often using libraries like `Fairlearn` or `AIF360`).

### Group Fairness Metrics
Comparing metrics between a **Privileged Group** (e.g., Male) and an **Unprivileged Group** (e.g., Female).

*   **Disparate Impact (DI):** Ratio of positive outcomes.
*   $$ \frac{P(\hat{Y}=1 | D=\text{unprivileged})}{P(\hat{Y}=1 | D=\text{privileged})} $$
*   *Rule of thumb:* DI < 0.8 (Four-fifths rule) usually indicates bias.
*   **Equal Opportunity:** True Positive Rates (TPR) should be equal across groups.
*   **Demographic Parity:** The likelihood of a positive prediction should be equal regardless of the group.

> [!TIP] Which metric to choose?
> *   Use **Equal Opportunity** if you care about **errors** (e.g., not incorrectly flagging valid transactions as fraud).
> *   Use **Demographic Parity** if you care about **outcomes** (e.g., hiring equal numbers of men and women).

---

## 4. Mitigation Strategies
How to fix bias once detected.

### A. Pre-Processing (Fix the Data)
*   **Re-sampling:** Oversample the minority group or undersample the majority.
*   **Reweighting:** Assign higher weights to training examples from the unprivileged group.
*   **Suppression:** Removing the sensitive attribute (Gender/Race).
> [!WARNING] Blindness Trap
> Removing the column often FAILS because of **Proxy Variables** (e.g., Zip Code correlates with Race).

### B. In-Processing (Fix the Model)
*   **Adversarial Debiasing:** Train two models—one to predict the target, another to predict the sensitive attribute. The first model tries to "fool" the second.
*   **Regularization:** Add a fairness penalty term to the Loss Function.

### C. Post-Processing (Fix the Predictions)
*   **Threshold Adjustment:** Use different classification thresholds (0.4 vs 0.6) for different groups to achieve Equal Opportunity.

---

## 5. Python Implementation (Fairlearn)
Example using `fairlearn` (Standard in the Databricks ecosystem).

```python
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score

# 1. Detection: Measure Disparate Impact
# y_true, y_pred, and sensitive_features (e.g., 'sex')
metrics = MetricFrame(
metrics=accuracy_score,
y_true=y_test,
y_pred=y_pred,
sensitive_features=X_test['sex']
)
print(metrics.by_group) # Check accuracy for Male vs Female

# 2. Mitigation: Post-processing
# Optimize for Equalized Odds
optimizer = ThresholdOptimizer(
estimator=my_trained_model,
constraints="equalized_odds",
predict_method='predict_proba'
)
optimizer.fit(X_train, y_train, sensitive_features=X_train['sex'])

# 3. Fair Predictions
y_pred_fair = optimizer.predict(X_test, sensitive_features=X_test['sex'])
