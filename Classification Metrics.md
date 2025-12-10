# Classification Metrics

## 1. The Foundation: Confusion Matrix
Everything starts here. It compares **Predicted** values vs **Actual** values.

| | **Predicted: Positive (1)** | **Predicted: Negative (0)** |
| :--- | :--- | :--- |
| **Actual: Positive (1)** | **True Positive (TP)**<br>*(Hit)* | **False Negative (FN)**<br>*(Missed Detection)* |
| **Actual: Negative (0)** | **False Positive (FP)**<br>*(False Alarm)* | **True Negative (TN)**<br>*(Correct Rejection)* |

> [!NOTE] 🧠 Type I vs Type II (The Standard Mnemonic)
> *   **Type I Error (False Positive):** Predicting **YES** when it's **NO**.
>     *   *Example:* Telling a man he is pregnant.
>     *   *Impact:* Lowers **Precision**.
> *   **Type II Error (False Negative):** Predicting **NO** when it's **YES**.
>     *   *Example:* Telling a pregnant woman she is NOT pregnant.
>     *   *Impact:* Lowers **Recall**.

---

## 2. Threshold Metrics (Single Point)
These metrics depend on a fixed threshold (usually 0.5). If probability > 0.5 $\rightarrow$ 1.

### Accuracy (The misleading one)
$$ \text{Accuracy} = \frac{TP + TN}{\text{Total}} $$
*   **Definition:** Percentage of correct predictions overall.
*   **When to use:** ONLY when classes are balanced (50/50).
> [!FAILURE] Trap
> In **[[Imbalanced Datasets]]**, Accuracy is useless. A model predicting "No Fraud" always has 99.9% accuracy but 0 value.

### Precision (Quality of Positives)
$$ \text{Precision} = \frac{TP}{TP + FP} $$
*   **Question:** "Of all the ones we labeled as fraud, how many were actually fraud?"
*   **Focus:** Minimizing **Type I Errors** (False Positives).
> [!TIP] Business Case
> Use when **False Alarms are expensive**.
> *   *Example:* Spam Filter (You don't want to send a real work email to spam).

### Recall / Sensitivity (Coverage of Positives)
$$ \text{Recall} = \frac{TP}{TP + FN} $$
*   **Question:** "Of all the real frauds that happened, how many did we catch?"
*   **Focus:** Minimizing **Type II Errors** (False Negatives).
> [!TIP] Business Case
> Use when **Missing a case is dangerous**.
> *   *Example:* Cancer Diagnosis or Terrorist Detection. Better to double-check a healthy person than miss a sick one.

### F1-Score (The Balance)
$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
*   **Definition:** Harmonic mean of Precision and Recall.
*   **When to use:** When you need a balance (see **[[Imbalanced Datasets]]**). It punishes extreme values (if Recall is 0, F1 is 0).

---

## 3. Rank Metrics (Probabilities / Curves)
These metrics evaluate the model **at all possible thresholds**. They measure how well the model separates classes.

### ROC - AUC (Receiver Operating Characteristic)
*   **Axes:** TPR (Recall) vs FPR (False Positive Rate).
*   **Definition:** Probability that a random Positive example is ranked higher than a random Negative example.
*   **When to use:** Balanced datasets.
*   **Note:** High ROC-AUC can mask **[[Bias in Machine Learning]]** if the negative class is huge.

### PR - AUC (Precision-Recall AUC)
*   **Axes:** Precision vs Recall.
*   **Definition:** Area under the Precision-Recall curve.
*   **When to use:** **[[Imbalanced Datasets]]**.
> [!IMPORTANT] ROC vs PR
> If you have **1% Fraud** and **99% Legit**:
> *   **ROC-AUC** might look great (0.95) because it includes TN (Correct Legit) which are easy.
> *   **PR-AUC** will tell the truth (e.g., 0.20) because it ignores TN and focuses on how well you handle the Frauds.

---

## 4. Implementation

### 🐼 Python (Pandas / Scikit-Learn)
Sklearn offers individual functions and a master report.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# 1. Prediction (Hard Labels 0/1)
y_pred = model.predict(X_test)
# 2. Probabilities (For AUC)
y_prob = model.predict_proba(X_test)[:, 1]

# --- The "One-Shot" Report (Best for quick analysis) ---
print(classification_report(y_test, y_pred))

# --- Individual Metrics ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred)) # Add pos_label=... if not 1
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))     # Uses Probabilities!
print("PR AUC:", average_precision_score(y_test, y_prob)) # Uses Probabilities!
```

### ⚡ PySpark (Databricks)
Spark splits evaluators into Multiclass (Point metrics) and Binary (Rank metrics).

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# --- A. Point Metrics (Acc, F1, Prec, Rec) ---
# Tip: MulticlassEvaluator works for Binary too and gives access to F1/Acc
multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
print(f"Accuracy: {acc} | F1: {f1}")

# --- B. Rank Metrics (ROC, PR) ---
# Tip: Needs Raw Probabilities, not predictions!
binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability")

roc_auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
pr_auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})

print(f"ROC AUC: {roc_auc} | PR AUC (For Imbalanced): {pr_auc}")
```