# Imbalanced Datasets

## 1. What is it?
A dataset is imbalanced when the classification categories are not approximately equally represented.
*   **Majority Class:** The prevalent class (e.g., "Legit Transaction").
*   **Minority Class:** The rare class of interest (e.g., "Fraud").

> [!FAILURE] The Accuracy Paradox
> **NEVER use Accuracy** as the primary metric for imbalanced data.
> *   *Scenario:* 99% Legit, 1% Fraud.
> *   *Dumb Model:* Predicts "Legit" for everyone.
> *   *Result:* **99% Accuracy**, but **0% Recall** (catches 0 frauds). The model is useless.

---

## 2. Evaluation Metrics (How to measure success)
Since Accuracy lies, we must use metrics that focus on the **Minority Class**. (See full details in **[[Classification Metrics]]**).

*   **[[Classification Metrics|Confusion Matrix]]:** Focus specifically on **False Negatives (FN)** (the frauds we missed).
*   **[[Classification Metrics|Precision and Recall]]:**
*   *Precision:* Of all flagged frauds, how many were real?
*   *Recall (Sensitivity):* Of all real frauds, how many did we catch?
*   **[[Classification Metrics|F1-Score]]:** The harmonic mean, useful to find a balance between Precision and Recall.
*   **AUC Curves:**
*   **[[Classification Metrics|ROC-AUC]]:** Good, but can be optimistic if the imbalance is extreme.
*   **[[Classification Metrics|PR-AUC]] (Precision-Recall AUC):** The **Gold Standard** for highly imbalanced datasets. Focuses purely on the minority class performance.

---

## 3. Strategies to Handle it

### A. Data Level (Resampling)
Modify the training data to balance the class distribution.
*   **Undersampling:** Randomly remove samples from the Majority Class.
*   *Pro:* Fast training (less data).
*   *Con:* Loss of information.
*   **Oversampling:** Duplicate samples from the Minority Class.
*   *Pro:* No info loss.
*   *Con:* High risk of **Overfitting** (model memorizes the duplicates).
*   **Synthetic Data Generation ([[SMOTE]]):**
*   *Concept:* Instead of duplicating, create *new* synthetic points interpolating between existing minority samples.
*   *Link:* See details in **[[SMOTE]]** (Synthetic Minority Over-sampling Technique).

> [!WARNING] Data Leakage Risk
> **CRITICAL:** ONLY apply Resampling (SMOTE/Undersampling) on the **TRAINING SET**. Never touch the Test Set/Validation Set, or your metrics will be fake.

### B. Algorithmic Level (Cost-Sensitive Learning)
Instead of changing data, change the **Loss Function**. Tell the algorithm that a mistake on the Minority Class is expensive.

*   **Class Weights:** Assign a higher weight (penalty) to the minority class.
*   *Weight Formula:* $W_{minority} = \frac{N_{total}}{N_{minority}}$

### C. Ensemble Methods
*   **Balanced Random Forest:** Internally undersamples the majority class for each tree in the forest.
*   **XGBoost / LightGBM:** Have built-in parameters (`scale_pos_weight`) to handle this natively.

---

## 4. Implementation

### Python (Scikit-Learn / Imblearn)

**1. Class Weights (Easiest & Production Ready)**
Most models accept a `class_weight` parameter.
```python
from sklearn.linear_model import LogisticRegression

# 'balanced' automatically calculates weights inversely proportional to frequencies
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
```

**2. SMOTE (Using `imblearn`)**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Pipeline is mandatory to avoid Data Leakage!
pipeline = ImbPipeline([
('smote', SMOTE(random_state=42)),
('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

### PySpark (Big Data approach)

Spark ML does not have SMOTE built-in (requires heavy custom implementation). The standard approach is **Class Weights**.

**1. Calculate Weights Manually**
```python
from pyspark.sql.functions import col, when

# 1. Count class ratios
total_count = df.count()
fraud_count = df.filter(col("label") == 1).count()
legit_count = total_count - fraud_count

# 2. Calculate balancing ratio
balancing_ratio = legit_count / fraud_count

# 3. Add Weight Column
df_weighted = df.withColumn("classWeight", 
when(col("label") == 1, balancing_ratio).otherwise(1.0)
)
```

**2. Train with Weight Column**
```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="classWeight")
model = lr.fit(df_weighted)
```