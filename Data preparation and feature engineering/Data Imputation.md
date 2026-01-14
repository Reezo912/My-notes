# Handling Missing Data

## 1. How does it impact?
It impacts the performance and reliability of ML models in the following ways:

> [!FAILURE] Systemic Risks
> *   **Reduced model performance:** Gaps in features hinder pattern learning.
> *   **Biased inferences:** Can skew predictions towards certain groups (leading to **[[Bias in Machine Learning]]**).
> *   **Imbalanced representations:** Uneven missingness across classes can exacerbate **[[Imbalanced Datasets]]** issues.
> *   **Increased complexity:** Requires complex code to handle these missing values.

---
![[Captura de pantalla 2025-12-10 a las 13.30.51.png]]

## 2. How to handle it
The best practices involve **understanding missing data patterns** and **selecting the right technique**.

### Option A: Dropping (The Nuclear Option)
**Concept:** Removing incomplete data points entirely.
**Strategies:**
*   **Listwise Deletion (Rows):** Remove the user/row if it has missing values. Use when data loss is minimal.
*   **Dropping Features (Columns):** Remove the variable entirely if it has too many missing values (>40-50%).

> [!WARNING] The Risk
> If done on large missing data, it introduces **[[Selection Bias]]** (you only model "complete" users) and results in significant information loss.

**Code Implementation:**
```python
# --- Pandas ---
df.dropna(axis=0)  # Drop Rows
df.dropna(axis=1)  # Drop Columns

# --- PySpark ---
df.na.drop("any")  # Drop row if ANY value is null
df.drop("col_name") # Drop specific column
```

### Option B: New Category (Tagging)
**Concept:** Sometimes missing values can carry information on their own ([[Types of Missing Data|MNAR]]).
**Strategy:**
*   **Categorical:** Replace `null` with a new label like `"Unknown"`.
*   **Numerical:** Replace with `-1` or `-999`.

> [!TIP] Model Compatibility
> Using `-1` is valid for **[[Tree-based Models]]** (Decision Trees, RF, XGBoost), but **dangerous** for **[[Linear Models]]** (Regression, Neural Nets) as they will interpret -1 as a mathematical value, ruining weights.

**Code Implementation:**
```python
# --- Pandas ---
df['category_col'].fillna("Unknown", inplace=True)
df['numeric_col'].fillna(-1, inplace=True)

# --- PySpark (DataFrame API) ---
# Note: Spark ML Imputer cannot handle strings. Use DataFrame API directly.
df_filled = df.na.fill("Unknown", ["category_col"]) \
          .na.fill(-1, ["numeric_col"])
```

### Option C: Imputation (Statistical Filling)
**Concept:** Fill gaps using statistical properties of the column.

**Standard Approaches:**
*   **Mean:** For **[[Normal Distribution]]** (Gaussian).
*   **Median:** For **[[Skewness|Skewed Data]]** or data with **[[Outliers]]**.
*   **Mode:** For **[[Categorical Data]]**.

**Advanced Methods:**
*   **Regression Imputation:** Estimates based on relationships inferred using **[[Linear Models]]**.
*   **KNN Imputation:** Estimates using the most similar data points found via **[[KNN]]**.
> [!CAUTION] Scalability Warning
> KNN imputation is computationally expensive ($O(N^2)$). In **Big Data / Spark** environments with millions of rows, this is often unfeasible.

**Spark Implementation (The Estimator Pattern):**
> [!NOTE] Spark Architecture
> In Spark ML, `Imputer` is an **Estimator**. See **[[Spark ML Estimators vs Transformers]]** to understand why we need `.fit()` and `.transform()`.

```python
from pyspark.ml.feature import Imputer

# 1. Define the Estimator (Configure strategy)
# Spark Imputer only supports 'mean' and 'median' natively
imputer = Imputer(inputCols=["age"], outputCols=["age_imputed"])
imputer.setStrategy("median")

# 2. Fit (Compute the median)
model = imputer.fit(train_df)

# 3. Transform (Apply to data)
train_filled = model.transform(train_df)
test_filled = model.transform(test_df) 
```

---

## 3. Factors Influencing Imputation Method
Choosing the right method depends on the nature of your data and the goal.

### Nature of Data
*   **Type of data:**
*   *Categorical* $\rightarrow$ Mode or Hot-deck.
*   *Continuous* $\rightarrow$ Mean or Regression imputation.
*   **Distribution:**
> [!EXAMPLE] Skewness Rule
> If data has **[[Skewness|Skewed Distribution]]** or Large **[[Outliers]]** $\rightarrow$ Use **Median** imputation (Mean is sensitive to outliers).

### Amount of Missing Data
*   **Low (<5%):** Safe to Drop, Simple Imputation (Mean), or Hot-deck.
*   **High (>20-30%):** Drop the column entirely OR use a specific "Missing" category.
*   **Note:** Consider the **Trade-Off** between accuracy (Advanced methods) and computational complexity.

### Form of Missingness
*   **Random ([[Types of Missing Data|MCAR]]):** Standard imputation works well.
*   **Not Random ([[Types of Missing Data|MNAR]]):** Use specialized techniques like Pattern Mixture Models or Selection Models.

> [!IMPORTANT] Critical Rule
> If data is **[[Types of Missing Data|MNAR]]**, you MUST use a specific flag or category so the model learns *why* it is missing. Imputing with the mean here introduces bias.