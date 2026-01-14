## 1. The Core Architecture
Spark ML uses a standardized API to distinguish between **"Learning"** algorithms and **"Applying"** algorithms.

| Component | Method | Action | State | Analogy |
| :--- | :--- | :--- | :--- | :--- |
| **Estimator** | `.fit()` | Learns from data. | **Stateful** (Needs to see the data to calculate parameters). | The **Chef** (Needs ingredients to cook the meal). |
| **Transformer** | `.transform()` | Modifies data. | **Stateless** (Or typically immutable once created). | The **Meal** (Ready to be eaten/served). |

---

## 2. Estimators (The Learners)
An Estimator is an algorithm that must look at the data to learn parameters.

*   **Logic:** It accepts a `DataFrame` and produces a `Transformer`.
*   **Examples:**
*   `Imputer`: Needs to see all rows to calculate the **Mean**.
*   `StandardScaler`: Needs to see all rows to calculate **Std Dev** and **Mean**.
*   `LogisticRegression`: Needs to run Gradient Descent to find **Weights**.

> [!WARNING] Lazy vs Eager
> Calling `.fit()` triggers an **Eager Execution** (usually). Spark must scan the data (Action) to compute the parameters immediately.

```python
from pyspark.ml.feature import Imputer

# 1. Instantiate the Estimator (It knows NOTHING yet)
imputer_estimator = Imputer(inputCols=["age"], outputCols=["age_imputed"])

# 2. Learn (Fit)
# Spark scans the 'train_df' to calculate the mean age (e.g., 34.5)
# It returns a MODEL (which is a Transformer)
imputer_model = imputer_estimator.fit(train_df)
```

---

## 3. Transformers (The Doers)
A Transformer is an algorithm that can read a `DataFrame`, append a new column, and output a new `DataFrame`.

*   **Logic:** $f(x) = y$. It maps input rows to output rows.
*   **Types:**
1.  **Feature Transformers:** Tools that don't need learning (e.g., `VectorAssembler`, `Binarizer`, `Tokenizer`).
2.  **Trained Models:** The result of fitting an estimator (e.g., `ImputerModel`, `LogisticRegressionModel`).

> [!NOTE] Why is a Model a Transformer?
> Once a Linear Regression "learns" the weights ($y = 2x + 1$), it doesn't need to learn anymore. To make predictions, it just **transforms** input $x$ into output $y$ using the formula.

```python
# 3. Apply (Transform)
# The model uses the learnt mean (34.5) to fill nulls.
# This is LAZY evaluation (adds to the DAG).
df_clean = imputer_model.transform(test_df)
```

---

## 4. The Pipeline Concept
This distinction exists to allow **Pipelines**. A Pipeline is just a list of stages.

When you call `pipeline.fit(data)`:
1.  Spark looks at the stages.
2.  If it's a **Transformer**, it calls `.transform()` and passes the data to the next stage.
3.  If it's an **Estimator**, it calls `.fit()` to get a Transformer, then calls `.transform()` on that new Transformer to pass data to the next stage.

> [!TIP] The Production Benefit
> By saving the **Fitted Pipeline**, you save the state (Means, StdDevs, Weights).
> When you load it in production, you don't need the training data anymore. You just apply the transformations.

---

## 5. Cheat Sheet: Which one is it?

| Class Name | Type | Why? |
| :--- | :--- | :--- |
| `VectorAssembler` | **Transformer** | Just concatenates columns. No math required. |
| `StringIndexer` | **Estimator** | Needs to scan data to know all unique categories. |
| `OneHotEncoder` | **Estimator*** | *Usually requires fitting to know category cardinality.* |
| `StandardScaler` | **Estimator** | Needs to calculate Mean/Std. |
| `LinearRegression` | **Estimator** | Needs to find weights. |
| `LinearRegressionModel` | **Transformer** | Already has the weights. |