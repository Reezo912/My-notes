## 1. What is it?
The process of rescaling one or more features so that they share a common scale (usually $\mu=0, \sigma=1$ or range $[0,1]$).

> [!FAILURE] Why do we need it?
> Machine Learning algorithms calculate **distances** (Euclidean) between data points.
> *   **Scenario:** Feature A is "Age" (0-100). Feature B is "Salary" (0-1,000,000).
> *   **The Problem:** Mathematically, a change of 1€ in Salary looks identical to a change of 1 year in Age. But 1 year is massive, 1€ is noise.
> *   **The Result:** The model ignores Age and focuses entirely on Salary.

---

## 2. Benefits

### A. Optimization Speed (Convergence)
In algorithms using **Gradient Descent** (Neural Nets, Linear Regression), if features have vastly different scales, the error surface becomes an elongated valley (elliptical). The gradient bounces back and forth, taking forever to reach the minimum.
*   **With Scaling:** The error surface becomes spherical. The gradient goes straight to the center.

### B. Fair Regularization
L1 (Lasso) and L2 (Ridge) penalize large weights.
*   If a feature has a small scale (e.g., 0.001), the model gives it a huge weight to compensate.
*   Regularization will unfairly crush that huge weight, killing the feature. Scaling ensures the penalty is applied equally.

### C. Distance Validity
Algorithms like **[[KNN]]** or K-Means Clustering are purely distance-based. Without scaling, the distance is dominated by the feature with the largest numbers, rendering the algorithm useless.

---

## 3. Techniques: Standard vs MinMax

| Technique | Formula | Distribution Result | Use Case |
| :--- | :--- | :--- | :--- |
| **StandardScaler** (Z-Score) | $z = \frac{x - \mu}{\sigma}$ | Mean = 0<br>Std Dev = 1 | **Default choice.**<br>Best for [[Linear Models]], PCA, Neural Nets.<br>Preserves outliers (doesn't cap them). |
| **MinMaxScaler** (Normalization) | $x' = \frac{x - min}{max - min}$ | Range [0, 1] | **Images** (Pixel intensity 0-255).<br>Algorithms that need bounded inputs.<br>Sensitive to outliers (they squish data). |
| **RobustScaler** | $x' = \frac{x - Q2}{Q3 - Q1}$ | Median = 0<br>IQR = 1 | Data with **massive [[Outliers]]**. Uses Median/IQR instead of Mean/Std. |

---

## 4. Model Sensitivity Cheat Sheet

| 🚨 Sensitive (MUST SCALE)                    | ✅ Robust (NO SCALING NEEDED)            |
| :------------------------------------------- | :-------------------------------------- |
| **[[Linear Models]]** (Regression, Logistic) | **[[Tree-based Models]]** (RF, XGBoost) |
| **[[Neural Networks]]**                      | Naive Bayes                             |
| **[[KNN]]** / K-Means (Distance Based)       |                                         |
| **SVM** (Support Vector Machines)            |                                         |
| **PCA** (Principal Component Analysis)       |                                         |


---

## 5. Implementation Workflow (The Leakage Trap)

The most common mistake in Junior ML is applying scaling to the whole dataset BEFORE splitting. This causes **Data Leakage** (the test set info leaks into the training mean).

### Correct Steps:
1.  **Split Data:** Train / Test.
2.  **Fit Scaler on TRAIN:** Calculate Mean/Std only from Training data.
3.  **Transform TRAIN:** Apply math.
4.  **Transform TEST:** Apply **TRAIN's** Mean/Std to Test data. (Do NOT re-fit on Test).

### 🐼 Scikit-Learn Code
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

# 1. Fit only on Train
scaler.fit(X_train)

# 2. Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ⚡ PySpark Code (Estimator Pattern)
```python
from pyspark.ml.feature import StandardScaler, VectorAssembler

# 1. Assembler (Spark requires a single vector column 'features')
assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="unscaled_features")

# 2. Scaler Definition
# withMean=True centers data (Dense output, heavy on RAM)
# withStd=True scales to unit variance (Standard)
scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features", 
withStd=True, withMean=False)

# 3. Pipeline
pipeline = Pipeline(stages=[assembler, scaler, model])
```

---

## 6. Drawbacks & Trade-offs
*   **Loss of Interpretation:** A coefficient of `0.5` on "Scaled Age" is hard to explain to business ("What does +1 Standard Deviation of Age mean?").
*   **Sparse Data:** Standardizing sparse data (withMean=True) destroys sparsity (turns 0s into non-zeros), exploding memory usage.
*   **Normality Assumption:** StandardScaler works best if data is Gaussian (Bell Curve). If data is extremely skewed, `LogTransform` might be better than scaling.