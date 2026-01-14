# Linear Models

## 1. Theoretical Foundation
Linear models assume the target $y$ is a **linear combination** (weighted sum) of the input features. Unlike trees, they rely heavily on geometry (distances and hyperplanes).

### The Master Equation
$$ \hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} + b $$
*   **Goal:** Find the optimal weights vector $\mathbf{w}$ that minimizes the error.
*   **Optimization Engine:** **Gradient Descent**. The algorithm iteratively updates weights by moving in the direction of the steepest descent (negative gradient).
$$ \mathbf{w}_{next} = \mathbf{w}_{current} - \alpha \nabla J(\mathbf{w}) $$
*   $\alpha$: Learning Rate (Step size).
*   $\nabla J$: Gradient of the Cost Function.

> [!FAILURE] Sensitivity Weakness
> Because they rely on **Euclidean distance**:
> *   **Unscaled Data:** Large features dominate the gradient. **[[StandardScaler]] is mandatory**.
> *   **Outliers:** Squared errors maximize the penalty of outliers, pulling the model significantly.

---

## 2. Variations by Loss Function
The difference between Regression and Classification is purely mathematical: the **Activation Function** and the **Cost Function ($J$)**.

### A. Linear Regression (Continuous Target)
*   **Activation:** None (Identity). Output is $\in (-\infty, \infty)$.
*   **Cost Function:** **Mean Squared Error (MSE)**.
$$ J(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $$
*   **Why Squared?** It makes the error surface **Convex** (bowl-shaped), guaranteeing a global minimum.
* **The metrics for regression can be found here** -> [[Regression metrics]]

### B. Logistic Regression (Binary Classification)
*   **Activation:** Wraps the linear equation in a **Sigmoid** function to squash output to $[0,1]$.
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
*   **Cost Function:** **Log-Loss (Binary Cross-Entropy)**.
$$ J(\mathbf{w}) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})] $$
*   **Why not MSE?** If we used MSE with Sigmoid, the function would be "wavy" (non-convex), causing Gradient Descent to get stuck in local minima.
* **The metrics for classification can be found here** -> [[Classification Metrics]]

---

## 3. Regularization (L1 vs L2)
In linear models, Overfitting happens when coefficients $\mathbf{w}$ become too large. We add a **penalty term** to the Cost Function to constrain them.

| Method | Also known as... | Math Penalty | Effect on Features |
| :--- | :--- | :--- | :--- |
| **L2** | **Ridge** | $+ \lambda \sum w_j^2$ | **Shrinkage.** Reduces weights towards 0 but keeps them. Best for multicollinearity. |
| **L1** | **Lasso** | $+ \lambda \sum |w_j|$ | **Selection.** Forces weak features to **Exactly Zero**. Creates sparse models. |
| **ElasticNet** | -- | $\lambda_1 L1 + \lambda_2 L2$ | Combines both. Best for complex, high-dimensional data. |

---

## 4. Scikit-Learn Implementation

```python
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 🛑 CRITICAL: Always use a Pipeline with Scaler for Linear Models

# 1. Linear Regression (Basic OLS)
lin_reg = make_pipeline(StandardScaler(), LinearRegression())

# 2. Logistic Regression (Classification)
# Note: 'C' is the inverse of regularization strength (1/lambda). 
# Smaller C = Stronger Regularization.
log_reg = make_pipeline(
StandardScaler(), 
LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
)

# 3. Lasso (L1 Regularization for Feature Selection)
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))

# Training
log_reg.fit(X_train, y_train)
```