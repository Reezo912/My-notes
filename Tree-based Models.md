# Tree-based Models

## 1. Theoretical Foundation
Tree models are **non-parametric** algorithms. Unlike linear models, they do not assume data follows a specific function ($y = wx+b$); instead, they recursively partition the feature space into hyper-rectangles using **Greedy Logic**.

### The Splitting Mechanism
At each node, the algorithm searches for the tuple $\theta = (j, t_m)$ (feature $j$ and threshold $t_m$) that maximizes **Information Gain** (or minimizes Impurity).

> [!NOTE] 🧮 Mathematical Criteria for Splitting
> The objective function depends on the task:
>
> | Criterion | Task | Formula | Logic |
> | :--- | :--- | :--- | :--- |
> | **Gini Impurity** | Classification | $G = 1 - \sum_{k=1}^{C} p_k^2$ | Measures probability of misclassification. **0 = Pure Node**. Computationally fast. |
> | **Entropy** | Classification | $H = - \sum_{k=1}^{C} p_k \log_2(p_k)$ | Measures "disorder" (Shannon Info Theory). Tends to create more balanced trees. |
> | **Variance** | Regression | $\text{MSE} = \sum (y_i - \bar{y})^2$ | Minimizes the squared error within the leaf node. |

---

## 2. Ensemble Mathematics (Bagging vs Boosting)
A single Decision Tree has **High Variance** (it changes drastically with small data changes). We use Ensembles to fix this.

### A. Bagging (Bootstrap Aggregating)
*   **Flagship Algorithm:** **[[Random Forest]]**.
*   **Philosophy:** "Parallel Democracy".
*   **The Math (Variance Reduction):**
It relies on the statistical property that averaging uncorrelated random variables reduces variance.
$$ \hat{y}_{bagging} = \frac{1}{B} \sum_{b=1}^{B} f_b(x) $$
*   **Bootstrapping:** Each tree $f_b$ is trained on a random sample drawn *with replacement*.
*   **Feature Randomness:** At each split, only a random subset of features is considered. This decorrelates the trees, ensuring $\text{Var}(\text{Ensemble}) \ll \text{Var}(\text{Single Tree})$.

### B. Boosting
*   **Flagship Algorithm:** **[[XGBoost]]** / GBM.
*   **Philosophy:** "Sequential Improvement".
*   **The Math (Bias Reduction via Gradients):**
The model is constructed as an additive sum of weak learners.
$$ F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x) $$
1.  **$F_{m-1}(x)$**: The prediction of the ensemble at step $m-1$.
2.  **$h_m(x)$**: A new tree trained to predict the **Pseudo-Residuals** (the error of the previous step).
    $$ r_{i} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right] $$
3.  **$\eta$ (Learning Rate):** Scales the contribution of the new tree to prevent overfitting.

---

## 3. Scikit-Learn Implementation

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Single Decision Tree (Base Estimator)
# High Interpretability, High Variance
dt = DecisionTreeClassifier(criterion='gini', max_depth=None)

# 2. Random Forest (Bagging)
# Parallel training. Good default for most problems.
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# 3. Gradient Boosting (Boosting)
# Sequential training. High accuracy, needs tuning (learning_rate).
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Training (Interface is identical)
rf.fit(X_train, y_train)
```