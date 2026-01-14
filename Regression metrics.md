## 1. The Goal
Unlike classification (Right/Wrong), in Regression we measure **Error Magnitude**: "How far off was the prediction from the real number?".

---

## 2. Key Metrics

### A. RMSE (Root Mean Squared Error)
The standard metric for most industries.
$$ RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2} $$
*   **Characteristic:** Since it squares the error, it **punishes large errors (Outliers) heavily**.
*   **When to use:** When big mistakes are unacceptable (e.g., Safety systems, High-Frequency Trading).

### B. MAE (Mean Absolute Error)
$$ MAE = \frac{1}{n} \sum |y - \hat{y}| $$
*   **Characteristic:** Linear penalty. It is **robust to outliers**.
*   **When to use:** When you have messy data with outliers (e.g., Salary prediction where billionaires distort the mean).

### C. R-Squared ($R^2$)
$$ R^2 = 1 - \frac{\text{Unexplained Variation}}{\text{Total Variation}} $$
*   **Definition:** Percentage of the variance in the target explained by the features.
*   **Range:** $-\infty$ to $1.0$. (1.0 is perfect).
*   **Trap:** $R^2$ always increases if you add more features (even useless ones). Use **Adjusted $R^2$** for feature selection.

---

## 3. Comparison Cheat Sheet

| Metric | Penalizes Outliers? | Unit of Measure | Use Case |
| :--- | :--- | :--- | :--- |
| **RMSE** | 🚨 YES (Heavily) | Same as Target | Default. Good for Gaussian errors. |
| **MSE** | 🚨 YES (Extreme) | Target Squared | Math optimization (Gradient Descent), not for reporting. |
| **MAE** | ✅ NO (Robust) | Same as Target | Financial data, Skewed distributions. |
| **R2** | -- | Percentage % | Explaining model quality to business stakeholders. |

---

## 4. Implementation

### Python (Scikit-Learn)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Sklearn doesn't have direct RMSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse} (Penalty for outliers)")
print(f"MAE: {mae} (Real average error)")