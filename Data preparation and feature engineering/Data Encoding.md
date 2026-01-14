# Data Encoding

## 1. Why Encoding?
Most Machine Learning algorithms (Linear Regression, Neural Nets, SVM) operate on mathematical equations ($y = wx + b$). They cannot understand text like "Red" or "Blue". We must translate these categories into numbers.

> [!FAILURE] Systemic Risks of Bad Encoding
> Choosing the wrong strategy can destroy your model:
> *   **Curse of Dimensionality:** Creating 10,000 columns for Zip Code crashes the RAM.
> *   **False Relationships:** Telling the model that $Blue (2) > Red (1)$ when colors have no order.
> *   **Overfitting:** Memorizing the target value of rare categories (Target Leakage).

---

## 2. Pre-Encoding Challenges

### A. High Cardinality
When a feature has thousands of unique values (e.g., User ID, Zip Code).
*   **The Problem:** One-Hot Encoding would explode the feature space.
*   **Strategies:**
1.  **Group Rare:** Keep top-N categories and rename the rest to `"Other"`.
2.  **Frequency Encoding:** Replace category with its count.
3.  **Embeddings:** (See Section 4).

### B. Missing Values
Before encoding, you must handle nulls.
*   See **[[Handling Missing Data]]**.
*   *Specific Trick:* For trees, you can treat "Missing" as just another category to encode.

---

## 3. Standard Encoding Strategies

### Option A: One-Hot Encoding (OHE)
Creates a new binary column (0/1) for each unique category.
*   **Analogy:** "Is it Red?" (Yes/No), "Is it Blue?" (Yes/No).

> [!TIP] Best For
> *   Nominal data (No order).
> *   Low Cardinality (< 10-20 categories).
> *   **[[Linear Models]]** (They need OHE to avoid assuming order).

> [!WARNING] Drawbacks
> *   **Sparsity:** Creates a massive matrix of mostly zeros.
> *   **Tree Inefficiency:** **[[Tree-based Models]]** struggle with sparse data (need deeper trees to split).

### Option B: Ordinal / Label Encoding
Assigns an integer to each category (e.g., Low=1, Med=2, High=3).

> [!TIP] Best For
> *   **Ordinal Data:** When there is a clear rank (S, M, L, XL).
> *   **[[Tree-based Models]]**: Trees handle integers natively perfectly.

> [!FAILURE] The "Label" Trap
> If you use this on **Nominal Data** (e.g., Cities: Paris=1, London=2) with a **[[Linear Models|Linear Model]]**, the model will learn that $London > Paris$. This is false bias.

### Option C: Target Encoding (Mean Encoding)
Replaces the category with the **mean of the target variable** for that group.
*   *Example:* If "Madrid" users buy 80% of the time, Madrid becomes `0.80`.

> [!CAUTION] Overfitting Risk (Data Leakage)
> If a category appears only once and has Target=1, it gets encoded as `1.0`. The model "memorizes" the answer.
> *   **Fix:** Use **Smoothing** (blend with global mean) and compute on **Cross-Validation folds**.

---

## 4. Advanced: Entity Embeddings
The state-of-the-art for High Cardinality in Deep Learning (and increasingly in Tabular ML).

> [!INFO] Concept
> An embedding maps each category to a **Dense Vector** of float numbers in a lower-dimensional space.
> *   Instead of 10,000 columns (OHE), you represent "Zip Code" with just 5 numbers.
> *   These numbers are **weights learned by a Neural Network**.

### Why is it magic?
1.  **Semantic Similarity:** The network learns to place similar categories closer together geometrically.
*   *Example:* "Monday" and "Tuesday" vectors will be close. "Saturday" and "Sunday" vectors will be close to each other but far from "Monday".
2.  **Dimensionality Reduction:** Compresses massive sparse data into compact dense info.
3.  **Transfer Learning:** You can train embeddings on one task and reuse them on another.

### Implementation Logic
Usually implemented as the first layer of a **[[Neural Networks|Neural Network]]** (Look-up Table).
```python
# Keras / TensorFlow Concept
model.add(Embedding(input_dim=10000, output_dim=5)) 
# Learns a 5-number vector for each of the 10,000 zip codes
```

### Model Compatibility & Workflow
Embeddings are not just for Neural Networks. You can use them in two ways:

**1. Native Integration (End-to-End Deep Learning)**
*   **Models:** [[Neural Networks]] (Keras, PyTorch, TensorFlow).
*   **Workflow:** The Embedding is just the **first layer** of the network. The vector values (weights) are updated via **Backpropagation** simultaneously with the rest of the model during training.

**2. As a Feature Extractor (The "Two-Stage" Pattern)**
*   **Models:** **[[Tree-based Models]]** (XGBoost, RF) or **[[Linear Models]]**.
*   **Workflow:**
1.  Train a small Neural Network to predict the target using the categorical column.
2.  **Extract the weights** (the learned vectors) from the Embedding layer.
3.  **Replace** the original categorical column in your dataset with these new vector columns (e.g., `City` becomes `City_Vec_1`, `City_Vec_2`...).
4.  Train your standard XGBoost/Linear model on this new numeric dataset.
> [!TIP] Engineering Benefit
> This trick allows a **[[Linear Models|Linear Regression]]** to understand non-linear semantic relationships (e.g., that "Monday" is close to "Tuesday") without complex feature engineering.

---

## 5. Summary Cheat Sheet

| Data Type | Cardinality | Model | Recommended Method |
| :--- | :--- | :--- | :--- |
| **Nominal** (Color) | Low | Linear | One-Hot |
| **Nominal** (Zip Code) | High | Tree | Target Encoding / Frequency |
| **Nominal** (Zip Code) | High | Neural Net | **Embeddings** |
| **Ordinal** (T-Shirt) | Any | Any | Ordinal (Integer) |

![[Captura de pantalla 2025-12-17 a las 13.23.12.png]]