## 1. Theoretical Foundation
Neural Networks (Deep Learning) are **Universal Function Approximators**. Theoretically, a network with enough neurons can approximate *any* mathematical function, no matter how complex or non-linear.

### The Atomic Unit: The Perceptron
A single neuron is essentially a **[[Linear Models|Linear Regression]]** wrapped in a generic "Activation Function".

$$ z = \sum (w_i x_i) + b $$
$$ \hat{y} = \sigma(z) $$

*   **$w$ (Weights):** The importance of each feature (Learned).
*   **$b$ (Bias):** The activation threshold (Learned).
*   **$\sigma$ (Activation):** The non-linear transformation.

> [!FAILURE] The Linear Trap
> If you stack 100 layers of neurons **without** non-linear activation functions, mathematically you just have **one single Linear Regression**.
> $$ W_2(W_1 x) = W_{combined} x $$
> The **Activation Function** is what allows the network to learn curves and complex patterns.

---

## 2. Key Components

### A. Activation Functions
They dictate whether a neuron should "fire" or not.

| Function | Formula | Use Case | Pros/Cons |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | Output Layer (Binary Class) | ❌ Vanishing Gradient problem in deep nets. |
| **ReLU** | $\max(0, z)$ | Hidden Layers (Standard) | ✅ Fast, solves Vanishing Gradient. ❌ Dead Neurons (if inputs < 0). |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Output Layer (Multi-Class) | ✅ Returns probabilities that sum to 1. |

### B. The Engine: Backpropagation
How does the network learn millions of weights?
1.  **Forward Pass:** Data flows through the network $\rightarrow$ Prediction $\hat{y}$.
2.  **Loss Calculation:** Compare $\hat{y}$ vs real $y$ (e.g., using MSE or Log-Loss).
3.  **Backward Pass (Backprop):** Calculate the **Gradient** of the loss with respect to *every* weight using the **Chain Rule**.
$$ \frac{\partial Loss}{\partial w} = \frac{\partial Loss}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w} $$
4.  **Update:** Adjust weights in the opposite direction of the gradient (Gradient Descent).

---

## 3. Common Architectures

### A. MLP (Multi-Layer Perceptron)
*   **Structure:** Dense layers fully connected.
*   **Use Case:** **Tabular Data** (Alternatives to [[Tree-based Models]]).
*   **Input:** Flat vectors (Rows).

### B. CNN (Convolutional Neural Networks)
*   **Structure:** Filters (Kernels) that slide over the input.
*   **Use Case:** **Images** / Spatial Data.
*   **Feature:** Spatial Invariance (a cat is a cat whether it's in the top-left or bottom-right).

### C. RNN / Transformers
*   **Structure:** Loops (RNN) or Attention Mechanisms (Transformers).
*   **Use Case:** **Sequences** (Time Series, NLP, Audio).
*   **Feature:** Context awareness (Position $t$ depends on $t-1$).

---

## 4. Preprocessing Mandates
Neural Networks are "Divas". Unlike [[Tree-based Models]], they require perfect data hygiene.

1.  **No Missing Values:** Math breaks with NaNs. See **[[Data Imputation]]**.
2.  **Scaling:** Inputs must be normalized (0-1) or standardized (Mean 0, Std 1). Large inputs cause gradients to explode or vanish. (See **[[Data Standardization]]**).
3.  **Encoding:** Categorical variables must be OHE or use **[[Data Encoding#4. Advanced: Entity Embeddings|Embeddings]]**.

---

## 5. Implementation

### ⚡ PySpark ML (MultilayerPerceptron)
> [!WARNING] Databricks Limitation
> Spark ML's native `MultilayerPerceptronClassifier` is **very limited** (only supports basic MLP). For real Deep Learning (CNN, BERT), Databricks uses **TorchDistributor** or **HorovodRunner** to distribute PyTorch/TensorFlow code.

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Define Architecture
# Input layer: 4 features
# Hidden layers: 5 neurons, 4 neurons
# Output layer: 3 classes
layers = [4, 5, 4, 3] 

# 2. Define Model
mlp = MultilayerPerceptronClassifier(
layers=layers, 
seed=1234, 
blockSize=128 # Batch size
)

# 3. Fit
model = mlp.fit(train_df)

# 4. Evaluate (Using Classification Metrics)
result = model.transform(test_df)
predictionAndLabels = result.select("prediction", "label")
```

### 🧠 Keras / TensorFlow (Standard)
The API you will likely use inside a Single Node or with a distributor.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Define Architecture
model = Sequential([
Dense(12, input_dim=8, activation='relu'), # Hidden 1
Dense(8, activation='relu'),               # Hidden 2
Dense(1, activation='sigmoid')             # Output (Binary)
])

# 2. Compile (Define Optimizer and Loss)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Fit
model.fit(X_train, y_train, epochs=150, batch_size=10)
```