## 1. The Hierarchy of Missingness
Before fixing nulls in **[[Data Imputation]]**, you must diagnose *why* they are there. Statisticians divide missing data into three categories based on the **relationship between the missing data and the values**.

| Type | Name | Risk Level | Logic |
| :--- | :--- | :--- | :--- |
| **MCAR** | Missing **Completely** At Random | 🟢 Low | Pure bad luck. No pattern. |
| **MAR** | Missing At Random | 🟡 Medium | Missingness depends on *observed* data (e.g., Gender). |
| **MNAR** | Missing **Not** At Random | 🔴 High | Missingness depends on the *missing value itself*. |

---

## 2. 🟢 MCAR (Missing Completely At Random)
The probability of being missing is the same for all observations. There is **no correlation** between missingness and any values (observed or unobserved).

*   **Analogy:** A lab assistant trips and drops a random test tube. It didn't break because the liquid was hot or blue; it broke because of random chance.
*   **Math:** $P(Missing | Data) = P(Missing)$
*   **Solution:**
*   Safe to drop rows (**Listwise Deletion**).
*   Safe to impute with Mean/Median.
*   Does not introduce **[[Bias in Machine Learning]]**.

---

## 3. 🟡 MAR (Missing At Random)
The name is misleading. It is **NOT** random. The probability of missingness depends on **other observed variables** in your dataset.

*   **Scenario:** You are collecting "Weight" and "Gender".
*   **The Pattern:** Women are less likely to disclose their weight than men.
*   **Why it's MAR:** If you look at just the "Weight" column, it looks random. But if you group by "Gender", you see the pattern. Since you *have* the Gender column, you can account for it.
*   **Math:** $P(Missing | Y) = P(Missing | X)$
*   *(Missingness of Y depends on X)*.
*   **Solution:**
*   **Do NOT drop rows** (You will under-represent women).
*   Use **[[Regression Imputation]]** or **[[KNN Imputation]]** (Predict Weight based on Gender).

---

## 4. 🔴 MNAR (Missing Not At Random)
The probability of missingness depends on the **value of the missing data itself**. This is the most dangerous type.

*   **Scenario:** A survey asks: *"Have you ever committed a crime?"*.
*   **The Pattern:** People who *have* committed a crime are likely to leave it blank to hide the truth.
*   **Why it's MNAR:** The missingness correlates with the answer "Yes". You cannot predict this using other columns because the cause is hidden.
*   **Math:** $P(Missing | Y) = P(Missing | Y)$
*   *(Missingness of Y depends on Y)*.
*   **Solution:**
*   **NEVER Impute with Mean** (You will bias the result towards "No Crime").
*   Create a specific **"Unknown" Category** or Flag.
*   Use specialized models like **Selection Models** or acknowledge the **[[Selection Bias]]**.

---

## 5. Summary & Decision Logic

> [!TIP] Diagnostic Test (Little's Test)
> In Python, you can perform **Little's MCAR Test** (library `pyampute` or `statsmodels`).
> *   p-value < 0.05 $\rightarrow$ Not MCAR (It's MAR or MNAR).
> *   p-value > 0.05 $\rightarrow$ MCAR.

### Decision Flowchart
1.  **Is the missingness pure luck?**
*   YES $\rightarrow$ **MCAR**. (Use `dropna` or Mean).
2.  **Can I explain the missingness using other columns I have?**
*   YES $\rightarrow$ **MAR**. (Use Regression/KNN Imputation).
3.  **Is the data missing because of the value itself?**
*   YES $\rightarrow$ **MNAR**. (Use Flag/Category `-1`).