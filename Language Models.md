# Language Models

## 1. The Building Block: Tokens
The basic unit of a language model is a **token**. It is not necessarily a word; it can be a character, a word, or a part of a word.

![[Captura de pantalla 2025-12-29 a las 23.40.54.png]]

> [!INFO] Definition: Tokenization
> The process of breaking the original text into tokens.
> *   **Vocabulary:** The set of all distinct tokens a model can work with.

**The Alphabet Analogy:**
You can use a small number of tokens to construct a large number of distinct words, similar to how you can use a few letters in the alphabet to construct many words.
*   **Mixtral 8x7B:** ~32,000 vocabulary size.
*   **GPT-4:** ~100,256 vocabulary size.
*(Note: Tokenization method and vocabulary size are architectural decisions made by developers).*

---

## 2. Main Types of Language Models
There are two distinct families based on how they process these tokens.

### A. Masked Language Model (MLM)
**"The Fill-in-the-Blank Expert"**
*   **Mechanism:** Predicts missing tokens anywhere in a sequence using context from **both before and after** (bidirectional).
*   **Example:** Input *"My favorite __ is blue"* $\rightarrow$ Predicts *"color"*.
*   **Flagship Model:** **BERT** (Bidirectional Encoder Representations from Transformers).
*   **Use Cases:**
*   Text Classification.
*   Sentiment Analysis.
*   Code Debugging.

### B. Autoregressive Language Model (CLM)
**"The Next-Token Predictor"**
*   **Mechanism:** Predicts the **next token** in a sequence using **only the preceding tokens** (left-to-right).
*   **Capability:** They can continually generate tokens one after another.
*   **Flagship Models:** GPT Series.
*   **Use Cases:** Text Generation (Most popular type today).

---
