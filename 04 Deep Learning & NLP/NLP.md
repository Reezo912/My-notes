---
type: concept
domain: dl-nlp
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# NLP

Natural Language Processing (NLP) is the field of building systems that analyze, represent, and generate human language.

> [!INFO] Core idea
> NLP sits between general machine learning and language-specific systems. It turns raw text into something models can represent, compare, classify, search, or generate from.

## Why It Matters
NLP is the bridge from general learning systems into language tasks such as search, classification, extraction, translation, question answering, and language modeling.

## Concept Map
| Layer | Main Question | Typical Concepts |
| :--- | :--- | :--- |
| text representation | how do we turn language into model inputs? | tokenization, embeddings |
| sequence understanding | how do we preserve order and context? | recurrence, attention |
| language tasks | what do we want the system to do? | classification, retrieval, generation |

> [!IMPORTANT] NLP is broader than language models
> Modern LLMs dominate today’s spotlight, but NLP also includes classical text pipelines, representation learning, retrieval, and task-specific modeling.

## Core Concepts
### Representation
- tokenization
- embeddings
- text as sequences rather than unordered features

### Sequence Modeling
- recurrence in older architectures such as [[LSTMs (Long Short-Term Memory)]]
- context mixing through [[Attention]]
- scaling toward modern [[Language Models]]

### Typical Tasks
- text classification
- information extraction
- semantic search and retrieval
- generation and dialogue

> [!WARNING] Language is not just “text features”
> Order, ambiguity, context, and grounding matter much more in NLP than in many simpler tabular workflows.

## Tradeoffs And Decision Rules
### When To Start Here
- use this note when you need the conceptual bridge from generic ML into text systems
- use it before diving into language models, retrieval, or agentic language systems

### When Not To Stop Here
- do not treat NLP as one single model family
- do not assume token generation alone solves retrieval, grounding, or task design

> [!CAUTION] Better language generation does not remove system design
> Even with strong language models, problems like retrieval, evaluation, and tool use still live at the system layer.

> [!TIP] Practical default
> Start with the question “Do I need understanding, retrieval, or generation?” That choice usually determines which NLP path matters next.

## Related Notes
- Prerequisites: [[Neural Networks]]
- Related: [[Language Models]], [[LSTMs (Long Short-Term Memory)]], [[Attention]], [[RAG (Retrieval Augmented Generation)]]

## Sources
- Jurafsky and Martin, *Speech and Language Processing*.
- Tunstall, von Werra, and Wolf, *Natural Language Processing with Transformers*.

## Last Reviewed
- 2026-04-10
