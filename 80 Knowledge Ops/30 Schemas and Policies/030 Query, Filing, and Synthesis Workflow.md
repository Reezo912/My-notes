---
type: policy
domain: knowledge-ops
audience:
  - builder
  - data-strategy
aliases:
  - Query, Filing, and Synthesis Workflow
status: evergreen
last_reviewed: 2026-04-20
---
# Query, Filing, and Synthesis Workflow

This workflow governs what happens after a question, exploration session, or prompt yields a reusable result.

> [!INFO] Good answers should compound
> If a query produces a durable comparison, synthesis, or open-question frame, it should be filed back into the workspace instead of disappearing into chat history.

## Filing Workflow
1. Start from `hot context`, not the whole vault.
2. Read the relevant source notes or candidate pages.
3. Generate the answer.
4. Decide whether the output is ephemeral, draft-worthy, or promotion-worthy.
5. File it into the domain workspace if it deserves persistence.

## Source-Driven Ingest Default
For a newly introduced source, the default flow is:
1. create the source note
2. summarize and discuss key takeaways with the user
3. capture the emphasis in the workspace
4. create a draft or canonical candidate if warranted
5. stop there unless the user explicitly approves promotion

> [!IMPORTANT] Discussion comes before promotion
> A good `full Karpathy` ingest is not “read source, write canon.” It is “read source, discuss what matters, file the durable result, then promote only with explicit approval.”

## Output Classes
| Output | Where It Lives | Promotion Rule |
| :--- | :--- | :--- |
| ephemeral chat answer | nowhere | do not file |
| draft synthesis | domain workspace | file with `knowledge_state: draft` |
| comparison or decision note | domain workspace | promote only if it will be reused |
| patch candidate for canon | domain workspace plus promotion queue | review before canonical edit |
| question note | domain workspace | keep if it guides future ingest or lint |

> [!IMPORTANT] File only durable value
> Do not turn every response into a permanent note. The storage layer should compound knowledge, not archive every moment of model output.

## Save-Back Criteria
Save an output only when it does at least one of these:
- compresses repeated reasoning
- resolves a recurring comparison
- clarifies contradictions between sources
- proposes a canonical patch that is likely to survive review
- creates a useful question or watchlist for future research

## Default Stop Points
| Situation | Default Stop Point |
| :--- | :--- |
| simple query with no durable value | ephemeral answer only |
| useful answer from existing sources | workspace draft if reusable |
| first ingest of a new source | source note plus workspace candidate |
| canon-worthy source-backed update | promotion queue, then human approval |

## Related Notes
- Related: [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/030 Agentic Systems Draft Inbox|Agentic Systems Draft Inbox]], [[80 Knowledge Ops/30 Schemas and Policies/040 Promotion and Canon Policy|Promotion and Canon Policy]], [[80 Knowledge Ops/40 Registries and Logs/030 Promotion Queue|Promotion Queue]]

## Sources
- [LLM Wiki | Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [Effective context engineering for AI agents | Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

## Last Reviewed
- 2026-04-21
