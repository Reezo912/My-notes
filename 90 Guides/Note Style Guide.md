---
type: guide
domain: guides
audience:
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# Note Style Guide

This note is the living authoring guide for the vault. It is meant for both humans and AI agents and should be updated whenever the note-writing rules change.

## Purpose Of The Vault
This vault is a shared AI and ML knowledge base designed to work in two modes:
- **Reference mode:** a reader can jump into one note and understand the concept quickly.
- **Study mode:** a reader can follow the index notes as a guided learning path.

## Note Taxonomy
| Note Class | Purpose | Default Style |
| :--- | :--- | :--- |
| **Substantive note** | Main concept, method, metric, or system note | High visual density |
| **Index note** | Navigation, grouping, and study-path note | High visual density |
| **Dashboard note** | Operational browsing, maintenance, or review note powered by metadata | Structured support layer |
| **Micro bridge note** | Short connector note for a narrow concept | Compact visual style |

## Vault Structure
- `00 Home`: root navigation and top-level index notes
- `01 Foundations`: data and statistical foundations
- `02 Data Preparation`: encoding, imputation, standardization, and imbalance handling
- `03 Classical ML`: model families, metrics, and tabular ML notes
- `04 Deep Learning & NLP`: neural networks, sequence models, NLP, language models, and `RAG`
- `05 Agentic Systems`: agents, orchestration, evaluation, and research log notes
- `90 Guides`: shared documentation like this guide
- `99 Archive`: deprecated or replaced notes only

## Metadata Rules
Use frontmatter on canonical notes.

Minimum fields:
```yaml
---
type: concept
domain: agentic-systems
audience:
  - learner
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
```

Allowed values:
- `type`: `index`, `concept`, `bridge`, `guide`, `research-log`, `dashboard`
- `domain`: `home`, `foundations`, `data-preparation`, `classical-ml`, `dl-nlp`, `agentic-systems`, `guides`
- `status`: `draft`, `evergreen`, `roadmap`, `archived`

## Metadata Layer: Bases And Dataview
Frontmatter is the canonical metadata source. The dashboard layer reads from it and should not introduce parallel classification.

Use the dashboard stack like this:
- `Bases`: browse, group, sort, and edit note properties interactively.
- `Dataview`: query note inventories, review queues, and editorial exceptions.
- `Home.md` and the index notes: remain the main human navigation layer.
- `Dashboard notes`: operational support notes that present the Bases and Dataview layer without replacing the pedagogical index layer.

Canonical dashboard files:
- `00 Home/Vault Dashboard.md`
- `90 Guides/Editorial Dashboard.md`
- `00 Home/Vault Catalog.base`
- `90 Guides/Editorial Review.base`

Rules:
- If note classes, domains, or review fields change, update the dashboard layer in the same change.
- Prefer Bases for stable grouped views by `domain`, `type`, `status`, and `last_reviewed`.
- Prefer Dataview for review debt, missing-section checks, and editorial maintenance.

## Required Structure For Substantive Notes
Substantive notes should usually follow this order:
1. `# Title`
2. short definition or framing sentence
3. `Why It Matters`
4. one visual anchor near the top
5. core concepts or mechanics
6. tradeoffs, pitfalls, or decision rules
7. related notes
8. source section when provenance matters

## Visual Grammar
### High Visual Density
Use this for substantive notes and index notes.
- 3 to 5 meaningful callouts
- at least one strong visual anchor
- explicit sections for intuition, mechanics, pitfalls, and use cases when relevant
- tables for comparisons, decision rules, or cheat sheets
- Mermaid only when it clarifies flow, architecture, or decision logic

### Compact Visual Style
Use this for micro bridge notes.
- 0 to 1 callouts
- at most one visual anchor
- short explanation focused on definition, why it matters, and related notes
- avoid over-designing narrow bridge concepts

## Callout Grammar
| Callout | Use |
| :--- | :--- |
| `> [!INFO]` | definition, framing, mental model |
| `> [!IMPORTANT]` | critical distinction or rule |
| `> [!WARNING]` | common mistake or misuse |
| `> [!CAUTION]` | tradeoff or subtle risk |
| `> [!FAILURE]` | failure mode or anti-pattern |
| `> [!TIP]` | practical default or useful shortcut |
| `> [!NOTE]` | local context or implementation note |

Rules:
- callouts must add scanability, not decoration
- do not stack many callouts with the same message
- place the strongest callouts near the sections where the decision matters

## Mermaid Rules
- Prefer `flowchart TD` or `flowchart TB` when labels are longer than a few words.
- Shorten node labels aggressively.
- If a diagram is still too wide, split it into two smaller diagrams or replace it with a table.
- Use Mermaid only for:
  - flow
  - architecture
  - decision logic
- Do not use Mermaid for simple comparisons that a table explains better.

## Table Rules
- Use tables for metric comparisons, model-family comparisons, or decision heuristics.
- Keep headers short and focused on the decision the reader needs to make.
- Prefer one strong table over several weak tables.

## Linking Rules
- Link to prerequisites and adjacent concepts intentionally.
- Substantive notes should usually list both prerequisites and related notes.
- Do not create duplicate concept ecosystems; link back to the canonical note.
- Micro bridge notes should remain connected but lightweight.

## Curriculum Principles
The top-level learning progression for the vault is:
- foundations -> data preparation -> classical ML -> deep learning and NLP -> agentic systems

Each major index in `00 Home` should answer:
- who the branch is for
- what the prerequisites are
- what comes next after the branch

Rules:
- Keep prerequisite blocks compact and explicit.
- Keep a short `Where This Leads` or `Read Next` block in each major index.
- Let `Home.md` act as the global portal and each index act as a branch-specific guide.
- Treat `RAG` as useful context before agentic systems, not as a mandatory gate for first-pass learning.

## Validation Workflow
- Agents may deploy sub-agents proactively when research, verification, or multi-perspective review materially improves the note or structure change.
- The vault workflow does not require the user to re-authorize sub-agent validation on each substantial task.
- Use at least one validation pass for substantial changes when correctness, structure, or audience fit could drift.
- Use multi-perspective review for high-impact changes such as:
  - new branches
  - major structure migrations
  - large content rewrites
  - strategy-sensitive notes
- Add a live Obsidian-facing QA pass when the change materially affects navigation, dashboards, Mermaid sizing, or workspace ergonomics.
- Skip sub-agent deployment for trivial fixes that do not benefit from extra review.
- After validation, summarize the conclusions and close review agents that are no longer needed.

## Source And Citation Rules
- Use a `Source` section when the note is meaningfully derived from a book, paper, or canonical technical source.
- Light citations are enough; the goal is provenance, not academic formatting.

## When To Use High Vs Compact
Use **high** for:
- metrics
- comparison-heavy notes
- data-preparation notes
- model-family notes
- system and pipeline notes
- index notes

Use **compact** for:
- narrow bridge concepts
- small auxiliary notes whose main role is linking, not teaching at length

## Templates
### Substantive Note Template
- `# Title`
- one-sentence framing
- `> [!INFO] Core idea`
- `## Why It Matters`
- `## Visual Map`
- one Mermaid diagram or one strong comparison table
- `## Core Concepts`
- `> [!WARNING]` or another callout where the main mistake or tradeoff appears
- `## Tradeoffs Or Decision Rules`
- `## Related Notes`
- `## Source` when provenance matters

### Index Note Template
- `# Index Title`
- `> [!INFO] Start here`
- compact prerequisite block
- `## Study Path`
- one Mermaid study-path block
- `## Where This Leads`
- `## Reference Groups`
- one compact grouping table

### Dashboard Note Template
- `# Dashboard Title`
- `> [!INFO] Start here`
- one short explanation of what the dashboard is for
- one compact view table with links to `.base` files or related dashboards
- one or more embedded Bases views
- one or more Dataview blocks for maintenance or browsing
- `## Last Reviewed`

### Micro Bridge Note Template
- `# Title`
- short definition
- one useful callout, usually `TIP`, `WARNING`, or `CAUTION`
- short related-notes section

## Style Change Log
| Date | Change | Rationale |
| :--- | :--- | :--- |
| 2026-04-10 | Promoted older bridge notes into substantive notes, added Bases and Dataview dashboards, formalized curriculum blocks in major indexes, and required live Obsidian QA for navigation-heavy changes. | Bring the vault to a scaling checkpoint where content quality, metadata browsing, curriculum design, and visual QA all reinforce each other. |
| 2026-04-10 | Moved the vault to a domain-folder structure, added frontmatter metadata rules, and formalized proactive sub-agent validation as part of the vault workflow. | Improve physical navigation, make classification explicit, and let AI agents validate research and structure changes without waiting for repeated user prompts. |
| 2026-04-09 | Moved the vault to a high-visual authoring system for substantive and index notes; kept micro bridge notes compact; added Mermaid sizing policy and vertical-diagram preference. | Improve scanability in Obsidian while avoiding horizontal page scrolling and keeping bridge notes lightweight. |
