---
type: guide
domain: guides
audience:
  - builder
status: evergreen
last_reviewed: 2026-04-21
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
- `80 Knowledge Ops`: operational layer for source intake, domain workspaces, promotion queues, and lint
- `90 Guides`: shared documentation like this guide
- `99 Archive`: deprecated or replaced notes only

Large branches may introduce numbered subfolders when a flat folder stops being easy to maintain. The folder hierarchy should still support the index layer rather than replacing it.

Current example:
- `05 Agentic Systems/00 Core`
- `05 Agentic Systems/10 Software Engineering Agents`
- `05 Agentic Systems/20 Applied Agentic Architectures`
- `05 Agentic Systems/90 Research and Roadmap`

Within those subfolders, notes may use zero-padded numeric filename prefixes such as `010`, `020`, and `030` to make reading order visible in Obsidian's file explorer. Keep the `# Title` clean and add an `aliases` entry with the unnumbered note name so existing wikilinks remain stable.

Repository-facing publication files such as `README.md` and localized variants like `README.es.md` are allowed at root, but they are not canonical vault notes.

`80 Knowledge Ops` is an operational branch, not a curriculum branch. Readers should still navigate by `Home.md` and the major indexes. `Knowledge Ops` exists to support ingest, draft maintenance, lint, and canonical promotion.

## Metadata Rules
Use frontmatter on canonical notes.

Exception:
- `README.md` and localized publication files are GitHub-facing repository documentation and do not need canonical note frontmatter.

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
- `type`: `index`, `concept`, `bridge`, `guide`, `research-log`, `dashboard`, `source`, `ops-log`, `policy`
- `domain`: `home`, `foundations`, `data-preparation`, `classical-ml`, `dl-nlp`, `agentic-systems`, `knowledge-ops`, `guides`
- `status`: `draft`, `evergreen`, `roadmap`, `archived`

Optional `Knowledge Ops` fields:
- `knowledge_state`: `raw`, `normalized`, `draft`, `canonical-candidate`, `promoted`, `stale`, `archived`
- `review_state`: `unreviewed`, `in-review`, `approved`, `rejected`
- `source_kind`: `pdf`, `web-article`, `paper`, `book-chapter`, `video-transcript`, `podcast-transcript`, `github-repo`, `thread`, `meeting-note`, `other`
- `target_domains`: list of target domains for routing or promotion
- `source_url` and/or `source_path`
- `canonical_targets`: optional list of likely canonical destinations

## Audience Contract
The `audience` field is a navigation contract, not decorative metadata. Use it to signal who should confidently enter a note and what kind of help they should expect from it.

| Audience | Use When | Main Reader Question |
| :--- | :--- | :--- |
| `learner` | the note supports guided study, shared mental models, prerequisites, or reading order | "How do I understand this in sequence?" |
| `builder` | the note helps design, implement, operate, validate, or troubleshoot a real system or workflow | "How do I use this to build or run something?" |
| `data-strategy` | the note supports decision-making about investment, readiness, ROI, governance, rollout, or operating model | "Should we do this, and under what constraints?" |

Rules:
- `builder` can be broad, but do not use it as an automatic default when the note does not materially help implementation or operations.
- `learner` should appear on index notes and substantive notes that genuinely teach a sequence, a prerequisite chain, or a shared conceptual foundation.
- `data-strategy` should be used deliberately. Add it only when the note contains decision framing, economics, governance, rollout, or operating-model guidance that a strategy reader can act on directly.
- Revisit `audience` whenever a note gains or loses an executive lens, a study-path role, or practical operator guidance.

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

Operational `Knowledge Ops` dashboards:
- `80 Knowledge Ops/90 Dashboards/010 Knowledge Ops Dashboard.md`
- `80 Knowledge Ops/90 Dashboards/Source Intake.base`
- `80 Knowledge Ops/90 Dashboards/Promotion Queue.base`
- `80 Knowledge Ops/90 Dashboards/Lint Review.base`

Rules:
- If note classes, domains, or review fields change, update the dashboard layer in the same change.
- Prefer Bases for stable grouped views by `domain`, `type`, `status`, and `last_reviewed`.
- Prefer Dataview for review debt, missing-section checks, and editorial maintenance.
- Do not rely on folder paths as the primary curriculum. Use folder structure for maintainability and use indexes for the human learning path.

## Knowledge Ops Workflow
Use `80 Knowledge Ops` as the runtime layer for `full Karpathy` work:
- `00 Intake/raw`: immutable source intake
- `10 Source Notes`: one normalized note per source
- `20 Domain Workspaces`: draft and synthesis layer by canonical domain
- `30 Schemas and Policies`: operating contract for ingest, query, lint, and promotion
- `40 Registries and Logs`: global index, activity log, promotion queue, lint queue, and canonical target map
- `90 Dashboards`: Bases and Dataview views for operational maintenance

Rules:
- `80 Knowledge Ops` is agent-owned working infrastructure
- canonical branches `01` through `05` remain human-supervised canon
- do not promote meaningful canonical changes outside the explicit promotion workflow
- use `knowledge_state` and `review_state` to model lifecycle without replacing the vault-wide `status` field

## Required Structure For Substantive Notes
Substantive notes should usually follow this order:
1. `# Title`
2. short definition or framing sentence
3. `Why It Matters`
4. one visual anchor near the top
5. core concepts or mechanics
6. tradeoffs, pitfalls, or decision rules
7. related notes
8. `Sources` section when provenance matters

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
- how `learner`, `builder`, and `data-strategy` should enter it

Rules:
- Keep `Home.md` audience-first. The first visible routes should be `learner`, `builder`, and `data-strategy`, with branch indexes presented as the second navigation layer.
- Each major index should include a visible `Best Route By Audience` section with the same three questions answered for each role: where to start, when not to start there, and where to continue next.
- Keep prerequisite blocks compact and explicit.
- Keep a short `Where This Leads` or `Read Next` block in each major index.
- Let `Home.md` act as the global portal and each index act as a branch-specific guide.
- Treat `RAG` as useful context before agentic systems, not as a mandatory gate for first-pass learning.
- When a branch grows into multiple internal tracks, keep the branch index responsible for exposing both the conceptual track map and the physical folder map.
- When a note uses `Apprenticeship / Advanced / Mastery` tables, state whether the table is self-contained or assumes the shared branch core. If it assumes the core, list the core handoff notes immediately before or inside the table.

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

## Multi-Agent Operating Contract
- Default to `single-agent` execution for small local changes, narrow note edits, and low-risk maintenance work.
- Prefer multi-agent work when the task benefits from:
  - recency-sensitive research
  - broad source review
  - multi-audience editorial review
  - structure, link, or navigation validation
  - large branch expansions or migrations
- Recommended role patterns for this vault:
  - `research/source agent`
  - `audience/editorial reviewer`
  - `structure/link validator`
  - `domain specialist`
- Ownership rules:
  - only one writing agent should own a given file or path scope at a time
  - review agents should stay read-only by default
  - the principal agent is responsible for integrating outputs, resolving conflicts, and preserving the canonical editorial voice
- Validation rules:
  - substantial changes should include at least one explicit validation pass
  - high-impact changes should usually include multi-perspective review
  - navigation, dashboard, workspace, or large Mermaid changes should include real Obsidian QA when feasible
- Closure rule:
  - summarize what each review or research pass concluded and close sub-agents once they are no longer needed

## Source And Citation Rules
- Use a `## Sources` section when the note is meaningfully derived from a book, paper, or canonical technical source.
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
- `## Sources` when provenance matters

### Index Note Template
- `# Index Title`
- `> [!INFO] Start here`
- `## Best Route By Audience`
- one compact table covering `learner`, `builder`, and `data-strategy`
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
| 2026-04-21 | Formalized `learner`, `builder`, and `data-strategy` as a first-class audience contract; made `Home` audience-first; required `Best Route By Audience` in major indexes; and standardized `## Sources` as the canonical provenance heading. | Make the vault easier to enter by role, turn `audience` into a real navigation interface, and keep editorial validation aligned with the body-section contract used across dashboards and notes. |
| 2026-04-20 | Added `80 Knowledge Ops` as the operational layer for full-Karpathy workflows, extended metadata with `knowledge_state`, `review_state`, and `source_kind`, and formalized the `raw -> source -> workspace -> promotion` model. | Turn the vault into a supervised, agent-maintained knowledge system without replacing the curated curriculum and canonical branch structure. |
| 2026-04-18 | Formalized the multi-agent operating contract for this vault, including default role patterns, ownership rules, validation expectations, and closure rules. | Make multi-agent work explicit and safe for large content, navigation, and research changes without turning every task into ad hoc coordination. |
| 2026-04-18 | Allowed numbered subfolders inside large branches, formalized track-based organization within `05 Agentic Systems`, and clarified that indexes remain the primary navigation layer over physical folders. | Support deeper curriculum branches without letting a flat file layout or raw folder browsing become the main study path. |
| 2026-04-10 | Promoted older bridge notes into substantive notes, added Bases and Dataview dashboards, formalized curriculum blocks in major indexes, and required live Obsidian QA for navigation-heavy changes. | Bring the vault to a scaling checkpoint where content quality, metadata browsing, curriculum design, and visual QA all reinforce each other. |
| 2026-04-10 | Moved the vault to a domain-folder structure, added frontmatter metadata rules, and formalized proactive sub-agent validation as part of the vault workflow. | Improve physical navigation, make classification explicit, and let AI agents validate research and structure changes without waiting for repeated user prompts. |
| 2026-04-09 | Moved the vault to a high-visual authoring system for substantive and index notes; kept micro bridge notes compact; added Mermaid sizing policy and vertical-diagram preference. | Improve scanability in Obsidian while avoiding horizontal page scrolling and keeping bridge notes lightweight. |

## Last Reviewed
- 2026-04-21
