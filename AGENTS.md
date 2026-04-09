---
type: guide
domain: guides
audience:
  - builder
status: evergreen
last_reviewed: 2026-04-10
---
# AGENTS.md

This vault uses [[Note Style Guide]] as the canonical style reference for both humans and agents.

## Operating Rules
- Treat `Note Style Guide.md` as the source of truth for note structure, visual grammar, linking rules, and Mermaid usage.
- Follow the vault folder structure when creating or moving notes:
  - `00 Home`
  - `01 Foundations`
  - `02 Data Preparation`
  - `03 Classical ML`
  - `04 Deep Learning & NLP`
  - `05 Agentic Systems`
  - `90 Guides`
  - `99 Archive`
- Keep the root almost empty. Leave `AGENTS.md`, `.obsidian`, and `images` at root; place notes inside the domain folders.
- `README.md` and localized publication files such as `README.es.md` are allowed at root as GitHub-facing onboarding documents for the vault.
- Ensure all canonical notes, including new notes and migrated or normalized existing notes, use frontmatter with at least: `type`, `domain`, `audience`, `status`, and `last_reviewed`.
- Treat `README.md` and localized publication files as repository documentation, not as canonical vault notes. They do not need note frontmatter.
- Update `Note Style Guide.md` first whenever the authoring rules change.
- After updating the style guide, sync the summary in this file so agents keep the short operational contract current.

## Note Classes
- **Substantive note:** full concept notes, comparison notes, and systems notes. Default to high visual density.
- **Index note:** navigation and study-path notes. Also default to high visual density.
- **Dashboard note:** operational note backed by Bases or Dataview. Support the vault, but do not replace the index layer.
- **Micro bridge note:** short connector notes for adjacent concepts. Keep them compact with at most one visual anchor.

## Default Behaviors
- Use high visual density for substantive notes and index notes.
- Use dashboard notes for metadata browsing, maintenance, and review workflows.
- Use compact visual style for micro bridge notes.
- Prefer native Obsidian callouts, tables, and Mermaid over screenshots.
- Keep Mermaid diagrams vertical when labels are long and avoid page-level horizontal scrolling.
- Treat frontmatter as the canonical metadata layer. Keep `type`, `domain`, `audience`, `status`, and `last_reviewed` accurate whenever notes are created, promoted, moved, or materially rewritten.
- Keep `00 Home/Home.md` and the four top-level indexes as the main navigation system. Dashboards and Bases support this layer, but should not replace it.

## Required Visual Rules
- Substantive notes should usually have 3 to 5 meaningful callouts.
- Each substantive note should include at least one visual anchor:
  - Mermaid for flow, architecture, or decision logic
  - table or cheat sheet for comparisons
- Micro bridge notes should remain short and should not be turned into full dashboards.

## Metadata And Dashboard Rules
- The canonical dashboard stack is:
  - `00 Home/Vault Dashboard.md`
  - `90 Guides/Editorial Dashboard.md`
  - `00 Home/Vault Catalog.base`
  - `90 Guides/Editorial Review.base`
- Keep Bases and Dataview aligned with the frontmatter schema. If note classes, domains, or review fields change, update the dashboard layer in the same work.
- Use Bases for browsing, grouping, and quick editing by metadata.
- Use Dataview for review queues, maintenance queries, and editorial checks that depend on note content.
- Do not let dashboards become the primary study path. Readers should still enter through `Home.md` and the index notes.

## Curriculum And Navigation Rules
- Each major index in `00 Home` should answer three questions clearly:
  - who the branch is for
  - what the prerequisites are
  - what the branch leads to next
- When a branch expands, update both the relevant index and `Home.md`.
- Keep learning progression coherent across the vault:
  - foundations -> data preparation -> classical ML -> deep learning and NLP -> agentic systems
- Treat `RAG` as useful context for agentic systems, not a universal prerequisite.

## Validation And Review
- Agents may deploy sub-agents at their own discretion for research, validation, review, and multi-perspective analysis inside this vault workflow.
- The user does not need to re-authorize sub-agent deployment each time when it materially improves the work.
- Use sub-agents when they materially improve factual verification, structure review, audience review, strategy or technical cross-checking, or link and architecture validation.
- Substantial changes should usually include at least one validation pass. High-impact changes should usually include multi-perspective review.
- Typical defaults:
  - use one agent for research when recency or breadth matters
  - use one or more agents for review when a note, branch, or structure change affects multiple audiences
  - skip sub-agent deployment for trivial typo fixes or very small local edits
- For material changes to navigation, dashboards, Mermaid behavior, or workspace setup, prefer a real Obsidian-facing QA pass in addition to file-level validation when feasible.
- After validation, summarize the conclusions and close review sub-agents when they are no longer needed.

## Maintenance Rule
- When new note-writing conventions are introduced, record them in `Note Style Guide.md` with a dated changelog entry.
