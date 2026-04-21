---
type: index
domain: knowledge-ops
audience:
  - builder
aliases:
  - Source Notes Index
status: evergreen
last_reviewed: 2026-04-20
---
# Source Notes Index

Each meaningful source should get one normalized source note. This preserves provenance, source quality, and the ability to promote or reject downstream syntheses without losing the original trail.

> [!INFO] One source, one note
> A source note is the normalized metadata and summary layer for one source item. It should always point back to the immutable raw source or external URL.

## Required Fields
| Field | Why |
| :--- | :--- |
| `source_kind` | classify the source for downstream handling |
| `source_path` or `source_url` | preserve provenance |
| `target_domains` | route the source into the right workspaces |
| `knowledge_state` | distinguish normalized source notes from drafts and promoted artifacts |
| `review_state` | keep explicit whether the source note was checked |

## Workflow
1. Normalize the source into one note.
2. Add a short evidence-oriented summary.
3. Link canonical targets only when they are plausible, not mandatory.
4. Route the source into one or more domain workspaces.

## Recent Normalized Sources
| Source Note | Source Kind | Target Domains | Likely Canonical Target |
| :--- | :--- | :--- | :--- |
| [[2026-01-26 The Complete Guide to Building Skills for Claude\|The Complete Guide to Building Skills for Claude]] | `pdf` | `agentic-systems` | [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] |

## Related Notes
- Related: [[80 Knowledge Ops/00 Intake/010 Intake Workspace|Intake Workspace]], [[80 Knowledge Ops/30 Schemas and Policies/010 Knowledge Ops Schema|Knowledge Ops Schema]], [[80 Knowledge Ops/40 Registries and Logs/010 Global Knowledge Index|Global Knowledge Index]]

## Last Reviewed
- 2026-04-21
