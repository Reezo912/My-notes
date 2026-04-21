---
type: policy
domain: knowledge-ops
audience:
  - builder
  - data-strategy
aliases:
  - Knowledge Ops Schema
status: evergreen
last_reviewed: 2026-04-20
---
# Knowledge Ops Schema

This schema defines the file classes, metadata expectations, and lifecycle contract for the `full Karpathy` operating layer.

> [!INFO] Core rule
> `status` stays global to the vault. The knowledge lifecycle is modeled separately with `knowledge_state` and `review_state`.

## Why It Matters
Without an explicit schema, a wiki-maintenance workflow becomes an accumulation of ad hoc prompts. The schema is what turns the agent into a disciplined maintainer instead of a generic writer.

## Core Object Types
| Type | Role | Default Location |
| :--- | :--- | :--- |
| `source` | normalized note for one source item | `80 Knowledge Ops/10 Source Notes` |
| `ops-log` | queues, logs, hot context, and registries | `20 Domain Workspaces`, `40 Registries and Logs` |
| `policy` | operating contract, lifecycle rules, and schemas | `30 Schemas and Policies` |
| `dashboard` | Bases and Dataview operational surfaces | `90 Dashboards` |
| `index` | human-readable operational hub | root of `80 Knowledge Ops` and workspace hubs |

## Metadata Contract
```yaml
---
type: source
domain: knowledge-ops
audience:
  - builder
status: draft
knowledge_state: normalized
review_state: unreviewed
source_kind: pdf
target_domains:
  - agentic-systems
source_path: 80 Knowledge Ops/00 Intake/raw/example.pdf
canonical_targets:
  - 090 LLM Wiki and Agentic Knowledge Bases
last_reviewed: 2026-04-20
---
```

### Required Fields By Class
| Field | Source | Ops log | Policy |
| :--- | :--- | :--- | :--- |
| `type` | required | required | required |
| `domain` | `knowledge-ops` | `knowledge-ops` | `knowledge-ops` |
| `status` | required | required | required |
| `knowledge_state` | required | recommended | optional |
| `review_state` | required | recommended | optional |
| `source_kind` | required | optional | optional |
| `source_path` or `source_url` | required | optional | optional |
| `target_domains` | required | recommended | optional |
| `canonical_targets` | optional | optional | optional |
| `last_reviewed` | required | required | required |

> [!IMPORTANT] One source note per source item
> Do not merge multiple source items into one source note. Cross-source synthesis belongs in workspaces and candidate pages, not in the provenance layer.

## Lifecycle Values
| Field | Allowed Values |
| :--- | :--- |
| `knowledge_state` | `raw`, `normalized`, `draft`, `canonical-candidate`, `promoted`, `stale`, `archived` |
| `review_state` | `unreviewed`, `in-review`, `approved`, `rejected` |
| `source_kind` | `pdf`, `web-article`, `paper`, `book-chapter`, `video-transcript`, `podcast-transcript`, `github-repo`, `thread`, `meeting-note`, `other` |

## Naming Rules
- use stable, human-readable filenames
- normalize numbered operational notes when reading order matters
- preserve clean aliases for notes that are likely to be linked often
- keep canonical note titles stable; candidate notes can be more tactical

## Related Notes
- Related: [[80 Knowledge Ops/30 Schemas and Policies/020 Source Ingestion and Media Normalization|Source Ingestion and Media Normalization]], [[80 Knowledge Ops/30 Schemas and Policies/040 Promotion and Canon Policy|Promotion and Canon Policy]], [[80 Knowledge Ops/40 Registries and Logs/050 Canonical Target Map|Canonical Target Map]]

## Sources
- [LLM Wiki | Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [Properties | Obsidian Help](https://obsidian.md/help/properties)

## Last Reviewed
- 2026-04-20
