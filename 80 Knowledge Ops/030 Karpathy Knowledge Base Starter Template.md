---
type: guide
domain: knowledge-ops
audience:
  - builder
  - data-strategy
aliases:
  - Karpathy Knowledge Base Starter Template
status: evergreen
last_reviewed: 2026-04-21
---
# Karpathy Knowledge Base Starter Template

This note turns the current `Knowledge Ops` system into a reusable template for other repos, vaults, or team knowledge bases.

> [!INFO] Use this as a starter kit, not as dogma
> The pattern generalizes well. The exact folder names, frontmatter fields, and editorial rigor should be adapted to the size and seriousness of the target knowledge base.

> [!IMPORTANT] The universal part
> The durable pattern is the separation between `raw sources`, `normalized source notes`, `workspace drafts`, `policies`, and `canonical promotion`. That structure is much more reusable than the exact Obsidian-specific implementation details.

> [!TIP] Three adoption levels
> Most people should not start with the full stack. A `lite` or `standard` version is often enough until the base grows.

## Three Adoption Levels
| Level | Best For | Minimum Components |
| :--- | :--- | :--- |
| `lite` | solo notes, small research projects | `raw`, `sources`, `drafts`, `canon` |
| `standard` | serious personal vaults, small teams | `raw`, `source notes`, `workspaces`, `promotion queue`, `schema` |
| `full` | shared vaults, multi-agent maintenance, auditable knowledge systems | full lifecycle metadata, policies, logs, lint, and dashboards |

## Recommended Folder Tree
```text
Knowledge Ops/
  00 Intake/
    raw/
    assets/
  10 Source Notes/
  20 Domain Workspaces/
    Domain A/
    Domain B/
  30 Schemas and Policies/
  40 Registries and Logs/
  90 Dashboards/
```

## Minimal Lifecycle Model
| Field | Suggested Values |
| :--- | :--- |
| `knowledge_state` | `raw`, `normalized`, `draft`, `canonical-candidate`, `promoted`, `stale`, `archived` |
| `review_state` | `unreviewed`, `in-review`, `approved`, `rejected` |
| `source_kind` | `pdf`, `web-article`, `paper`, `thread`, `meeting-note`, `other` |

## Minimal Object Types
| Note Type | Role |
| :--- | :--- |
| `source` | normalized note for one source |
| `ops-log` | workspace log, queue, or hot context |
| `policy` | ingest, query, lint, and promotion rules |
| `guide` | onboarding or team-facing instructions |
| `index` | human-readable hub for the operating layer |

## Starter Templates
### Source note
```yaml
---
type: source
domain: knowledge-ops
status: draft
knowledge_state: normalized
review_state: unreviewed
source_kind: pdf
target_domains:
  - your-domain
source_path: Knowledge Ops/00 Intake/raw/example.pdf
canonical_targets:
  - Your Canonical Note
last_reviewed: 2026-04-21
---
```

### Workspace draft
```yaml
---
type: ops-log
domain: knowledge-ops
status: draft
knowledge_state: draft
review_state: unreviewed
target_domains:
  - your-domain
last_reviewed: 2026-04-21
---
```

### Promotion candidate
```yaml
---
type: concept
domain: knowledge-ops
status: draft
knowledge_state: canonical-candidate
review_state: in-review
target_domains:
  - your-domain
canonical_targets:
  - Your Canonical Note
last_reviewed: 2026-04-21
---
```

## Required Team Rules
1. `ingest` and `promote` are separate approvals.
2. One source gets one source note.
3. New-source ingest includes a takeaway discussion before canon changes.
4. Workspace drafts are allowed to be volatile; canon is not.
5. Rejections stay visible in the queue instead of being silently erased.

> [!WARNING] The biggest failure mode
> Most failed `full Karpathy` systems collapse because they skip the policy layer. The agent starts writing everything everywhere, and the repository stops having a stable truth layer.

## What To Keep Repo-Specific
| Keep Generic | Customize Per Repo |
| :--- | :--- |
| lifecycle states | domain names and workspace folders |
| ingest vs promotion split | note classes and frontmatter details |
| source note pattern | naming conventions |
| promotion queue logic | dashboards and views |
| takeaway discussion rule | canonical branch ownership |

## Adoption Advice For Teams
- start with one domain, not the whole company wiki
- make one person own the promotion step at first
- keep the first few ingests manual and source-by-source
- use a worked example as training material
- add dashboards only after the schema is stable

## Related Notes
- Related: [[80 Knowledge Ops/020 Knowledge Ops Quickstart|Knowledge Ops Quickstart]], [[80 Knowledge Ops/040 Worked Example - PDF Ingest|Worked Example - PDF Ingest]], [[80 Knowledge Ops/30 Schemas and Policies/010 Knowledge Ops Schema|Knowledge Ops Schema]]

## Sources
- [LLM Wiki | Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [[80 Knowledge Ops/30 Schemas and Policies/010 Knowledge Ops Schema|Knowledge Ops Schema]]
- [[80 Knowledge Ops/30 Schemas and Policies/040 Promotion and Canon Policy|Promotion and Canon Policy]]

## Last Reviewed
- 2026-04-21
