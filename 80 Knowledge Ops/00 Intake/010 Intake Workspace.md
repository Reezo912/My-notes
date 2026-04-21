---
type: guide
domain: knowledge-ops
audience:
  - builder
aliases:
  - Intake Workspace
status: evergreen
last_reviewed: 2026-04-20
---
# Intake Workspace

The intake layer is where new source material lands before the agent normalizes it into source notes and domain workspaces.

> [!INFO] Core rule
> `raw/` is immutable. The agent reads from it, classifies it, and writes normalized notes elsewhere. It should not rewrite the source of truth.

## Folder Rules
| Folder | Use | Rule |
| :--- | :--- | :--- |
| `raw/` | incoming source files | keep immutable after placement |
| `assets/` | locally downloaded images and attachments | normalize filenames and keep them referentially stable |

## Operational Defaults
1. Drop the source into `raw/`.
2. Create a normalized source note in [[80 Knowledge Ops/10 Source Notes/010 Source Notes Index|Source Notes Index]].
3. Update the appropriate domain workspace.
4. Append the ingest event to [[80 Knowledge Ops/40 Registries and Logs/020 Activity Log|Activity Log]].

## Related Notes
- Related: [[80 Knowledge Ops/30 Schemas and Policies/020 Source Ingestion and Media Normalization|Source Ingestion and Media Normalization]], [[80 Knowledge Ops/10 Source Notes/010 Source Notes Index|Source Notes Index]]

## Last Reviewed
- 2026-04-20
