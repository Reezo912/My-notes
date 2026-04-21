---
type: policy
domain: knowledge-ops
audience:
  - builder
aliases:
  - Lint and Maintenance Workflow
status: evergreen
last_reviewed: 2026-04-20
---
# Lint and Maintenance Workflow

This workflow governs periodic health checks over the intake, workspace, and canonical handoff layers.

> [!INFO] Lint is not optional
> A knowledge base that compounds without lint also compounds drift, duplicates, and hidden editorial debt.

## What To Check
| Check | Why |
| :--- | :--- |
| broken wikilinks | navigation failures break trust quickly |
| orphan pages | useful knowledge disappears from normal traversal |
| stale claims | newer sources may have superseded old syntheses |
| duplicate concepts | semantic duplication increases drift and query noise |
| missing provenance | the wiki starts citing itself instead of evidence |
| oversized pages | large pages become context bottlenecks |
| weak canonical routing | candidate notes never reach the right target |

## Queue Rules
1. Log material lint findings in [[80 Knowledge Ops/40 Registries and Logs/040 Lint Queue|Lint Queue]].
2. Keep trivial fixes local.
3. Promote only the fixes that change canonical truth or structure.
4. Treat repeated lint failures as schema or workflow problems, not just cleanup tasks.

## Maintenance Cadence
| Cadence | Default Work |
| :--- | :--- |
| per ingest batch | link check, source-note sanity, target-domain check |
| periodic | orphan scan, stale scan, duplicate scan |
| pre-promotion | target-note fit, alias check, scope check |

## Related Notes
- Related: [[80 Knowledge Ops/40 Registries and Logs/040 Lint Queue|Lint Queue]], [[80 Knowledge Ops/90 Dashboards/010 Knowledge Ops Dashboard|Knowledge Ops Dashboard]], [[100 Evaluation, Observability, and Governance for Agent Systems|Evaluation, Observability, and Governance for Agent Systems]]

## Sources
- [LLM Wiki | Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

## Last Reviewed
- 2026-04-20
