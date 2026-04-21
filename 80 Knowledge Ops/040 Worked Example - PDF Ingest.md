---
type: guide
domain: knowledge-ops
audience:
  - builder
aliases:
  - Worked Example - PDF Ingest
status: evergreen
last_reviewed: 2026-04-21
---
# Worked Example - PDF Ingest

This note documents the first real `Knowledge Ops` ingest in the vault so future sessions and future teammates can see the intended end-to-end workflow.

> [!INFO] Example source
> The source was [The-Complete-Guide-to-Building-Skill-for-Claude.pdf](</Users/carloslopezdelizaga/Documents/Obsidian Vault/80 Knowledge Ops/00 Intake/raw/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>), routed into the `Agentic Systems` operator-playbooks track.

> [!IMPORTANT] What this example demonstrates
> This example shows the intended flow of `raw -> source note -> takeaway discussion -> workspace candidate -> promotion`. It also captures the mistake that happened the first time: promotion happened before the takeaway discussion, and the policy was later tightened so that future ingests do not repeat that error.

> [!WARNING] Use this as a workflow example, not as a claim that every ingest should create a new canonical note
> This PDF was rich enough to justify a new note. Many ingests should stop at source note plus workspace candidate.

## Artifact Chain
| Step | Artifact |
| :--- | :--- |
| raw source | [The-Complete-Guide-to-Building-Skill-for-Claude.pdf](</Users/carloslopezdelizaga/Documents/Obsidian Vault/80 Knowledge Ops/00 Intake/raw/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>) |
| source note | [[2026-01-26 The Complete Guide to Building Skills for Claude\|The Complete Guide to Building Skills for Claude]] |
| workspace candidate | [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/040 Building Effective Skills for Claude Promotion Candidate\|Building Effective Skills for Claude Promotion Candidate]] |
| canonical result | [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] |

## What The Agent Extracted
| Takeaway | Why It Mattered |
| :--- | :--- |
| skills are packaging, not just prompting | clarified the role of `SKILL.md` plus support files |
| progressive disclosure is central | strengthened the design model for skill authoring |
| trigger quality is part of skill quality | justified a dedicated note on authoring, not only on surface choice |
| skills need testing and distribution discipline | pushed the track beyond `when do I use skills?` into `how do I build one well?` |

## Canonical Impact
| Note | Change |
| :--- | :--- |
| [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] | created as new canonical owner |
| [[130 Skills, Commands, and Hooks in Practice\|Skills, Commands, and Hooks in Practice]] | patched to bridge from surface selection into skill authoring |
| [[100 Claude Code Setup and Repo Contracts\|Claude Code Setup and Repo Contracts]] | patched to point skill-building guidance to the new canonical owner |
| [[090 Operating Agentic Coding Environments\|Operating Agentic Coding Environments]] | updated operator-playbook map |
| [[010 Software Engineering Agents\|Software Engineering Agents]] | surfaced the new note in the track |
| [[Agentic Systems Index]] | surfaced the note in the operator-playbook path |

## What Went Wrong In The First Pass
The first pass promoted the content too early. The system ingested the source and promoted it before discussing the key takeaways with the user. That broke the intended Karpathy-style loop.

### Fix applied afterward
- `ingest` and `promote` were codified as separate approvals
- takeaway discussion was made mandatory before canon changes
- `vamos a ello` was explicitly documented as enough for ingest, but not for promotion

## Recommended Prompt Pattern
```text
Ingest this source into Knowledge Ops.
Give me the key takeaways first.
Stop at source note plus workspace candidate until I approve promotion.
```

## When To Use This Example With Teammates
- onboarding a colleague to the system
- explaining the difference between source processing and canon changes
- showing how one source can touch multiple notes without bypassing review
- demonstrating why the policy layer matters

## Related Notes
- Related: [[80 Knowledge Ops/020 Knowledge Ops Quickstart|Knowledge Ops Quickstart]], [[80 Knowledge Ops/030 Karpathy Knowledge Base Starter Template|Karpathy Knowledge Base Starter Template]], [[80 Knowledge Ops/30 Schemas and Policies/020 Source Ingestion and Media Normalization|Source Ingestion and Media Normalization]], [[80 Knowledge Ops/30 Schemas and Policies/040 Promotion and Canon Policy|Promotion and Canon Policy]]

## Sources
- [[2026-01-26 The Complete Guide to Building Skills for Claude|The Complete Guide to Building Skills for Claude]]
- [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/040 Building Effective Skills for Claude Promotion Candidate|Building Effective Skills for Claude Promotion Candidate]]
- [[135 Building Effective Skills for Claude|Building Effective Skills for Claude]]

## Last Reviewed
- 2026-04-21
