---
type: concept
domain: knowledge-ops
audience:
  - builder
aliases:
  - Building Effective Skills for Claude Promotion Candidate
status: draft
knowledge_state: canonical-candidate
review_state: approved
target_domains:
  - agentic-systems
canonical_targets:
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/135 Building Effective Skills for Claude
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/130 Skills, Commands, and Hooks in Practice
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/100 Claude Code Setup and Repo Contracts
last_reviewed: 2026-04-21
---
# Building Effective Skills for Claude Promotion Candidate

This candidate packages the first `full Karpathy` ingest in the vault into a source-backed canonical patch set for the `Operator Playbooks` line.

> [!INFO] Promotion decision
> The PDF is strong enough to justify a new canonical note, not just a small patch to an existing one.

> [!IMPORTANT] New canonical owner
> Promote [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] as the stable owner for skill design, testing, and distribution. Keep [[130 Skills, Commands, and Hooks in Practice\|Skills, Commands, and Hooks in Practice]] as the higher-level surface-selection note.

> [!TIP] Scope boundary
> This ingest is not mainly about `CLAUDE.md` contracts or general hooks policy. It is specifically about how to build, test, and ship better `skills`.

## Why This Deserves Canon
| Reason | Explanation |
| :--- | :--- |
| concept gap | the track explains where skills fit, but not how to author one well |
| operational value | the user can apply this directly in `Claude Code`, `Claude.ai`, and the API |
| source quality | the local PDF is dense, structured, and strongly aligned with current docs |
| current-practice fit | the guide plus new docs and repo examples produce a strong operator playbook |

## Canonical Patch Map
| Target | Patch Type | Why |
| :--- | :--- | :--- |
| [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] | new canonical note | stable owner for the whole topic |
| [[130 Skills, Commands, and Hooks in Practice\|Skills, Commands, and Hooks in Practice]] | bridge patch | clarify that `skill choice` and `skill authoring` are different layers |
| [[100 Claude Code Setup and Repo Contracts\|Claude Code Setup and Repo Contracts]] | bridge patch | connect repo setup to actual skill-building practice |
| [[090 Operating Agentic Coding Environments\|Operating Agentic Coding Environments]] | route patch | add the new note to the operator map |
| [[010 Software Engineering Agents\|Software Engineering Agents]] | track patch | surface the note in the operator-playbooks workflow |
| [[Agentic Systems Index]] | branch patch | make the note visible from the specialization path |
| [[010 Agentic Systems Sources and Research Log\|Agentic Systems Sources and Research Log]] | source patch | register the PDF and open-standard sources explicitly |

## Evidence Base
| Source | Role |
| :--- | :--- |
| [[2026-01-26 The Complete Guide to Building Skills for Claude\|The Complete Guide to Building Skills for Claude]] | main design, testing, and distribution playbook |
| [Extend Claude with skills | Claude Code Docs](https://code.claude.com/docs/en/skills) | newer field-level behavior and Claude Code extensions |
| [Agent Skills Overview](https://agentskills.io/) | open-standard framing and portability layer |
| [anthropics/skills | GitHub](https://github.com/anthropics/skills) | living example repository and packaging patterns |

## Related Notes
- Related: [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/020 Agentic Systems Hot Context\|Agentic Systems Hot Context]], [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/030 Agentic Systems Draft Inbox\|Agentic Systems Draft Inbox]], [[80 Knowledge Ops/40 Registries and Logs/030 Promotion Queue\|Promotion Queue]]

## Last Reviewed
- 2026-04-21
