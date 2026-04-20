---
type: concept
domain: agentic-systems
audience:
  - builder
aliases:
  - Skills, Commands, and Hooks in Practice
status: evergreen
last_reviewed: 2026-04-20
---
# Skills, Commands, and Hooks in Practice

Skills, commands, and hooks are different workflow surfaces. The point is not to use all of them. The point is to place each kind of logic where it stays reliable, inspectable, and cheap to maintain.

> [!INFO] Core idea
> Put reusable procedure in skills, direct invocation shortcuts in commands, and hard enforcement in hooks. Do not ask one surface to do the job of another.

## Why It Matters
Many agentic coding setups degrade because every new need gets pushed into the same place. The result is usually one of two bad states: an oversized instruction file or a messy runtime full of overlapping hooks and half-maintained skills.

## Surface Comparison
| Surface | Best For | Avoid |
| :--- | :--- | :--- |
| skill | reusable playbook with optional supporting files | always-on facts or hard enforcement |
| command | quick explicit invocation for a known action | deep policy or hidden side effects |
| subagent or worker | isolated context, independent review, or bounded delegated work | minor tasks that one session can finish cleanly |
| hook | blocking, validation, or post-action automation | long advisory prose the model may ignore |
| repo doc | stable reference the agent can open on demand | things the agent must always remember |
| task prompt | current job brief and temporary context | permanent repo conventions |

> [!IMPORTANT] Procedure vs policy
> If the model may choose to ignore it, it is not policy. Put policy in hooks, permissions, or config. Use skills and commands for repeatable workflow, not enforcement.

## Decision Table
| If The Need Is... | Prefer... | Why |
| :--- | :--- | :--- |
| repeatable multi-step workflow | skill | the agent can load it when relevant |
| one-click or one-slash action | command | explicit invocation is enough |
| isolated delegated task with clear scope | subagent or separate session | keeps context narrower and ownership clearer |
| block dangerous paths or actions | hook | runtime enforcement beats advisory text |
| team-wide repo fact | repo contract | always-on and versioned with the repo |
| current task nuance | local prompt | should expire with the task |

## Patterns That Age Well
- keep skills small, stable, and scoped to one job family
- move large examples or reference docs out of `SKILL.md` and into supporting files
- use commands for convenience, not for hidden complexity
- move from skills to subagents only when isolated context or independent review is genuinely useful
- keep hooks short and testable
- review skills and hooks together when repo workflows change

## From Reuse To Delegation
| Need | Better Surface |
| :--- | :--- |
| same workflow repeated often | skill or command |
| dangerous action must be blocked | hook |
| separate reviewer or researcher role | subagent or separate session |
| broad work split across paths or modules | worktree-backed multi-agent workflow |

### Small Stable Skill Vs Overgrown Skill
| Skill Shape | Characteristics | Likely Outcome |
| :--- | :--- | :--- |
| small stable skill | narrow purpose, short trigger rules, optional supporting docs | easier selective loading and less drift |
| overgrown skill | many edge cases, duplicated repo facts, mixed policy and procedure | lower adherence and unclear ownership |

> [!WARNING] Overlapping surfaces create drift
> If the same workflow exists partly in a skill, partly in a hook, and partly in a repo contract, the agent is forced to reconcile conflicting signals instead of doing the job.

## Experimental Community Pattern
Some teams use prompt-submission hooks to remind the model about relevant skills before work begins. This can help on large repos, but treat it as an emerging practice rather than a canonical default.

## Maintenance Rules
- give each skill or hook an owner
- retire workflows that the repo no longer uses
- keep side-effect-heavy hooks observable and easy to disable for debugging
- prefer fewer stronger surfaces over many overlapping ones

> [!TIP] Practical default
> Start with one or two skills for repeated workflows and one or two hooks for hard safety or validation boundaries. Add more surfaces only when a clear maintenance owner exists.

## Related Notes
- Prerequisites: [[090 Operating Agentic Coding Environments|Operating Agentic Coding Environments]], [[120 Writing Effective CLAUDE and AGENTS Contracts|Writing Effective CLAUDE and AGENTS Contracts]]
- Related: [[100 Claude Code Setup and Repo Contracts|Claude Code Setup and Repo Contracts]], [[140 Context Engineering and Session Hygiene for Coding Agents|Context Engineering and Session Hygiene for Coding Agents]], [[150 Parallel Sessions, Worktrees, and Multi-Agent Workflows|Parallel Sessions, Worktrees, and Multi-Agent Workflows]], [[065 Delegation and Role Specialization|Delegation and Role Specialization]], [[160 Tool Design and MCP Integration in Practice|Tool Design and MCP Integration in Practice]], [[030 Approvals, Permissions, and Sandboxing for Coding Agents|Approvals, Permissions, and Sandboxing for Coding Agents]]

## Sources
- [Extend Claude with skills | Claude Code Docs](https://code.claude.com/docs/en/skills)
- [Hooks guide | Claude Code Docs](https://code.claude.com/docs/en/hooks-guide)
- [CLI slash commands | OpenAI Developers](https://developers.openai.com/codex/cli/slash-commands)
- [OpenCode Commands](https://opencode.ai/docs/commands/)
- [OpenCode Agent Skills](https://opencode.ai/docs/skills)
- [Writing effective tools for AI agents - using AI agents | Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [peterkrueck/Claude-Code-Development-Kit | GitHub](https://github.com/peterkrueck/Claude-Code-Development-Kit)
- [trailofbits/skills | GitHub](https://github.com/trailofbits/skills/blob/main/CLAUDE.md)
- See [[010 Agentic Systems Sources and Research Log|Agentic Systems Sources and Research Log]]

## Last Reviewed
- 2026-04-20
