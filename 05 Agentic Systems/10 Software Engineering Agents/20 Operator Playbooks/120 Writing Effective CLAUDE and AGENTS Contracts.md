---
type: concept
domain: agentic-systems
audience:
  - builder
aliases:
  - Writing Effective CLAUDE and AGENTS Contracts
  - Writing Effective CLAUDE.md and AGENTS.md
status: evergreen
last_reviewed: 2026-04-20
---
# Writing Effective CLAUDE and AGENTS Contracts

`CLAUDE.md` and `AGENTS.md` are repo contracts, not dumping grounds. They should define the stable rules that improve agent behavior across sessions and across humans, while leaving local tactics and temporary context elsewhere.

> [!INFO] Core idea
> The question is not “what can I tell the model?” The better question is “what must stay true often enough that it deserves always-on or repo-level instruction space?”

## Why It Matters
Instruction files are expensive context surfaces. If they are short and sharp, they improve behavior consistently. If they become bloated or duplicated, they reduce adherence, create contradictions, and hide the real contract of the repository.

## Instruction Layer Map
| Instruction Layer | Best Home | Why |
| :--- | :--- | :--- |
| repo-wide invariants | `AGENTS.md` or `CLAUDE.md` | should apply on every meaningful task |
| tool-specific or vendor-specific behavior | the tool's own repo file or a thin shim | keep generic and tool-local guidance separate |
| path-specific rules | nested `AGENTS.md` or `.claude/rules/` | narrower scope improves adherence |
| repeatable workflow | skill, command, hook, or script | do not spend always-on context on procedures |
| temporary task detail | current prompt or short artifact brief | should expire with the task |
| hard enforcement | hooks, config, permissions, runtime policy | models should not be the final guardrail |

> [!IMPORTANT] Stable rule vs temporary reminder
> If a rule should survive many sessions and apply to most work in the repo, it belongs in the repo contract. If it is only useful for the current task, keep it out of the permanent layer.

## What Belongs In Each File
| File | Good Content | Weak Content |
| :--- | :--- | :--- |
| `CLAUDE.md` | commands, architecture invariants, review workflow, repo gotchas, Claude-specific behavior | every repo detail, giant prose tutorials, local credentials |
| `AGENTS.md` | team conventions, done criteria, note or code style rules, path and validation norms | unstable personal preferences, runtime config values, task-specific requests |

## Platform Nuance
| Surface | Contract Bias |
| :--- | :--- |
| Claude Code | `CLAUDE.md` is a native first-class surface, with repo-specific behavior often split into `.claude` rules, skills, and hooks |
| Codex | `AGENTS.md` is the main repo contract, with runtime behavior pushed into config and host-level policy |
| OpenCode | `AGENTS.md` is primary, `CLAUDE.md` is a compatibility fallback, and `opencode.json` can modularize extra instructions |

## Good Patterns
### Thin-shim pattern
If a repo already has `AGENTS.md`, keep `CLAUDE.md` short and point to it, then add only Claude-specific behavior.

```md
@AGENTS.md

- Prefer project skills before adding long local instructions.
- Use the shared validation commands before proposing completion.
```

### Short-contract pattern
Keep only:
- how to search the repo or vault
- which validation commands matter
- which files or actions are risky
- what a finished handoff must include

## What Should Live Elsewhere
- setup steps that belong in bootstrap docs or local environment config
- temporary architecture debates for one feature
- deep tutorials better represented as skills or supporting docs
- enforcement rules that should actually be hooks or permission policy

> [!CAUTION] Duplication is worse than omission
> Two partially overlapping instruction files feel safer than one, but they create hidden contradictions. When in doubt, keep the permanent contract smaller and push detail into reusable workflows.

## Short Templates
### `AGENTS.md`
```md
# AGENTS.md

- State the repo-wide conventions that should survive every session.
- Name the validation commands and the finish criteria.
- Keep risky paths, secrets, and deployment boundaries explicit.
```

### `CLAUDE.md`
```md
# CLAUDE.md

@AGENTS.md

- Add only Claude-specific workflow behavior.
- Point to skills, rules, or commands instead of copying them here.
```

## Failure Modes
- turning the file into a memory dump
- repeating the same rules in root and nested contracts
- using instruction files for runtime policy that should be enforced elsewhere
- keeping outdated commands after the repo changes
- confusing user preference with team contract

> [!TIP] Review question
> Read the file and ask: if I deleted this line, would the agent become worse in a repeatable way? If not, it probably does not deserve permanent instruction space.

## Related Notes
- Prerequisites: [[090 Operating Agentic Coding Environments|Operating Agentic Coding Environments]]
- Related: [[100 Claude Code Setup and Repo Contracts|Claude Code Setup and Repo Contracts]], [[110 Codex Setup and Repo Contracts|Codex Setup and Repo Contracts]], [[115 OpenCode Setup and Repo Contracts|OpenCode Setup and Repo Contracts]], [[130 Skills, Commands, and Hooks in Practice|Skills, Commands, and Hooks in Practice]], [[140 Context Engineering and Session Hygiene for Coding Agents|Context Engineering and Session Hygiene for Coding Agents]]

## Sources
- [AGENTS.md | OpenAI Developers](https://developers.openai.com/codex/guides/agents-md)
- [Claude Code memory | Claude Code Docs](https://code.claude.com/docs/en/memory)
- [Claude Code best practices | Claude Code Docs](https://code.claude.com/docs/en/best-practices)
- [Claude Code: Best practices for agentic coding | Anthropic](https://www.anthropic.com/engineering/claude-code-best-practices?curius=1527)
- [OpenCode Rules](https://opencode.ai/docs/rules/)
- [OpenCode Config](https://opencode.ai/docs/config/)
- [agentsmd/agents.md | GitHub](https://github.com/agentsmd/agents.md)
- [openai/codex agents_md docs | GitHub](https://github.com/openai/codex/blob/main/docs/agents_md.md)
- See [[010 Agentic Systems Sources and Research Log|Agentic Systems Sources and Research Log]]

## Last Reviewed
- 2026-04-20
