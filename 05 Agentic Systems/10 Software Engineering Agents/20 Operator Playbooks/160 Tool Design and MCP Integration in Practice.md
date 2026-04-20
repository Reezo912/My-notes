---
type: concept
domain: agentic-systems
audience:
  - builder
aliases:
  - Tool Design and MCP Integration in Practice
status: evergreen
last_reviewed: 2026-04-20
---
# Tool Design and MCP Integration in Practice

Tool design matters because the tool interface becomes part of the model's reasoning surface. In practice, better tool schemas and clearer approval boundaries usually matter more than minor prompt tuning.

> [!INFO] Core idea
> A good tool is easy for the model to choose correctly, easy for the runtime to validate, and easy for a human to supervise. A bad tool creates ambiguity before the agent even starts reasoning.

## Why It Matters
Coding agents are often judged as if the model alone caused success or failure. In reality, weak tool interfaces create a large fraction of the mistakes: wrong parameter choice, vague side effects, duplicated capability, and poor observability.

## Good Tool Vs Weak Tool
| Tool Shape | Characteristics | Likely Outcome |
| :--- | :--- | :--- |
| good tool | clear purpose, explicit side effects, tight schema, readable outputs | high correct-use rate and lower review burden |
| weak tool | overlapping semantics, vague descriptions, many optional fields, hidden side effects | misuse, retries, and unpredictable traces |

> [!IMPORTANT] Tool schema is part of capability
> The tool description, parameters, examples, and output structure shape the agent's real capability surface. A weak schema can make a strong model look worse than it is.

## Design Rules
| Design Question | Strong Default |
| :--- | :--- |
| can two tools be confused? | merge them behind an explicit `action` or `mode` |
| can the model tell when to use it? | state when to use it and when not to use it |
| can inputs be validated? | use constrained enums, required fields, and clear path semantics |
| can humans inspect it? | produce outputs with readable status, paths, and side-effect summaries |
| is the action risky? | keep approval boundaries outside the prompt layer |

## MCP In Practice
| MCP Is Good For | Why |
| :--- | :--- |
| shared interoperability across hosts and agents | avoids rewriting the same integration many times |
| standard tool exposure | keeps tool contracts inspectable and reusable |
| hosted or external systems | lets the coding harness call beyond the local repo cleanly |

| MCP Is Not By Itself | Why |
| :--- | :--- |
| a session model | the harness still needs threads, approvals, artifacts, and recovery |
| a policy layer | permission posture still lives in the host or runtime |
| proof of good tool design | poor schemas remain poor even behind a standard protocol |

> [!CAUTION] Protocol does not rescue a bad interface
> Standardization helps reuse, not clarity. If the tool contract is ambiguous before MCP, it stays ambiguous after MCP.

## MCP Vs Local CLI Or Script
| Choose... | When It Is Better |
| :--- | :--- |
| local CLI or script | the action is repo-local, trusted, and tightly coupled to one environment |
| MCP tool | the integration should be reusable across hosts, agents, or remote systems |
| both | the CLI is the implementation, but MCP is the stable external interface |

## Protocol Boundary Vs Host Boundary
The `MCP` protocol standardizes how tools are exposed and called. It does not decide:
- how sessions are modeled
- how approvals are granted
- how artifacts are stored
- how retries, traces, or handoffs are handled

Those choices belong to the host or harness. That distinction matters because two hosts can expose the same `MCP` tool but supervise it very differently.

## Example Shape
```json
{
  "name": "repo_action",
  "description": "Use for repository operations that need explicit action selection. Do not use for free-form shell exploration.",
  "input_schema": {
    "type": "object",
    "properties": {
      "action": { "type": "string", "enum": ["search", "diff", "read_file"] },
      "path": { "type": "string" }
    },
    "required": ["action"]
  }
}
```

## Approval And Observability
- keep risky tools behind explicit human or policy approval
- record tool name, parameters, side effects, and outputs in traces
- prefer tools that return structured outputs humans can review
- design tool errors so the agent learns what to fix next instead of retrying blindly

## Debugging Failing Integrations
| Failure Shape | First Check |
| :--- | :--- |
| wrong tool chosen | description overlap or poor examples |
| right tool, wrong parameters | schema clarity and enum design |
| tool works but review is poor | side-effect summary and output readability |
| tool fails only in one host | host policy, auth, or environment boundary rather than MCP itself |

> [!TIP] Practical default
> Start with fewer, clearer tools. Expand the surface only after you can see where the current tools are genuinely too weak, not merely under-described.

## Failure Modes
- many overlapping tools with fuzzy descriptions
- tool schemas that encourage optional-parameter guessing
- side effects that are not visible to the reviewer
- pushing approval logic into prompt text instead of runtime policy
- assuming MCP replaces host-level supervision

## Related Notes
- Prerequisites: [[030 Tool Use and Environment Interaction|Tool Use and Environment Interaction]], [[040 MCP and Connector Protocols|MCP and Connector Protocols]]
- Related: [[130 Skills, Commands, and Hooks in Practice|Skills, Commands, and Hooks in Practice]], [[060 Building Coding Agent Harnesses|Building Coding Agent Harnesses]], [[050 Tool Ecosystems and Harness Engineering|Tool Ecosystems and Harness Engineering]], [[170 Eval Hygiene for Agentic Coding Systems|Eval Hygiene for Agentic Coding Systems]]

## Sources
- [Define tools | Anthropic Docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)
- [Writing effective tools for AI agents - using AI agents | Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [MCP | OpenAI Developers](https://platform.openai.com/docs/mcp/)
- [MCP specification](https://modelcontextprotocol.io/specification/latest)
- [Unlocking the Codex harness: how we built the App Server | OpenAI](https://openai.com/index/unlocking-the-codex-harness/)
- See [[010 Agentic Systems Sources and Research Log|Agentic Systems Sources and Research Log]]

## Last Reviewed
- 2026-04-20
