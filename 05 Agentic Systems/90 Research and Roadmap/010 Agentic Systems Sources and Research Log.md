---
type: research-log
domain: agentic-systems
audience:
  - builder
  - data-strategy
aliases:
  - Agentic Systems Sources and Research Log
status: evergreen
last_reviewed: 2026-04-18
---
# Agentic Systems Sources and Research Log

This note is the research baseline for the `Agentic Systems` branch.

> [!INFO] What this note is for
> Use this note as the source registry for the branch, the current-practice snapshot, and the watchlist for future updates.

## Why It Matters
Agentic systems are moving quickly. Without an explicit source log, notes drift into mixed layers of timeless concepts, current product behavior, and emerging practice.

## Current Practice Snapshot
### Last reviewed: 2026-04-18
- Current official guidance converges on a simple rule: start with the smallest workflow or agent loop that solves the task.
- Tool quality, permissions, and observability matter as much as model capability.
- Multi-agent systems are useful when specialization or verification clearly helps, not as a default upgrade path.
- Long-running and background agents are now a practical systems concern, not just a research topic.
- MCP / app-style interoperability is becoming a practical integration layer for agent systems.
- Coding-agent products now expose a distinct harness layer across CLI, IDE, app, and cloud surfaces; skills, permissions, hooks, and isolation are part of the architecture rather than implementation detail.
- The branch now reads best as a core conceptual trunk plus two deeper applied tracks: software engineering agents and applied agentic architectures.
- Decision economics is now an explicit branch concern rather than an implicit side note inside architecture selection.
- Computer-use systems now have real official product surfaces, but they should still be treated as a higher-risk `current practice + emerging` area.

> [!IMPORTANT] How to read recency
> This branch should distinguish between `canonical` ideas that age slowly, `current practice` that may change within months, and `emerging` topics that need active monitoring.

## How To Read The Log
| Tag | Meaning |
| :--- | :--- |
| `canonical` | foundational concept that should remain useful over time |
| `current practice` | state of practical implementation as of the review date |
| `emerging` | active area that may change quickly |

## Canonical Foundations
| Source | Type | Date | Tag | Why It Matters | Note Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [MRKL Systems](https://arxiv.org/abs/2205.00445) | paper | 2022-05-01 | canonical | modular tool-using agent architecture | [[AI Agents]], [[Agent Architectures and Orchestration Patterns]] |
| [ReAct](https://arxiv.org/abs/2210.03629) | paper | 2022-10-06 | canonical | reasoning plus acting loop | [[AI Agents]], [[Planning and Control Flow in Agent Systems]], [[Applied Agentic Architectures]] |
| [Toolformer](https://arxiv.org/abs/2302.04761) | paper | 2023-02-09 | canonical | model-driven tool use | [[Tool Use and Environment Interaction]] |
| [Reflexion](https://arxiv.org/abs/2303.11366) | paper | 2023-03-20 | canonical | reflection and episodic improvement | [[Planning and Control Flow in Agent Systems]], [[Memory in Agent Systems]] |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | paper | 2023-04-07 | canonical | memory, reflection, and planning in agents | [[Memory in Agent Systems]] |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | paper | 2023-05-17 | canonical | search-based reasoning control | [[Planning and Control Flow in Agent Systems]] |
| [Voyager](https://arxiv.org/abs/2305.16291) | paper | 2023-05-25 | canonical | skill library and open-ended embodied agent loop | [[Memory in Agent Systems]], [[AI Agents]] |
| [Language Agent Tree Search](https://arxiv.org/abs/2310.04406) | paper | 2023-10-06 | canonical | planning and search for agent systems | [[Planning and Control Flow in Agent Systems]] |
| [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) | paper | 2023-09-05 | canonical | conceptual architecture framing | [[AI Agents]], [[Memory in Agent Systems]], [[Agent Architectures and Orchestration Patterns]] |

## Surveys
| Source | Type | Date | Tag | Why It Matters | Note Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) | survey | 2024-03-22 | canonical | broad survey of construction, applications, evaluation | all core notes |
| [Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/abs/2402.01680) | survey | 2024-02-04 | canonical | strongest broad MAS survey for this branch | [[Multi-Agent Systems]], [[Agent Architectures and Orchestration Patterns]] |
| [Large Language Model-Brained GUI Agents: A Survey](https://arxiv.org/abs/2411.18279) | survey | 2024-11-27 | emerging | GUI/computer-use branch precursor | [[Computer Use and GUI Agents]] |
| [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) | survey | 2025-01-15 | emerging | bridge between `RAG` and agents | [[AI Agents]], [[Tool Use and Environment Interaction]] |
| [Survey on Evaluation of LLM-based Agents](https://arxiv.org/abs/2503.16416) | survey | 2025-03-20 | current practice | evaluation taxonomy and gaps | [[Evaluation, Observability, and Governance for Agent Systems]] |
| [Large Language Model Agent: A Survey on Methodology, Applications and Challenges](https://arxiv.org/abs/2503.21460) | survey | 2025-03-27 | current practice | recent methodology overview | all core notes |

## Current Practice / Official Docs
| Source | Type | Date | Tag | Why It Matters | Note Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [OpenAI, New tools for building agents](https://openai.com/index/new-tools-for-building-agents/) | official | 2025-03-11 | current practice | current OpenAI framing for tools, tracing, and agents | [[When to Use Agentic Systems]], [[Economic and ROI Analysis for Agentic Systems]], [[Tool Use and Environment Interaction]], [[Evaluation, Observability, and Governance for Agent Systems]], [[Applied Agentic Architectures]], [[Validation and Eval Design for Agent Architectures]] |
| [OpenAI, A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) | official guide | 2026 | current practice | practical framing for use-case selection, orchestration, and guardrails | [[When to Use Agentic Systems]], [[Economic and ROI Analysis for Agentic Systems]], [[Applied Agentic Architectures]], [[Architecture Design Methods for Agent Systems]], [[Proposal-to-Production for Agent Systems]], [[Human-in-the-Loop and Approval Flows]], [[Orchestration Trade-offs and Pattern Selection]], [[Delegation and Role Specialization]], [[Applied Agentic Architecture Case Studies]], [[Software Engineering Agents]] |
| [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) | official | 2025-01-23 | current practice | product and benchmark framing for GUI agents | [[Computer Use and GUI Agents]] |
| [OpenAI, Introducing Codex](https://openai.com/index/introducing-codex/) | official | 2025-05-16 | current practice | cloud coding-agent pattern with sandboxed parallel tasks and PR artifacts | [[Software Engineering Agents]], [[Repo Operating Model for Coding Agents]], [[CI, Pull Requests, and Human Review for Coding Agents]], [[Operating Coding Agents in Teams]] |
| [OpenAI, Introducing AgentKit](https://openai.com/index/introducing-agentkit/) | official | 2025-10-06 | current practice | modern orchestration and developer workflow example | [[Agent Architectures and Orchestration Patterns]], [[Orchestration Trade-offs and Pattern Selection]] |
| [OpenAI, Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/) | official | 2025-09-15 | current practice | code review and long-task behavior in a modern coding agent | [[Software Engineering Agents]], [[CI, Pull Requests, and Human Review for Coding Agents]], [[Long-Running and Background Coding Agents]], [[Operating Coding Agents in Teams]] |
| [OpenAI, Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/) | official | 2026-02-02 | current practice | multi-agent supervision, worktrees, and skills in a coding-agent surface | [[Software Engineering Agents]], [[Repo Operating Model for Coding Agents]], [[Building Coding Agent Harnesses]], [[Long-Running and Background Coding Agents]], [[Operating Coding Agents in Teams]], [[Tool Ecosystems and Harness Engineering]], [[Applied Agentic Architectures]], [[Applied Agentic Architecture Case Studies]] |
| [OpenAI, Unlocking the Codex harness: how we built the App Server](https://openai.com/index/unlocking-the-codex-harness/) | engineering post | 2026-02-04 | current practice | harness surfaces, session semantics, and agent integration layers | [[Software Engineering Agents]], [[Building Coding Agent Harnesses]], [[Long-Running and Background Coding Agents]], [[Tool Ecosystems and Harness Engineering]], [[Applied Agentic Architectures]] |
| [OpenAI, Background mode](https://platform.openai.com/docs/guides/background) | official docs | 2026 | current practice | long-running async execution pattern | [[Agent Architectures and Orchestration Patterns]], [[Proposal-to-Production for Agent Systems]], [[Building Coding Agent Harnesses]], [[Long-Running and Background Coding Agents]], [[Reliability, Checkpoints, and Recovery in Agent Systems]], roadmap |
| [OpenAI, Agent Builder](https://platform.openai.com/docs/guides/agent-builder) | official docs | 2026 | current practice | workflow and builder pattern reference | [[When to Use Agentic Systems]], [[Agent Architectures and Orchestration Patterns]], [[Architecture Design Methods for Agent Systems]], [[Proposal-to-Production for Agent Systems]], [[Orchestration Trade-offs and Pattern Selection]] |
| [OpenAI, Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/) | official docs | 2026 | current practice | programmable agent runtime and orchestration primitives | [[Building Coding Agent Harnesses]], [[Tool Ecosystems and Harness Engineering]], [[Long-Running and Background Coding Agents]] |
| [OpenAI, Computer use](https://developers.openai.com/api/docs/guides/tools-computer-use) | official docs | 2026 | current practice | API-level loop and safety framing for GUI agents | [[Tool Use and Environment Interaction]], [[Computer Use and GUI Agents]] |
| [OpenAI, Trace grading](https://platform.openai.com/docs/guides/trace-grading) | official docs | 2026 | current practice | product-level tracing and grading | [[Evaluation, Observability, and Governance for Agent Systems]], [[Validation and Eval Design for Agent Architectures]], [[Evaluating Software Engineering Agents]] |
| [OpenAI, Safety in building agents](https://platform.openai.com/docs/guides/agent-builder-safety) | official docs | 2026 | current practice | guardrails and safety framing | [[Evaluation, Observability, and Governance for Agent Systems]], [[Tool Use and Environment Interaction]], [[Approvals, Permissions, and Sandboxing for Coding Agents]], [[Human-in-the-Loop and Approval Flows]], [[Reliability, Checkpoints, and Recovery in Agent Systems]] |
| [OpenAI MCP docs](https://platform.openai.com/docs/mcp/) | official docs | 2026 | current practice | interoperability example | [[Tool Use and Environment Interaction]], roadmap |
| [OpenAI Help, Apps in ChatGPT](https://help.openai.com/en/articles/11487775-connectors-in-chatgpt) | help center | 2026 | current practice | distinction between MCP-backed custom apps and host-level app surfaces | [[MCP and Connector Protocols]], [[Tool Ecosystems and Harness Engineering]] |
| [Anthropic, Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents) | engineering guide | 2024-12-19 | current practice | strongest official workflow-vs-agent framing | [[When to Use Agentic Systems]], [[Economic and ROI Analysis for Agentic Systems]], [[AI Agents]], [[Planning and Control Flow in Agent Systems]], [[Applied Agentic Architectures]], [[Architecture Design Methods for Agent Systems]], [[Proposal-to-Production for Agent Systems]], [[Human-in-the-Loop and Approval Flows]], [[Orchestration Trade-offs and Pattern Selection]], [[Delegation and Role Specialization]], [[Applied Agentic Architecture Case Studies]], [[Software Engineering Agents]] |
| [Anthropic, Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet](https://www.anthropic.com/engineering/swe-bench-sonnet) | engineering guide | 2025-01-06 | current practice | coding-agent tool design and benchmark harness lessons | [[Software Engineering Agents]], [[Evaluating Software Engineering Agents]], [[CI, Pull Requests, and Human Review for Coding Agents]], [[Operating Coding Agents in Teams]] |
| [Anthropic, How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | engineering guide | 2025-06-13 | current practice | practical multi-agent architecture example | [[Economic and ROI Analysis for Agentic Systems]], [[Multi-Agent Systems]], [[Agent Architectures and Orchestration Patterns]], [[Proposal-to-Production for Agent Systems]], [[Human-in-the-Loop and Approval Flows]], [[Orchestration Trade-offs and Pattern Selection]], [[Delegation and Role Specialization]], [[Applied Agentic Architecture Case Studies]] |
| [Anthropic, Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) | engineering guide | 2025-11-24 | current practice | structured tool use and catalog design | [[Tool Use and Environment Interaction]] |
| [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | engineering guide | 2025-11-26 | current practice | durability and harness design | [[Memory in Agent Systems]], [[Agent Architectures and Orchestration Patterns]], [[Software Engineering Agents]], [[Building Coding Agent Harnesses]], [[Long-Running and Background Coding Agents]], [[Reliability, Checkpoints, and Recovery in Agent Systems]] |
| [Anthropic, Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | engineering guide | 2026-01-09 | current practice | current evaluation discipline | [[Evaluation, Observability, and Governance for Agent Systems]], [[Software Engineering Agents]], [[Validation and Eval Design for Agent Architectures]], [[Evaluating Software Engineering Agents]], [[Operating Coding Agents in Teams]] |
| [Anthropic, Managed Agents](https://www.anthropic.com/engineering/managed-agents) | engineering guide | 2025 | current practice | long-running managed-agent framing | [[Memory in Agent Systems]], [[Agent Architectures and Orchestration Patterns]], [[Long-Running and Background Coding Agents]], [[Reliability, Checkpoints, and Recovery in Agent Systems]] |
| [Claude Code overview](https://code.claude.com/docs/en/overview) | official docs | 2026 | current practice | current product surface across terminal, IDE, desktop, web, CI, and MCP-backed integrations | [[Software Engineering Agents]], [[Repo Operating Model for Coding Agents]], [[Long-Running and Background Coding Agents]], [[Operating Coding Agents in Teams]], [[Tool Ecosystems and Harness Engineering]] |
| [Claude Code settings](https://code.claude.com/docs/en/settings) | official docs | 2026 | current practice | permissions, hierarchy, and project-level policy model | [[Tool Ecosystems and Harness Engineering]], [[Approvals, Permissions, and Sandboxing for Coding Agents]], [[Operating Coding Agents in Teams]] |
| [Extend Claude with skills](https://code.claude.com/docs/en/skills) | official docs | 2026 | current practice | reusable workflow layer and selective loading model | [[Software Engineering Agents]], [[Building Coding Agent Harnesses]], [[Tool Ecosystems and Harness Engineering]] |
| [Create custom subagents](https://code.claude.com/docs/en/sub-agents) | official docs | 2026 | current practice | scoped workers, worktree isolation, and subagent config | [[Software Engineering Agents]], [[Building Coding Agent Harnesses]], [[Tool Ecosystems and Harness Engineering]], [[Long-Running and Background Coding Agents]] |
| [Anthropic Docs, Define tools](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use) | official docs | 2026 | current practice | tool schema quality guidance | [[Tool Use and Environment Interaction]] |
| [Anthropic Docs, Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) | official docs | 2026 | current practice | GUI-agent loop, sandboxing, and approval guidance | [[Tool Use and Environment Interaction]], [[Computer Use and GUI Agents]] |
| [Anthropic Docs, Hooks reference](https://docs.anthropic.com/en/docs/claude-code/hooks) | official docs | 2026 | current practice | pre-tool and post-tool permission controls in coding harnesses | [[Software Engineering Agents]], [[Approvals, Permissions, and Sandboxing for Coding Agents]], [[Tool Ecosystems and Harness Engineering]] |

## Benchmarks And Evaluation Anchors
| Source | Type | Date | Tag | Why It Matters | Note Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [AgentBench](https://arxiv.org/abs/2308.03688) | benchmark paper | 2023 | canonical | early broad agent benchmark | [[Evaluation, Observability, and Governance for Agent Systems]] |
| [WebArena](https://arxiv.org/abs/2307.13854) | benchmark paper | 2023 | canonical | realistic web environment benchmark | [[Computer Use and GUI Agents]] |
| [GAIA](https://arxiv.org/abs/2311.12983) | benchmark paper | 2023 | canonical | broad assistant-style evaluation | [[Evaluation, Observability, and Governance for Agent Systems]] |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | benchmark paper | 2023 | canonical | real software-engineering agent benchmark | [[Evaluation, Observability, and Governance for Agent Systems]], [[Software Engineering Agents]], [[Evaluating Software Engineering Agents]], [[Operating Coding Agents in Teams]] |
| [OSWorld](https://arxiv.org/abs/2404.07972) | benchmark paper | 2024 | current practice | open computer environment benchmark | [[Computer Use and GUI Agents]] |
| [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208) | paper | 2024 | current practice | instruction priority and prompt-injection defense | [[Evaluation, Observability, and Governance for Agent Systems]], [[Tool Use and Environment Interaction]] |

> [!WARNING] Current practice will age faster than canonical theory
> Official product docs, tooling surfaces, and operational recommendations can change much faster than the foundational papers. Review these entries first when refreshing the branch.

## Watchlist for Future Updates
- MCP and connector ecosystem maturity
- background and long-running execution primitives
- computer-use safety and benchmark drift
- evaluation frameworks for agents with real tools
- enterprise governance patterns for multi-agent systems

> [!TIP] Update order
> When refreshing the branch, review official docs first, then 2025+ surveys, then decide whether the core notes need conceptual changes or only source refreshes.

## Related Notes
- Related: [[Agentic Systems Index]], [[When to Use Agentic Systems]], [[Evaluation, Observability, and Governance for Agent Systems]]

## Sources
- This note is itself the source registry for the branch and contains the primary references used by the rest of the notes.

## Last Reviewed
- 2026-04-18
