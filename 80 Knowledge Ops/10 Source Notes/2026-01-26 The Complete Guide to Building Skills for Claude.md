---
type: source
domain: knowledge-ops
audience:
  - builder
aliases:
  - The Complete Guide to Building Skills for Claude
status: draft
knowledge_state: normalized
review_state: approved
source_kind: pdf
target_domains:
  - agentic-systems
source_path: 80 Knowledge Ops/00 Intake/raw/The-Complete-Guide-to-Building-Skill-for-Claude.pdf
canonical_targets:
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/135 Building Effective Skills for Claude
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/130 Skills, Commands, and Hooks in Practice
  - 05 Agentic Systems/10 Software Engineering Agents/20 Operator Playbooks/100 Claude Code Setup and Repo Contracts
last_reviewed: 2026-04-21
---
# The Complete Guide to Building Skills for Claude

This source note normalizes the local PDF guide into evidence that can be routed into the `Agentic Systems` operator-playbook line.

> [!INFO] Provenance
> Raw source: [The-Complete-Guide-to-Building-Skill-for-Claude.pdf](</Users/carloslopezdelizaga/Documents/Obsidian Vault/80 Knowledge Ops/00 Intake/raw/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>)

> [!IMPORTANT] Why this source matters
> This is the strongest single source in the vault so far for the practical question `how do I design, test, and distribute a good Claude skill?` It goes beyond basic docs by covering workflow shape, triggering quality, iteration loops, and distribution.

> [!WARNING] Treat this as `January 2026` guidance
> The PDF is still highly useful, but it predates some newer Claude Code doc details such as `when_to_use`, `disable-model-invocation`, `paths`, and `context: fork`. Use it together with current docs rather than as a frozen specification.

## Source Snapshot
| Field | Value |
| :--- | :--- |
| title | `The Complete Guide to Building Skills for Claude` |
| source kind | `pdf` |
| created | `2026-01-26` |
| extracted pages | `33` |
| primary target domain | `agentic-systems` |
| strongest canonical target | [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] |

## Core Claims
### What a skill is
- a skill is a folder, not a single markdown file
- `SKILL.md` is required
- `scripts/`, `references/`, and `assets/` are optional but important for stronger workflows
- the design model is `progressive disclosure`: frontmatter first, `SKILL.md` body when relevant, and supporting files only when needed

### How to design one well
- start from `2-3` concrete use cases before writing instructions
- choose whether the skill is `problem-first` or `tool-first`
- define both quantitative and qualitative success criteria
- keep the description field explicit about `what it does` and `when to use it`
- keep instructions specific, actionable, and explicit about error handling

### How to test one well
- run triggering tests
- run functional tests
- compare against a baseline without the skill
- iterate on one hard task before broadening coverage
- use deterministic scripts for validations when prose is too weak

### How to distribute one well
- skills work across `Claude.ai`, `Claude Code`, and the API
- GitHub hosting plus a repo-level `README.md` is recommended for sharing
- the skill folder itself should not contain `README.md`
- org-wide deployment and API-based usage are real current-practice paths, not just theory

## Practical Details Worth Preserving
| Topic | Key Detail |
| :--- | :--- |
| naming | folder and `name` should be `kebab-case` |
| entrypoint | `SKILL.md` must be exact-case |
| description | should include what it does, when to use it, and likely trigger phrases |
| frontmatter safety | avoid XML angle brackets and reserved product names in the skill name |
| testing | triggering, functional, and performance comparison are distinct checks |
| scale concern | too many enabled skills and oversized `SKILL.md` files degrade performance |

## Important Deltas Against Newer Docs
| Area | PDF Guide | Newer Claude Code Docs |
| :--- | :--- | :--- |
| minimal metadata | treats `name` and `description` as the minimal required pair | allows directory-name fallback and first-paragraph fallback, but still recommends explicit metadata |
| trigger guidance | centers everything on `description` | splits this into `description` plus `when_to_use` |
| manual-only vs model-only | does not emphasize manual-only skills strongly | supports `disable-model-invocation: true` for manual workflows |
| routing scope | focuses on portable folder design | adds `paths`, `context: fork`, `agent`, `hooks`, and other Claude Code extensions |
| size guidance | suggests keeping `SKILL.md` under `5,000` words | suggests keeping it under `500` lines |

> [!TIP] Best current reading
> Read this guide as the `design and testing playbook`, then use current docs for the latest field-level details and Claude Code-specific extensions.

## Likely Canonical Impact
| Canonical Target | Why |
| :--- | :--- |
| [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]] | no existing note owns full skill design, testing, and distribution |
| [[130 Skills, Commands, and Hooks in Practice\|Skills, Commands, and Hooks in Practice]] | needs a stronger bridge from surface selection into actual skill authoring |
| [[100 Claude Code Setup and Repo Contracts\|Claude Code Setup and Repo Contracts]] | should point to a dedicated skill-design note instead of keeping skill guidance too shallow |
| [[010 Agentic Systems Sources and Research Log\|Agentic Systems Sources and Research Log]] | source baseline should include this guide explicitly |

## Related Notes
- Related: [[80 Knowledge Ops/20 Domain Workspaces/05 Agentic Systems/040 Building Effective Skills for Claude Promotion Candidate\|Building Effective Skills for Claude Promotion Candidate]], [[135 Building Effective Skills for Claude\|Building Effective Skills for Claude]], [[130 Skills, Commands, and Hooks in Practice\|Skills, Commands, and Hooks in Practice]]

## Sources
- Local PDF: [The-Complete-Guide-to-Building-Skill-for-Claude.pdf](</Users/carloslopezdelizaga/Documents/Obsidian Vault/80 Knowledge Ops/00 Intake/raw/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>)
- [Extend Claude with skills | Claude Code Docs](https://code.claude.com/docs/en/skills)
- [Agent Skills Overview](https://agentskills.io/)
- [anthropics/skills | GitHub](https://github.com/anthropics/skills)

## Last Reviewed
- 2026-04-21
