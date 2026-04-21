# AI / ML Obsidian Vault

English version. Spanish version: [README.es.md](./README.es.md)

This repository is an Obsidian vault for AI, machine learning, NLP, and agentic systems notes.

It is designed to work in two modes:
- **Reference mode**: jump into a concept note and understand it quickly.
- **Study mode**: start from the home and index notes and follow curated learning paths.

## Choose Your Route
This vault is organized for three audiences:
- **Learner**: start at [`00 Home/Home.md`](./00%20Home/Home.md), use the curated foundations starter set, then move through the main indexes in sequence.
- **Builder**: start at [`00 Home/Home.md`](./00%20Home/Home.md), then enter [`00 Home/Data Preparation Index.md`](./00%20Home/Data%20Preparation%20Index.md) for data-quality problems or [`00 Home/Agentic Systems Index.md`](./00%20Home/Agentic%20Systems%20Index.md) for tool-rich systems, coding agents, and applied architectures.
- **Data Strategy**: start at [`00 Home/Home.md`](./00%20Home/Home.md), then use [`00 Home/Data Preparation Index.md`](./00%20Home/Data%20Preparation%20Index.md) for data-readiness and policy decisions or [`00 Home/Agentic Systems Index.md`](./00%20Home/Agentic%20Systems%20Index.md) for ROI, governance, and operating-model questions. Pull in the ML and DL/NLP indexes when model tradeoffs need deeper support.

## Quick Start
1. Clone or download this repository.
2. Open the folder in Obsidian as an existing vault.
3. Start at [`00 Home/Home.md`](./00%20Home/Home.md).

## First-Time Obsidian Setup
If you have never used Obsidian before, use this sequence:

1. Install Obsidian from [obsidian.md](https://obsidian.md/).
2. Open Obsidian.
3. On the welcome screen, choose **Open folder as vault** or **Open existing vault**.
4. Select the downloaded repository folder.
5. Once the vault opens, start at [`00 Home/Home.md`](./00%20Home/Home.md).

## Plugin Model
This repository ships with a shared `.obsidian` configuration. The core knowledge base remains readable without community plugins, but the metadata layer and some workspace conveniences depend on plugins.

### Required For The Metadata Layer
`Bases` is a core Obsidian plugin.

1. Open **Settings**.
2. Go to **Core plugins**.
3. Search for `Bases`.
4. Turn it on.

You need this for:
- [`00 Home/Vault Catalog.base`](./00%20Home/Vault%20Catalog.base)
- [`90 Guides/Editorial Review.base`](./90%20Guides/Editorial%20Review.base)

### Enable Community Plugins
`Dataview` is a community plugin. Obsidian keeps community plugins disabled in Restricted Mode by default.

1. Open **Settings**.
2. Go to **Community plugins**.
3. Read the warning and only continue if you trust this vault and its plugin setup.
4. Select **Turn on community plugins** or disable **Restricted mode**.

Official security reference:
- [Obsidian Plugin Security](https://obsidian.md/help/plugin-security)

### Enable Dataview
This repository already includes the Dataview plugin files in `.obsidian/plugins/dataview`, but Obsidian still needs the plugin enabled in your local vault.

If Dataview does not appear active automatically:
1. Open **Settings**.
2. Go to **Community plugins**.
3. Look for `Dataview` in the installed plugins list.
4. Enable it.

If it is not listed for any reason:
1. Open **Settings**.
2. Go to **Community plugins**.
3. Select **Browse**.
4. Search for `Dataview`.
5. Install it.
6. Enable it.

You need this for:
- [`00 Home/Vault Dashboard.md`](./00%20Home/Vault%20Dashboard.md)
- [`90 Guides/Editorial Dashboard.md`](./90%20Guides/Editorial%20Dashboard.md)

### Included Community Plugins In `.obsidian`
These plugins are present in the shared vault configuration:
- `dataview`
- `editing-toolbar`
- `obsidian-excalidraw-plugin`
- `obsidian-git`
- `obsidian-outliner`
- `obsidian-tasks-plugin`
- `quickadd`
- `table-editor-obsidian`
- `templater-obsidian`

These plugins are not required to follow the audience routes, read the canonical notes, or use the main indexes. They are included for the author's workspace and optional workflows. Enabling them is still a user trust decision.

### Recommended Obsidian Setup
- Keep the included `.obsidian` folder if you want the shared workspace behavior.
- Enable `Bases` and `Dataview` if you want the metadata and dashboard layer.
- If you prefer a minimal setup, keep community plugins off except `Dataview`.

## What To Expect If You Skip Plugins
- Without **Bases**, the `.base` files will not be useful.
- Without **Dataview**, the dashboard notes will open, but their query blocks will not render properly.
- Without both, the vault still works well as a Markdown knowledge base through the home page and index notes.

## What Works Even Without Extra Plugins
These parts work as normal Markdown notes even if Dataview is missing:
- folder structure
- wiki-links
- note content
- frontmatter metadata
- top-level index notes

If Dataview is not enabled, the vault is still usable, but the dashboard notes lose part of their value.

## Branch Shortcuts
- [`00 Home/Home.md`](./00%20Home/Home.md): audience-first portal
- [`00 Home/Data Preparation Index.md`](./00%20Home/Data%20Preparation%20Index.md): preprocessing, data quality, and data-readiness branch
- [`00 Home/Machine Learning Index.md`](./00%20Home/Machine%20Learning%20Index.md): central ML path from prepared data into model families and metrics
- [`00 Home/Deep Learning & NLP Index.md`](./00%20Home/Deep%20Learning%20%26%20NLP%20Index.md): deep learning, NLP, language models, and `RAG`
- [`00 Home/Agentic Systems Index.md`](./00%20Home/Agentic%20Systems%20Index.md): tools, planning, memory, orchestration, coding agents, and applied architectures
- [`80 Knowledge Ops/010 Knowledge Ops.md`](./80%20Knowledge%20Ops/010%20Knowledge%20Ops.md): operational layer for ingest, drafts, lint, and supervised promotion

## Vault Structure
- `00 Home`: main portal, top-level indexes, dashboards, and Bases views
- `01 Foundations`: statistics, bias, and data concepts
- `02 Data Preparation`: encoding, imputation, scaling, imbalance handling
- `03 Classical ML`: metrics, linear models, tree-based models, and tabular ML
- `04 Deep Learning & NLP`: neural networks, sequence models, NLP, language models, and RAG
- `05 Agentic Systems`: agents, tool use, planning, memory, orchestration, evaluation, and specialization tracks
- `80 Knowledge Ops`: operational layer for intake, source notes, domain workspaces, promotion queues, and lint
- `90 Guides`: style guide and editorial dashboard
- `99 Archive`: reserved for deprecated notes

## Contributor Notes
If you want to extend or maintain the vault, use:
- [`AGENTS.md`](./AGENTS.md): short operational rules for AI agents
- [`90 Guides/Note Style Guide.md`](./90%20Guides/Note%20Style%20Guide.md): canonical authoring, curriculum, metadata, and dashboard rules

## Notes On Portability
- The vault can be opened directly by another Obsidian user.
- The included `.obsidian/workspace.json` is opinionated; if someone prefers a clean local layout, they can change it without affecting the content.
- Some visual behavior depends on theme, plugin availability, and local Obsidian settings.
- Menu labels may vary slightly across Obsidian versions, but the overall plugin flow is the same: open the vault, enable `Bases` if you want Bases views, then enable `Dataview` if you want dashboards.
