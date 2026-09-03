# Cortex — WebMCP Challenge submission notes

**One-liner.** A researcher's second brain (notes with LaTeX, a PDF library, projects, a daily page) where the browser agent is a first-class user: the same ten tools a person uses are registered with `navigator.modelContext`, so ChatGPT's browser or Chrome can search the brain, read a paper, file a new arXiv paper, write a note, or update a project verdict, and the person watches it happen in the UI with a ledger of every call.

## Use case fit
Research work is a loop of reading, filing, and deciding. Today each step lives in a different tool and the agent lives in a chat box beside them. Cortex puts the vault (plain markdown + PDFs) behind one page and exposes the loop as tools, so a browser agent can do the mechanical half (find, file, summarize, record the verdict) while the person keeps the judgment half.

## Human-agent experience
- Every agent tool navigates the center pane (`open_item`, `write_note`, `file_paper`, `update_project` all show their result), so the human sees exactly what changed.
- An **agent ledger** lists each call with verb, target, and outcome, next to the app's own chat ledger.
- Writes are additive and reversible (files in a folder, git-friendly); read tools are annotated `readOnlyHint`.
- Agent input is treated as untrusted: every argument is coerced and bounded before it touches the vault.

## WebMCP implementation
`web/src/lib/webmcp.ts` registers ten tools with `navigator.modelContext.registerTool` (feature-detected; the app works without it):

| tool | what it does |
|---|---|
| `search_brain` | FTS5 search over notes, projects, and extracted PDF text |
| `open_item` | show a note, paper, or project |
| `read_note` / `write_note` / `append_today` | read, create, append markdown (LaTeX-aware) |
| `read_paper` / `file_paper` / `set_paper` | read a paper, ingest from arXiv (download + index), set status/rating/takeaway |
| `list_projects` / `update_project` | one-line verdicts and next actions |

Results are returned as MCP content blocks; failures return `{error}` text and never throw into the page.

## Stack
FastAPI + SQLite FTS5 over a plain-file vault; Vite + React + CodeMirror 6 + KaTeX; chat via any OpenAI-compatible model (OpenCode Go by default) with the same tools server-side; optional hand-off to Codex / OpenCode / Claude Code running inside the vault.

## Prior vs new work
Everything in this repository was created during the submission period (first commit timestamps in `git log`).

## Demo
Live URL: https://cortex.aftersave.app (public demo vault of arXiv papers) · Repo: https://github.com/thehimalayanleo/cortex · Video: (< 3 min, to add)

Try it with an agent (ChatGPT's browser, or Chrome 149+ with `chrome://flags/#enable-webmcp-testing`): "Which papers here are about deception? Open the best one and set its status to reading." Without an agent, the same tools are callable from DevTools: `await cortex.call("search_brain", {query: "probes"})`.
