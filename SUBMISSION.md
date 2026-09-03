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
`web/src/lib/webmcp.ts` registers ten tools with `document.modelContext.registerTool` (falling back to `navigator.modelContext`) (feature-detected; the app works without it):

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

## Video script (under 3 minutes)
0:00 The problem: reading, filing, deciding live in four tools; the agent lives in a fifth. Show the vault folder: plain markdown and PDFs.
0:25 The app: daily page, a note with LaTeX rendering live, the library with a PDF open and its status/rating strip.
0:55 WebMCP: open the same page in ChatGPT's browser. Ask: "Which papers here are about deception? Open the strongest one." Watch the paper open in the center pane and the agent ledger list `search_brain` and `open_item`.
1:30 Ask: "Set it to reading with the takeaway 'probes beat output monitoring', then file arXiv 2511.13653 and append a line to today's daily page." Show `set_paper`, `file_paper` (PDF downloads and opens), `append_today`, and the daily page updating.
2:10 The built-in chat does the same with the same tools on any OpenAI-compatible model; "Hand to Codex" runs a coding agent inside the vault folder.
2:35 Close: one vault, one page, the same ten tools for the person and the agent. Repo link, MIT.

## Demo
Live URL: https://cortex.aftersave.app (public demo vault of arXiv papers) · Repo: https://github.com/thehimalayanleo/cortex · Video: (< 3 min, to add)

Try it with an agent (ChatGPT's browser, or Chrome 149+ with `chrome://flags/#enable-webmcp-testing`): "Which papers here are about deception? Open the best one and set its status to reading." Without an agent, the same tools are callable from DevTools: `await cortex.call("search_brain", {query: "probes"})`.
