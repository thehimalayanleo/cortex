# Cortex: WebMCP Challenge submission notes

Cortex is a researcher's second brain: papers on the left, the open PDF in the middle, chat on the right. The browser agent gets the same ten tools the person uses. They are registered with `document.modelContext`, so ChatGPT's browser or Chrome can search the library, read a paper, file a new arXiv paper, write a note, or update a project, and the person watches each step happen in the app with a ledger of every call.

## Use case fit
Research is a loop: read, file, decide. Each step usually lives in a different tool, and the agent lives in a chat box next to them. Cortex keeps the vault (plain markdown and PDFs) behind one page and exposes the loop as tools. The agent does the mechanical half, finding, filing, summarizing, and recording the verdict. The person keeps the judgment half.

## Human-agent experience
The agent works on the same page the person is looking at. Tools that change something (`open_item`, `write_note`, `file_paper`, `update_project`) show their result in the center pane, and an agent ledger lists each call with its verb, target, and outcome. Writes are additive and easy to undo because everything is a file in a folder. Read-only tools carry `readOnlyHint`. Agent input is treated as untrusted: every argument is coerced and bounded before it touches the vault.

## WebMCP implementation
`web/src/lib/webmcp.ts` registers ten tools with `document.modelContext.registerTool`, falling back to `navigator.modelContext` on older Chrome builds. Registration is feature-detected, so the app works without it.

| tool | what it does |
|---|---|
| `search_brain` | full-text search over notes, projects, and extracted PDF text |
| `open_item` | show a note, paper, or project |
| `read_note` / `write_note` / `append_today` | read, create, or append markdown (LaTeX aware) |
| `read_paper` / `file_paper` / `set_paper` | read a paper, ingest one from arXiv (download and index), set status, rating, takeaway |
| `list_projects` / `update_project` | one-line verdicts and next actions |

Results come back as MCP content blocks. Failures return `{error}` text and never throw into the page.

## Stack
FastAPI with SQLite FTS5 over a plain-file vault; Vite, React, CodeMirror 6, and KaTeX on the front end; chat through any OpenAI-compatible model (OpenCode Go by default) using the same tools server-side; optional hand-off to Codex, OpenCode, or Claude Code running inside the vault folder.

## Prior vs new work
Everything in this repository was written during the submission period. See the commit timestamps in `git log`.

## Video script (under 3 minutes)
0:00 The problem. Reading, filing, and deciding live in four tools, and the agent lives in a fifth. Show the vault folder: markdown and PDFs, nothing else.
0:25 The app. Papers on the left, a PDF open in the middle with its status, rating, and one-line takeaway, chat on the right.
0:55 WebMCP. Open the same page in ChatGPT's browser. Ask: "Which papers here are about deception? Open the strongest one." The paper opens in the center pane and the ledger shows `search_brain` and `open_item`.
1:30 Ask: "Set it to reading with the takeaway 'probes beat output monitoring', then file arXiv 2511.13653 and add a line to today's page." Show `set_paper`, `file_paper` (the PDF downloads and opens), `append_today`, and today's page updating.
2:10 Drag a PDF from Finder onto the window. It files itself. The built-in chat uses the same ten tools on any OpenAI-compatible model.
2:35 Close: one vault, one page, the same ten tools for the person and the agent. Repo link, MIT.

## Demo
Live: https://cortex.aftersave.app (a public demo vault of arXiv papers). Repo: https://github.com/thehimalayanleo/cortex. Video: under 3 minutes, to add.

Try it with an agent in ChatGPT's browser, or in Chrome 149+ with `chrome://flags/#enable-webmcp-testing`: "Which papers here are about deception? Open the best one and set its status to reading." Without an agent, the same tools are callable from DevTools: `await cortex.call("search_brain", {query: "probes"})`.
