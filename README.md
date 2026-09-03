# Cortex

A local second brain for research. Papers on the left, the open PDF in the middle, chat on the right. Projects are spaces: a named group of papers with a one-line verdict and a next action. Everything is plain files in `~/Cortex` (see SPEC.md), so Claude Code, Codex, and OpenCode can all work on the same brain.

## Run

```bash
cd cortex && ./run.sh            # server on http://127.0.0.1:8788, serves web/dist
cd cortex/web && pnpm install && pnpm build   # build the front end once (pnpm dev for live reload on :5173)
```

Chat uses OpenCode Go by default. The key is read from `~/.local/share/opencode/auth.json` or `OPENCODE_API_KEY`. Any OpenAI-compatible provider works: set `CORTEX_BASE_URL`, `CORTEX_API_KEY`, and `CORTEX_MODEL`.

## Getting papers in

Drop a PDF anywhere on the window, paste an arXiv link into the Add box, or put PDFs in `~/Cortex/inbox` and they file themselves within a few seconds. Images and files pasted into a note are stored under `~/Cortex/assets`.

## First-time import

```bash
CORTEX_VAULT=~/Cortex uv run --python 3.11 --with fastapi --with openai --with pypdf --with pyyaml \
  python scripts/import_local.py --queue <triage queue.json> --bear <bear_notes.json>
```

The import is idempotent. It copies PDFs into `library/<id>/paper.pdf` with arXiv metadata where the id is known, seeds topics and projects, and imports Bear notes. `scripts/import_paperpilot.py` pulls in everything PaperPilot's reading lists reference.

## Deploy (public demo)

The `Dockerfile` builds the web app and serves it with the API. With `CORTEX_DEMO=1` the server seeds a public sample vault of arXiv papers on first boot. `render.yaml` describes a free Render web service: connect this repo on Render, set `CORTEX_API_KEY` (any OpenAI-compatible key; OpenCode Go works), and point a CNAME such as `cortex.aftersave.app` at the service.

## WebMCP

`web/src/lib/webmcp.ts` registers ten tools with `document.modelContext` (or `navigator.modelContext` on older Chrome builds). A browser agent in ChatGPT's browser, or in Chrome 149+ with the WebMCP flag, can search the library, open or read a paper, file a new arXiv paper, write a note, or update a project. The person watches it happen and sees every call in the agent ledger. Details are in `SUBMISSION.md`.

## Layout

- `server/` FastAPI. `vault.py` handles files and the FTS5 index, `chat.py` the model and vault tools with SSE streaming, `agents.py` the Codex, OpenCode, and Claude runners, `app.py` the routes.
- `web/` Vite, React, and TypeScript front end.
- `scripts/` the initial vault build and the PaperPilot importer.
