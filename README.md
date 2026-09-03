# Cortex

One local app for the whole second brain: notes with LaTeX, a PDF library, projects, a daily page, and a chat that reads and writes all of it. Everything is plain files in `~/Cortex` (see SPEC.md), so Claude Code, Codex, and OpenCode work on the same brain.

## Run

```bash
cd cortex && ./run.sh            # server on http://127.0.0.1:8788, serves web/dist
cd cortex/web && pnpm install && pnpm build   # build the front-end once (pnpm dev for live reload on :5173)
```

Chat uses OpenCode Go by default (key read from `~/.local/share/opencode/auth.json` or `OPENCODE_API_KEY`). Any OpenAI-compatible provider works: set `CORTEX_BASE_URL`, `CORTEX_API_KEY`, `CORTEX_MODEL`.

## First-time import

```bash
CORTEX_VAULT=~/Cortex uv run --python 3.11 --with fastapi --with openai --with pypdf --with pyyaml \
  python scripts/import_local.py --queue <triage queue.json> --bear <bear_notes.json>
```

Idempotent. Drops PDFs into `library/<id>/paper.pdf` with arXiv metadata where known, seeds topics and projects, imports Bear notes.

## Deploy (public demo)

The `Dockerfile` builds the web app and serves it with the API; `CORTEX_DEMO=1` seeds a public sample vault of arXiv papers on first boot. `render.yaml` describes a free Render web service: connect this repo on Render, set `CORTEX_API_KEY` (an OpenAI-compatible key; OpenCode Go works), and point a CNAME (for example `cortex.aftersave.app`) at the service.

## WebMCP

`web/src/lib/webmcp.ts` registers ten tools with `navigator.modelContext` so a browser agent (ChatGPT's browser, Chrome 149+ with the WebMCP flag) can search the brain, open or read a paper, file a new arXiv paper, write a note, or update a project, while the person watches it happen and sees every call in the agent ledger. Details in `SUBMISSION.md`.

## Layout

- `server/` FastAPI: `vault.py` (files + FTS5 index), `chat.py` (model + vault tools, SSE), `agents.py` (Codex / OpenCode / Claude runners), `app.py` (routes).
- `web/` Vite + React + TypeScript front-end.
- `scripts/import_local.py` initial vault build.
