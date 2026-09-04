# Cortex

A local second brain for research. Papers on the left, the open PDF in the middle, chat on the right. Projects are spaces: a named group of papers with a one-line verdict and a next action. Everything is plain files in `~/Cortex` (see SPEC.md), so Claude Code, Codex, and OpenCode can all work on the same brain.

## Run

```bash
cd cortex && ./run.sh            # server on http://127.0.0.1:8788, serves web/dist
cd cortex/web && pnpm install && pnpm build   # build the front end once (pnpm dev for live reload on :5173)
```

Chat uses OpenCode Go by default. The key is read from `~/.local/share/opencode/auth.json` or `OPENCODE_API_KEY`. Any OpenAI-compatible provider works: set `CORTEX_BASE_URL`, `CORTEX_API_KEY`, and `CORTEX_MODEL`.

## Run it as a login service (macOS)

```bash
cd cortex && scripts/install_service.sh   # starts at login, restarts if it dies; `scripts/install_service.sh remove` to undo
```

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

## Training Lab

`Lab` in the sidebar (or Cmd+K, "Training Lab") opens a curriculum that lives inside the brain:

- In the browser: seven stations train a 2-layer, width-48 character transformer with tf.js so you can watch each stage of an LLM pipeline: data, pretraining, mid-training with a cooldown, SFT then DPO, an encoder trained with masked LM then contrastive pairs, k-means on the embeddings, and a paint-with-code station that runs GRPO against a rendered reward.
- Chapters: twenty teaching chapters in `lab/chapters/` (data curation, pretraining, mid-training, SFT loop design, PPO/DPO/GRPO, tool use with the NVIDIA Nemotron agentic collection, embeddings, clustering and retrieval, evals, red-teaming, architecture including looped transformers, optimizers including Muon, GPUs/kernels/KV cache, Lean and code verification, paint with code, the Puro-2B one-GPU pretraining recipe, speculative decoding, mixture of experts, multimodal post-training with Miles, agentic cinema). Each ends with a ten-question self-test and a section on what will and will not change. They are synced into the vault as notes, so search and chat see them.
- GPU runs: `lab/recipes/*.py` are short, readable scripts that print `METRIC {...}` lines. Launch them from the Runs tab on this machine, on your own GPU box over SSH (`CORTEX_SSH_HOST=your-box CORTEX_SSH_PYTHON=~/lab-venv/bin/python ./run.sh`; Tailscale SSH works well), or on Modal (`modal token set` once). Logs and loss curves stream into the app and are stored under `runs/` in the vault.
- Agent tools: `open_lab`, `lab_train`, `lab_status`, `list_lab_chapters`, `list_runs`, `start_run`, `read_run` are registered through WebMCP alongside the library tools, so an agent can train the browser model, read its loss, launch a real run, and open the chapter that explains it, all on the page you are looking at.

## WebMCP

`web/src/lib/webmcp.ts` registers eleven tools with `document.modelContext` (or `navigator.modelContext` on older Chrome builds). A browser agent in ChatGPT's browser, or in Chrome 149+ with the WebMCP flag, can search the library, open or read a paper, file a new arXiv paper, write a note, or update a project. The person watches it happen and sees every call in the agent ledger. Details are in `SUBMISSION.md`.

## Layout

- `server/` FastAPI. `vault.py` handles files and the FTS5 index, `chat.py` the model and vault tools with SSE streaming, `agents.py` the Codex, OpenCode, and Claude runners, `app.py` the routes.
- `web/` Vite, React, and TypeScript front end.
- `scripts/` the initial vault build and the PaperPilot importer.
