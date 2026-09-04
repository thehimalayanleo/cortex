"""Chat over the vault: an OpenAI-compatible model (OpenCode Go by default) with vault tools, streamed as events."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Iterator

from openai import OpenAI

from . import agents, vault

UA = "cortex/0.1"


def _load_key() -> str:
    k = os.environ.get("CORTEX_API_KEY") or os.environ.get("OPENCODE_API_KEY")
    if k:
        return k
    p = Path("~/.local/share/opencode/auth.json").expanduser()
    if p.exists():
        try:
            return json.loads(p.read_text())["opencode-go"]["key"]
        except Exception:
            pass
    return ""


BASE_URL = os.environ.get("CORTEX_BASE_URL", "https://opencode.ai/zen/go/v1")
DEFAULT_MODEL = os.environ.get("CORTEX_MODEL", "glm-5.3")
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=_load_key() or "missing", default_headers={"User-Agent": UA})
    return _client


_models_cache: tuple[float, list[dict]] = (0.0, [])


def list_models() -> list[dict]:
    global _models_cache
    if time.time() - _models_cache[0] < 3600 and _models_cache[1]:
        return _models_cache[1]
    try:
        ms = [{"id": m.id, "name": m.id} for m in client().models.list().data]
        ms.sort(key=lambda m: m["id"])
    except Exception:
        ms = [{"id": DEFAULT_MODEL, "name": DEFAULT_MODEL}]
    _models_cache = (time.time(), ms)
    return ms


CHANNELS = [
    {"id": "general", "name": "general", "desc": "Anything. Ask the brain, think out loud."},
    {"id": "papers", "name": "papers", "desc": "What to read, what a paper said, what to file."},
    {"id": "projects", "name": "projects", "desc": "Verdicts, next actions, what is stuck."},
    {"id": "ideas", "name": "ideas", "desc": "Half-formed research ideas. Promote the good ones to notes."},
    {"id": "daily", "name": "daily", "desc": "Today. What happened, what is next."},
]

# ---------------------------------------------------------------- tools

TOOLS = [
    {"type": "function", "function": {"name": "search_vault", "description": "Full-text search over notes, projects and papers in the vault. Returns hits with type, id, title, snippet. Use before answering anything about what the user knows, read, or is doing.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string", "enum": ["all", "note", "project", "paper"]}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_note", "description": "Read one note (or daily page) by slug. Returns frontmatter and full markdown body.",
        "parameters": {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}}},
    {"type": "function", "function": {"name": "write_note", "description": "Create a note. kind: fleeting (default), permanent, literature, meeting. body is markdown with $LaTeX$. Only when the user asks to save, note, or remember.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "kind": {"type": "string"}, "body": {"type": "string"}, "topics": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "body"]}}},
    {"type": "function", "function": {"name": "append_daily", "description": "Append a line or paragraph to today's daily page.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "list_lab_chapters", "description": "List the Training Lab chapters (the curriculum: data, pretraining, mid-training, SFT, RL, tool use, embeddings, clustering, evals, red-teaming, architecture, optimizers, GPU/KV cache, Lean, paint-with-code). Each is a note with slug lab-NN-name; read one with read_note.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "gpu_status", "description": "Check the user's GPU box (the home RTX 5090 over Tailscale): reachable, GPU name and memory, whether PyTorch is ready, whether a run is in progress. Call before start_run on the ssh executor.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "gpu_setup", "description": "Prepare the GPU box for runs (installs uv, a Python 3.11 venv, CUDA 12.8 PyTorch, and the training libraries over SSH; idempotent; takes minutes the first time). Returns the last log lines and the final status.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "lab_plan", "description": "The user's learning plan: a kanban of cards (read a chapter, train the station, run the snippet, run the recipe on the GPU, pass the self-test) in columns todo|doing|done, with counts. Use it to suggest what to do next and to track progress.",
        "parameters": {"type": "object", "properties": {"col": {"type": "string", "enum": ["todo", "doing", "done", "all"]}}}}},
    {"type": "function", "function": {"name": "lab_plan_move", "description": "Move a learning card to a column (todo|doing|done) with an optional comment, for example after the user passes a quiz or finishes a run.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "col": {"type": "string"}, "comment": {"type": "string"}}, "required": ["id", "col"]}}},
    {"type": "function", "function": {"name": "list_runs", "description": "List training runs launched from the lab (recipe, executor, status, last metric).",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "start_run", "description": "Launch a lab recipe (a training or evaluation script under lab/recipes, for example pretrain_nano, sft_lora, dpo, grpo_tool, embed_contrastive, eval_suite, redteam_suite, kernel_bench, optim_bench, paint_grpo, lean_eval) on an executor: local (this machine), ssh (the user's GPU box), or modal (rented GPU). args is the script's command line, e.g. '--smoke --steps 200'. Only when the user asks to train, run, or benchmark something.",
        "parameters": {"type": "object", "properties": {"recipe": {"type": "string"}, "args": {"type": "string"}, "executor": {"type": "string", "enum": ["local", "ssh", "modal"]}}, "required": ["recipe"]}}},
    {"type": "function", "function": {"name": "read_run", "description": "Read a run: status, the parsed metrics (loss curves etc.), the final result, and the last log lines.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "tail": {"type": "integer"}}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "read_paper", "description": "Read a library paper: metadata, reading notes, and up to 12000 characters of extracted text starting at offset.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "offset": {"type": "integer"}}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "file_paper", "description": "Add a paper to the library from an arXiv id/URL (downloaded) or a local PDF path. Returns its metadata.",
        "parameters": {"type": "object", "properties": {"arxiv": {"type": "string"}, "path": {"type": "string"}, "takeaway": {"type": "string"}, "topics": {"type": "array", "items": {"type": "string"}}}}}},
    {"type": "function", "function": {"name": "set_paper", "description": "Update a paper's status (inbox|reading|read|reference), rating (1-5), takeaway, or topics.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "status": {"type": "string"}, "rating": {"type": "integer"}, "takeaway": {"type": "string"}, "topics": {"type": "array", "items": {"type": "string"}}}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "list_projects", "description": "List projects with status, verdict and next action. status: active|paused|done|banked|refuted|all.",
        "parameters": {"type": "object", "properties": {"status": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "update_project", "description": "Update a project's verdict, next_action, or status.",
        "parameters": {"type": "object", "properties": {"slug": {"type": "string"}, "verdict": {"type": "string"}, "next_action": {"type": "string"}, "status": {"type": "string"}}, "required": ["slug"]}}},
    {"type": "function", "function": {"name": "key_passages", "description": "The most important passages of a paper: theorems, main results, central claim, method, limitation. Verbatim quotes with page numbers, cached per paper. Use when asked what a paper proves or shows, or to highlight it.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "refresh": {"type": "boolean"}}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "run_agent", "description": "Hand a longer task to a coding agent (codex, opencode, or claude) that runs inside the vault with file tools. Use for multi-file edits, literature sweeps, or anything that needs a shell. Returns its output (bounded).",
        "parameters": {"type": "object", "properties": {"agent": {"type": "string", "enum": ["codex", "opencode", "claude"]}, "task": {"type": "string"}}, "required": ["agent", "task"]}}},
]


HIGHLIGHT_KINDS = ("theorem", "result", "claim", "method", "limitation")


def extract_highlights(pid: str, model: str | None = None, force: bool = False) -> dict:
    """Pull the theorems, main results, and key claims out of a paper as verbatim passages with page numbers.

    Cached in library/<id>/highlights.json. Each passage is checked against the extracted text, so the model
    cannot invent a quote; the page comes from the form feeds pdftotext leaves between pages.
    """
    import re
    p = vault.get_paper(pid)
    if not p:
        raise ValueError(f"no paper {pid}")
    cached = vault.read_highlights(pid)
    if cached and not force:
        return cached
    full = vault.paper_text(pid, max_chars=200000)
    if not full.strip():
        raise ValueError("no extracted text for this paper yet")
    pages = full.split("\f")
    # keep the model's input bounded: first ~45k chars plus the tail (results/conclusion often sit late)
    body = full if len(full) <= 60000 else full[:45000] + "\n\n[... middle omitted ...]\n\n" + full[-15000:]
    prompt = (
        "You are reading a research paper. Extract the 6 to 12 most important passages: theorems, lemmas, main results with their numbers, "
        "the central claim, the key method statement, and the main stated limitation. Quote each passage VERBATIM from the text "
        "(20 to 60 words, exact characters, no paraphrase, no ellipsis). Reply with only a JSON array of objects: "
        '{"kind": one of theorem|result|claim|method|limitation, "quote": "<verbatim>", "why": "<one line on why it matters>"}. '
        "Order by importance.\n\nPAPER TEXT:\n" + body
    )
    resp = client().chat.completions.create(model=model or DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1)
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    try:
        items = json.loads(m.group(0) if m else raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return JSON: {e}")

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    npages = [norm(pg) for pg in pages]
    out = []
    for it in items if isinstance(items, list) else []:
        q = str(it.get("quote", "")).strip()
        if len(q) < 15:
            continue
        nq = norm(q)
        page = next((i + 1 for i, pg in enumerate(npages) if nq in pg), None)
        if page is None:  # try a shorter core of the quote before giving up on it
            core = " ".join(nq.split()[:12])
            page = next((i + 1 for i, pg in enumerate(npages) if core and core in pg), None)
        if page is None:
            continue  # not in the paper: drop it rather than show an invented quote
        kind = str(it.get("kind", "claim")).lower()
        out.append({"kind": kind if kind in HIGHLIGHT_KINDS else "claim", "quote": q[:600], "why": str(it.get("why", ""))[:300], "page": page})
    result = {"id": pid, "model": model or DEFAULT_MODEL, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": out[:12]}
    vault.write_highlights(pid, result)
    return result


def _exec(name: str, a: dict) -> tuple[object, str, str]:
    if name == "key_passages":
        r = extract_highlights(str(a["id"]), force=bool(a.get("refresh")))
        return r["items"], f"{len(r['items'])} passages · {str(a['id'])[:30]}", f"cortex://paper/{a['id']}"
    """Run a tool. Returns (result_for_model, summary_for_ledger, link)."""
    if name == "search_vault":
        types = None if a.get("scope") in (None, "all") else [a["scope"]]
        hits = vault.search(str(a.get("query", "")), limit=10, types=types)
        return hits, f"{a.get('scope','all')}: {a.get('query','')} · {len(hits)} hits", ""
    if name == "read_note":
        n = vault.get_note(str(a["slug"]))
        if not n:
            raise ValueError(f"no note {a['slug']}")
        return {"frontmatter": n["frontmatter"], "body": n["body"][:12000]}, n["frontmatter"].get("title", n["slug"]), f"cortex://note/{n['slug']}"
    if name == "write_note":
        n = vault.create_note(str(a["title"]), str(a.get("kind") or "fleeting"), str(a.get("body", "")), list(a.get("topics") or []))
        return {"slug": n["slug"], "link": f"cortex://note/{n['slug']}"}, a["title"], f"cortex://note/{n['slug']}"
    if name == "append_daily":
        n = vault.append_daily(str(a["text"]))
        return {"slug": n["slug"]}, str(a["text"])[:80], f"cortex://note/{n['slug']}"
    if name == "read_paper":
        p = vault.get_paper(str(a["id"]))
        if not p:
            raise ValueError(f"no paper {a['id']}")
        off = int(a.get("offset") or 0)
        text = vault.paper_text(str(a["id"]))[off: off + 12000]
        return {"meta": p["meta"], "notes": p["notes"], "text": text, "next_offset": off + len(text)}, p["meta"].get("title", a["id"]), f"cortex://paper/{a['id']}"
    if name == "file_paper":
        extra = {k: a[k] for k in ("takeaway", "topics") if a.get(k)}
        if a.get("arxiv"):
            m = vault.ingest_arxiv(str(a["arxiv"]), **extra)
        elif a.get("path"):
            m = vault.ingest_pdf(Path(str(a["path"])).expanduser(), extra)
        else:
            raise ValueError("give arxiv or path")
        return m, m["title"], f"cortex://paper/{m['id']}"
    if name == "set_paper":
        patch = {k: a[k] for k in ("status", "rating", "takeaway", "topics") if a.get(k) is not None}
        p = vault.update_paper(str(a["id"]), patch)
        if not p:
            raise ValueError(f"no paper {a['id']}")
        return p["meta"], f"{p['meta'].get('title','')[:50]} · {', '.join(patch)}", f"cortex://paper/{a['id']}"
    if name == "list_projects":
        st = a.get("status") or "active"
        ps = vault.list_projects(None if st == "all" else st)
        rows = [{"slug": p["slug"], **{k: p["frontmatter"].get(k) for k in ("title", "status", "type", "verdict", "next_action", "deadline")}} for p in ps]
        return rows, f"projects · {st} · {len(rows)}", ""
    if name == "update_project":
        patch = {k: a[k] for k in ("verdict", "next_action", "status") if a.get(k)}
        p = vault.update_project(str(a["slug"]), patch)
        if not p:
            raise ValueError(f"no project {a['slug']}")
        return p["frontmatter"], f"{p['frontmatter'].get('title','')} · {', '.join(patch)}", f"cortex://project/{a['slug']}"
    if name == "list_lab_chapters":
        from . import runs
        ch = runs.chapters()
        return ch, f"lab · {len(ch)} chapters", "cortex://lab"
    if name == "gpu_status":
        from . import runs
        g = runs.gpu_status()
        return g, g.get("message", ""), "cortex://lab/runs"
    if name == "gpu_setup":
        from . import runs
        lines = []
        final = None
        for ev in runs.gpu_setup_stream():
            if ev.get("type") == "log":
                lines.extend(ev["lines"])
            elif ev.get("type") in ("status", "error"):
                final = ev
        return {"log": lines[-40:], "final": final}, f"gpu setup · {(final or {}).get('status', 'error')}", "cortex://lab/runs"
    if name == "lab_plan":
        from . import runs
        pl = runs.plan()
        col = a.get("col") or "all"
        cards = [c for c in pl["cards"] if col == "all" or c["col"] == col]
        return {"done": pl["done"], "total": pl["total"], "cards": cards}, f"plan · {pl['done']}/{pl['total']} done", "cortex://lab/plan"
    if name == "lab_plan_move":
        from . import runs
        pl = runs.plan_move(str(a["id"]), str(a["col"]), a.get("comment"))
        return {"done": pl["done"], "total": pl["total"]}, f"{a['id']} → {a['col']}", "cortex://lab/plan"
    if name == "list_runs":
        from . import runs
        rs = runs.list_runs(int(a.get("limit") or 20))
        rows = [{k: r.get(k) for k in ("id", "recipe", "args", "executor", "status", "started", "ended", "last")} for r in rs]
        return rows, f"runs · {len(rows)}", "cortex://lab"
    if name == "start_run":
        from . import runs
        m = runs.start(str(a["recipe"]), str(a.get("args") or ""), str(a.get("executor") or "local"))
        return m, f"{m['recipe']} on {m['executor']} · {m['id']}", f"cortex://lab/run/{m['id']}"
    if name == "read_run":
        from . import runs
        r = runs.read_run(str(a["id"]), int(a.get("tail") or 60))
        if not r:
            raise ValueError(f"no run {a['id']}")
        m = r["metrics"]
        thin = m if len(m) <= 60 else [m[int(i * len(m) / 60)] for i in range(60)] + [m[-1]]
        return {**{k: r[k] for k in ("id", "recipe", "args", "executor", "status", "started", "ended", "exit")}, "result": r.get("result"), "metrics": thin, "log": r["log"]}, f"run {r['id']} · {r['status']}", f"cortex://lab/run/{r['id']}"
    if name == "run_agent":
        out = agents.run_capture(str(a["agent"]), str(a["task"]))
        return {"output": out}, f"{a['agent']}: {str(a['task'])[:70]}", ""
    raise ValueError(f"unknown tool {name}")


WRITE_TOOLS = {"write_note", "append_daily", "file_paper", "set_paper", "update_project", "run_agent", "start_run", "gpu_setup", "lab_plan_move"}


def system_prompt(channel: str) -> str:
    ch = next((c for c in CHANNELS if c["id"] == channel), CHANNELS[0])
    c = vault.counts()
    return (
        "You are Cortex, the research partner living inside Ajinkya's second brain: a local vault of markdown notes, "
        "a PDF library, and project pages. Ajinkya is a senior ML research scientist (sparse methods, continual learning, "
        f"interpretability, safety, GPU kernels). The vault holds {c['notes']} notes, {c['papers']} papers, {c['projects']} projects.\n"
        f"Today is {date.today().isoformat()}. Channel #{ch['name']}: {ch['desc']}\n"
        "Rules: ground answers in the vault (search_vault first, read_note/read_paper for detail); say plainly when the vault has nothing. "
        "Write only when asked to save, note, file, or remember, and confirm what you wrote with its link. "
        "Cite notes and papers by title with links of the form cortex://note/<slug>, cortex://paper/<id>, cortex://project/<slug>. "
        "Be direct and specific: short paragraphs, no filler, no headers unless the answer is long. Markdown with $LaTeX$ is fine. "
        "Tone: a sharp colleague, not an assistant."
    )


def _context_line(ctx: dict | None) -> str:
    """Describe what the user has open so 'this paper' / 'this note' resolve to it."""
    if not ctx or not ctx.get("kind"):
        return ""
    kind, cid, title = str(ctx.get("kind")), str(ctx.get("id") or ""), str(ctx.get("title") or "")
    if kind == "paper" and cid:
        p = vault.get_paper(cid)
        if p:
            m = p["meta"]
            return (f"\nOPEN IN THE APP RIGHT NOW: paper {cid} \"{m.get('title','')}\" ({m.get('authors','')}, {m.get('year','')}), "
                    f"status {m.get('status')}, takeaway: {m.get('takeaway') or 'none'}. When the user says 'this paper' they mean it; "
                    f"call read_paper(\"{cid}\") for its text. Link: cortex://paper/{cid}")
    if kind in ("note", "daily") and cid:
        return f"\nOPEN IN THE APP RIGHT NOW: note \"{title or cid}\" (slug {cid}). 'This note' means it; call read_note(\"{cid}\") for the body. Link: cortex://note/{cid}"
    if kind == "project" and cid:
        return f"\nOPEN IN THE APP RIGHT NOW: project \"{title or cid}\" (slug {cid}). 'This project' means it. Link: cortex://project/{cid}"
    if kind == "lab":
        where = f"station '{cid}'" if cid else "the overview"
        return (f"\nOPEN IN THE APP RIGHT NOW: the Training Lab, {where}. The lab has in-browser stations (data, pretrain, midtrain, posttrain, encoder, cluster, paint) "
                "that train a tiny transformer with tf.js, 15 teaching chapters (list_lab_chapters, then read_note on a slug like lab-05-preference-and-rl), "
                "and GPU runs (list_runs, start_run, read_run). Teach like a pedantic, careful instructor: define terms, derive, and quiz the user when they ask to be tested. "
                "Link: cortex://lab")
    return ""


def _space_line(ctx: dict | None) -> str:
    """The active space (a project used as a workspace): its verdict, next action, and papers."""
    slug = str((ctx or {}).get("space") or "")
    if not slug or slug == "all":
        return ""
    p = vault.get_project(slug)
    if not p:
        return ""
    fm = p["frontmatter"]
    papers = vault.list_library(project=slug)
    titles = "; ".join(f"{m.get('title','')[:60]} ({m.get('id')})" for m in papers[:25])
    return (f"\nACTIVE SPACE: \"{fm.get('title', slug)}\" (project slug {slug}); status {fm.get('status')}; verdict: {fm.get('verdict') or 'none yet'}; "
            f"next action: {fm.get('next_action') or 'none'}. Papers in this space ({len(papers)}): {titles or 'none yet'}. "
            f"Prefer these when the user speaks about 'this project' or 'these papers'. Link: cortex://project/{slug}")


def stream(channel: str, content: str, model: str | None = None, context: dict | None = None) -> Iterator[dict]:
    model = model or DEFAULT_MODEL
    history = vault.read_chat(channel, limit=40)
    user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": content, "ts": int(time.time() * 1000)}
    vault.append_chat(channel, user_msg)
    messages: list[dict] = [{"role": "system", "content": system_prompt(channel) + _space_line(context) + _context_line(context)}]
    for m in history[-16:]:
        if m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:6000]})
    messages.append({"role": "user", "content": content})

    text_all, trace = "", []
    try:
        for _round in range(8):
            resp = client().chat.completions.create(model=model, messages=messages, tools=TOOLS, stream=True, temperature=0.3)
            round_text, calls = "", {}
            for chunk in resp:
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                if d.content:
                    round_text += d.content; text_all += d.content
                    yield {"type": "text", "delta": d.content}
                for tc in d.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": tc.id or "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
            if not calls:
                break
            messages.append({"role": "assistant", "content": round_text or None,
                             "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for i, c in calls.items()]})
            for i, c in calls.items():
                cid = c["id"] or f"call_{i}"
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool", "id": cid, "name": c["name"], "input": args, "status": "running"}
                try:
                    result, summary, link = _exec(c["name"], args)
                    ev = {"type": "tool", "id": cid, "name": c["name"], "input": args, "status": "ok", "summary": summary, "link": link, "write": c["name"] in WRITE_TOOLS}
                    payload = json.dumps(result, ensure_ascii=False)[:16000]
                except Exception as e:  # tool errors go back to the model, not to the user as a crash
                    ev = {"type": "tool", "id": cid, "name": c["name"], "input": args, "status": "error", "summary": str(e)[:160], "link": "", "write": False}
                    payload = f"Error: {e}"
                trace.append({k: ev[k] for k in ("name", "status", "summary", "link", "write")})
                yield ev
                messages.append({"role": "tool", "tool_call_id": cid, "content": payload})
            if text_all:
                text_all += "\n\n"; yield {"type": "text", "delta": "\n\n"}
    except Exception as e:
        yield {"type": "error", "code": type(e).__name__, "message": str(e)[:300]}
    msg = {"id": uuid.uuid4().hex, "role": "assistant", "content": text_all.strip(), "ts": int(time.time() * 1000), "trace": trace, "model": model}
    if msg["content"] or trace:
        vault.append_chat(channel, msg)
    yield {"type": "done", "message": msg}
