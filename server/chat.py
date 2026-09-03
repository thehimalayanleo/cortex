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
    {"type": "function", "function": {"name": "run_agent", "description": "Hand a longer task to a coding agent (codex, opencode, or claude) that runs inside the vault with file tools. Use for multi-file edits, literature sweeps, or anything that needs a shell. Returns its output (bounded).",
        "parameters": {"type": "object", "properties": {"agent": {"type": "string", "enum": ["codex", "opencode", "claude"]}, "task": {"type": "string"}}, "required": ["agent", "task"]}}},
]


def _exec(name: str, a: dict) -> tuple[object, str, str]:
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
    if name == "run_agent":
        out = agents.run_capture(str(a["agent"]), str(a["task"]))
        return {"output": out}, f"{a['agent']}: {str(a['task'])[:70]}", ""
    raise ValueError(f"unknown tool {name}")


WRITE_TOOLS = {"write_note", "append_daily", "file_paper", "set_paper", "update_project", "run_agent"}


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


def stream(channel: str, content: str, model: str | None = None) -> Iterator[dict]:
    model = model or DEFAULT_MODEL
    history = vault.read_chat(channel, limit=40)
    user_msg = {"id": uuid.uuid4().hex, "role": "user", "content": content, "ts": int(time.time() * 1000)}
    vault.append_chat(channel, user_msg)
    messages: list[dict] = [{"role": "system", "content": system_prompt(channel)}]
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
