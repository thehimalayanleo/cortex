"""Cortex server. Run:  cd cortex && ./run.sh   (FastAPI on :8788, serves web/dist at /)"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agents, chat, runs, vault

app = FastAPI(title="Cortex", version="0.1")


INBOX = None  # set at startup: a folder you can drop PDFs into from Finder; they get filed automatically


def _watch_inbox(folder: Path) -> None:
    """Every few seconds, file any PDF that landed in the inbox folder. Non-PDFs are left alone."""
    import time
    seen: dict[str, float] = {}
    while True:
        try:
            for f in sorted(folder.glob("*.pdf")):
                key = str(f)
                mtime = f.stat().st_mtime
                # wait until the file stopped growing (Finder copies are not atomic)
                if seen.get(key) != mtime:
                    seen[key] = mtime
                    continue
                try:
                    m = vault.ingest_pdf(f, {}, move=True)
                    if f.exists():  # already in the library under this id: the vault copy wins, drop the duplicate
                        f.unlink()
                        print(f"inbox: {f.name} is already filed as {m.get('id')}; removed the duplicate")
                    else:
                        print(f"inbox: filed {f.name} as {m.get('id')}")
                except Exception as e:  # keep the watcher alive; a bad file just stays put
                    print("inbox: could not ingest", f.name, e)
                    (folder / (f.name + ".failed")).write_text(str(e))
                    f.rename(folder / ("failed-" + f.name))
                seen.pop(key, None)
        except Exception as e:
            print("inbox watcher:", e)
        time.sleep(4)


@app.on_event("startup")
def _startup() -> None:
    global INBOX
    vault.ensure_dirs()
    import threading
    if os.environ.get("CORTEX_DEMO") == "1" and vault.counts()["papers"] == 0:
        from scripts.demo_vault import seed  # public sample vault for the hosted demo
        threading.Thread(target=seed, daemon=True).start()
    vault.rebuild_index()
    INBOX = vault.VAULT / "inbox"
    INBOX.mkdir(exist_ok=True)
    (vault.VAULT / "assets").mkdir(exist_ok=True)
    threading.Thread(target=_watch_inbox, args=(INBOX,), daemon=True).start()
    try:
        n = runs.sync_chapters_into_vault()
        if n:
            vault.rebuild_index()
            print(f"lab: synced {n} chapter(s) into the vault")
    except Exception as e:
        print("lab: chapter sync failed", e)


def _sse(gen: Iterator[dict]) -> StreamingResponse:
    def body():
        for ev in gen:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- health / topics

@app.get("/api/health")
def health():
    return {"ok": True, "vault": str(vault.VAULT), "inbox": str(INBOX or vault.VAULT / "inbox"), "counts": vault.counts(), "model": chat.DEFAULT_MODEL, "provider": chat.BASE_URL}


# ---------------------------------------------------------------- attachments (images and files inside notes)

ASSETS = lambda: vault.VAULT / "assets"  # noqa: E731
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def _safe_name(name: str) -> str:
    import re
    base = re.sub(r"[^\w.\-]+", "-", name.strip())[:120].strip("-.") or "file"
    return base


@app.post("/api/notes/{slug}/attach")
async def note_attach(slug: str, file: UploadFile):
    """Store a pasted or dropped file next to the vault and return the markdown to embed it."""
    folder = ASSETS() / _safe_name(slug)
    folder.mkdir(parents=True, exist_ok=True)
    name = _safe_name(file.filename or "pasted")
    dst = folder / name
    stem, ext = dst.stem, dst.suffix
    n = 1
    while dst.exists():
        n += 1
        dst = folder / f"{stem}-{n}{ext}"
    dst.write_bytes(await file.read())
    url = f"/api/assets/{folder.name}/{dst.name}"
    kind = "image" if dst.suffix.lower() in IMAGE_EXT else "file"
    md = f"![{dst.stem}]({url})" if kind == "image" else f"[{dst.name}]({url})"
    return {"url": url, "name": dst.name, "kind": kind, "markdown": md, "bytes": dst.stat().st_size}


@app.get("/api/assets/{slug}/{name}")
def asset_get(slug: str, name: str):
    p = ASSETS() / _safe_name(slug) / _safe_name(name)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "no such asset")
    return FileResponse(str(p), headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/topics")
def topics():
    return vault.list_topics()


# ---------------------------------------------------------------- notes

class NoteIn(BaseModel):
    title: str
    kind: str | None = "fleeting"
    body: str | None = ""
    topics: list[str] | None = None


class DocPatch(BaseModel):
    frontmatter: dict[str, Any] | None = None
    body: str | None = None


@app.get("/api/notes")
def notes(kind: str | None = None, q: str | None = None, limit: int = 200):
    return vault.list_notes(kind, q, limit)


@app.post("/api/notes")
def note_create(n: NoteIn):
    return vault.create_note(n.title, n.kind or "fleeting", n.body or "", n.topics or [])


@app.get("/api/daily/today")
def daily_today():
    return vault.today_note()


@app.get("/api/notes/{slug}")
def note_get(slug: str):
    n = vault.get_note(slug)
    if not n:
        raise HTTPException(404, "no such note")
    return n


@app.put("/api/notes/{slug}")
def note_put(slug: str, p: DocPatch):
    n = vault.update_note(slug, p.frontmatter, p.body)
    if not n:
        raise HTTPException(404, "no such note")
    return n


@app.delete("/api/notes/{slug}")
def note_delete(slug: str):
    if not vault.delete_note(slug):
        raise HTTPException(404, "no such note")
    return {"ok": True}


# ---------------------------------------------------------------- projects

class ProjectIn(BaseModel):
    title: str
    status: str | None = "active"
    type: str | None = "research"
    verdict: str | None = None
    next_action: str | None = None
    deadline: str | None = None
    repo: str | None = None
    topics: list[str] | None = None
    body: str | None = ""


@app.get("/api/projects")
def projects(status: str | None = None):
    return vault.list_projects(status)


@app.post("/api/projects")
def project_create(p: ProjectIn):
    d = p.model_dump()
    body = d.pop("body") or ""
    title = d.pop("title")
    return vault.create_project(title, body, **d)


@app.get("/api/projects/{slug}")
def project_get(slug: str):
    p = vault.get_project(slug)
    if not p:
        raise HTTPException(404, "no such project")
    return p


@app.put("/api/projects/{slug}")
def project_put(slug: str, p: DocPatch):
    d = vault.update_project(slug, p.frontmatter, p.body)
    if not d:
        raise HTTPException(404, "no such project")
    return d


# ---------------------------------------------------------------- library

class IngestIn(BaseModel):
    path: str | None = None
    arxiv: str | None = None
    url: str | None = None
    takeaway: str | None = None
    topics: list[str] | None = None


class PaperPatch(BaseModel):
    meta: dict[str, Any] | None = None
    notes: str | None = None


@app.get("/api/library")
def library(status: str | None = None, topic: str | None = None, q: str | None = None, project: str | None = None):
    return vault.list_library(status, topic, q, project)


@app.get("/api/library/{pid}")
def paper_get(pid: str):
    p = vault.get_paper(pid)
    if not p:
        raise HTTPException(404, "no such paper")
    return p


@app.put("/api/library/{pid}")
def paper_put(pid: str, p: PaperPatch):
    r = vault.update_paper(pid, p.meta, p.notes)
    if not r:
        raise HTTPException(404, "no such paper")
    return r


class HighlightsIn(BaseModel):
    model: str | None = None
    refresh: bool = False
    pages: str | None = None  # e.g. "1-5" or "first 5"; None = whole paper (long papers auto-scope to the head and tail)


@app.get("/api/library/{pid}/highlights")
def paper_highlights(pid: str):
    """Cached key passages, or {items: null} when they have not been extracted yet."""
    return vault.read_highlights(pid) or {"id": pid, "items": None}


@app.post("/api/library/{pid}/highlights")
def paper_highlights_make(pid: str, h: HighlightsIn):
    try:
        return chat.extract_highlights(pid, h.model, force=h.refresh or bool(h.pages), pages=h.pages)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"highlighting failed: {e}")


@app.get("/api/library/{pid}/pdf")
def paper_pdf(pid: str):
    p = vault.paper_pdf_path(pid)
    if not p:
        raise HTTPException(404, "no pdf")
    # PDFs are immutable per id: let the browser keep them for a day so reopening a paper is instant.
    return FileResponse(str(p), media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{pid}.pdf"', "Cache-Control": "private, max-age=86400"})


@app.post("/api/library/ingest")
def library_ingest(i: IngestIn):
    extra = {k: v for k, v in (("takeaway", i.takeaway), ("topics", i.topics)) if v}
    try:
        if i.arxiv or (i.url and "arxiv.org" in i.url):
            return vault.ingest_arxiv(i.arxiv or i.url, **extra)
        if i.path:
            src = Path(i.path).expanduser()
            if not src.exists():
                raise HTTPException(400, "path does not exist")
            return vault.ingest_pdf(src, extra)
        if i.url:
            import urllib.request
            tmp = vault.VAULT / ".cortex" / "download.pdf"
            tmp.write_bytes(urllib.request.urlopen(urllib.request.Request(i.url, headers={"User-Agent": chat.UA}), timeout=120).read())
            return vault.ingest_pdf(tmp, extra, move=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"ingest failed: {e}")
    raise HTTPException(400, "give path, arxiv, or url")


@app.post("/api/library/upload")
async def library_upload(file: UploadFile):
    tmp = Path(tempfile.mkdtemp()) / (file.filename or "upload.pdf")
    tmp.write_bytes(await file.read())
    try:
        return vault.ingest_pdf(tmp, {}, move=True)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


# ---------------------------------------------------------------- search

@app.get("/api/search")
def search(q: str, limit: int = 20):
    return vault.search(q, limit)


# ---------------------------------------------------------------- chat

class ChatIn(BaseModel):
    content: str
    model: str | None = None
    context: dict[str, Any] | None = None  # what is open in the app: {kind: note|paper|project|daily, id, title}


@app.get("/api/chat/channels")
def channels():
    return [{**c, "count": len(vault.read_chat(c["id"]))} for c in chat.CHANNELS]


@app.get("/api/models")
def models():
    return chat.list_models()


class ToolResultIn(BaseModel):
    result: Any = None


@app.post("/api/chat/tool_result/{cid}")
def chat_tool_result(cid: str, r: ToolResultIn):
    """The page reports the outcome of a client-executed tool (open_lab, lab_train, lab_status)."""
    return {"ok": chat.client_tool_result(cid, r.result)}


@app.get("/api/chat/{channel}")
def chat_get(channel: str):
    return vault.read_chat(channel)


@app.delete("/api/chat/{channel}")
def chat_clear(channel: str):
    vault.clear_chat(channel)
    return {"ok": True}


@app.post("/api/chat/{channel}")
def chat_post(channel: str, m: ChatIn):
    if not m.content.strip():
        raise HTTPException(400, "empty message")
    return _sse(chat.stream(channel, m.content.strip(), m.model, m.context))


# ---------------------------------------------------------------- agents

class AgentIn(BaseModel):
    agent: str
    task: str


@app.get("/api/agents")
def agents_list():
    return agents.available()


@app.post("/api/agents/run")
def agents_run(a: AgentIn):
    def gen():
        code = 0
        for line in agents.run(a.agent, a.task):
            yield {"type": "log", "line": line}
            if line.startswith("[cortex]") and "exited with code" in line:
                try:
                    code = int(line.rsplit(" ", 1)[1])
                except ValueError:
                    code = -1
        yield {"type": "done", "code": code}
    return _sse(gen())


# ---------------------------------------------------------------- training lab: runs on a GPU (this machine, the 5090 over SSH, or Modal)

class RunIn(BaseModel):
    recipe: str
    args: str | None = ""
    executor: str | None = "local"


@app.get("/api/lab/executors")
def lab_executors():
    return runs.executors()


@app.get("/api/lab/gpu")
def lab_gpu():
    return runs.gpu_status()


@app.post("/api/lab/gpu/setup")
def lab_gpu_setup():
    """SSE: bootstrap the GPU box (uv, python, CUDA torch, training libs). Safe to re-run."""
    return _sse(runs.gpu_setup_stream())


class PlanMove(BaseModel):
    id: str
    col: str
    comment: str | None = None


@app.get("/api/lab/plan")
def lab_plan():
    return runs.plan()


@app.post("/api/lab/plan/move")
def lab_plan_move(m: PlanMove):
    try:
        return runs.plan_move(m.id, m.col, m.comment)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/lab/recipes")
def lab_recipes():
    return runs.recipes()


@app.get("/api/lab/chapters")
def lab_chapters():
    return runs.chapters()


@app.get("/api/lab/runs")
def lab_runs(limit: int = 50):
    return runs.list_runs(limit)


@app.post("/api/lab/runs")
def lab_run_start(r: RunIn):
    try:
        return runs.start(r.recipe, r.args or "", r.executor or "local")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/lab/runs/{rid}")
def lab_run_get(rid: str, tail: int = 200):
    r = runs.read_run(rid, tail)
    if not r:
        raise HTTPException(404, "no such run")
    return r


@app.post("/api/lab/runs/{rid}/stop")
def lab_run_stop(rid: str):
    return {"ok": runs.stop(rid)}


@app.delete("/api/lab/runs/{rid}")
def lab_run_delete(rid: str):
    if not runs.delete_run(rid):
        raise HTTPException(404, "no such run")
    return {"ok": True}


@app.get("/api/lab/runs/{rid}/events")
def lab_run_events(rid: str):
    """SSE: log lines and metrics as they arrive, then a final status."""
    import time as _t

    def gen():
        seen_log = 0
        seen_m = 0
        while True:
            r = runs.read_run(rid, tail=100000)
            if not r:
                yield {"type": "error", "message": "no such run"}
                return
            log = r["log"]
            if len(log) > seen_log:
                yield {"type": "log", "lines": log[seen_log:]}
                seen_log = len(log)
            m = r["metrics"]
            if len(m) > seen_m:
                yield {"type": "metrics", "rows": m[seen_m:]}
                seen_m = len(m)
            if r["status"] in ("done", "failed", "stopped"):
                yield {"type": "status", "status": r["status"], "result": r.get("result"), "exit": r.get("exit")}
                return
            _t.sleep(1.0)

    return _sse(gen())


# ---------------------------------------------------------------- the training lab (static page + chapters)

LAB = Path(__file__).resolve().parent.parent / "lab"


@app.get("/lab")
@app.get("/lab/")
def lab_page():
    p = LAB / "index.html"
    if not p.exists():
        raise HTTPException(404, "lab not built")
    return FileResponse(str(p), media_type="text/html")


# ---------------------------------------------------------------- static frontend

DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.exists() and f.is_file():
            return FileResponse(str(f))
        return FileResponse(str(DIST / "index.html"))
else:
    @app.get("/")
    def root():
        return JSONResponse({"ok": True, "hint": "build the web app: cd web && pnpm build"})
