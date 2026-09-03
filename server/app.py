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

from . import agents, chat, vault

app = FastAPI(title="Cortex", version="0.1")


@app.on_event("startup")
def _startup() -> None:
    vault.ensure_dirs()
    if os.environ.get("CORTEX_DEMO") == "1" and vault.counts()["papers"] == 0:
        import threading
        from scripts.demo_vault import seed  # public sample vault for the hosted demo
        threading.Thread(target=seed, daemon=True).start()
    vault.rebuild_index()


def _sse(gen: Iterator[dict]) -> StreamingResponse:
    def body():
        for ev in gen:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- health / topics

@app.get("/api/health")
def health():
    return {"ok": True, "vault": str(vault.VAULT), "counts": vault.counts(), "model": chat.DEFAULT_MODEL, "provider": chat.BASE_URL}


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
def library(status: str | None = None, topic: str | None = None, q: str | None = None):
    return vault.list_library(status, topic, q)


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


@app.get("/api/library/{pid}/pdf")
def paper_pdf(pid: str):
    p = vault.paper_pdf_path(pid)
    if not p:
        raise HTTPException(404, "no pdf")
    return FileResponse(str(p), media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{pid}.pdf"'})


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


@app.get("/api/chat/channels")
def channels():
    return [{**c, "count": len(vault.read_chat(c["id"]))} for c in chat.CHANNELS]


@app.get("/api/models")
def models():
    return chat.list_models()


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
    return _sse(chat.stream(channel, m.content.strip(), m.model))


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
