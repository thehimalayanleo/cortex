"""Cortex vault: plain files on disk plus a small FTS5 index. No database of record, only files."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT = Path(os.environ.get("CORTEX_VAULT", "~/Cortex")).expanduser()
KINDS = ["fleeting", "literature", "permanent", "meeting", "daily"]
STATUSES = ["active", "paused", "done", "banked", "refuted"]
LIB_STATUSES = ["inbox", "reading", "read", "reference"]
LIB_TYPES = ["paper", "book", "blog", "talk", "doc", "course"]


def ensure_dirs() -> None:
    for d in ("notes", "daily", "projects", "library", "chats", ".cortex"):
        (VAULT / d).mkdir(parents=True, exist_ok=True)
    if not (VAULT / "topics.json").exists():
        (VAULT / "topics.json").write_text("[]\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "untitled"


# ---------------------------------------------------------------- markdown files

def read_md(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            fm = yaml.safe_load(raw[4:end]) or {}
            body = raw[end + 4 :].lstrip("\n")
            return (fm if isinstance(fm, dict) else {}), body
    return {}, raw


def write_md(path: Path, fm: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in fm.items() if v not in (None, "", [])}
    head = yaml.safe_dump(clean, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{head}\n---\n\n{body.rstrip()}\n", encoding="utf-8")


def _note_path(slug: str) -> Path:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", slug):
        return VAULT / "daily" / f"{slug}.md"
    return VAULT / "notes" / f"{slug}.md"


def _doc(path: Path, kind_dir: str) -> dict:
    fm, body = read_md(path)
    slug = path.stem
    return {"slug": slug, "frontmatter": fm, "body": body}


def _summary(path: Path) -> dict:
    fm, body = read_md(path)
    preview = re.sub(r"\s+", " ", body)[:160]
    return {
        "slug": path.stem,
        "title": fm.get("title") or path.stem,
        "kind": fm.get("kind", "daily" if path.parent.name == "daily" else "fleeting"),
        "topics": fm.get("topics", []),
        "updated": fm.get("updated") or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "preview": preview,
    }


# ---------------------------------------------------------------- notes

def list_notes(kind: str | None = None, q: str | None = None, limit: int = 200) -> list[dict]:
    paths = list((VAULT / "notes").glob("*.md")) + list((VAULT / "daily").glob("*.md"))
    out = [_summary(p) for p in paths]
    if kind:
        out = [n for n in out if n["kind"] == kind]
    if q:
        ql = q.lower()
        out = [n for n in out if ql in n["title"].lower() or ql in n["preview"].lower()]
    out.sort(key=lambda n: n["updated"], reverse=True)
    return out[:limit]


def get_note(slug: str) -> dict | None:
    p = _note_path(slug)
    return _doc(p, "notes") if p.exists() else None


def create_note(title: str, kind: str = "fleeting", body: str = "", topics: list[str] | None = None, **extra) -> dict:
    slug = slugify(title)
    p = _note_path(slug)
    i = 2
    while p.exists():
        slug = f"{slugify(title)}-{i}"; p = _note_path(slug); i += 1
    fm = {"title": title, "kind": kind if kind in KINDS else "fleeting", "topics": topics or [],
          "created": now_iso(), "updated": now_iso(), **extra}
    write_md(p, fm, body)
    index_upsert("note", slug, title, body)
    return _doc(p, "notes")


def update_note(slug: str, frontmatter: dict | None = None, body: str | None = None) -> dict | None:
    p = _note_path(slug)
    if not p.exists():
        return None
    fm, old_body = read_md(p)
    if frontmatter:
        fm.update({k: v for k, v in frontmatter.items() if k not in ("created",)})
    fm["updated"] = now_iso()
    new_body = old_body if body is None else body
    write_md(p, fm, new_body)
    index_upsert("note", slug, fm.get("title", slug), new_body)
    return _doc(p, "notes")


def delete_note(slug: str) -> bool:
    p = _note_path(slug)
    if not p.exists():
        return False
    p.unlink(); index_delete("note", slug)
    return True


def today_note() -> dict:
    slug = date.today().isoformat()
    p = _note_path(slug)
    if not p.exists():
        write_md(p, {"title": slug, "kind": "daily", "topics": [], "created": now_iso(), "updated": now_iso()}, "")
    return _doc(p, "daily")


def append_daily(text: str) -> dict:
    n = today_note()
    body = (n["body"].rstrip() + "\n\n" + text.strip() + "\n").lstrip("\n")
    return update_note(n["slug"], body=body)


# ---------------------------------------------------------------- projects

def list_projects(status: str | None = None) -> list[dict]:
    out = [_doc(p, "projects") for p in (VAULT / "projects").glob("*.md")]
    if status:
        out = [d for d in out if d["frontmatter"].get("status") == status]
    out.sort(key=lambda d: d["frontmatter"].get("updated", ""), reverse=True)
    return out


def get_project(slug: str) -> dict | None:
    p = VAULT / "projects" / f"{slug}.md"
    return _doc(p, "projects") if p.exists() else None


def create_project(title: str, body: str = "", **fm_extra) -> dict:
    slug = slugify(title)
    p = VAULT / "projects" / f"{slug}.md"
    fm = {"title": title, "status": "active", "type": "research", "created": now_iso(), "updated": now_iso()}
    fm.update({k: v for k, v in fm_extra.items() if v is not None})
    write_md(p, fm, body)
    index_upsert("project", slug, title, f"{fm.get('verdict','')}\n{fm.get('next_action','')}\n{body}")
    return _doc(p, "projects")


def update_project(slug: str, frontmatter: dict | None = None, body: str | None = None) -> dict | None:
    p = VAULT / "projects" / f"{slug}.md"
    if not p.exists():
        return None
    fm, old = read_md(p)
    if frontmatter:
        fm.update({k: v for k, v in frontmatter.items() if k != "created"})
    fm["updated"] = now_iso()
    new_body = old if body is None else body
    write_md(p, fm, new_body)
    index_upsert("project", slug, fm.get("title", slug), f"{fm.get('verdict','')}\n{fm.get('next_action','')}\n{new_body}")
    return _doc(p, "projects")


# ---------------------------------------------------------------- library

def _lib_dir(pid: str) -> Path:
    return VAULT / "library" / pid


def read_meta(pid: str) -> dict | None:
    p = _lib_dir(pid) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(pid: str, meta: dict) -> None:
    d = _lib_dir(pid); d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def list_library(status: str | None = None, topic: str | None = None, q: str | None = None, project: str | None = None) -> list[dict]:
    out = []
    for d in (VAULT / "library").iterdir():
        m = read_meta(d.name) if d.is_dir() else None
        if m:
            m["has_pdf"] = (d / "paper.pdf").exists()
            m.setdefault("projects", [])
            out.append(m)
    if status:
        out = [m for m in out if m.get("status") == status]
    if topic:
        out = [m for m in out if topic in (m.get("topics") or [])]
    if project:  # a "space": papers assigned to that project slug; "none" = unassigned
        out = [m for m in out if (not (m.get("projects") or []) if project == "none" else project in (m.get("projects") or []))]
    if q:
        ql = q.lower()
        out = [m for m in out if ql in (m.get("title", "") + " " + m.get("authors", "") + " " + m.get("takeaway", "")).lower()]
    out.sort(key=lambda m: m.get("added", ""), reverse=True)
    return out


def get_paper(pid: str) -> dict | None:
    m = read_meta(pid)
    if not m:
        return None
    d = _lib_dir(pid)
    notes = (d / "notes.md").read_text() if (d / "notes.md").exists() else ""
    text = (d / "text.txt").read_text(errors="ignore")[:4000] if (d / "text.txt").exists() else ""
    m["has_pdf"] = (d / "paper.pdf").exists()
    return {"meta": m, "notes": notes, "text_preview": text}


def update_paper(pid: str, patch: dict | None = None, notes: str | None = None) -> dict | None:
    m = read_meta(pid)
    if not m:
        return None
    if patch:
        for k, v in patch.items():
            if k in ("id", "added"):
                continue
            m[k] = v
    write_meta(pid, m)
    if notes is not None:
        (_lib_dir(pid) / "notes.md").write_text(notes)
    index_upsert("paper", pid, m.get("title", pid), _paper_index_text(pid, m))
    return get_paper(pid)


def paper_pdf_path(pid: str) -> Path | None:
    p = _lib_dir(pid) / "paper.pdf"
    return p if p.exists() else None


def paper_text(pid: str, max_chars: int = 60000) -> str:
    p = _lib_dir(pid) / "text.txt"
    return p.read_text(errors="ignore")[:max_chars] if p.exists() else ""


def extract_text(pdf: Path, out: Path) -> str:
    try:
        r = subprocess.run(["pdftotext", "-layout", "-q", str(pdf), "-"], capture_output=True, text=True, timeout=120)
        text = r.stdout
    except Exception:
        text = ""
    if not text.strip():
        try:
            from pypdf import PdfReader
            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(pdf)).pages[:80])
        except Exception:
            text = ""
    out.write_text(text, errors="ignore")
    return text


def _paper_index_text(pid: str, m: dict) -> str:
    d = _lib_dir(pid)
    notes = (d / "notes.md").read_text() if (d / "notes.md").exists() else ""
    text = (d / "text.txt").read_text(errors="ignore")[:20000] if (d / "text.txt").exists() else ""
    return "\n".join([m.get("authors", ""), m.get("takeaway", ""), notes, text])


ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def arxiv_meta(aid: str) -> dict:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    req = urllib.request.Request(f"https://export.arxiv.org/api/query?id_list={aid}", headers={"User-Agent": "cortex/0.1"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=60).read())
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return {}
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    return {
        "title": re.sub(r"\s+", " ", e.find("a:title", ns).text).strip(),
        "authors": ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
        "year": int(e.find("a:published", ns).text[:4]),
        "abstract": re.sub(r"\s+", " ", e.find("a:summary", ns).text).strip(),
        "link": f"https://arxiv.org/abs/{aid}",
    }


def ingest_pdf(src: Path, meta: dict | None = None, pid: str | None = None, move: bool = False) -> dict:
    """Copy a PDF into the library, fill metadata (arXiv when possible), extract text, index."""
    meta = dict(meta or {})
    head = ""
    try:
        head = subprocess.run(["pdftotext", "-l", "1", "-q", str(src), "-"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        pass
    aid = meta.get("arxiv") or next((m.group(1) for m in [re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", head or "", re.I)] if m), None) \
        or next((m.group(1) for m in [ARXIV_RE.search(src.name)] if m), None)
    if aid and not meta.get("title"):
        try:
            meta.update({k: v for k, v in arxiv_meta(aid).items() if v})
        except Exception:
            pass
    if not meta.get("title"):
        first = next((l.strip() for l in (head or "").splitlines() if 12 < len(l.strip()) < 160), None)
        meta["title"] = first or src.stem.replace("_", " ").replace("-", " ")
    # Display-caps titles come out of pdftotext as "R EPRESENTATION E NGINEERING": collapse them.
    if re.search(r"\b[A-Z] [A-Z]{2,}", meta["title"]):
        meta["title"] = re.sub(r"\b([A-Z]) (?=[A-Z]{2,})", r"\1", meta["title"]).title().replace("Llm", "LLM").replace("Ai ", "AI ")
    pid = pid or aid or slugify(meta["title"])
    d = _lib_dir(pid); d.mkdir(parents=True, exist_ok=True)
    dst = d / "paper.pdf"
    if not dst.exists():
        (shutil.move if move else shutil.copy2)(str(src), str(dst))
    m = read_meta(pid) or {}
    m.update({
        "id": pid, "title": meta["title"], "authors": meta.get("authors", ""), "year": meta.get("year"),
        "arxiv": aid or "", "link": meta.get("link", ""), "type": meta.get("type", "paper" if aid else "doc"),
        "status": meta.get("status", "inbox"), "rating": meta.get("rating"), "takeaway": meta.get("takeaway", ""),
        "topics": meta.get("topics", []), "added": m.get("added") or now_iso(), "source_path": str(src),
    })
    if meta.get("abstract") and not m.get("abstract"):
        m["abstract"] = meta["abstract"][:1500]
    write_meta(pid, m)
    if not (d / "text.txt").exists():
        extract_text(dst, d / "text.txt")
    index_upsert("paper", pid, m["title"], _paper_index_text(pid, m))
    return m


def ingest_arxiv(aid: str, **meta) -> dict:
    aid = ARXIV_RE.search(aid).group(1)
    tmp = VAULT / ".cortex" / f"{aid}.pdf"
    req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "cortex/0.1"})
    tmp.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    m = ingest_pdf(tmp, {"arxiv": aid, **meta}, pid=aid, move=True)
    m["source_path"] = f"https://arxiv.org/pdf/{aid}"
    write_meta(aid, m)
    return m


# ---------------------------------------------------------------- topics

def list_topics() -> list[dict]:
    p = VAULT / "topics.json"
    return json.loads(p.read_text()) if p.exists() else []


# ---------------------------------------------------------------- chats

def chat_path(channel: str) -> Path:
    return VAULT / "chats" / f"{slugify(channel)}.jsonl"


def read_chat(channel: str, limit: int = 500) -> list[dict]:
    p = chat_path(channel)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()[-limit:]
    return [json.loads(l) for l in lines if l.strip()]


def append_chat(channel: str, msg: dict) -> None:
    with chat_path(channel).open("a") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def clear_chat(channel: str) -> None:
    p = chat_path(channel)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------- search index (FTS5)

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(VAULT / ".cortex" / "index.sqlite")
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(type, id UNINDEXED, title, body, tokenize='porter unicode61')")
    return con


def index_upsert(dtype: str, did: str, title: str, body: str) -> None:
    con = _db()
    con.execute("DELETE FROM docs WHERE type=? AND id=?", (dtype, did))
    con.execute("INSERT INTO docs(type,id,title,body) VALUES (?,?,?,?)", (dtype, did, title or "", (body or "")[:200000]))
    con.commit(); con.close()


def index_delete(dtype: str, did: str) -> None:
    con = _db(); con.execute("DELETE FROM docs WHERE type=? AND id=?", (dtype, did)); con.commit(); con.close()


def rebuild_index() -> int:
    con = _db(); con.execute("DELETE FROM docs"); con.commit(); con.close()
    n = 0
    for p in list((VAULT / "notes").glob("*.md")) + list((VAULT / "daily").glob("*.md")):
        fm, body = read_md(p); index_upsert("note", p.stem, fm.get("title", p.stem), body); n += 1
    for p in (VAULT / "projects").glob("*.md"):
        fm, body = read_md(p)
        index_upsert("project", p.stem, fm.get("title", p.stem), f"{fm.get('verdict','')}\n{fm.get('next_action','')}\n{body}"); n += 1
    for d in (VAULT / "library").iterdir():
        m = read_meta(d.name) if d.is_dir() else None
        if m:
            index_upsert("paper", d.name, m.get("title", d.name), _paper_index_text(d.name, m)); n += 1
    return n


def search(q: str, limit: int = 20, types: list[str] | None = None) -> list[dict]:
    q = q.strip()
    if not q:
        return []
    # tolerant query: quote each term so punctuation in titles does not break FTS syntax
    terms = [t for t in re.split(r"\s+", q) if t]
    match = " ".join(f'"{t.replace(chr(34), "")}"' for t in terms)
    con = _db()
    sql = "SELECT type, id, title, snippet(docs, 3, '[', ']', '…', 18) AS snip, bm25(docs) AS score FROM docs WHERE docs MATCH ?"
    args: list[Any] = [match]
    if types:
        sql += " AND type IN (%s)" % ",".join("?" * len(types)); args += types
    sql += " ORDER BY score LIMIT ?"; args.append(limit)
    try:
        rows = con.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [{"type": r[0], "id": r[1], "title": r[2], "snippet": r[3], "score": round(-r[4], 3)} for r in rows]


def counts() -> dict:
    return {
        "notes": len(list((VAULT / "notes").glob("*.md"))),
        "daily": len(list((VAULT / "daily").glob("*.md"))),
        "projects": len(list((VAULT / "projects").glob("*.md"))),
        "papers": sum(1 for d in (VAULT / "library").iterdir() if d.is_dir() and (d / "meta.json").exists()),
    }
