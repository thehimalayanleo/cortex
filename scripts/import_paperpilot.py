"""Import everything PaperPilot lists (its reading catalog plus the GPU-perf and LLM-RL books) into the vault.

arXiv items get the PDF downloaded; direct .pdf links are downloaded too; web posts become link-only
library entries (no file). Idempotent: items already in the vault (same arXiv id or link) are skipped.

Usage: python scripts/import_paperpilot.py [--dry-run] [--src DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import vault  # noqa: E402

SRC = Path("~/Documents/New project/PaperPilot/App/Sources/PaperPilot").expanduser()
FILES = ["ReadingCatalog.swift", "GPUPerfCatalog.swift", "LLMRLCatalog.swift"]
TOPIC_BY_BOOK = {"gpuperf": ["gpu"], "llmrl": ["post-training"], "apollo": ["safety"], "arxiv": [], "": []}
UA = {"User-Agent": "cortex/0.1 (+https://github.com/thehimalayanleo/cortex)"}


def parse(src: Path) -> list[dict]:
    items: list[dict] = []
    for name in FILES:
        text = (src / name).read_text()
        # Named-argument style: id: "...", title: "...", source: "...", kind: .paper, url: "..."
        for m in re.finditer(r'id:\s*"([^"]+)",\s*title:\s*"((?:[^"\\]|\\.)+)",\s*source:\s*"((?:[^"\\]|\\.)*)",\s*kind:\s*\.(\w+),\s*url:\s*"([^"]+)"', text, re.S):
            items.append(dict(id=m.group(1), title=m.group(2).replace('\\"', '"'), source=m.group(3), kind=m.group(4), url=m.group(5)))
        # Positional helper style used by the books: item("id", "title", "source", .kind, "url") or similar order
        for m in re.finditer(r'\(\s*"([\w\-.]+)",\s*"((?:[^"\\]|\\.)+)",\s*"((?:[^"\\]|\\.)*)",\s*\.(\w+),\s*"(https?://[^"]+)"(?:,\s*pdf:\s*"(https?://[^"]+)")?', text, re.S):
            items.append(dict(id=m.group(1), title=m.group(2).replace('\\"', '"'), source=m.group(3), kind=m.group(4), url=m.group(5), pdf=m.group(6) or "", book=name.replace("Catalog.swift", "").lower()))
    seen: set[str] = set()
    out = []
    for it in items:
        key = it["url"].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    return m.group(1) if m else None


def book_of(it: dict) -> str:
    if it.get("book"):
        return it["book"]
    i = it["id"]
    for k in ("gpuperf", "llmrl", "apollo"):
        if i.startswith(k):
            return k
    return "arxiv" if i.startswith("arxiv") else ""


def link_only(it: dict, topics: list[str]) -> dict:
    """A library row without a file: the link is the copy. Uses the same meta.json shape as PDFs."""
    slug = re.sub(r"[^a-z0-9]+", "-", it["title"].lower()).strip("-")[:60] or it["id"]
    d = vault.VAULT / "library" / slug
    if d.exists():
        return json.loads((d / "meta.json").read_text())
    d.mkdir(parents=True)
    kind = it.get("kind", "post")
    meta = {"id": slug, "title": it["title"], "authors": it.get("source", ""), "year": None, "arxiv": "", "link": it["url"],
            "type": "paper" if kind == "paper" else "doc", "status": "reference", "rating": None, "takeaway": "", "topics": topics,
            "added": time.strftime("%Y-%m-%dT%H:%M:%S"), "pages": None, "source_path": "paperpilot"}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=str(SRC))
    a = ap.parse_args()
    items = parse(Path(a.src))
    existing = vault.list_library()
    have_arxiv = {m.get("arxiv") for m in existing if m.get("arxiv")}
    have_link = {(m.get("link") or "").rstrip("/") for m in existing}
    plan = []
    for it in items:
        aid = arxiv_id(it["url"])
        if aid and aid in have_arxiv:
            continue
        if it["url"].rstrip("/") in have_link:
            continue
        plan.append((it, aid))
    is_pdf = lambda it: it["url"].lower().endswith(".pdf") or bool(it.get("pdf"))  # noqa: E731
    kinds = {"arxiv": sum(1 for _, x in plan if x), "pdf": sum(1 for it, x in plan if not x and is_pdf(it)),
             "link": sum(1 for it, x in plan if not x and not is_pdf(it))}
    print(f"catalog {len(items)} items; already in vault {len(items) - len(plan)}; to import {len(plan)} = {kinds}", flush=True)
    if a.dry_run:
        for it, aid in plan[:200]:
            print(f"  {book_of(it):8s} {aid or ('pdf' if is_pdf(it) else 'link'):12s} {it['title'][:70]}")
        return
    ok = fail = 0
    for it, aid in plan:
        topics = TOPIC_BY_BOOK.get(book_of(it), [])
        try:
            if aid:
                vault.ingest_arxiv(aid, title=it["title"], status="inbox", topics=topics)
                time.sleep(1.5)  # be polite to arXiv
            elif it["url"].lower().endswith(".pdf") or it.get("pdf"):
                tmp = vault.VAULT / ".cortex" / "pp-download.pdf"
                tmp.parent.mkdir(exist_ok=True)
                tmp.write_bytes(urllib.request.urlopen(urllib.request.Request(it.get("pdf") or it["url"], headers=UA), timeout=120).read())
                vault.ingest_pdf(tmp, {"title": it["title"], "link": it["url"], "authors": it.get("source", ""), "status": "inbox", "topics": topics}, move=True)
            else:
                link_only(it, topics)
            ok += 1
            print(f"ok   {it['title'][:70]}", flush=True)
        except Exception as e:
            fail += 1
            print(f"FAIL {it['title'][:60]}: {e}", flush=True)
    vault.rebuild_index()
    print(f"done: {ok} imported, {fail} failed; vault now {vault.counts()}")


if __name__ == "__main__":
    main()
