"""Traces: a long-horizon data collector for one person's work, taste, and decisions.

Anything can be sent here: a note, a preference ("I liked A over B because…"), a prompt/response pair, a rating,
a file, a screenshot, a decision, a chat turn. Every record is one JSON line under traces/YYYY-MM/traces.jsonl in
the vault, with attachments under traces/YYYY-MM/files/. Nothing is deleted or rewritten by the app.

Record shape (all optional except kind, ts, id):
  {"id", "ts", "kind": note|preference|pair|rating|decision|file|chat|taste|feedback|...,
   "content": free text, "data": any JSON, "tags": [...], "source": "chat|webmcp|ui|cli|api", "context": where it happened,
   "prompt", "response", "chosen", "rejected", "rating"}

Export shapes (for post-training later):
  sft   -> [{"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}]  from pair/chat records
  dpo   -> [{"prompt", "chosen", "rejected"}]                                                                 from preference records
  all   -> every record as stored
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from . import vault

KINDS = ["note", "preference", "pair", "rating", "decision", "file", "chat", "taste", "feedback", "idea", "link", "screenshot"]


def _root() -> Path:
    d = vault.VAULT / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _month_dir(ts: float | None = None) -> Path:
    d = _root() / time.strftime("%Y-%m", time.localtime(ts))
    (d / "files").mkdir(parents=True, exist_ok=True)
    return d


def add(rec: dict[str, Any]) -> dict[str, Any]:
    ts = time.time()
    clean: dict[str, Any] = {"id": uuid.uuid4().hex[:12], "ts": ts, "when": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)), "kind": str(rec.get("kind") or "note")[:30]}
    for k in ("content", "source", "context", "prompt", "response", "chosen", "rejected"):
        v = rec.get(k)
        if v is not None and str(v).strip():
            clean[k] = str(v)[:20000]
    if rec.get("data") is not None:
        clean["data"] = rec["data"]
    if rec.get("tags"):
        clean["tags"] = [str(t)[:40] for t in rec["tags"]][:20]
    if rec.get("rating") is not None:
        clean["rating"] = float(rec["rating"])
    if rec.get("file"):
        clean["file"] = rec["file"]
    if len(clean) <= 4:
        raise ValueError("an empty trace is not worth keeping: give content, data, a pair, or a file")
    with (_month_dir(ts) / "traces.jsonl").open("a") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return clean


def add_file(name: str, data: bytes, kind: str = "file", tags: list[str] | None = None, context: str = "") -> dict[str, Any]:
    safe = re.sub(r"[^\w.\-]+", "-", name)[:120] or "file"
    d = _month_dir()
    p = d / "files" / f"{int(time.time())}-{safe}"
    p.write_bytes(data)
    return add({"kind": kind, "file": str(p.relative_to(_root())), "content": name, "tags": tags or [], "context": context, "source": "upload", "data": {"bytes": len(data)}})


def _iter(limit: int | None = None):
    files = sorted(_root().glob("*/traces.jsonl"), reverse=True)
    n = 0
    for fp in files:
        lines = fp.read_text(errors="ignore").splitlines()
        for line in reversed(lines):
            try:
                yield json.loads(line)
            except Exception:
                continue
            n += 1
            if limit and n >= limit:
                return


def list_traces(limit: int = 100, kind: str | None = None, q: str | None = None) -> list[dict[str, Any]]:
    out = []
    ql = (q or "").lower()
    for r in _iter():
        if kind and r.get("kind") != kind:
            continue
        if ql and ql not in json.dumps(r, ensure_ascii=False).lower():
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def stats() -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_month: dict[str, int] = {}
    total = 0
    first = None
    for fp in sorted(_root().glob("*/traces.jsonl")):
        month = fp.parent.name
        for line in fp.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            total += 1
            by_kind[r.get("kind", "?")] = by_kind.get(r.get("kind", "?"), 0) + 1
            by_month[month] = by_month.get(month, 0) + 1
            if first is None or r.get("ts", 0) < first:
                first = r.get("ts")
    pairs = by_kind.get("pair", 0) + by_kind.get("chat", 0)
    prefs = by_kind.get("preference", 0)
    return {"total": total, "by_kind": by_kind, "by_month": by_month, "since": time.strftime("%Y-%m-%d", time.localtime(first)) if first else None,
            "sft_pairs": pairs, "dpo_pairs": prefs, "root": str(_root())}


def export(fmt: str = "sft") -> dict[str, Any]:
    rows = []
    for r in _iter():
        if fmt == "sft" and r.get("prompt") and r.get("response"):
            rows.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}], "tags": r.get("tags", []), "id": r["id"]})
        elif fmt == "dpo" and r.get("chosen") and r.get("rejected"):
            rows.append({"prompt": r.get("prompt") or r.get("content") or "", "chosen": r["chosen"], "rejected": r["rejected"], "id": r["id"]})
        elif fmt == "all":
            rows.append(r)
    return {"format": fmt, "n": len(rows), "rows": rows}
