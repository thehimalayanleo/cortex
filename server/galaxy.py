"""The paper galaxy: every paper embedded, clustered into solar systems, grouped into universes.

The index itself is produced by lab/recipes/galaxy_index.py (bge-small on the CPU + DBSCAN + agglomerative), run as an
ordinary local run so the server's own environment stays light. This module reads the result, serves it, rebuilds on
demand, and rebuilds by itself when the library changes (checked every few minutes).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import runs, vault

_state = {"last_count": -1, "last_build": 0.0, "run": None}
_lock = threading.Lock()


def _path() -> Path:
    return vault.VAULT / ".cortex" / "galaxy.json"


def read() -> dict[str, Any]:
    p = _path()
    if not p.exists():
        return {"generated": None, "n": 0, "papers": [], "clusters": [], "universes": [], "building": _building()}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {"generated": None, "n": 0, "papers": [], "clusters": [], "universes": [], "building": _building(), "error": "index unreadable"}
    d["building"] = _building()
    d["stale"] = vault.counts()["papers"] != d.get("n")
    return d


def _building() -> str | None:
    rid = _state.get("run")
    if not rid:
        return None
    r = runs.read_run(rid, tail=0, max_metrics=0)
    if r and r["status"] in ("queued", "running"):
        return rid
    return None


def rebuild(smoke: bool = False, origin: str = "ui") -> dict[str, Any]:
    """Start a galaxy_index run (local executor; needs uv). Returns the run."""
    with _lock:
        if _building():
            return runs.read_run(_state["run"], tail=0, max_metrics=0) or {}
        args = f"--vault {vault.VAULT} --out {_path()}" + (" --smoke" if smoke else "")
        py = os.environ.get("CORTEX_LOCAL_PYTHON")
        if not py:  # the index needs sentence-transformers and scikit-learn; give the local run those
            os.environ["CORTEX_GALAXY_PYTHON"] = "uv run --python 3.11 --with numpy --with scikit-learn --with sentence-transformers python"
        m = runs.start("galaxy_index", args, "local", origin=origin)
        _state["run"] = m["id"]
        _state["last_build"] = time.time()
        _state["last_count"] = vault.counts()["papers"]
        return m


def _watch() -> None:
    """Rebuild when the paper count changes (after a quiet period), at most once every 10 minutes."""
    time.sleep(60)
    while True:
        try:
            n = vault.counts()["papers"]
            d = read()
            if n and (d.get("n") != n) and not _building() and time.time() - _state["last_build"] > 600:
                print(f"galaxy: library changed ({d.get('n')} -> {n} papers); rebuilding")
                rebuild(origin="auto")
        except Exception as e:
            print("galaxy watcher:", e)
        time.sleep(180)


def start_watcher() -> None:
    threading.Thread(target=_watch, daemon=True, name="galaxy-watch").start()


def summary() -> dict[str, Any]:
    d = read()
    return {"generated": d.get("generated"), "n": d.get("n"), "model": d.get("model"), "stale": d.get("stale"), "building": d.get("building"),
            "universes": [{"id": u["id"], "label": u["label"], "size": u["size"]} for u in d.get("universes", [])],
            "solar_systems": [{"id": c["id"], "label": c["label"], "size": c["size"], "universe": c.get("universe")} for c in d.get("clusters", []) if c["id"] != -1]}


def system_papers(cluster_id: int) -> list[dict[str, Any]]:
    d = read()
    return [{"id": p["id"], "title": p["title"], "year": p.get("year"), "status": p.get("status")} for p in d.get("papers", []) if p.get("cluster") == cluster_id]
