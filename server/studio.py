"""The Studio: a shot list, takes rendered on the GPU box, critic scores, and a director's verdict.

Storage (inside the vault):
  studio/shots.json                the shot list: id, title, prompt, keyframe, status, takes[]
  studio/takes/<run_id>/           what came back from a render run: results.json, contact.png, clip.mp4 (when fetched)
  studio/assets/                   keyframes and reference images the person drops in

A "take" is just a run of the cinema_render recipe. Rendering therefore reuses the runs machinery: executor
(ssh = the 5090), streamed logs, METRIC lines, ROLLOUT-style scores, and telemetry to Grafana when configured.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from . import runs, vault

STATUSES = ["planned", "rendering", "rendered", "approved", "reshoot"]


def _dir() -> Path:
    d = vault.VAULT / "studio"
    (d / "takes").mkdir(parents=True, exist_ok=True)
    (d / "assets").mkdir(parents=True, exist_ok=True)
    return d


def _load() -> dict[str, Any]:
    p = _dir() / "shots.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"logline": "", "shots": []}


def _save(data: dict[str, Any]) -> None:
    (_dir() / "shots.json").write_text(json.dumps(data, indent=1))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "shot"


# ---------------------------------------------------------------- shots

def board() -> dict[str, Any]:
    data = _load()
    for s in data["shots"]:
        s["takes"] = [_take_summary(t) for t in s.get("takes", [])]
    counts = {st: sum(1 for s in data["shots"] if s.get("status") == st) for st in STATUSES}
    return {**data, "counts": counts, "assets": sorted(p.name for p in (_dir() / "assets").iterdir() if p.is_file())}


def set_logline(text: str) -> dict[str, Any]:
    data = _load()
    data["logline"] = text.strip()[:2000]
    _save(data)
    return board()


def add_shot(title: str, prompt: str, keyframe: str | None = None, frames: int = 49, size: str = "832x480", notes: str | None = None) -> dict[str, Any]:
    data = _load()
    n = len(data["shots"]) + 1
    sid = f"s{n:02d}-{_slug(title)}"
    while any(s["id"] == sid for s in data["shots"]):
        n += 1
        sid = f"s{n:02d}-{_slug(title)}"
    shot = {"id": sid, "title": title.strip()[:120], "prompt": prompt.strip()[:1500], "keyframe": keyframe, "frames": int(frames), "size": size,
            "notes": (notes or "")[:1000], "status": "planned", "takes": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    data["shots"].append(shot)
    _save(data)
    return shot


def update_shot(sid: str, patch: dict[str, Any]) -> dict[str, Any]:
    data = _load()
    for s in data["shots"]:
        if s["id"] == sid:
            for k in ("title", "prompt", "keyframe", "frames", "size", "notes", "status", "verdict", "director_note"):
                if k in patch and patch[k] is not None:
                    s[k] = patch[k]
            if s.get("status") not in STATUSES:
                s["status"] = "planned"
            _save(data)
            return s
    raise ValueError(f"no shot {sid}")


def remove_shot(sid: str) -> None:
    data = _load()
    before = len(data["shots"])
    data["shots"] = [s for s in data["shots"] if s["id"] != sid]
    if len(data["shots"]) == before:
        raise ValueError(f"no shot {sid}")
    _save(data)


def get_shot(sid: str) -> dict[str, Any] | None:
    return next((s for s in _load()["shots"] if s["id"] == sid), None)


# ---------------------------------------------------------------- takes (renders)

def render(sid: str, executor: str | None = None, smoke: bool | None = None, origin: str = "ui", force: bool = False) -> dict[str, Any]:
    """Start a take: a cinema_render run for the shot. The 5090 renders for real; anywhere else runs the smoke brick.
    A real render needs ~24 GB of GPU memory; when someone else's job holds the box, refuse unless force=True."""
    shot = get_shot(sid)
    if not shot:
        raise ValueError(f"no shot {sid}")
    ex = runs.executors()
    executor = executor or ("ssh" if ex["ssh"]["available"] else "local")
    if smoke is None:
        smoke = executor != "ssh" or not shot.get("keyframe")
    if executor == "ssh" and not smoke and not force:
        g = runs.gpu_status()
        held = int(g.get("foreign_load_mib") or 0)
        if not g.get("ready"):
            raise ValueError(f"the GPU box is not ready: {g.get('message')}")
        if held > 6000:
            raise ValueError(f"the GPU box is busy: another job holds {held} MiB and a Wan render needs about 24 GB; try when it is idle (or force the take)")
    args = [f"--shot {sid}", f"--prompt {json.dumps(shot['prompt'])}", f"--frames {shot.get('frames', 49)}", f"--size {shot.get('size', '832x480')}"]
    if smoke:
        args.append("--smoke")
    else:
        kf = shot["keyframe"]
        if not kf.startswith("~") and not kf.startswith("/"):  # an asset in the vault: it must be on the box; sync it
            src = _dir() / "assets" / kf
            if not src.exists():
                raise ValueError(f"keyframe {kf} is not in studio/assets")
            _push_asset(src)
            kf = f"~/cortex-lab/assets/{src.name}"
        args.append(f"--keyframe {kf}")
        proto = os.environ.get("CINEMA_PROTO")
        if proto:
            args.append(f"--proto {proto}")
    args.append(f"--out out/cinema/{sid}/{time.strftime('%H%M%S')}")
    m = runs.start("cinema_render", " ".join(args), executor, origin=origin)
    data = _load()
    for s in data["shots"]:
        if s["id"] == sid:
            s.setdefault("takes", []).append(m["id"])
            s["status"] = "rendering"
    _save(data)
    return m


def _push_asset(src: Path) -> None:
    host = os.environ.get("CORTEX_SSH_HOST", "").strip()
    if not host:
        raise ValueError("no GPU box configured")
    subprocess.run(["ssh", *runs.SSH_OPTS, host, "mkdir -p ~/cortex-lab/assets"], check=True, timeout=60)
    scp_opts = [("-P" if o == "-p" else o) for o in runs.SSH_OPTS]
    subprocess.run(["scp", *scp_opts, str(src), f"{host}:~/cortex-lab/assets/{src.name}"], check=True, timeout=120)


def _take_summary(rid: str) -> dict[str, Any]:
    r = runs.read_run(rid, tail=0, max_metrics=0)
    if not r:
        return {"id": rid, "status": "missing"}
    res = r.get("result") or {}
    local = _dir() / "takes" / rid
    return {"id": rid, "status": r["status"], "started": r.get("started"), "ended": r.get("ended"), "executor": r.get("executor"), "origin": r.get("origin"),
            "verdict": res.get("verdict"), "identity_mean": res.get("identity_mean"), "identity_min": res.get("identity_min"), "flicker_mean": res.get("flicker_mean"),
            "gen_s": res.get("gen_s"), "model": res.get("model"), "contact": f"/api/studio/takes/{rid}/contact.png" if (local / "contact.png").exists() else None,
            "clip": f"/api/studio/takes/{rid}/clip.mp4" if (local / "clip.mp4").exists() else None, "fetched": local.exists()}


def refresh(sid: str) -> dict[str, Any]:
    """After takes finish: pull artifacts back from the box, set the shot's status from the latest verdict."""
    shot = get_shot(sid)
    if not shot:
        raise ValueError(f"no shot {sid}")
    latest = None
    for rid in shot.get("takes", []):
        r = runs.read_run(rid, tail=0, max_metrics=0)
        if not r:
            continue
        if r["status"] == "done":
            fetch_take(rid, r)
            latest = r
    states = [(runs.read_run(rid, 0, 0) or {}).get("status") for rid in shot.get("takes", [])]
    if any(st in ("queued", "running") for st in states):
        update_shot(sid, {"status": "rendering"})
    elif latest:
        res = latest.get("result") or {}
        update_shot(sid, {"status": "rendered" if res.get("verdict") == "keep" else "reshoot"})
    elif states and states[-1] == "failed":
        update_shot(sid, {"status": "reshoot", "director_note": "the last take failed; open it in GPU runs for the log"})
    return next(s for s in board()["shots"] if s["id"] == sid)


def fetch_take(rid: str, r: dict[str, Any] | None = None) -> Path | None:
    """Copy contact.png, results.json and clip.mp4 for a finished take into the vault (from the box when remote)."""
    r = r or runs.read_run(rid, tail=0, max_metrics=0)
    if not r or r["status"] != "done":
        return None
    res = r.get("result") or {}
    out = res.get("out")
    if not out:
        return None
    local = _dir() / "takes" / rid
    if (local / "contact.png").exists() and ((local / "clip.mp4").exists() or not res.get("clip")):
        return local
    local.mkdir(parents=True, exist_ok=True)
    names = ["contact.png", "results.json"] + (["clip.mp4"] if res.get("clip") else [])
    if r.get("executor") == "ssh":
        host = os.environ.get("CORTEX_SSH_HOST", "").strip()
        scp_opts = [("-P" if o == "-p" else o) for o in runs.SSH_OPTS]
        for n in names:
            try:
                subprocess.run(["scp", *scp_opts, f"{host}:~/cortex-lab/{out}/{n}", str(local / n)], check=True, timeout=300, capture_output=True)
            except Exception:
                pass
    else:
        base = Path(out) if Path(out).is_absolute() else runs.ROOT / out
        for n in names:
            if (base / n).exists():
                shutil.copyfile(base / n, local / n)
    return local


def take_file(rid: str, name: str) -> Path | None:
    if name not in ("contact.png", "clip.mp4", "results.json"):
        return None
    p = _dir() / "takes" / rid / name
    return p if p.exists() else None


def add_asset(name: str, data: bytes) -> str:
    safe = re.sub(r"[^\w.\-]+", "-", name)[:100] or "asset.png"
    p = _dir() / "assets" / safe
    n = 1
    while p.exists():
        n += 1
        p = _dir() / "assets" / f"{Path(safe).stem}-{n}{Path(safe).suffix}"
    p.write_bytes(data)
    return p.name


# ---------------------------------------------------------------- planning (a shot list from a logline, by the chat model)

def plan(logline: str, n: int = 4, model: str | None = None) -> list[dict[str, Any]]:
    """Ask the model for n shots; each becomes a planned shot. Keyframes are chosen later by the person or the director."""
    from . import chat
    prompt = (
        f"You are a film director planning {n} shots for a short animated piece. Logline: {logline}\n"
        "Reply with only a JSON array of objects {\"title\": short, \"prompt\": a concrete image-to-video prompt (subject, action, camera, light, mood; 25 to 60 words), \"notes\": one line on why this shot}. "
        "Keep continuity of character and setting across shots."
    )
    resp = chat.client().chat.completions.create(model=model or chat.DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.4, extra_headers=chat.session_headers("studio:plan"))
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    items = json.loads(m.group(0) if m else raw)
    out = []
    set_logline(logline)
    for it in items[:n] if isinstance(items, list) else []:
        out.append(add_shot(str(it.get("title", "Shot")), str(it.get("prompt", "")), notes=str(it.get("notes", ""))))
    return out
