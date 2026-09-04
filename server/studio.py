"""The Studio: a shot list, takes rendered on the GPU box, critic scores, and a director's verdict.

Storage (inside the vault):
  studio/shots.json                the shot list: id, title, prompt, keyframe, status, takes[]
  studio/takes/<run_id>/           what came back from a render run: results.json, contact.png, clip.mp4 (when fetched)
  studio/assets/                   keyframes and reference images the person drops in
  studio/characters.json           the character bible: id, name, description (IDENTITY), style, negative, hero, heroset_dir, lora_dir, status
  studio/characters/<id>/          what a build run brought back: hero.png, contact.png, results.json
  studio/scenes.json               scenes: id, title, set, characters[], shots[] (ordered shot ids), kind filler|full, duration_s, status
  studio/scenes/<id>/              the assembled scene.mp4 and its contact strip.png

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
    return {**data, "counts": counts, "assets": sorted(p.name for p in (_dir() / "assets").iterdir() if p.is_file()),
            "characters": list_characters(), "scenes": list_scenes()}


def set_logline(text: str) -> dict[str, Any]:
    data = _load()
    data["logline"] = text.strip()[:2000]
    _save(data)
    return board()


def add_shot(title: str, prompt: str, keyframe: str | None = None, frames: int = 49, size: str = "832x480", notes: str | None = None, character: str | None = None) -> dict[str, Any]:
    data = _load()
    n = len(data["shots"]) + 1
    sid = f"s{n:02d}-{_slug(title)}"
    while any(s["id"] == sid for s in data["shots"]):
        n += 1
        sid = f"s{n:02d}-{_slug(title)}"
    shot = {"id": sid, "title": title.strip()[:120], "prompt": prompt.strip()[:1500], "keyframe": keyframe, "frames": int(frames), "size": size,
            "notes": (notes or "")[:1000], "character": character or None, "status": "planned", "takes": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    data["shots"].append(shot)
    _save(data)
    return shot


def update_shot(sid: str, patch: dict[str, Any]) -> dict[str, Any]:
    data = _load()
    for s in data["shots"]:
        if s["id"] == sid:
            for k in ("title", "prompt", "keyframe", "frames", "size", "notes", "status", "verdict", "director_note", "character"):
                if k in patch and patch[k] is not None:
                    s[k] = patch[k]
            if patch.get("character") == "":
                s["character"] = None
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
    ch = get_character(shot["character"]) if shot.get("character") else None
    if smoke is None:
        smoke = executor != "ssh" or not (shot.get("keyframe") or (ch and ch.get("hero")))
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
        # the keyframe wins; without one, a character's hero image is the key and its hero set the identity prototype
        kf = shot.get("keyframe") or (ch or {}).get("hero")
        if not kf:
            raise ValueError("no keyframe and no character with a hero image")
        if not kf.startswith("~") and not kf.startswith("/"):  # an asset in the vault: it must be on the box; sync it
            src = _dir() / "assets" / kf
            if not src.exists():
                raise ValueError(f"keyframe {kf} is not in studio/assets")
            _push_asset(src)
            kf = f"~/cortex-lab/assets/{src.name}"
        args.append(f"--keyframe {kf}")
        proto = (ch or {}).get("heroset_dir") or os.environ.get("CINEMA_PROTO")
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


# ---------------------------------------------------------------- characters (the bible; hero, hero set, LoRA built on the box)

CHARACTER_STATUSES = ["draft", "building", "hero", "heroset", "lora", "failed"]
CHARACTER_STAGES = ["hero", "heroset", "lora"]


def _load_characters() -> list[dict[str, Any]]:
    p = _dir() / "characters.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("characters", [])
        except Exception:
            pass
    return []


def _save_characters(chars: list[dict[str, Any]]) -> None:
    (_dir() / "characters.json").write_text(json.dumps({"characters": chars}, indent=1))


def _character_summary(c: dict[str, Any]) -> dict[str, Any]:
    local = _dir() / "characters" / c["id"]
    hero = c.get("hero")
    hero_url = None
    if (local / "hero.png").exists():
        hero_url = f"/api/studio/characters/{c['id']}/hero.png"
    elif hero and not hero.startswith("~") and not hero.startswith("/"):
        hero_url = f"/api/studio/assets/{hero}"
    builds = [_build_summary(rid) for rid in c.get("builds", [])]
    return {**c, "hero_url": hero_url, "contact": f"/api/studio/characters/{c['id']}/contact.png" if (local / "contact.png").exists() else None, "builds": builds}


def _build_summary(rid: str) -> dict[str, Any]:
    r = runs.read_run(rid, tail=0, max_metrics=0)
    if not r:
        return {"id": rid, "status": "missing"}
    res = r.get("result") or {}
    m = re.search(r"--stage (\w+)", r.get("args") or "")
    stage = res.get("stage") or (m.group(1) if m else None)
    return {"id": rid, "status": r["status"], "stage": stage, "started": r.get("started"), "ended": r.get("ended"), "executor": r.get("executor"),
            "proto_mean": res.get("proto_mean"), "proto_min": res.get("proto_min"), "p_own": res.get("p_own"), "n_kept": res.get("n_kept"), "elapsed_s": res.get("elapsed_s")}


def list_characters() -> list[dict[str, Any]]:
    return [_character_summary(c) for c in _load_characters()]


def get_character(cid: str) -> dict[str, Any] | None:
    return next((c for c in _load_characters() if c["id"] == cid), None)


def add_character(name: str, description: str, style: str = "", negative: str = "", hero: str | None = None, hero_src: str | None = None,
                  heroset_dir: str | None = None, lora_dir: str | None = None, workdir: str | None = None) -> dict[str, Any]:
    chars = _load_characters()
    cid = _slug(name)
    n = 1
    while any(c["id"] == cid for c in chars):
        n += 1
        cid = f"{_slug(name)}-{n}"
    work = workdir or f"~/cortex-lab/out/characters/{cid}"
    c = {"id": cid, "name": name.strip()[:80], "description": description.strip()[:1500], "style": (style or "").strip()[:300], "negative": (negative or "").strip()[:600],
         "hero": hero or None, "hero_src": hero_src or None, "workdir": work, "heroset_dir": heroset_dir or None, "lora_dir": lora_dir or None,
         "status": "hero" if hero else "draft", "builds": [], "scores": {}, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    chars.append(c)
    _save_characters(chars)
    return _character_summary(c)


def update_character(cid: str, patch: dict[str, Any]) -> dict[str, Any]:
    chars = _load_characters()
    for c in chars:
        if c["id"] == cid:
            for k in ("name", "description", "style", "negative", "hero", "hero_src", "workdir", "heroset_dir", "lora_dir", "status"):
                if k in patch and patch[k] is not None:
                    c[k] = patch[k] or None if k in ("hero", "hero_src", "heroset_dir", "lora_dir") else patch[k]
            if c.get("status") not in CHARACTER_STATUSES:
                c["status"] = "draft"
            _save_characters(chars)
            return _character_summary(c)
    raise ValueError(f"no character {cid}")


def remove_character(cid: str) -> None:
    chars = _load_characters()
    if not any(c["id"] == cid for c in chars):
        raise ValueError(f"no character {cid}")
    _save_characters([c for c in chars if c["id"] != cid])
    shutil.rmtree(_dir() / "characters" / cid, ignore_errors=True)


def build_character(cid: str, stage: str = "heroset", executor: str | None = None, smoke: bool | None = None, origin: str = "ui", force: bool = False) -> dict[str, Any]:
    """Start a build run: cinema_character --stage hero|heroset|lora for the character, on the box when it is there."""
    c = get_character(cid)
    if not c:
        raise ValueError(f"no character {cid}")
    if stage not in CHARACTER_STAGES:
        raise ValueError(f"stage must be one of {CHARACTER_STAGES}")
    ex = runs.executors()
    executor = executor or ("ssh" if ex["ssh"]["available"] else "local")
    if smoke is None:
        smoke = executor != "ssh"
    if executor == "ssh" and not smoke and not force:
        g = runs.gpu_status()
        if not g.get("ready"):
            raise ValueError(f"the GPU box is not ready: {g.get('message')}")
        if int(g.get("foreign_load_mib") or 0) > 6000:
            raise ValueError("the GPU box is busy: another job holds it; try when it is idle (or force the build)")
    # on the box the character lives in its workdir (~/celwright_v3b for the original Okuun); a local (smoke) build stays in this repo's out/
    work = (c.get("workdir") or f"~/cortex-lab/out/characters/{cid}") if executor == "ssh" else f"out/characters/{cid}"
    args = [f"--character {cid}", f"--stage {stage}", f"--work {work}", f"--identity {json.dumps(c.get('description') or c['name'])}"]
    if c.get("style"):
        args.append(f"--style {json.dumps(c['style'])}")
    if c.get("negative"):
        args.append(f"--negative {json.dumps(c['negative'])}")
    if c.get("hero_src"):
        args.append(f"--hero-src {c['hero_src']}")
    if smoke:
        args.append("--smoke")
    args.append(f"--out out/characters/{cid}/{stage}-{time.strftime('%H%M%S')}")
    m = runs.start("cinema_character", " ".join(args), executor, origin=origin)
    chars = _load_characters()
    for cc in chars:
        if cc["id"] == cid:
            cc.setdefault("builds", []).append(m["id"])
            cc["status"] = "building"
    _save_characters(chars)
    return m


def refresh_character(cid: str) -> dict[str, Any]:
    """After builds finish: pull hero.png, contact.png and results back, and set the character's paths and status from the latest one."""
    c = get_character(cid)
    if not c:
        raise ValueError(f"no character {cid}")
    latest, latest_res = None, None
    for rid in c.get("builds", []):
        r = runs.read_run(rid, tail=0, max_metrics=0)
        if r and r["status"] == "done" and (r.get("result") or {}).get("out"):
            _fetch_build(cid, rid, r)
            latest, latest_res = r, r.get("result") or {}
    states = [(runs.read_run(rid, 0, 0) or {}).get("status") for rid in c.get("builds", [])]
    patch: dict[str, Any] = {}
    if any(st in ("queued", "running") for st in states):
        patch["status"] = "building"
    elif latest_res:
        stage = latest_res.get("stage") or "hero"
        patch["status"] = stage
        # paths from a build on the box are the character's paths; a local smoke build only fills blanks, never clobbers box paths
        on_box = latest.get("executor") == "ssh"
        for key, field in (("hero", "hero"), ("heroset", "heroset_dir"), ("lora", "lora_dir")):
            if latest_res.get(key) and (on_box or not c.get(field)):
                patch[field] = latest_res[key]
    elif states and states[-1] == "failed":
        patch["status"] = "failed"
    chars = _load_characters()
    for cc in chars:
        if cc["id"] == cid:
            cc.update({k: v for k, v in patch.items()})
            if latest_res:
                cc["scores"] = {k: latest_res.get(k) for k in ("proto_mean", "proto_min", "p_own", "n_kept", "elapsed_s") if latest_res.get(k) is not None}
    _save_characters(chars)
    return _character_summary(next(cc for cc in chars if cc["id"] == cid))


def _fetch_build(cid: str, rid: str, r: dict[str, Any]) -> None:
    res = r.get("result") or {}
    out = res.get("out")
    local = _dir() / "characters" / cid
    stamp = local / ".fetched"
    if stamp.exists() and stamp.read_text().strip() == rid:
        return
    local.mkdir(parents=True, exist_ok=True)
    names = ["hero.png", "contact.png", "results.json"]
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
    stamp.write_text(rid)


def character_file(cid: str, name: str) -> Path | None:
    if name not in ("hero.png", "contact.png", "results.json"):
        return None
    p = _dir() / "characters" / cid / name
    return p if p.exists() else None


# ---------------------------------------------------------------- scenes (ordered shots; filler b-roll or a full scene with dialogue)

SCENE_KINDS = ["filler", "full"]
SCENE_STATUSES = ["planned", "rendering", "rendered", "assembled"]
FPS = 24  # Wan 2.2 TI2V-5B clips
_scene_threads: dict[str, Any] = {}


def _load_scenes() -> list[dict[str, Any]]:
    p = _dir() / "scenes.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("scenes", [])
        except Exception:
            pass
    return []


def _save_scenes(scenes: list[dict[str, Any]]) -> None:
    (_dir() / "scenes.json").write_text(json.dumps({"scenes": scenes}, indent=1))


def _scene_summary(sc: dict[str, Any], shots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    shots = shots if shots is not None else _load()["shots"]
    by_id = {s["id"]: s for s in shots}
    rows = []
    frames = 0
    for sid in sc.get("shots", []):
        s = by_id.get(sid)
        if not s:
            rows.append({"id": sid, "title": "(missing)", "status": "missing", "verdict": None})
            continue
        frames += int(s.get("frames") or 0)
        last = _take_summary(s["takes"][-1]) if s.get("takes") else None
        rows.append({"id": sid, "title": s["title"], "status": s.get("status"), "verdict": (last or {}).get("verdict"), "take": (last or {}).get("id"), "frames": s.get("frames")})
    local = _dir() / "scenes" / sc["id"]
    return {**sc, "duration_s": round(frames / FPS, 2), "shot_rows": rows,
            "video": f"/api/studio/scenes/{sc['id']}/scene.mp4" if (local / "scene.mp4").exists() else None,
            "strip": f"/api/studio/scenes/{sc['id']}/strip.png" if (local / "strip.png").exists() else None}


def list_scenes() -> list[dict[str, Any]]:
    shots = _load()["shots"]
    return [_scene_summary(sc, shots) for sc in _load_scenes()]


def get_scene(sid: str) -> dict[str, Any] | None:
    return next((sc for sc in _load_scenes() if sc["id"] == sid), None)


def add_scene(title: str, kind: str = "filler", set_name: str = "", splat: str | None = None, characters: list[str] | None = None, shots: list[str] | None = None,
              dialogue: list[dict[str, str]] | None = None, continuity: str = "", logline: str = "") -> dict[str, Any]:
    scenes = _load_scenes()
    n = len(scenes) + 1
    sid = f"sc{n:02d}-{_slug(title)}"
    while any(sc["id"] == sid for sc in scenes):
        n += 1
        sid = f"sc{n:02d}-{_slug(title)}"
    sc = {"id": sid, "title": title.strip()[:120], "kind": kind if kind in SCENE_KINDS else "filler", "set": {"name": (set_name or "").strip()[:120], "splat": splat or None},
          "characters": list(characters or []), "shots": list(shots or []), "dialogue": list(dialogue or []), "continuity": (continuity or "")[:2000], "logline": (logline or "")[:2000],
          "status": "planned", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    scenes.append(sc)
    _save_scenes(scenes)
    return _scene_summary(sc)


def update_scene(sid: str, patch: dict[str, Any]) -> dict[str, Any]:
    scenes = _load_scenes()
    for sc in scenes:
        if sc["id"] == sid:
            for k in ("title", "kind", "characters", "shots", "dialogue", "continuity", "status", "logline"):
                if k in patch and patch[k] is not None:
                    sc[k] = patch[k]
            if "set" in patch and isinstance(patch["set"], dict):
                sc["set"] = {"name": str(patch["set"].get("name") or sc["set"].get("name") or ""), "splat": patch["set"].get("splat", sc["set"].get("splat"))}
            if sc.get("kind") not in SCENE_KINDS:
                sc["kind"] = "filler"
            if sc.get("status") not in SCENE_STATUSES:
                sc["status"] = "planned"
            _save_scenes(scenes)
            return _scene_summary(sc)
    raise ValueError(f"no scene {sid}")


def remove_scene(sid: str, with_shots: bool = False) -> None:
    scenes = _load_scenes()
    sc = next((x for x in scenes if x["id"] == sid), None)
    if not sc:
        raise ValueError(f"no scene {sid}")
    if with_shots:
        for shot_id in sc.get("shots", []):
            try:
                remove_shot(shot_id)
            except ValueError:
                pass
    _save_scenes([x for x in scenes if x["id"] != sid])
    shutil.rmtree(_dir() / "scenes" / sid, ignore_errors=True)


def plan_scene(logline: str, kind: str = "filler", n: int | None = None, characters: list[str] | None = None, set_name: str = "", model: str | None = None) -> dict[str, Any]:
    """Ask the chat model for a scene: filler = 2 to 4 short b-roll shots (17 to 33 frames, no dialogue);
    full = n shots with dialogue lines and continuity notes. Every shot is added to the board and attached to the scene."""
    from . import chat
    kind = kind if kind in SCENE_KINDS else "filler"
    n = int(n or (3 if kind == "filler" else 5))
    n = max(2, min(4, n)) if kind == "filler" else max(2, min(12, n))
    chars = [c for c in _load_characters() if not characters or c["id"] in characters]
    bible = "\n".join(f"- {c['id']}: {c['name']}: {c.get('description') or ''}" + (f" (style: {c['style']})" if c.get("style") else "") for c in chars)
    if kind == "filler":
        ask = (f"You are a film director planning a FILLER scene: {n} short b-roll shots (each 1 to 1.5 seconds; 17 to 33 frames) meant as cutaways for a post-production edit. Logline: {logline}\n"
               + (f"Characters (use their ids in the character field when they appear; empty for empty frames):\n{bible}\n" if bible else "")
               + "Reply with only a JSON object {\"title\": short scene title, \"set\": one line naming the location, \"continuity\": one line of what must hold across the shots, "
               "\"shots\": [{\"title\": short, \"prompt\": a concrete image-to-video prompt (subject, action, camera, light, mood; 20 to 45 words), \"frames\": 17|25|33, \"character\": id or \"\", \"notes\": one line}]}. "
               "Filler shots are quiet: inserts, textures, weather, empty set, a detail of the character; nothing that advances the story.")
    else:
        ask = (f"You are a film director planning a FULL scene of {n} shots for a short animated piece. Logline: {logline}\n"
               + (f"Characters (use their ids in the character field):\n{bible}\n" if bible else "")
               + "Reply with only a JSON object {\"title\": short scene title, \"set\": one line naming the location, \"continuity\": two or three lines of what must hold across the shots (who is where, carrying what, marked how, time of day), "
               "\"dialogue\": [{\"who\": character id or name, \"line\": the spoken line}], "
               "\"shots\": [{\"title\": short, \"prompt\": a concrete image-to-video prompt (subject, action, camera, light, mood; 25 to 60 words), \"frames\": 33|49|81, \"character\": id or \"\", \"dialogue\": the line spoken in this shot or \"\", \"notes\": one line on why this shot}]}. "
               "Keep continuity of character, setting and light across shots.")
    resp = chat.client().chat.completions.create(model=model or chat.DEFAULT_MODEL, messages=[{"role": "user", "content": ask}], temperature=0.4, extra_headers=chat.session_headers("studio:plan_scene"))
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    plan_obj = json.loads(m.group(0) if m else raw)
    return scene_from_plan(plan_obj, logline, kind, n, [c["id"] for c in chars], set_name)


def scene_from_plan(plan_obj: dict[str, Any], logline: str, kind: str, n: int, char_ids: list[str], set_name: str = "") -> dict[str, Any]:
    items = plan_obj.get("shots") if isinstance(plan_obj, dict) else plan_obj
    items = items[:n] if isinstance(items, list) else []
    shot_ids = []
    used: list[str] = []
    for it in items:
        frames = int(it.get("frames") or (25 if kind == "filler" else 49))
        frames = max(17, min(33, frames)) if kind == "filler" else max(17, min(81, frames))
        ch = str(it.get("character") or "") or None
        if ch and ch not in char_ids:
            ch = next((c for c in char_ids if c in ch or ch in c), None)
        notes = str(it.get("notes") or "")
        if it.get("dialogue"):
            notes = (f"Line: {it['dialogue']}. " + notes).strip()
        sh = add_shot(str(it.get("title", "Shot")), str(it.get("prompt", "")), frames=frames, notes=notes, character=ch)
        shot_ids.append(sh["id"])
        if ch and ch not in used:
            used.append(ch)
    title = str((plan_obj.get("title") if isinstance(plan_obj, dict) else None) or logline[:60] or "Scene")
    return add_scene(title, kind, set_name or str((plan_obj.get("set") if isinstance(plan_obj, dict) else "") or ""), characters=used, shots=shot_ids,
                     dialogue=[d for d in (plan_obj.get("dialogue") or []) if isinstance(d, dict)] if isinstance(plan_obj, dict) else [],
                     continuity=str((plan_obj.get("continuity") if isinstance(plan_obj, dict) else "") or ""), logline=logline)


def render_scene(sid: str, executor: str | None = None, smoke: bool | None = None, origin: str = "ui", only_missing: bool = False) -> dict[str, Any]:
    """Render every shot of the scene in order, one take at a time, waiting for each; runs in a thread and marks the scene rendering -> rendered."""
    import threading
    sc = get_scene(sid)
    if not sc:
        raise ValueError(f"no scene {sid}")
    t = _scene_threads.get(sid)
    if t and t.is_alive():
        raise ValueError(f"scene {sid} is already rendering")
    shot_ids = [x for x in sc.get("shots", []) if get_shot(x)]
    if not shot_ids:
        raise ValueError(f"scene {sid} has no shots")

    def work() -> None:
        try:
            for shot_id in shot_ids:
                sh = get_shot(shot_id)
                if not sh:
                    continue
                if only_missing and sh.get("status") in ("rendered", "approved"):
                    continue
                m = render(shot_id, executor, smoke, origin=origin)
                while True:
                    time.sleep(3)
                    r = runs.read_run(m["id"], tail=0, max_metrics=0)
                    if not r or r["status"] not in ("queued", "running"):
                        break
                try:
                    refresh(shot_id)
                except Exception:
                    pass
        finally:
            update_scene(sid, {"status": "rendered"})

    update_scene(sid, {"status": "rendering"})
    th = threading.Thread(target=work, daemon=True)
    _scene_threads[sid] = th
    th.start()
    return _scene_summary(get_scene(sid) or sc)


def _kept_clip(shot: dict[str, Any]) -> Path | None:
    """The latest kept take's clip (fetched into the vault), else the latest fetched clip of any finished take."""
    best = None
    for rid in reversed(shot.get("takes", [])):
        r = runs.read_run(rid, tail=0, max_metrics=0)
        if not r or r["status"] != "done":
            continue
        fetch_take(rid, r)
        clip = _dir() / "takes" / rid / "clip.mp4"
        if not clip.exists():
            continue
        if (r.get("result") or {}).get("verdict") == "keep":
            return clip
        best = best or clip
    return best


def assemble_scene(sid: str) -> dict[str, Any]:
    """Concatenate the latest kept take of each shot into studio/scenes/<id>/scene.mp4 (ffmpeg, here on the Mac) and write a contact strip."""
    sc = get_scene(sid)
    if not sc:
        raise ValueError(f"no scene {sid}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is not installed here")
    clips, missing = [], []
    for shot_id in sc.get("shots", []):
        sh = get_shot(shot_id)
        c = _kept_clip(sh) if sh else None
        if c:
            clips.append(c)
        else:
            missing.append(shot_id)
    if not clips:
        raise ValueError("no shot of this scene has a fetched clip yet (render, then refresh the shots)")
    local = _dir() / "scenes" / sid
    local.mkdir(parents=True, exist_ok=True)
    lst = local / "concat.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    out = local / "scene.mp4"
    # re-encode so takes of different sizes or codecs still join; scale to the first clip's width, even dimensions
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", str(out)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise ValueError("ffmpeg failed: " + (r.stderr or r.stdout).strip()[-600:])
    total = sum(int((get_shot(x) or {}).get("frames") or 0) for x in sc.get("shots", [])) or FPS
    every = max(1, total // 8)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(out), "-vf", f"select='not(mod(n\\,{every}))',scale=192:-2,tile=8x1", "-frames:v", "1", str(local / "strip.png")],
                   capture_output=True, text=True, timeout=300)
    update_scene(sid, {"status": "assembled"})
    return {**_scene_summary(get_scene(sid) or sc), "clips": len(clips), "missing": missing, "path": str(out)}


def scene_file(sid: str, name: str) -> Path | None:
    if name not in ("scene.mp4", "strip.png"):
        return None
    p = _dir() / "scenes" / sid / name
    return p if p.exists() else None
