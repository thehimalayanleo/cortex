"""Physical AI: the SO-ARM101 in the Lab. Joints, poses, episodes, a planner, and the bridge to real servos.

Storage (inside the vault):
  arm/state.json            joints (degrees), mode, port, bridge run id, poses{name: joints}
  arm/cmd.json, obs.json    what the bridge (lab/recipes/arm_bridge.py) reads and writes
  arm/calibration.json      per-joint tick offset and sign for real servos
  arm/episodes/<id>/        meta.json + frames.jsonl (t, joints), LeRobot-style: fps, joint names, task text

The bridge is a normal run (local executor, `--sim` when no port), so its log, metrics and stop button are the
ordinary run ones. The planner turns an instruction into waypoints with a language model: NVIDIA Nemotron on Nebius
Token Factory when NEBIUS_API_KEY is set (the Nebius x NVIDIA track), otherwise the app's chat model.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import runs, vault

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LIMITS = {"shoulder_pan": (-110, 110), "shoulder_lift": (-100, 100), "elbow_flex": (-100, 100), "wrist_flex": (-100, 100), "wrist_roll": (-160, 160), "gripper": (0, 100)}
# link lengths in metres, from the SO-ARM101 STEP files (approximate; used for the visualiser's forward kinematics)
LINKS = {"base": 0.0585, "upper": 0.1165, "forearm": 0.1350, "wrist": 0.0600, "gripper": 0.0900}
DEFAULT_POSES = {
    "rest": {"shoulder_pan": 0, "shoulder_lift": -95, "elbow_flex": 95, "wrist_flex": 60, "wrist_roll": 0, "gripper": 10},
    "home": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 30},
    "reach": {"shoulder_pan": 0, "shoulder_lift": 45, "elbow_flex": -30, "wrist_flex": -15, "wrist_roll": 0, "gripper": 60},
    "look_left": {"shoulder_pan": 70, "shoulder_lift": 20, "elbow_flex": 10, "wrist_flex": -30, "wrist_roll": 0, "gripper": 30},
    "look_right": {"shoulder_pan": -70, "shoulder_lift": 20, "elbow_flex": 10, "wrist_flex": -30, "wrist_roll": 0, "gripper": 30},
}
_lock = threading.Lock()
_motion = {"token": 0, "busy": None}
_rec: dict[str, Any] = {"id": None, "thread": None, "stop": False}


def _dir() -> Path:
    d = vault.VAULT / "arm"
    (d / "episodes").mkdir(parents=True, exist_ok=True)
    return d


def _load() -> dict[str, Any]:
    p = _dir() / "state.json"
    st = {"joints": dict(DEFAULT_POSES["home"]), "mode": "sim", "port": "", "bridge": None, "poses": dict(DEFAULT_POSES)}
    if p.exists():
        try:
            st.update(json.loads(p.read_text()))
        except Exception:
            pass
    for k, v in DEFAULT_POSES.items():
        st["poses"].setdefault(k, v)
    return st


def _save(st: dict[str, Any]) -> None:
    (_dir() / "state.json").write_text(json.dumps(st, indent=1))


def clamp(j: str, v: float) -> float:
    lo, hi = LIMITS[j]
    return float(max(lo, min(hi, float(v))))


def ports() -> list[str]:
    pats = ["/dev/tty.usbmodem*", "/dev/cu.usbmodem*", "/dev/tty.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"]
    out: list[str] = []
    for p in pats:
        out += glob.glob(p)
    return sorted(set(out))


def _obs() -> dict[str, Any] | None:
    p = _dir() / "obs.json"
    if not p.exists():
        return None
    try:
        o = json.loads(p.read_text())
        if time.time() - float(o.get("t", 0)) > 3.0:
            return None
        return o
    except Exception:
        return None


def _bridge_alive(st: dict[str, Any]) -> bool:
    rid = st.get("bridge")
    if not rid:
        return False
    r = runs.read_run(rid, tail=0, max_metrics=0)
    return bool(r and r["status"] in ("queued", "running"))


def fk(j: dict[str, float]) -> dict[str, Any]:
    """Forward kinematics (planar chain rotated by the pan) → joint positions in metres, for the visualiser and checks."""
    import math
    pan = math.radians(j.get("shoulder_pan", 0))
    a1 = math.radians(j.get("shoulder_lift", 0))
    a2 = a1 + math.radians(j.get("elbow_flex", 0))
    a3 = a2 + math.radians(j.get("wrist_flex", 0))
    pts = [(0.0, 0.0), (0.0, LINKS["base"])]
    r, z = 0.0, LINKS["base"]
    for ang, L in ((a1, LINKS["upper"]), (a2, LINKS["forearm"]), (a3, LINKS["wrist"] + LINKS["gripper"])):
        r += L * math.sin(ang)
        z += L * math.cos(ang)
        pts.append((r, z))
    xyz = [[round(rr * math.cos(pan), 4), round(rr * math.sin(pan), 4), round(zz, 4)] for rr, zz in pts]
    return {"points": xyz, "tip": xyz[-1], "reach_m": round(math.hypot(xyz[-1][0], xyz[-1][1]), 4)}


def status() -> dict[str, Any]:
    st = _load()
    obs = _obs()
    alive = _bridge_alive(st)
    joints = obs["joints"] if (obs and alive and obs.get("joints")) else st["joints"]
    joints = {j: round(float(joints.get(j, 0)), 2) for j in JOINTS}
    return {"joints": joints, "target": st["joints"], "mode": (obs or {}).get("mode", st["mode"]) if alive else "offline", "bridge": st.get("bridge") if alive else None,
            "port": st.get("port", ""), "ports": ports(), "poses": st["poses"], "limits": LIMITS, "links": LINKS, "joint_names": JOINTS, "fk": fk(joints),
            "recording": _rec["id"], "moving": _motion["busy"], "host": platform.node(), "planner": planner_name(), "episodes": len(list((_dir() / "episodes").iterdir()))}


def connect(port: str = "", sim: bool = False, executor: str = "local") -> dict[str, Any]:
    """Start the bridge run. With sim (or no port) the servos are emulated."""
    st = _load()
    if _bridge_alive(st):
        return runs.read_run(st["bridge"], tail=0, max_metrics=0) or {}
    args = f"--dir {_dir()}" + (" --sim" if (sim or not port) else f" --port {port}")
    if not os.environ.get("CORTEX_LOCAL_PYTHON"):
        os.environ["CORTEX_ARM_PYTHON"] = "uv run --python 3.11 --with pyserial python"
    m = runs.start("arm_bridge", args, executor, origin="arm")
    st["bridge"] = m["id"]
    st["port"] = port
    st["mode"] = "sim" if (sim or not port) else "hardware"
    _save(st)
    (_dir() / "cmd.json").write_text(json.dumps({"joints": st["joints"], "t": time.time()}))
    return m


def disconnect() -> dict[str, Any]:
    st = _load()
    if st.get("bridge"):
        try:
            runs.stop(st["bridge"])
        except Exception:
            pass
    st["bridge"] = None
    _save(st)
    return status()


def set_joints(joints: dict[str, float], immediate: bool = True) -> dict[str, Any]:
    with _lock:
        st = _load()
        for j, v in joints.items():
            if j in LIMITS:
                st["joints"][j] = clamp(j, v)
        _save(st)
        (_dir() / "cmd.json").write_text(json.dumps({"joints": st["joints"], "t": time.time()}))
    return status()


def goto(target: dict[str, float], seconds: float = 1.2, steps: int = 24) -> dict[str, Any]:
    """Interpolate from the current target to `target` over `seconds` (cosine ease) in a background thread."""
    st = _load()
    start = dict(st["joints"])
    end = {j: clamp(j, target.get(j, start[j])) for j in JOINTS}
    _motion["token"] += 1
    token = _motion["token"]
    _motion["busy"] = "goto"

    def run() -> None:
        import math
        for k in range(1, steps + 1):
            if _motion["token"] != token:
                return
            e = 0.5 - 0.5 * math.cos(math.pi * k / steps)
            set_joints({j: start[j] + (end[j] - start[j]) * e for j in JOINTS})
            time.sleep(seconds / steps)
        if _motion["token"] == token:
            _motion["busy"] = None
    threading.Thread(target=run, daemon=True).start()
    return status()


def save_pose(name: str, joints: dict[str, float] | None = None) -> dict[str, Any]:
    st = _load()
    nm = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:32] or "pose"
    st["poses"][nm] = {j: clamp(j, (joints or st["joints"]).get(j, 0)) for j in JOINTS}
    _save(st)
    return st["poses"]


def remove_pose(name: str) -> dict[str, Any]:
    st = _load()
    if name in DEFAULT_POSES:
        raise ValueError("built-in pose")
    st["poses"].pop(name, None)
    _save(st)
    return st["poses"]


# ---------------------------------------------------------------- episodes (LeRobot-style: fps, joint names, frames)

def record_start(task: str = "", fps: int = 10) -> dict[str, Any]:
    if _rec["id"]:
        raise ValueError("already recording")
    eid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = _dir() / "episodes" / eid
    d.mkdir(parents=True)
    meta = {"id": eid, "task": task, "fps": fps, "joint_names": JOINTS, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": 0, "mode": status()["mode"]}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    _rec.update(id=eid, stop=False)

    def loop() -> None:
        n = 0
        with (d / "frames.jsonl").open("a") as f:
            t0 = time.time()
            while not _rec["stop"]:
                s = status()
                f.write(json.dumps({"t": round(time.time() - t0, 3), "joints": s["joints"], "target": s["target"], "tip": s["fk"]["tip"]}) + "\n")
                f.flush()
                n += 1
                time.sleep(1.0 / fps)
        meta["frames"] = n
        meta["seconds"] = round(time.time() - t0, 2)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        _rec.update(id=None, thread=None)
    th = threading.Thread(target=loop, daemon=True)
    _rec["thread"] = th
    th.start()
    return meta


def record_stop() -> dict[str, Any]:
    if not _rec["id"]:
        raise ValueError("not recording")
    eid = _rec["id"]
    _rec["stop"] = True
    th = _rec.get("thread")
    if th:
        th.join(timeout=3)
    return episode(eid)


def list_episodes() -> list[dict[str, Any]]:
    out = []
    for d in sorted((_dir() / "episodes").iterdir(), reverse=True):
        mp = d / "meta.json"
        if mp.exists():
            try:
                out.append(json.loads(mp.read_text()))
            except Exception:
                pass
    return out


def episode(eid: str, frames: bool = True) -> dict[str, Any]:
    d = _dir() / "episodes" / eid
    if not (d / "meta.json").exists():
        raise KeyError(eid)
    m = json.loads((d / "meta.json").read_text())
    if frames and (d / "frames.jsonl").exists():
        m["frames_data"] = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
    return m


def remove_episode(eid: str) -> None:
    import shutil
    d = _dir() / "episodes" / eid
    if d.exists():
        shutil.rmtree(d)


def play(eid: str, speed: float = 1.0) -> dict[str, Any]:
    ep = episode(eid)
    fr = ep.get("frames_data") or []
    if not fr:
        raise ValueError("empty episode")
    _motion["token"] += 1
    token = _motion["token"]
    _motion["busy"] = f"play {eid}"

    def run() -> None:
        t0 = time.time()
        for f in fr:
            if _motion["token"] != token:
                return
            wait = f["t"] / speed - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            set_joints(f.get("target") or f["joints"])
        if _motion["token"] == token:
            _motion["busy"] = None
    threading.Thread(target=run, daemon=True).start()
    return status()


def stop_motion() -> dict[str, Any]:
    _motion["token"] += 1
    _motion["busy"] = None
    return status()


# ---------------------------------------------------------------- planner (Nemotron on Nebius when configured)

def planner_name() -> str:
    if os.environ.get("NEBIUS_API_KEY"):
        return os.environ.get("NEBIUS_MODEL", "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5") + " @ nebius"
    return "chat model (set NEBIUS_API_KEY for Nemotron on Nebius)"


def _llm(messages: list[dict[str, str]]) -> str:
    from openai import OpenAI
    key = os.environ.get("NEBIUS_API_KEY")
    if key:
        c = OpenAI(base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"), api_key=key, timeout=90.0)
        model = os.environ.get("NEBIUS_MODEL", "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5")
    else:
        from . import chat
        c = chat.client()
        model = chat.DEFAULT_MODEL
    # reasoning models think first; give the answer room and ask for the thinking to be brief or off
    kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 6000}
    if not key:
        kw["extra_body"] = {"reasoning_effort": "low"}
    r = c.chat.completions.create(**kw)
    m = r.choices[0].message
    txt = m.content or ""
    if not txt.strip():  # some proxies put the whole reply in the reasoning field
        txt = str(getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning_content") or "")
    return txt


def plan(instruction: str, execute: bool = False) -> dict[str, Any]:
    st = status()
    sysmsg = ("You plan motions for a 6-joint SO-ARM101 desk robot arm. Joints in degrees: " + json.dumps(LIMITS) +
              ". Zero for every joint is the arm pointing straight up; positive shoulder_lift leans the arm forward, positive elbow_flex folds the forearm back, "
              "gripper 0 closed 100 open. Link lengths (m): " + json.dumps(LINKS) + ". Named poses you may reuse: " + json.dumps(st["poses"]) +
              ". Current joints: " + json.dumps(st["joints"]) +
              ". Answer with JSON only: {\"waypoints\": [{\"name\": str, \"joints\": {joint: deg}, \"seconds\": float}], \"note\": str}. "
              "Use 2 to 8 waypoints, gentle motions, and stay inside the limits. Open the gripper before approaching an object, close it to grasp.")
    txt = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": instruction}])
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"planner returned no JSON: {txt[:200]}")
    data = json.loads(m.group(0))
    wps = []
    for w in data.get("waypoints", [])[:8]:
        joints = {j: clamp(j, float(w.get("joints", {}).get(j, st["target"][j]))) for j in JOINTS}
        wps.append({"name": str(w.get("name", ""))[:40], "joints": joints, "seconds": float(max(0.3, min(6.0, w.get("seconds", 1.5)))), "tip": fk(joints)["tip"]})
    out = {"instruction": instruction, "waypoints": wps, "note": str(data.get("note", ""))[:400], "planner": planner_name(), "executed": False}
    if execute and wps:
        run_waypoints(wps)
        out["executed"] = True
    return out


def run_waypoints(wps: list[dict[str, Any]]) -> dict[str, Any]:
    _motion["token"] += 1
    token = _motion["token"]
    _motion["busy"] = "plan"

    def run() -> None:
        import math
        cur = dict(_load()["joints"])
        for w in wps:
            end = {j: clamp(j, w["joints"].get(j, cur[j])) for j in JOINTS}
            steps = max(6, int(w.get("seconds", 1.5) * 20))
            for k in range(1, steps + 1):
                if _motion["token"] != token:
                    return
                e = 0.5 - 0.5 * math.cos(math.pi * k / steps)
                set_joints({j: cur[j] + (end[j] - cur[j]) * e for j in JOINTS})
                time.sleep(w.get("seconds", 1.5) / steps)
            cur = end
        if _motion["token"] == token:
            _motion["busy"] = None
    threading.Thread(target=run, daemon=True).start()
    return status()
