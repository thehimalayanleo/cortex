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

from . import arm_world as W
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
            "recording": _rec["id"], "moving": _motion["busy"], "host": platform.node(), "planner": planner_name(), "world": world(), "episodes": len(list((_dir() / "episodes").iterdir()))}


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


# ---------------------------------------------------------------- the tabletop (arm_world holds the physics)

def _world_path() -> Path:
    return _dir() / "world.json"


def world() -> dict[str, Any]:
    p = _world_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return W.make_scene(1)


def set_world(w: dict[str, Any]) -> dict[str, Any]:
    _world_path().write_text(json.dumps(w))
    return w


def reset_world(seed: int = 1, far: bool = False) -> dict[str, Any]:
    return set_world(W.make_scene(seed, far="purple" if far else None))


# ---------------------------------------------------------------- models: Nemotron on Nebius Token Factory

KEY_FILE = Path("~/.config/cortex/nebius.key").expanduser()
TF_BASE = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
# fast = everyday planning calls, deep = the escalation when repairs keep failing (the track's own advice:
# Ultra for serious reasoning, Super/Nano for the fast path). Overridable; otherwise discovered from /models.
TIER_ENV = {"fast": "NEBIUS_MODEL_FAST", "deep": "NEBIUS_MODEL_DEEP"}
TIER_DEFAULT = {"fast": "nvidia/nemotron-3-super-120b-a12b", "deep": "nvidia/Nemotron-3-Ultra-550b-a55b"}
TIER_MATCH = {"fast": ("nemotron-3-super", "nemotron-3-nano", "nemotron"), "deep": ("nemotron-3-ultra", "ultra", "nemotron-3-super")}
_tf: dict[str, Any] = {"client": None, "models": None}


def nebius_key() -> str:
    k = os.environ.get("NEBIUS_API_KEY", "")
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
    return k


def set_nebius_key(key: str) -> dict[str, Any]:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key.strip())
    os.chmod(KEY_FILE, 0o600)
    _tf.update(client=None, models=None)
    return nebius_status()


def _tf_client():
    from openai import OpenAI
    if _tf["client"] is None:
        _tf["client"] = OpenAI(base_url=TF_BASE, api_key=nebius_key(), timeout=120.0, max_retries=1)
    return _tf["client"]


def _tf_models() -> list[str]:
    if _tf["models"] is None:
        try:
            _tf["models"] = sorted(m.id for m in _tf_client().models.list().data)
        except Exception:
            _tf["models"] = []
    return _tf["models"]


def tier_model(tier: str) -> str:
    if os.environ.get(TIER_ENV[tier]):
        return os.environ[TIER_ENV[tier]]
    ids = [m for m in _tf_models() if "nvidia" in m.lower()]
    for pat in TIER_MATCH[tier]:
        hit = [m for m in ids if pat in m.lower() and "embed" not in m.lower()]
        if hit:
            return hit[0]
    return TIER_DEFAULT[tier]


def nebius_status() -> dict[str, Any]:
    if not nebius_key():
        return {"connected": False, "provider": provider(), "planner": planner_name(), "base_url": TF_BASE, "models": []}
    ms = _tf_models()
    return {"connected": bool(ms), "provider": provider(), "base_url": TF_BASE, "models": [m for m in ms if "nvidia" in m.lower()], "fast": tier_model("fast"), "deep": tier_model("deep"),
            "error": None if ms else "the key did not list any models"}


# Self-hosted Nemotron 3 Nano until a Token Factory key is set: vLLM on the 5090 over Tailscale by default, or the
# Modal deployment (serve/nemotron_modal.py). One model; the deep tier is the same model with thinking on.
SELFHOST = {
    # ARM-FT-1's fine-tuned student (Nemotron 3 Nano 4B + LoRA on simulator-verified programs; best of 5 seeds, 0.95 held-out)
    "student": {"url": "http://100.86.175.54:8002/v1", "model": "nemotron-3-nano-4b-arm-lora", "prompt": "student"},
    "5090": {"url": "http://100.86.175.54:8001/v1", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"},
    "modal": {"url": "https://ajinkyamulay123--cortex-nemotron-serve.modal.run/v1", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"},
}
SELFHOST_KEY_FILE = Path("~/.config/cortex/selfhost.key").expanduser()
_self: dict[str, Any] = {"client": None, "host": None}


def selfhost() -> tuple[str, dict[str, str]]:
    host = os.environ.get("CORTEX_NEMOTRON_HOST", "student")
    return host, SELFHOST[host]


def provider() -> str:
    if nebius_key():
        return "nebius"
    if SELFHOST_KEY_FILE.exists():
        return "selfhost"
    return "chat"


def planner_name() -> str:
    pv = provider()
    if pv == "nebius":
        return f"{tier_model('fast')} → {tier_model('deep')} @ Nebius Token Factory"
    if pv == "selfhost":
        host, cfg = selfhost()
        if cfg.get("prompt") == "student":
            return f"{cfg['model']} @ 5090 (fine-tuned on simulator-verified programs)"
        return f"{cfg['model']} (thinking off → on) @ {host}, self-hosted vLLM"
    return "chat model (add a Nebius Token Factory key for Nemotron)"


def _llm(messages: list[dict[str, str]], tier: str = "fast") -> tuple[str, str]:
    """One chat call. Returns (text, model id)."""
    from openai import OpenAI
    pv = provider()
    if pv == "nebius":
        c, model = _tf_client(), tier_model(tier)
        kw: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 8000}
    elif pv == "selfhost":
        host, cfg = selfhost()
        if _self["host"] != host:  # cold starts can take minutes
            _self.update(host=host, client=OpenAI(base_url=cfg["url"], api_key=SELFHOST_KEY_FILE.read_text().strip(), timeout=900.0, max_retries=1))
        c, model = _self["client"], cfg["model"]
        kw = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 10000,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": tier == "deep"}}}
    else:
        from . import chat
        c, model = chat.client(), chat.DEFAULT_MODEL
        kw = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 6000, "extra_body": {"reasoning_effort": "low"}}
    r = c.chat.completions.create(**kw)
    m = r.choices[0].message
    txt = m.content or ""
    if not txt.strip():  # some providers put the whole reply in the reasoning field
        txt = str(getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning_content") or "")
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    return (txt, model + (" +thinking" if pv == "selfhost" and tier == "deep" and selfhost()[1].get("prompt") != "student" else ""))


def _json(txt: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"no JSON in the reply: {txt[:200]}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------- planners

SKILL_PROMPT = """You control a 6-joint SO-ARM101 desk robot arm by writing a short skill program. Frame: x forward, y left, z up, metres, origin at the arm's base on the table.
The gripper tip can reach about 0.15 to 0.38 m from the base (pointing down it reaches about 0.15 to 0.23 m at table height).

Scene:
{scene}

Skills (JSON objects, run in order):
  {{"op": "open"}} / {{"op": "close"}}  open or close the gripper; closing grasps a block whose centre is within 2 cm of the tip
  {{"op": "move", "to": "<object name>", "above": <metres>}}  tip to that object's centre (for the bin: to the top of its walls), raised by "above"
  {{"op": "move", "to": [x, y, z]}}  tip to a point; a held block's centre follows the tip
  {{"op": "home"}}  the upright home pose
  {{"op": "refuse", "reason": "..."}}  when the task cannot be done safely or is out of reach (then nothing moves)

Physics: a released block falls onto whatever is under it. A closed gripper pushed into a block, a held block pushed into another block or the table, and a target out of reach are all failures. To grasp: open, move above the block, move down to its centre, close, lift. A block you are holding cannot be a move target.
Also declare what done looks like, so the simulator can check the program achieves it: "goal" is a list of
  {{"pred": "in", "obj": "<block>", "target": "bin"}}   {{"pred": "on", "obj": "<block>", "target": "<block>"}}   {{"pred": "at", "obj": "<block>", "xy": [x, y]}} (resting on the table, within 2 cm)
Answer with JSON only: {{"goal": [...], "program": [...], "note": "one sentence"}}. When you refuse, the goal is []."""

STUDENT_PROMPT = """You plan for a 6-joint desk robot arm. Frame: x forward, y left, z up, metres, base at the origin on the table. The gripper reaches about 0.15 to 0.38 m from the base.
Scene:
{scene}
Ops: {{"op":"open"}} {{"op":"close"}} (grasps a block whose centre is within 2 cm of the tip) {{"op":"move","to":"<object>" or [x,y,z],"above":metres}} {{"op":"home"}} {{"op":"refuse","reason":"..."}}
To grasp: open, move above the block, move down to its centre, close, lift. A released block falls onto what is below it. Refuse tasks that are out of reach or unsafe.
Answer with JSON only: {{"goal":[...],"program":[...]}}. Goal items: {{"pred":"in","obj":"<block>","target":"bin"}} {{"pred":"on","obj":"<block>","target":"<block>"}} {{"pred":"at","obj":"<block>","xy":[x,y]}}. When refusing, the goal is []."""  # verbatim from ARM-FT-1 training (specialist-lab/outputs/armft/armft.py)

JOINT_PROMPT = """You plan motions for a 6-joint SO-ARM101 desk robot arm by writing joint angles directly. Joint limits in degrees: {limits}.
Zero for every joint is the arm pointing straight up; positive shoulder_lift leans the arm forward, positive elbow_flex bends the forearm further forward, wrist_flex likewise; shoulder_pan positive turns toward +y. Gripper 0 closed, 100 open; closing grasps a block within 2 cm of the tip.
Link lengths (m): {links}. Frame: x forward, y left, z up, origin at the base on the table.
Scene:
{scene}
Current joints: {joints}.
Answer with JSON only: {{"waypoints": [{{"name": str, "joints": {{joint: deg}}, "seconds": float}}], "note": str}}, 2 to 16 waypoints. If the task cannot be done, answer {{"refuse": "reason"}}."""


def _check(w: dict[str, Any], program: list[dict[str, Any]], start: dict[str, float]) -> dict[str, Any]:
    wps, issues, refused = W.compile_program(w, program, start)
    sim = W.simulate(w, wps, start) if not refused else {"world": w, "issues": [], "timeline": []}
    issues = issues + sim["issues"]
    for wp in wps:
        wp["tip"] = [round(v, 4) for v in W.fk(wp["joints"])[-1]]
    return {"waypoints": wps, "issues": issues, "refused": refused, "world_after": sim["world"], "timeline": sim["timeline"],
            "ok": not any(i["level"] == "error" for i in issues)}


def plan_skills(instruction: str, w: dict[str, Any], start: dict[str, float], repairs: int = 2) -> dict[str, Any]:
    """Nemotron writes a program; the simulator checks it; errors go back for a rewrite. Attempt 1 and 2 use the fast
    tier, the last repair escalates to the deep tier."""
    student = provider() == "selfhost" and selfhost()[1].get("prompt") == "student"
    msgs = [{"role": "system", "content": (STUDENT_PROMPT if student else SKILL_PROMPT).format(scene=W.describe(w))}, {"role": "user", "content": instruction}]
    attempts: list[dict[str, Any]] = []
    chk: dict[str, Any] = {}
    for k in range(repairs + 1):
        tier = "deep" if (k == repairs and k > 0) else "fast"
        t0 = time.time()
        try:
            txt, model = _llm(msgs, tier)
            data = _json(txt)
            prog = [p for p in data.get("program", []) if isinstance(p, dict)]
            chk = _check(w, prog, start)
            note = str(data.get("note", ""))[:300]
            goal = [g for g in data.get("goal", []) if isinstance(g, dict)]
            if not chk["refused"]:  # the model's own postcondition must hold in the simulated scene
                if not goal:
                    chk["issues"].append({"step": 0, "level": "error", "msg": "no goal declared: say what the scene should look like when the task is done"})
                else:
                    held, fails = W.check_goal(chk["world_after"], goal, False)
                    chk["issues"] += [{"step": 0, "level": "error", "msg": f"your declared goal does not hold after the program: {f}"} for f in fails]
                chk["ok"] = not any(i["level"] == "error" for i in chk["issues"])
        except Exception as e:
            txt, model, prog, note, goal = "", tier, [], "", []
            chk = {"waypoints": [], "issues": [{"step": 0, "level": "error", "msg": f"unusable reply: {e}"[:300]}], "refused": False, "world_after": w, "timeline": [], "ok": False}
        attempts.append({"tier": tier, "model": model, "seconds": round(time.time() - t0, 1), "program": prog, "goal": goal, "note": note, "issues": chk["issues"], "ok": chk["ok"]})
        if chk["ok"]:
            break
        errs = "\n".join(f"- step {i['step']}: {i['msg']}" for i in chk["issues"] if i["level"] == "error")
        msgs += [{"role": "assistant", "content": json.dumps({"goal": goal, "program": prog, "note": note})},
                 {"role": "user", "content": f"The simulator rejected that program:\n{errs}\nScene after the program ran:\n{W.describe(chk['world_after'])}\nWrite the full corrected program that still does the whole task: {instruction}"}]
    return {**chk, "attempts": attempts, "program": attempts[-1]["program"], "goal": attempts[-1]["goal"], "note": attempts[-1]["note"]}


def plan_joints(instruction: str, w: dict[str, Any], start: dict[str, float]) -> dict[str, Any]:
    """The original planner: the model writes joint angles itself. Kept as the baseline."""
    t0 = time.time()
    sysmsg = JOINT_PROMPT.format(limits=json.dumps(LIMITS), links=json.dumps(LINKS), scene=W.describe(w), joints=json.dumps({j: round(v, 1) for j, v in start.items()}))
    try:
        txt, model = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": instruction}])
        data = _json(txt)
        refused = bool(data.get("refuse"))
        wps = [{"name": str(x.get("name", ""))[:40], "joints": {j: float(x.get("joints", {}).get(j, start[j])) for j in JOINTS}, "seconds": float(max(0.3, min(6.0, x.get("seconds", 1.5)))), "step": n}
               for n, x in enumerate(data.get("waypoints", [])[:16], 1) if isinstance(x, dict)]
        sim = W.simulate(w, wps, start) if not refused else {"world": w, "issues": [], "timeline": []}
        issues = sim["issues"]
    except Exception as e:
        model, refused, wps, sim = "?", False, [], {"world": w, "timeline": []}
        issues = [{"step": 0, "level": "error", "msg": f"unusable reply: {e}"[:300]}]
    for wp in wps:
        wp["tip"] = [round(v, 4) for v in W.fk(wp["joints"])[-1]]
    return {"waypoints": wps, "issues": issues, "refused": refused, "world_after": sim["world"], "timeline": sim["timeline"], "ok": not any(i["level"] == "error" for i in issues),
            "attempts": [{"tier": "fast", "model": model, "seconds": round(time.time() - t0, 1), "issues": issues, "ok": not any(i["level"] == "error" for i in issues)}]}


def plan(instruction: str, execute: bool = False, mode: str = "skills") -> dict[str, Any]:
    st = status()
    w = world()
    start = dict(st["target"])
    out = plan_joints(instruction, w, start) if mode == "joints" else plan_skills(instruction, w, start)
    out.update(instruction=instruction, mode=mode, planner=planner_name(), executed=False)
    out["waypoints"] = [{**wp, "joints": {j: clamp(j, v) for j, v in wp["joints"].items()}} for wp in out["waypoints"]]
    if execute and out["ok"] and not out["refused"] and out["waypoints"]:
        run_waypoints(out["waypoints"], out["timeline"])
        out["executed"] = True
    out.pop("timeline", None)
    _dir().joinpath("last_plan.json").write_text(json.dumps(out))
    return out


def run_waypoints(wps: list[dict[str, Any]], timeline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    _motion["token"] += 1
    token = _motion["token"]
    _motion["busy"] = "plan"

    def run() -> None:
        import math
        cur = dict(_load()["joints"])
        for n, w in enumerate(wps):
            end = {j: clamp(j, w["joints"].get(j, cur[j])) for j in JOINTS}
            steps = max(6, int(w.get("seconds", 1.5) * 20))
            for k in range(1, steps + 1):
                if _motion["token"] != token:
                    return
                e = 0.5 - 0.5 * math.cos(math.pi * k / steps)
                set_joints({j: cur[j] + (end[j] - cur[j]) * e for j in JOINTS})
                time.sleep(w.get("seconds", 1.5) / steps)
            cur = end
            if timeline and n < len(timeline):
                set_world(timeline[n])  # grasps and drops land when their waypoint does
        if _motion["token"] == token:
            _motion["busy"] = None
    threading.Thread(target=run, daemon=True).start()
    return status()


# ---------------------------------------------------------------- benchmark: does the model's plan actually do the task?

_bench: dict[str, Any] = {"thread": None}
MODES = ("oracle", "joints", "skills", "skills+repair")


def bench_path() -> Path:
    return _dir() / "bench.json"


def bench_result() -> dict[str, Any]:
    p = bench_path()
    return json.loads(p.read_text()) if p.exists() else {"status": "none"}


def _bench_one(task: dict[str, Any], mode: str) -> dict[str, Any]:
    start = dict(W.HOME)
    t0 = time.time()
    if mode == "oracle":
        r = _check(task["scene"], W.oracle_program(task), start)
        r["attempts"] = []
    elif mode == "joints":
        r = plan_joints(task["instruction"], task["scene"], start)
    else:
        r = plan_skills(task["instruction"], task["scene"], start, repairs=2 if mode == "skills+repair" else 0)
    ok, fails = W.check_goal(r["world_after"], task["goal"], r["refused"])
    ok = ok and (r["ok"] or r["refused"])  # a plan the simulator rejects never runs, so it cannot succeed
    return {"task": task["id"], "mode": mode, "success": ok, "valid": r["ok"], "refused": r["refused"], "fails": fails,
            "errors": [i["msg"] for i in r["issues"] if i["level"] == "error"][:4], "calls": len(r["attempts"]),
            "models": sorted({a["model"] for a in r["attempts"]}), "seconds": round(time.time() - t0, 1)}


def bench_start(seeds: tuple[int, ...] = (1, 2, 3), modes: tuple[str, ...] = MODES, workers: int = 4) -> dict[str, Any]:
    if _bench["thread"] and _bench["thread"].is_alive():
        raise ValueError("a benchmark is already running")
    ts = W.tasks(seeds)
    jobs = [(t, m) for m in modes for t in ts]
    state = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "planner": planner_name(), "n_tasks": len(ts), "modes": list(modes), "total": len(jobs), "rows": []}
    bench_path().write_text(json.dumps(state))

    def run() -> None:
        from concurrent.futures import ThreadPoolExecutor
        lock = threading.Lock()

        def one(job):
            row = _bench_one(*job)
            with lock:
                state["rows"].append(row)
                state["summary"] = bench_summary(state["rows"])
                bench_path().write_text(json.dumps(state))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, jobs))
        state["status"] = "done"
        state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        bench_path().write_text(json.dumps(state))
    th = threading.Thread(target=run, daemon=True)
    _bench["thread"] = th
    th.start()
    return state


def bench_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for m in MODES:
        rs = [r for r in rows if r["mode"] == m]
        if rs:
            out[m] = {"n": len(rs), "success": round(sum(r["success"] for r in rs) / len(rs), 3), "valid": round(sum(r["valid"] for r in rs) / len(rs), 3),
                      "calls": round(sum(r["calls"] for r in rs) / len(rs), 2), "seconds": round(sum(r["seconds"] for r in rs) / len(rs), 1)}
    return out
