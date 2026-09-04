"""Training runs: launch a lab recipe on an executor (this machine, the 5090 over SSH, or Modal), capture its
stdout into the vault, and parse the METRIC / STATUS / RESULT protocol lines the recipes print.

Layout inside the vault:
  runs/<id>/meta.json      id, recipe, args, executor, status (queued|running|done|failed|stopped), started, ended
  runs/<id>/log.txt        raw stdout+stderr
  runs/<id>/metrics.jsonl  one JSON object per METRIC line
  runs/<id>/result.json    the RESULT line, if any

Executors are configured by environment:
  CORTEX_SSH_HOST     e.g. ajinkya-5090 (a Tailscale SSH alias). Recipes are rsynced to ~/cortex-lab on the host.
  CORTEX_SSH_PYTHON   python on that host with torch installed (default: python3)
  CORTEX_LOCAL_PYTHON command used for local runs (default: uv run --python 3.11 --with torch --with numpy python)
  modal               available when the `modal` CLI is on PATH and ~/.modal.toml (or MODAL_TOKEN_ID) exists
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import vault

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "lab"
RECIPES = LAB / "recipes"

_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def runs_dir() -> Path:
    d = vault.VAULT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- executors

def executors() -> dict[str, Any]:
    ssh_host = os.environ.get("CORTEX_SSH_HOST", "").strip()
    modal_ok = bool(shutil.which("modal")) and (Path.home().joinpath(".modal.toml").exists() or bool(os.environ.get("MODAL_TOKEN_ID")))
    local_ok = bool(shutil.which("uv")) or bool(os.environ.get("CORTEX_LOCAL_PYTHON"))
    return {
        "local": {"available": local_ok, "note": "runs the recipe on this machine (CPU unless it has a GPU); use --smoke for a quick pass"},
        "ssh": {"available": bool(ssh_host), "host": ssh_host or None, "note": "your GPU box over SSH (the default is the home 5090 over Tailscale); recipes are synced to ~/cortex-lab there"},
        "modal": {"available": modal_ok, "note": "a rented GPU through Modal; needs `modal token set` once"},
        "demo": os.environ.get("CORTEX_DEMO") == "1",
    }


def recipes() -> list[dict[str, Any]]:
    out = []
    for p in sorted(RECIPES.glob("*.py")):
        if p.name in {"common.py", "__init__.py"}:
            continue
        doc = ""
        try:
            src = p.read_text(errors="ignore")
            m = re.match(r'\s*(?:#[^\n]*\n)*\s*[ru]?"""(.*?)"""', src, re.S)
            doc = (m.group(1).strip() if m else "").split("\n\n")[0].strip()
        except Exception:
            pass
        out.append({"name": p.stem, "file": f"lab/recipes/{p.name}", "doc": doc[:400]})
    out.append({"name": "scratch", "file": "", "doc": "Run your own Python (from a chapter's snippet, or pasted): the code is saved with the run and executed on the chosen machine. Print METRIC {...} lines to get charts."})
    out.append({"name": "shell", "file": "", "doc": "The terminal: one shell command at a time on this machine or the GPU box, output streamed here."})
    return out


def _command(executor: str, recipe: str, args: str, script: Path | None = None, cmd: str | None = None) -> list[str]:
    if cmd is not None:  # the terminal: one shell command, streamed, in the lab folder of the chosen machine
        if executor == "local":
            return ["bash", "-lc", cmd]
        if executor == "ssh":
            host = _ssh_host()
            if not host:
                raise ValueError("CORTEX_SSH_HOST is not set")
            py_dir = os.path.dirname(_ssh_python())
            return ["ssh", *SSH_OPTS, host, f"export PATH={py_dir}:$HOME/.local/bin:$PATH; mkdir -p ~/cortex-lab && cd ~/cortex-lab && bash -lc {shlex.quote(cmd)}"]
        raise ValueError("the terminal runs locally or on the GPU box (ssh)")
    script = script or (RECIPES / f"{recipe}.py")
    if not script.exists():
        raise ValueError(f"no such recipe: {recipe}")
    remote_script = f"recipes/{recipe}.py" if script.parent == RECIPES else f"scratch/{script.name}"
    if recipe == "modal_app":
        raise ValueError("modal_app is the Modal wrapper; pick a recipe and the modal executor instead")
    extra = shlex.split(args or "")
    if executor == "local":
        py = shlex.split(os.environ.get("CORTEX_LOCAL_PYTHON") or "uv run --python 3.11 --with torch --with numpy python")
        return [*py, str(script), *extra]
    if executor == "ssh":
        host = os.environ.get("CORTEX_SSH_HOST", "").strip()
        if not host:
            raise ValueError("CORTEX_SSH_HOST is not set; export it (for example ajinkya-5090) and restart the server")
        py = _ssh_python()
        remote = " && ".join([
            "mkdir -p ~/cortex-lab/out",
            "cd ~/cortex-lab",
            f"{py} {remote_script} " + " ".join(shlex.quote(a) for a in extra),
        ])
        return ["ssh", *SSH_OPTS, host, remote]
    if executor == "modal":
        return ["modal", "run", str(RECIPES / "modal_app.py"), "--recipe", recipe if script.parent == RECIPES else str(script), "--args", " ".join(extra)]
    raise ValueError(f"unknown executor: {executor}")


def _sync_recipes(host: str) -> str:
    """rsync the recipes folder to the GPU box; returns rsync's output (empty on success)."""
    cmd = ["rsync", "-az", "--delete", "--rsync-path", "mkdir -p ~/cortex-lab/recipes ~/cortex-lab/out && rsync", "-e", "ssh " + " ".join(SSH_OPTS),
           str(RECIPES) + "/", f"{host}:~/cortex-lab/recipes/"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("rsync to the GPU box failed: " + (r.stderr or r.stdout).strip()[-800:])
    return r.stdout



# ---------------------------------------------------------------- the GPU box: status and one-click bootstrap (the app makes the connection, not the user)

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-%r@%h:%p", "-o", "ControlPersist=600"]
SETUP_SCRIPT = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
echo "[1/4] uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
echo "[2/4] python 3.11 venv at ~/lab-venv"
[ -x "$HOME/lab-venv/bin/python" ] || uv venv "$HOME/lab-venv" --python 3.11
PY="$HOME/lab-venv/bin/python"
echo "[3/4] torch (CUDA 12.8 wheels for Blackwell)"
$PY -c "import torch" 2>/dev/null || uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu128
echo "[4/4] training libraries"
$PY -c "import numpy, transformers, datasets, trl, peft, sentence_transformers" 2>/dev/null || uv pip install --python "$PY" numpy transformers datasets trl peft sentence-transformers accelerate
mkdir -p "$HOME/cortex-lab/out"
$PY -c "import torch; print('READY torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
"""


def _ssh_host() -> str:
    return os.environ.get("CORTEX_SSH_HOST", "").strip()


def _ssh_python() -> str:
    return os.environ.get("CORTEX_SSH_PYTHON", "$HOME/lab-venv/bin/python")


def _tailscale_peer(host: str) -> dict[str, Any] | None:
    """How the box is reached: its Tailscale IP, OS, and whether the link is direct or relayed (from `tailscale status`)."""
    for exe in (shutil.which("tailscale"), "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if not exe or not Path(exe).exists():
            continue
        try:
            r = subprocess.run([exe, "status"], capture_output=True, text=True, timeout=8)
        except Exception:
            continue
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == host:
                state = line.split(None, 4)[4] if len(line.split(None, 4)) > 4 else ""
                return {"ip": parts[0], "os": parts[3] if len(parts) > 3 else None, "link": "direct" if "direct" in state else ("relayed" if "relay" in state else ("active" if "active" in state else "idle")), "state": state.strip()}
        return {"ip": None, "state": "not in this tailnet's peer list"}
    return None


def gpu_status() -> dict[str, Any]:
    """One SSH round trip: is the box reachable, what GPU, is the venv ready."""
    host = _ssh_host()
    if not host:
        return {"host": None, "reachable": False, "ready": False, "message": "no GPU box configured (CORTEX_SSH_HOST)"}
    py = _ssh_python()
    probe = (
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1; "
        f"if [ -x {py} ]; then {py} -c \"import torch;print('TORCH', torch.__version__, torch.cuda.is_available())\" 2>&1 | tail -1; else echo NOVENV; fi; "
        "pgrep -f '[c]ortex-lab/recipes/[a-z_]*[.]py' >/dev/null && echo BUSY || echo IDLE"
    )
    t0 = time.time()
    try:
        r = subprocess.run(["ssh", *SSH_OPTS, host, probe], capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        return {"host": host, "reachable": False, "ready": False, "message": f"{host} did not answer in 25 s (is it awake, and is Tailscale up?)"}
    if r.returncode != 0 and not r.stdout.strip():
        err = (r.stderr or "").strip().splitlines()
        msg = err[-1] if err else f"ssh exited {r.returncode}"
        if "check mode" in (r.stderr or "").lower() or "tailscale" in (r.stderr or "").lower():
            msg = f"Tailscale SSH wants a one-time interactive approval: run `ssh {host}` in a terminal once, then retry"
        return {"host": host, "reachable": False, "ready": False, "message": msg}
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    gpu = next((l for l in lines if "," in l and "TORCH" not in l), None)
    torch_line = next((l for l in lines if l.startswith("TORCH")), None)
    novenv = any(l == "NOVENV" for l in lines)
    busy = any(l == "BUSY" for l in lines)
    out: dict[str, Any] = {"host": host, "reachable": True, "python": py, "busy": busy, "ssh_round_trip_ms": int((time.time() - t0) * 1000), "tailscale": _tailscale_peer(host)}
    if gpu:
        name, total, used, util = [x.strip() for x in gpu.split(",")][:4]
        out["gpu"] = {"name": name, "memory_total": total, "memory_used": used, "utilization": util}
    if torch_line:
        _, ver, ok = torch_line.split()[:3]
        out["torch"] = ver
        out["cuda"] = ok == "True"
        out["ready"] = ok == "True"
        out["message"] = f"ready: {out.get('gpu', {}).get('name', 'gpu')} · torch {ver}"
    else:
        out["ready"] = False
        out["message"] = "reachable; PyTorch is not installed yet (run setup)" if novenv else "reachable; PyTorch import failed (run setup)"
    return out


def gpu_setup_stream():
    """Bootstrap the box over SSH, yielding log lines. Idempotent: each step is skipped when already done."""
    host = _ssh_host()
    if not host:
        yield {"type": "error", "message": "no GPU box configured"}
        return
    log_path = runs_dir() / "_gpu_setup.log"
    yield {"type": "log", "lines": [f"[cortex] connecting to {host}"]}
    try:
        proc = subprocess.Popen(["ssh", *SSH_OPTS, host, "bash -s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return
    assert proc.stdin and proc.stdout
    proc.stdin.write(SETUP_SCRIPT)
    proc.stdin.close()
    with log_path.open("w") as f:
        for line in proc.stdout:
            f.write(line)
            yield {"type": "log", "lines": [line.rstrip("\n")]}
    code = proc.wait()
    yield {"type": "status", "status": "done" if code == 0 else "failed", "exit": code, "gpu": gpu_status() if code == 0 else None}


# ---------------------------------------------------------------- run lifecycle

def _write_meta(d: Path, meta: dict) -> None:
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def start(recipe: str, args: str = "", executor: str = "local", code: str | None = None, cmd: str | None = None) -> dict[str, Any]:
    """Launch a recipe; with code= run that Python source as a one-off script (the "scratch" recipe); with cmd= run a shell command (the "shell" recipe)."""
    ex = executors()
    if executor not in ("local", "ssh", "modal"):
        raise ValueError("executor must be local, ssh, or modal")
    if not ex[executor]["available"]:
        raise ValueError(f"the {executor} executor is not available here: {ex[executor]['note']}")
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = runs_dir() / rid
    d.mkdir(parents=True)
    meta = {"id": rid, "recipe": recipe, "args": args, "executor": executor, "status": "queued", "started": _now(), "ended": None, "exit": None}
    if code is not None:
        if not code.strip():
            raise ValueError("no code given")
        meta["recipe"] = "scratch"
        meta["script"] = str(d / f"scratch_{rid}.py")
        meta["code_preview"] = code.strip().splitlines()[0][:80]
        Path(meta["script"]).write_text(code)
    elif recipe == "scratch":
        raise ValueError("the scratch recipe needs code")
    if cmd is not None:
        if not cmd.strip():
            raise ValueError("no command given")
        meta["recipe"] = "shell"
        meta["cmd"] = cmd.strip()[:4000]
        meta["code_preview"] = cmd.strip().splitlines()[0][:80]
    elif recipe == "shell":
        raise ValueError("the shell recipe needs a command")
    _write_meta(d, meta)
    threading.Thread(target=_run, args=(d, meta), daemon=True).start()
    return meta


def _run(d: Path, meta: dict) -> None:
    log = (d / "log.txt").open("a", buffering=1)
    metrics = (d / "metrics.jsonl").open("a", buffering=1)
    try:
        script = Path(meta["script"]) if meta.get("script") else None
        if meta["executor"] == "ssh":
            log.write("[cortex] syncing recipes to " + os.environ.get("CORTEX_SSH_HOST", "") + "\n")
            _sync_recipes(os.environ["CORTEX_SSH_HOST"])
            if script:
                host = os.environ["CORTEX_SSH_HOST"]
                subprocess.run(["ssh", *SSH_OPTS, host, "mkdir -p ~/cortex-lab/scratch"], check=True, timeout=60)
                subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-%r@%h:%p", str(script), f"{host}:~/cortex-lab/scratch/{script.name}"], check=True, timeout=60)
        cmd = _command(meta["executor"], meta["recipe"], meta["args"], script, meta.get("cmd"))
        log.write("[cortex] $ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=str(LAB if meta.get("cmd") else ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        with _lock:
            _procs[meta["id"]] = proc
        meta["status"] = "running"
        _write_meta(d, meta)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            s = line.strip()
            if s.startswith("METRIC "):
                try:
                    obj = json.loads(s[7:])
                    obj.setdefault("t", time.time())
                    metrics.write(json.dumps(obj) + "\n")
                except Exception:
                    pass
            elif s.startswith("ROLLOUT "):
                try:
                    with (d / "rollouts.jsonl").open("a") as rf:
                        rf.write(json.dumps(json.loads(s[8:])) + "\n")
                except Exception:
                    pass
            elif s.startswith("RESULT "):
                try:
                    (d / "result.json").write_text(json.dumps(json.loads(s[7:]), indent=2))
                except Exception:
                    pass
        code = proc.wait()
        meta["exit"] = code
        try:  # stop() marks the on-disk meta as stopping; honour it
            stopping = json.loads((d / "meta.json").read_text()).get("status") == "stopping"
        except Exception:
            stopping = False
        meta["status"] = "done" if code == 0 else ("stopped" if stopping else "failed")
    except Exception as e:
        log.write(f"[cortex] {e}\n")
        meta["status"] = "failed"
        meta["error"] = str(e)
    finally:
        meta["ended"] = _now()
        _write_meta(d, meta)
        log.close()
        metrics.close()
        with _lock:
            _procs.pop(meta["id"], None)


def stop(rid: str) -> bool:
    with _lock:
        p = _procs.get(rid)
    if not p:
        return False
    d = runs_dir() / rid
    try:
        meta = json.loads((d / "meta.json").read_text())
        meta["status"] = "stopping"
        _write_meta(d, meta)
    except Exception:
        pass
    p.terminate()
    return True


# ---------------------------------------------------------------- reading

def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    out = []
    for d in sorted(runs_dir().iterdir(), reverse=True):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        m["last"] = _last_metric(d)
        out.append(m)
        if len(out) >= limit:
            break
    return out


def _last_metric(d: Path) -> dict | None:
    p = d / "metrics.jsonl"
    if not p.exists():
        return None
    try:
        lines = p.read_text().strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def read_run(rid: str, tail: int = 200, max_metrics: int = 2000, max_rollouts: int = 64) -> dict[str, Any] | None:
    d = runs_dir() / rid
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    log_lines = (d / "log.txt").read_text(errors="ignore").splitlines() if (d / "log.txt").exists() else []
    metrics: list[dict] = []
    if (d / "metrics.jsonl").exists():
        for line in (d / "metrics.jsonl").read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except Exception:
                pass
    if len(metrics) > max_metrics:  # thin evenly so charts stay light
        step = len(metrics) / max_metrics
        metrics = [metrics[int(i * step)] for i in range(max_metrics)]
    rollouts: list[dict] = []
    if (d / "rollouts.jsonl").exists():
        lines = (d / "rollouts.jsonl").read_text().splitlines()[-max_rollouts:]
        for line in lines:
            try:
                rollouts.append(json.loads(line))
            except Exception:
                pass
    result = None
    if (d / "result.json").exists():
        try:
            result = json.loads((d / "result.json").read_text())
        except Exception:
            pass
    return {**meta, "log": log_lines[-tail:], "log_lines": len(log_lines), "metrics": metrics, "rollouts": rollouts, "result": result}


def delete_run(rid: str) -> bool:
    d = runs_dir() / rid
    if not d.exists():
        return False
    stop(rid)
    shutil.rmtree(d)
    return True


# ---------------------------------------------------------------- chapters (the lab's curriculum; also synced into the vault as notes)

def chapters() -> list[dict[str, Any]]:
    out = []
    for p in sorted((LAB / "chapters").glob("*.md")):
        fm, _ = vault.read_md(p)
        out.append({"slug": "lab-" + p.stem, "file": p.name, **{k: fm.get(k) for k in ("title", "chapter", "station", "recipe", "reading_time")}})
    return out


def sync_chapters_into_vault() -> int:
    """Copy lab chapters into notes/ (slug lab-NN-name) when the source is newer, so search, chat tools, and the note view see them."""
    n = 0
    notes = vault.VAULT / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    for p in sorted((LAB / "chapters").glob("*.md")):
        dst = notes / f"lab-{p.stem}.md"
        if not dst.exists() or p.stat().st_mtime > dst.stat().st_mtime:
            shutil.copyfile(p, dst)
            n += 1
    return n


# ---------------------------------------------------------------- learning plan (a kanban over the chapters, stored in the vault)

PLAN_COLUMNS = ["todo", "doing", "done"]


def _plan_path() -> Path:
    d = vault.VAULT / ".cortex"
    d.mkdir(parents=True, exist_ok=True)
    return d / "lab_plan.json"


def _default_cards() -> list[dict[str, Any]]:
    cards = []
    for ch in chapters():
        n = int(ch.get("chapter") or 0)
        title = str(ch.get("title") or ch["slug"])
        short = title.split(":", 1)[-1].strip() if ":" in title else title
        cards.append({"id": f"read-{n:02d}", "chapter": n, "kind": "read", "title": f"Read: {short}", "note": ch["slug"], "col": "todo"})
        if ch.get("station") and ch["station"] != "none":
            cards.append({"id": f"station-{n:02d}", "chapter": n, "kind": "station", "title": f"Train it in the browser ({ch['station']})", "station": ch["station"], "col": "todo"})
        cards.append({"id": f"build-{n:02d}", "chapter": n, "kind": "build", "title": "Run the 'Build it small' snippet and change one thing", "note": ch["slug"], "col": "todo"})
        if ch.get("recipe") and ch["recipe"] != "none":
            rec = str(ch["recipe"]).split("/")[-1].replace(".py", "").split(" ")[0]
            cards.append({"id": f"recipe-{n:02d}", "chapter": n, "kind": "recipe", "title": f"Run {rec} on the 5090 and read the curves", "recipe": rec, "col": "todo"})
        cards.append({"id": f"quiz-{n:02d}", "chapter": n, "kind": "quiz", "title": "Pass the self-test (ask the chat to quiz you)", "note": ch["slug"], "col": "todo"})
    return cards


def plan() -> dict[str, Any]:
    """Merge saved state onto the generated cards so new chapters appear and old progress is kept."""
    saved: dict[str, Any] = {}
    p = _plan_path()
    if p.exists():
        try:
            saved = json.loads(p.read_text())
        except Exception:
            saved = {}
    state = saved.get("cards", {})
    cards = _default_cards()
    for c in saved.get("custom", []):  # cards the user (or the chat) added; these can be deleted, built-ins cannot
        if isinstance(c, dict) and c.get("id"):
            cards.append({"id": c["id"], "chapter": int(c.get("chapter") or 0), "kind": c.get("kind") or "custom", "title": str(c.get("title") or "")[:200], "note": c.get("note"), "station": c.get("station"), "recipe": c.get("recipe"), "col": "todo", "custom": True, "created": c.get("created")})
    for c in cards:
        st = state.get(c["id"])
        if isinstance(st, dict):
            c["col"] = st.get("col", c["col"]) if st.get("col") in PLAN_COLUMNS else c["col"]
            if st.get("done_at"):
                c["done_at"] = st["done_at"]
            if st.get("comment"):
                c["comment"] = st["comment"]
    done = sum(1 for c in cards if c["col"] == "done")
    return {"columns": PLAN_COLUMNS, "cards": cards, "done": done, "total": len(cards), **_score(cards)}


XP = {"read": 10, "station": 15, "build": 20, "recipe": 30, "quiz": 25}
LEVELS = ["Reader", "Tinkerer", "Trainer", "Post-trainer", "Evaluator", "Red-teamer", "Kernel writer", "Prover", "Research lead"]


def _score(cards: list[dict]) -> dict[str, Any]:
    """XP per finished card by kind, a level ladder (150 XP a rung), and a day streak from done_at dates."""
    import datetime as _dt
    xp = sum(XP.get(c["kind"], 10) for c in cards if c["col"] == "done")
    total_xp = sum(XP.get(c["kind"], 10) for c in cards)
    rung = min(len(LEVELS) - 1, xp // 150)
    days = sorted({c["done_at"][:10] for c in cards if c.get("done_at")}, reverse=True)
    today = _dt.date.today()
    streak = 0
    if days:
        cur = _dt.date.fromisoformat(days[0])
        if (today - cur).days <= 1:
            streak = 1
            for d in days[1:]:
                nxt = _dt.date.fromisoformat(d)
                if (cur - nxt).days == 1:
                    streak += 1
                    cur = nxt
                else:
                    break
    today_n = sum(1 for c in cards if c.get("done_at", "")[:10] == today.isoformat() and c["col"] == "done")
    return {"xp": xp, "xp_total": total_xp, "level": rung + 1, "level_name": LEVELS[rung], "next_level_xp": (rung + 1) * 150, "streak": streak, "done_today": today_n, "xp_by_kind": XP}


def plan_move(card_id: str, col: str, comment: str | None = None) -> dict[str, Any]:
    if col not in PLAN_COLUMNS:
        raise ValueError("col must be todo, doing, or done")
    p = _plan_path()
    saved: dict[str, Any] = {}
    if p.exists():
        try:
            saved = json.loads(p.read_text())
        except Exception:
            saved = {}
    cards = saved.setdefault("cards", {})
    entry = cards.setdefault(card_id, {})
    entry["col"] = col
    if col == "done":
        entry["done_at"] = _now()
    if comment is not None:
        entry["comment"] = comment[:500]
    p.write_text(json.dumps(saved, indent=1))
    return plan()


def _load_plan_file() -> dict[str, Any]:
    p = _plan_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def plan_add(title: str, kind: str = "custom", note: str | None = None, station: str | None = None, recipe: str | None = None, chapter: int | None = None) -> dict[str, Any]:
    """Add a learning card on the fly (a topic, a paper to work through, a run to do). Returns the plan."""
    title = title.strip()
    if not title:
        raise ValueError("a card needs a title")
    if kind not in ("custom", "read", "station", "build", "recipe", "quiz"):
        kind = "custom"
    saved = _load_plan_file()
    custom = saved.setdefault("custom", [])
    cid = "custom-" + uuid.uuid4().hex[:8]
    custom.append({"id": cid, "title": title[:200], "kind": kind, "note": note, "station": station, "recipe": recipe, "chapter": chapter or 0, "created": _now()})
    _plan_path().write_text(json.dumps(saved, indent=1))
    out = plan()
    out["added"] = cid
    return out


def plan_remove(card_id: str) -> dict[str, Any]:
    """Delete a custom card. Built-in chapter cards cannot be deleted (move them to done instead)."""
    if not card_id.startswith("custom-"):
        raise ValueError("built-in chapter cards cannot be deleted; move them to done, or add your own cards")
    saved = _load_plan_file()
    before = len(saved.get("custom", []))
    saved["custom"] = [c for c in saved.get("custom", []) if c.get("id") != card_id]
    saved.get("cards", {}).pop(card_id, None)
    if len(saved["custom"]) == before:
        raise ValueError(f"no such card {card_id}")
    _plan_path().write_text(json.dumps(saved, indent=1))
    return plan()
