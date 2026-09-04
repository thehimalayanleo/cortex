"""Pipelines: the training pie. A pipeline is a DAG of stages; each stage is one lab recipe run (see runs.py) whose
arguments can reference artifacts printed in an earlier stage's RESULT line ("{pretrain:checkpoint}"), so a
checkpoint or a corpus path flows forward with `--init <ckpt>` style flags.

Layout inside the vault:
  pipelines/<id>/pipeline.json   id, template, title, executor, smoke, status, stages[] (each with its run id and status)
  pipelines/<id>/traces_all.jsonl the collector export handed to the data stage (local path, or scp'd to the GPU box)

Statuses
  pipeline: created | running | paused | done | failed
  stage:    pending | running | done | failed

The runner is a daemon thread (start_daemon) that every few seconds re-reads every non-final pipeline from disk,
refreshes stage statuses from the runs' meta.json, and starts every pending stage whose dependencies are done.
Because the state lives on disk and the run ids are recorded before a run starts, a server restart is harmless:
runs.mark_stale() marks in-flight runs failed, the next tick marks their stages failed, and retry() re-queues them.

Templates (TEMPLATES) hold, per stage, a smoke argument string and a real one. Placeholders:
  {out}           the pipeline's output folder (relative to the recipes' working directory on the executor)
  {traces}        the collector export file for the data stage
  {stage:field}   a field of that stage's RESULT (a path or a number)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import runs, traces, vault

TEMPLATES: dict[str, dict[str, Any]] = {
    "reasoning-nano": {
        "title": "Reasoning nano: data, pretrain, midtrain, SFT, RL, eval",
        "doc": "A small reasoning model trained end to end from your Traces plus a synthetic reasoning set with verifiable answers. "
               "Smoke: the character-level minimal GPT on a CPU in a few minutes. Real: the same stack scaled up on the 5090 "
               "(GPT-2 tokens, 6 layers, 384 wide; edit the stage arguments in pipeline.json before starting).",
        "stages": [
            {"name": "data", "recipe": "data_prep", "deps": [],
             "smoke": "--smoke --out {out}/data --traces-jsonl {traces}",
             "real": "--out {out}/data --traces-jsonl {traces} --n-reason 20000 --tokenizer gpt2 --hf-dataset roneneldan/TinyStories --max-samples 20000"},
            {"name": "pretrain", "recipe": "pretrain_nano", "deps": ["data"],
             "smoke": "--smoke --corpus {data:corpus} --out {out}/pretrain --steps 300 --seq-len 256",
             "real": "--corpus {data:corpus} --out {out}/pretrain --steps 3000 --n-layer 6 --d-model 384 --n-head 6 --seq-len 512 --batch 32"},
            {"name": "midtrain", "recipe": "midtrain", "deps": ["pretrain", "data"],
             "smoke": "--smoke --ckpt {pretrain:checkpoint} --text-a {data:corpus} --text-b {data:reason_text} --mix a=0.4,b=0.6 --steps 150 --out {out}/midtrain",
             "real": "--ckpt {pretrain:checkpoint} --text-a {data:corpus} --text-b {data:reason_text} --mix a=0.4,b=0.6 --steps 1000 --cooldown-frac 0.3 --out {out}/midtrain"},
            {"name": "sft", "recipe": "sft_lora", "deps": ["midtrain", "data"],
             "smoke": "--ckpt {midtrain:checkpoint} --pairs-jsonl {data:sft} --steps 300 --max-new 64 --out {out}/sft",
             "real": "--ckpt {midtrain:checkpoint} --pairs-jsonl {data:sft} --steps 2000 --batch 64 --max-new 96 --out {out}/sft"},
            {"name": "rl", "recipe": "grpo_reason", "deps": ["sft", "data"],
             "smoke": "--ckpt {sft:checkpoint} --tasks-jsonl {data:reason_train} --eval-jsonl {data:reason_heldout} --steps 30 --group 4 --batch 4 --out {out}/rl",
             "real": "--ckpt {sft:checkpoint} --tasks-jsonl {data:reason_train} --eval-jsonl {data:reason_heldout} --steps 300 --group 8 --batch 16 --max-new 96 --out {out}/rl"},
            {"name": "eval", "recipe": "eval_suite", "deps": ["rl", "midtrain", "data"],
             "smoke": "--ckpt {rl:checkpoint} --baseline-ckpt {midtrain:checkpoint} --custom-jsonl {data:reason_heldout} --max-new 64 --out {out}/eval",
             "real": "--ckpt {rl:checkpoint} --baseline-ckpt {midtrain:checkpoint} --custom-jsonl {data:reason_heldout} --max-new 96 --out {out}/eval"},
        ],
    },
    "embed-mine": {
        "title": "Embed mine: embed the vault, then contrastive fine-tuning on its pairs",
        "doc": "Chunk and embed your notes and papers, mine {query, positive} pairs from what the vault already implies, "
               "then fine-tune an encoder with InfoNCE on those pairs and report recall@1 on a held-out slice.",
        "stages": [
            {"name": "embed", "recipe": "embed_vault", "deps": [],
             "smoke": "--smoke --out {out}/embed",
             "real": "--vault {vault} --out {out}/embed"},
            {"name": "contrastive", "recipe": "embed_contrastive", "deps": ["embed"],
             "smoke": "--smoke --pairs-jsonl {embed:pairs} --steps 200 --out {out}/contrastive",
             "real": "--pairs-jsonl {embed:pairs} --model nomic-ai/nomic-embed-text-v1.5 --matryoshka 768,512,256,128,64 --out {out}/contrastive"},
        ],
    },
}

FINAL = {"done", "failed"}
_lock = threading.RLock()
_daemon: threading.Thread | None = None
TICK_S = 3.0


def _dir() -> Path:
    d = vault.VAULT / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _path(pid: str) -> Path:
    return _dir() / pid / "pipeline.json"


def _load(pid: str) -> dict[str, Any] | None:
    p = _path(pid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save(p: dict[str, Any]) -> None:
    p["updated"] = _now()
    d = _dir() / p["id"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "pipeline.json.tmp"
    tmp.write_text(json.dumps(p, indent=2))
    tmp.replace(d / "pipeline.json")


# ---------------------------------------------------------------- templates and creation

def templates() -> list[dict[str, Any]]:
    return [{"name": k, "title": v["title"], "doc": v["doc"], "stages": [{"name": s["name"], "recipe": s["recipe"], "deps": s["deps"]} for s in v["stages"]]}
            for k, v in TEMPLATES.items()]


def create(template: str, executor: str = "local", smoke: bool = True, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Instantiate a template. overrides: {"out": "...", "args": {"<stage>": "<argument string>"}, "title": "..."}."""
    if template not in TEMPLATES:
        raise ValueError(f"no such template: {template} (have {', '.join(TEMPLATES)})")
    if executor not in ("local", "ssh", "modal"):
        raise ValueError("executor must be local, ssh, or modal")
    ex = runs.executors()
    if not ex[executor]["available"]:
        raise ValueError(f"the {executor} executor is not available here: {ex[executor]['note']}")
    ov = overrides or {}
    t = TEMPLATES[template]
    pid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    arg_ov = ov.get("args") or {}
    stages = []
    for s in t["stages"]:
        args = str(arg_ov.get(s["name"]) or (s["smoke"] if smoke else s["real"]))
        stages.append({"name": s["name"], "recipe": s["recipe"], "deps": list(s["deps"]), "args_template": args, "args": None,
                       "run_id": None, "status": "pending", "started": None, "ended": None, "error": None, "attempts": 0})
    p = {"id": pid, "template": template, "title": str(ov.get("title") or t["title"]), "executor": executor, "smoke": bool(smoke),
         "out": str(ov.get("out") or f"out/pipelines/{pid}"), "status": "created", "created": _now(), "updated": _now(), "stages": stages, "error": None}
    with _lock:
        _save(p)
    return read(pid) or p


# ---------------------------------------------------------------- the collector export handed to the data stage

def _traces_file(p: dict[str, Any]) -> str:
    """Write the 'all' export next to the pipeline; on the ssh executor also copy it to the box and return the remote path."""
    local = _dir() / p["id"] / "traces_all.jsonl"
    rows = traces.export("all")["rows"]
    with local.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if p["executor"] == "ssh":
        host = os.environ.get("CORTEX_SSH_HOST", "").strip()
        remote_dir = f"~/cortex-lab/{p['out']}"
        subprocess.run(["ssh", *runs.SSH_OPTS, host, f"mkdir -p {remote_dir}"], check=True, timeout=60)
        scp_opts = [("-P" if o == "-p" else o) for o in runs._ssh_opts()]
        subprocess.run(["scp", *scp_opts, str(local), f"{host}:{remote_dir}/traces_all.jsonl"], check=True, timeout=120)
        return f"{p['out']}/traces_all.jsonl"
    if p["executor"] == "modal":
        return ""  # no file transfer to Modal here; data_prep tolerates a missing file (synthetic only)
    return str(local)


# ---------------------------------------------------------------- resolving arguments from earlier RESULT lines

_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)\}")


def _run_meta(rid: str | None) -> dict[str, Any] | None:
    if not rid:
        return None
    d = runs.runs_dir() / rid
    mp = d / "meta.json"
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return None
    m["last"] = runs._last_metric(d)
    rp = d / "result.json"
    if rp.exists():
        try:
            m["result"] = json.loads(rp.read_text())
        except Exception:
            m["result"] = None
    else:
        m["result"] = None
    return m


def _resolve(p: dict[str, Any], stage: dict[str, Any]) -> str:
    by_name = {s["name"]: s for s in p["stages"]}
    s = stage["args_template"]
    s = s.replace("{out}", p["out"]).replace("{vault}", str(vault.VAULT))
    if "{traces}" in s:
        s = s.replace("{traces}", _traces_file(p) or "/dev/null")

    def sub(m: re.Match) -> str:
        dep, field = m.group(1), m.group(2)
        if dep not in by_name:
            raise ValueError(f"{stage['name']}: unknown stage '{dep}' in {m.group(0)}")
        meta = _run_meta(by_name[dep].get("run_id"))
        res = (meta or {}).get("result") or {}
        if field not in res:
            raise ValueError(f"{stage['name']}: stage '{dep}' printed no '{field}' in its RESULT line (have {', '.join(res) or 'nothing'})")
        return str(res[field])

    return _PLACEHOLDER.sub(sub, s)


# ---------------------------------------------------------------- the runner

def _advance(p: dict[str, Any]) -> bool:
    """One step of the state machine for one pipeline. Returns True when anything changed."""
    changed = False
    by_name = {s["name"]: s for s in p["stages"]}
    for s in p["stages"]:  # refresh running stages from their run
        if s["status"] != "running":
            continue
        m = _run_meta(s["run_id"])
        if m is None:
            s["status"], s["error"], s["ended"] = "failed", "the run folder is gone", _now()
            changed = True
        elif m["status"] == "done":
            s["status"], s["ended"] = "done", m.get("ended") or _now()
            changed = True
        elif m["status"] in ("failed", "stopped"):
            s["status"], s["ended"], s["error"] = "failed", m.get("ended") or _now(), m.get("error") or f"run {m['status']} (exit {m.get('exit')})"
            changed = True
    if p["status"] == "running":
        for s in p["stages"]:
            if s["status"] != "pending" or not all(by_name[d]["status"] == "done" for d in s["deps"]):
                continue
            try:
                s["args"] = _resolve(p, s)
                m = runs.start(s["recipe"], s["args"], p["executor"], origin=f"pipeline:{p['id']}:{s['name']}")
                s.update(run_id=m["id"], status="running", started=m["started"], ended=None, error=None, attempts=s.get("attempts", 0) + 1)
            except Exception as e:
                s.update(status="failed", error=str(e)[:500], ended=_now(), attempts=s.get("attempts", 0) + 1)
            changed = True
        if any(s["status"] == "failed" for s in p["stages"]):
            failed = [s["name"] for s in p["stages"] if s["status"] == "failed"]
            p["status"], p["error"] = "failed", f"stage {failed[0]} failed"
            changed = True
        elif all(s["status"] == "done" for s in p["stages"]):
            p["status"], p["error"] = "done", None
            changed = True
    return changed


def tick() -> int:
    """Advance every non-final pipeline once. Returns how many pipelines changed."""
    n = 0
    with _lock:
        for d in _dir().iterdir():
            p = _load(d.name)
            if not p or p["status"] in FINAL or p["status"] == "created":
                continue
            if _advance(p):
                _save(p)
                n += 1
    return n


def _loop() -> None:
    while True:
        try:
            tick()
        except Exception as e:  # keep the runner alive; a bad pipeline file must not stop the others
            print("pipeline runner:", e)
        time.sleep(TICK_S)


def start_daemon() -> None:
    global _daemon
    with _lock:
        if _daemon is None or not _daemon.is_alive():
            _daemon = threading.Thread(target=_loop, daemon=True, name="pipeline-runner")
            _daemon.start()


# ---------------------------------------------------------------- control

def start(pid: str) -> dict[str, Any]:
    """Start (from created) or resume (from paused / failed after a retry) and advance once right away."""
    with _lock:
        p = _load(pid)
        if not p:
            raise ValueError(f"no such pipeline {pid}")
        if p["status"] == "done":
            raise ValueError("this pipeline is finished; create a new one")
        if p["status"] == "failed" and any(s["status"] == "failed" for s in p["stages"]):
            raise ValueError("a stage failed; retry it (POST /retry/<stage>) instead of start")
        p["status"], p["error"] = "running", None
        _advance(p)
        _save(p)
    start_daemon()
    return read(pid) or p


def pause(pid: str) -> dict[str, Any]:
    """Stop launching new stages; the stage that is running keeps running."""
    with _lock:
        p = _load(pid)
        if not p:
            raise ValueError(f"no such pipeline {pid}")
        if p["status"] == "running":
            p["status"] = "paused"
            _save(p)
    return read(pid) or p


def retry(pid: str, stage: str) -> dict[str, Any]:
    """Re-queue a failed stage (and every stage downstream of it) and resume."""
    with _lock:
        p = _load(pid)
        if not p:
            raise ValueError(f"no such pipeline {pid}")
        by_name = {s["name"]: s for s in p["stages"]}
        if stage not in by_name:
            raise ValueError(f"no stage {stage} in this pipeline")
        if by_name[stage]["status"] == "running":
            raise ValueError(f"stage {stage} is running")
        todo = {stage}
        grew = True
        while grew:  # downstream closure
            grew = False
            for s in p["stages"]:
                if s["name"] not in todo and any(d in todo for d in s["deps"]):
                    todo.add(s["name"])
                    grew = True
        for name in todo:
            by_name[name].update(status="pending", run_id=None, args=None, started=None, ended=None, error=None)
        p["status"], p["error"] = "running", None
        _advance(p)
        _save(p)
    start_daemon()
    return read(pid) or p


def delete(pid: str, delete_runs: bool = False) -> bool:
    with _lock:
        p = _load(pid)
        if not p:
            return False
        for s in p["stages"]:
            if s.get("run_id"):
                runs.stop(s["run_id"])
                if delete_runs:
                    runs.delete_run(s["run_id"])
        shutil.rmtree(_dir() / pid, ignore_errors=True)
    return True


# ---------------------------------------------------------------- reading

def _elapsed(s: dict[str, Any], m: dict[str, Any] | None) -> float | None:
    start = (m or {}).get("started") or s.get("started")
    if not start:
        return None
    end = (m or {}).get("ended") or s.get("ended")
    try:
        t0 = time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%S"))
        t1 = time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%S")) if end else time.time()
        return max(0.0, t1 - t0)
    except Exception:
        return None


def read(pid: str) -> dict[str, Any] | None:
    p = _load(pid)
    if not p:
        return None
    out = dict(p)
    stages = []
    data_result = None
    for s in p["stages"]:
        m = _run_meta(s.get("run_id"))
        st = dict(s)
        st["run_status"] = (m or {}).get("status")
        st["last"] = (m or {}).get("last")
        st["result"] = (m or {}).get("result")
        st["elapsed_s"] = _elapsed(s, m)
        stages.append(st)
        if s["recipe"] == "data_prep" and st["result"]:
            data_result = st["result"]
    out["stages"] = stages
    out["data"] = {"sources": data_result.get("sources"), "total_tokens": data_result.get("total_tokens"), "tokenizer": data_result.get("tokenizer")} if data_result and isinstance(data_result.get("sources"), dict) else None
    done = sum(1 for s in stages if s["status"] == "done")
    out["progress"] = {"done": done, "total": len(stages)}
    final = stages[-1]["result"] if stages and stages[-1]["status"] == "done" else None
    out["final"] = final
    return out


def list_pipelines(limit: int = 50) -> list[dict[str, Any]]:
    out = []
    for d in sorted(_dir().iterdir(), reverse=True):
        p = _load(d.name)
        if not p:
            continue
        out.append({k: p[k] for k in ("id", "template", "title", "executor", "smoke", "status", "created", "updated", "error")}
                   | {"progress": {"done": sum(1 for s in p["stages"] if s["status"] == "done"), "total": len(p["stages"])},
                      "current": next((s["name"] for s in p["stages"] if s["status"] == "running"), None)})
        if len(out) >= limit:
            break
    return out
