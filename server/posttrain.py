"""The post-training lab: pick a task (with its verifier), a method and a model; it runs where it fits.

Where: this Mac (MLX, lab/recipes/mlx_post.py) when the job's estimated peak fits the local cap, otherwise the 5090
(today: the arm task, lab/recipes/arm_train.py). Every job trains, scores base and tuned on held-out items with the
task's verifier, runs the refusal check, and ends with a verdict: promote, or blocked with the reason.
Jobs are kept in the vault at posttrain/jobs.json; each points at an ordinary run (log, metrics, result).
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from . import arm, runs, vault

TASKS = [
    {"id": "code", "label": "Python functions", "verifier": "unit tests (MBPP), run in a capped process", "safety": False},
    {"id": "arm", "label": "Arm planning", "verifier": "arm simulator (reach, collisions, grasps, hidden goal)", "safety": True},
    {"id": "gsm8k", "label": "Grade-school math", "verifier": "exact final number", "safety": False},
]
METHODS = [
    {"id": "sft", "label": "SFT", "doc": "supervised fine-tuning on verified answers (LoRA)"},
    {"id": "grpo", "label": "GRPO", "doc": "RL: sample answers, the verifier scores them, reinforce the better ones"},
]
MODELS = [
    {"id": "mlx-community/Qwen3-0.6B-4bit", "label": "Qwen3 0.6B (4-bit)", "engine": "mlx"},
    {"id": "mlx-community/Qwen3-1.7B-4bit", "label": "Qwen3 1.7B (4-bit)", "engine": "mlx"},
    {"id": "mlx-community/Qwen3-4B-4bit", "label": "Qwen3 4B (4-bit)", "engine": "mlx"},
    {"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "label": "Nemotron 3 Nano 4B", "engine": "5090", "tasks": ["arm"]},
]


def _mlx_post():
    spec = importlib.util.spec_from_file_location("mlx_post", runs.RECIPES / "mlx_post.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap() -> float:
    return float(os.environ.get("CORTEX_MLX_MAX_GB", "4"))


def _path():
    p = vault.VAULT / "posttrain"
    p.mkdir(parents=True, exist_ok=True)
    return p / "jobs.json"


def _jobs() -> list[dict[str, Any]]:
    p = _path()
    return json.loads(p.read_text()) if p.exists() else []


def overview() -> dict[str, Any]:
    est = _mlx_post().estimate_gb
    models = []
    for m in MODELS:
        if m["engine"] == "mlx":
            gb = {meth["id"]: est(m["id"], meth["id"], 768 if meth["id"] == "grpo" else 1024, 2, 1 if meth["id"] == "grpo" else 2) for meth in METHODS}
            models.append({**m, "estimate_gb": gb, "fits": {k: v <= _cap() for k, v in gb.items()}})
        else:
            models.append({**m, "estimate_gb": None, "fits": None})
    return {"tasks": TASKS, "methods": METHODS, "models": models, "cap_gb": _cap(), "jobs": jobs(), "datasets": datasets()}


def jobs() -> list[dict[str, Any]]:
    out = []
    for j in reversed(_jobs()[-40:]):
        r = runs.read_run(j["run"], tail=6, max_metrics=300) or {}
        out.append({**j, "status": r.get("status", "missing"), "result": r.get("result"), "metrics": r.get("metrics", []), "log": r.get("log", [])})
    return out


def start(task: str, method: str, model: str, iters: int = 60, safety: bool = True, seed: int = 7) -> dict[str, Any]:
    m = next((x for x in MODELS if x["id"] == model), None)
    if not m:
        raise ValueError(f"unknown model {model}")
    custom = task.startswith("custom:")
    if custom:
        if method != "sft":
            raise ValueError("uploaded data has no checker, so only SFT is available")
        if not (_data_dir() / f"{task[7:]}.jsonl").exists():
            raise ValueError("no such dataset")
    elif task not in {t["id"] for t in TASKS} or method not in {x["id"] for x in METHODS}:
        raise ValueError("unknown task or method")
    if m["engine"] == "5090":
        if task != "arm" or method != "sft":
            raise ValueError("on the 5090 the lab runs arm SFT today; pick an MLX model for other tasks")
        run = arm.train_start(seed=seed, safety=safety)
        where = "5090"
    else:
        task_arg = f"custom:{_data_dir() / (task[7:] + '.jsonl')}" if custom else task
        args = f"--mode {method} --task {task_arg} --model {model} --iters {int(iters)}" + (" --safety" if safety else "")
        if method == "grpo":
            args += " --seq 768 --max-completion 400"
        run = runs.start("mlx_post", args, "local", origin="posttrain")
        where = "this Mac (MLX)"
    job = {"id": time.strftime("%Y%m%d-%H%M%S"), "run": run["id"], "task": task, "method": method, "model": model,
           "where": where, "iters": iters, "safety": bool(safety), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    js = _jobs()
    js.append(job)
    _path().write_text(json.dumps(js, indent=1))
    return job


def try_job(job_id: str, prompt: str) -> dict[str, Any]:
    """Ask the same question to the base model and to the job's trained adapter (this Mac, MLX)."""
    import shlex
    import subprocess
    job = next((j for j in jobs() if j["id"] == job_id), None)
    if not job:
        raise ValueError("no such job")
    adapter = (job.get("result") or {}).get("adapter")
    if not adapter or not str(adapter).endswith(".safetensors"):
        raise ValueError("this job has no Mac adapter to try (the 5090 arm planner is in the Physical tab)")
    py = shlex.split(os.environ.get("CORTEX_MLX_PYTHON") or "uv run --python 3.12 --with mlx-lm-lora==5.8.6 --with mlx-lm==0.32.0 --with datasets python")
    out = {}
    for name, extra in (("before", []), ("after", ["--resume", adapter])):
        cmd = py + [str(runs.RECIPES / "mlx_post.py"), "--mode", "infer", "--task", job["task"], "--model", job["model"], "--prompt", prompt, *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            raise ValueError(f"inference failed: {(r.stderr or r.stdout)[-300:]}")
        out[name] = json.loads(line[7:]).get("reply", "")
    if job["task"] in ("arm", "gsm8k"):  # score both answers when the prompt is a task item we can check
        out["note"] = "Answers come from the same prompt. The task checker scores new items in the result, not this question."
    return out


def preview(task: str, model: str, adapter: str = "") -> dict[str, Any]:
    """Start a small Mac run that shows the data: SFT rows as training sees them, and RL groups scored by the verifier."""
    if task not in {t["id"] for t in TASKS}:
        raise ValueError("unknown task")
    m = next((x for x in MODELS if x["id"] == model), None)
    if not m or m["engine"] != "mlx":
        model = "mlx-community/Qwen3-0.6B-4bit"  # the preview samples on this Mac
    args = f"--mode preview --task {task} --model {model} --group 4 --group-prompts 3" + (f" --resume {adapter}" if adapter else "")
    return runs.start("mlx_post", args, "local", origin="posttrain-preview")


def compress_start(sparsity: float = 0.6, safety: bool = True) -> dict[str, Any]:
    """Compress the post-trained Qwen3-1.7B arm planner on the 5090 (lab/recipes/compress.py); kept as a job."""
    if not os.environ.get("CORTEX_SSH_HOST"):
        raise ValueError("compression runs on the GPU box (CORTEX_SSH_HOST)")
    sp = 0.5 if sparsity < 0.55 else 0.6
    run = runs.start("compress", f"--sparsity {sp}" + ("" if safety else " --no-safety"), "ssh", origin="posttrain")
    job = {"id": time.strftime("%Y%m%d-%H%M%S"), "run": run["id"], "task": "arm", "method": "compress", "model": "Qwen3-1.7B arm planner",
           "where": "5090", "iters": 0, "safety": bool(safety), "sparsity": sp, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    js = _jobs()
    js.append(job)
    _path().write_text(json.dumps(js, indent=1))
    return job


# ---------------------------------------------------------------- your data: upload, detect the format, normalize to chat rows

def _data_dir():
    d = vault.VAULT / "posttrain" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


PAIRS = [("prompt", "completion"), ("prompt", "response"), ("prompt", "answer"), ("instruction", "output"), ("question", "answer"), ("input", "output")]


def normalize(raw: bytes, filename: str) -> tuple[list[dict[str, Any]], str]:
    """Rows to {"messages": [..., {"role": "assistant", ...}]}; returns (rows, the format that was detected)."""
    import csv
    import io
    text = raw.decode("utf-8", errors="replace")
    if filename.lower().endswith(".csv"):
        rd = list(csv.DictReader(io.StringIO(text)))
        if not rd:
            raise ValueError("the CSV has no rows")
        cols = [c.lower() for c in rd[0].keys()]
        pair = next(((a, b) for a, b in PAIRS if a in cols and b in cols), None) or (list(rd[0].keys())[0].lower(), list(rd[0].keys())[1].lower())
        low = [{k.lower(): v for k, v in r.items()} for r in rd]
        return [{"messages": [{"role": "user", "content": r[pair[0]]}, {"role": "assistant", "content": r[pair[1]]}]} for r in low if r.get(pair[0]) and r.get(pair[1])], f"CSV columns {pair[0]} → {pair[1]}"
    rows, fmt = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if isinstance(r.get("messages"), list) and r["messages"] and r["messages"][-1].get("role") == "assistant":
            rows.append({"messages": [{"role": m["role"], "content": str(m["content"])} for m in r["messages"]]}); fmt = fmt or "chat messages"
            continue
        pair = next(((a, b) for a, b in PAIRS if a in r and b in r), None)
        if not pair:
            raise ValueError(f"unknown row format, keys {sorted(r)[:6]}: use messages, or prompt/completion, instruction/output, question/answer")
        user = str(r[pair[0]]) + (f"\n\n{r['input']}" if pair[0] == "instruction" and r.get("input") else "")
        rows.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": str(r[pair[1]])}]}); fmt = fmt or f"{pair[0]} → {pair[1]}"
    if not rows:
        raise ValueError("no rows found")
    return rows, fmt or "chat messages"


def upload(raw: bytes, filename: str) -> dict[str, Any]:
    import re as _re
    rows, fmt = normalize(raw, filename)
    slug = _re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "data"
    path = _data_dir() / f"{slug}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"name": slug, "rows": len(rows), "format": fmt, "file": filename, "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (_data_dir() / f"{slug}.meta.json").write_text(json.dumps(meta))
    return {**meta, "preview": rows[:3]}


def datasets() -> list[dict[str, Any]]:
    out = []
    for m in sorted(_data_dir().glob("*.meta.json")):
        meta = json.loads(m.read_text())
        rows = [json.loads(l) for l in (_data_dir() / f"{meta['name']}.jsonl").read_text().splitlines()[:3] if l.strip()]
        out.append({**meta, "preview": rows})
    return out
