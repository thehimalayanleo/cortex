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
# Mac models: each one passed a smoke test (one reply, 5 SFT steps, measured peak) on 2026-10-15. "top" ones show first.
MODELS = [
    {"id": "mlx-community/Qwen3-0.6B-4bit", "label": "Qwen3 0.6B", "maker": "Alibaba", "engine": "mlx", "top": True},
    {"id": "mlx-community/Llama-3.2-1B-Instruct-4bit", "label": "Llama 3.2 1B", "maker": "Meta", "engine": "mlx", "top": True},
    {"id": "mlx-community/gemma-3-1b-it-4bit", "label": "Gemma 3 1B", "maker": "Google", "engine": "mlx"},
    {"id": "mlx-community/LFM2-1.2B-4bit", "label": "LFM2 1.2B", "maker": "Liquid AI", "engine": "mlx"},
    {"id": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", "label": "Qwen2.5 Coder 1.5B", "maker": "Alibaba", "engine": "mlx"},
    {"id": "mlx-community/Qwen3-1.7B-4bit", "label": "Qwen3 1.7B", "maker": "Alibaba", "engine": "mlx", "top": True},
    {"id": "mlx-community/granite-3.3-2b-instruct-4bit", "label": "Granite 3.3 2B", "maker": "IBM", "engine": "mlx"},
    {"id": "mlx-community/Llama-3.2-3B-Instruct-4bit", "label": "Llama 3.2 3B", "maker": "Meta", "engine": "mlx"},
    {"id": "mlx-community/SmolLM3-3B-4bit", "label": "SmolLM3 3B", "maker": "Hugging Face", "engine": "mlx"},
    {"id": "mlx-community/Qwen3-4B-4bit", "label": "Qwen3 4B", "maker": "Alibaba", "engine": "mlx"},
    {"id": "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit", "label": "Nemotron 3 Nano 4B", "maker": "NVIDIA", "engine": "mlx", "top": True},
    {"id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "label": "Nemotron 3 Nano 4B (full)", "maker": "NVIDIA", "engine": "5090", "tasks": ["arm"], "top": True},
]
# settings to try, largest first (batch, tokens): the first that fits the local cap runs; longer examples get cut
FITS = {"sft": [(2, 1024), (1, 1024), (1, 512)], "grpo": [(2, 768), (1, 768), (1, 512)]}


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


def fit(model: str, method: str, task: str) -> dict[str, Any]:
    """The settings a Mac run uses: the largest batch and length whose estimated peak is under the cap."""
    mp = _mlx_post()
    est = mp.estimate_gb
    key = "custom" if task.startswith("custom:") else task
    shortest = min(1024, mp.TASK_TOKENS.get(key, 0))  # a shorter length cuts the answer off the end of each example
    for b, n in FITS[method]:
        if n < shortest:
            continue
        gb = est(model, method, n, 2, b, task=key)
        if gb <= _cap():
            return {"batch": b, "seq": n, "gb": gb, "fits": True}
    b, n = next(((b, n) for b, n in reversed(FITS[method]) if n >= shortest), FITS[method][0])
    return {"batch": b, "seq": n, "gb": est(model, method, n, 2, b, task=key), "fits": False}


def overview() -> dict[str, Any]:
    models = []
    for m in MODELS:
        if m["engine"] == "mlx":
            st = {t: {meth["id"]: fit(m["id"], meth["id"], t) for meth in METHODS} for t in [x["id"] for x in TASKS] + ["custom"]}
            f = st["arm"]
            models.append({**m, "estimate_gb": {k: v["gb"] for k, v in f.items()}, "fits": {k: v["fits"] for k, v in f.items()}, "settings": f, "by_task": st})
        else:
            models.append({**m, "estimate_gb": None, "fits": None})
    return {"tasks": TASKS, "methods": METHODS, "models": models, "cap_gb": _cap(), "jobs": jobs(), "datasets": datasets(), "promoted": promoted()}


def jobs() -> list[dict[str, Any]]:
    out = []
    for j in reversed(_jobs()[-40:]):
        r = runs.read_run(j["run"], tail=6, max_metrics=300) or {}
        x = {**j, "status": r.get("status", "missing"), "result": r.get("result"), "metrics": r.get("metrics", []), "log": r.get("log", [])}
        if j.get("export_run"):
            e = runs.read_run(j["export_run"], tail=4, max_metrics=0) or {}
            x["export"] = {"run": j["export_run"], "status": e.get("status", "missing"), "result": e.get("result"), "log": e.get("log", [])}
        out.append(x)
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
        f = fit(model, method, task)
        if not f["fits"]:
            raise ValueError(f"{m['label']} needs about {f['gb']} GB for {method.upper()}, above this Mac's cap of {_cap()} GB")
        args = f"--mode {method} --task {task_arg} --model {model} --iters {int(iters)} --batch {f['batch']} --seq {f['seq']}" + (" --safety" if safety else "")
        if method == "grpo":
            args += " --max-completion 400"
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


# ---------------------------------------------------------------- after a job: promote, export, compare

def _job(job_id: str) -> dict[str, Any]:
    job = next((j for j in jobs() if j["id"] == job_id), None)
    if not job:
        raise ValueError("no such job")
    return job


def _adapter(job: dict[str, Any]) -> str:
    adapter = str((job.get("result") or {}).get("adapter") or "")
    if not adapter.endswith(".safetensors") or not Path(adapter).exists():
        raise ValueError("this job has no Mac adapter (5090 jobs live in the Physical tab)")
    return adapter


def _promoted_path() -> Path:
    return _path().parent / "promoted.json"


def promoted() -> dict[str, Any]:
    p = _promoted_path()
    return json.loads(p.read_text()) if p.exists() else {}


GATE = 3  # lab/recipes/mlx_post.py GATE: verdicts from an older gate cannot promote


def promote(job_id: str) -> dict[str, Any]:
    """Make a job's model the one in use for its task. Only a job whose verdict passed can be promoted."""
    job = _job(job_id)
    r = job.get("result") or {}
    if job["status"] != "done":
        raise ValueError("the job has not finished")
    v = r.get("verdict") or {}
    if not v.get("promote"):
        raise ValueError("the checks did not pass: " + "; ".join(v.get("reasons") or ["no verdict"]))
    if v.get("gate", 1) < GATE:
        raise ValueError("this job was checked before the lab tested refusal of ordinary questions; train it again to check it with today's tests")
    pr = promoted()
    pr[job["task"]] = {"job": job_id, "model": job["model"], "adapter": _adapter(job), "method": job["method"],
                       "score": (r.get("tuned") or {}).get("task_score"), "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _promoted_path().write_text(json.dumps(pr, indent=1))
    return pr


def export_start(job_id: str, outtype: str = "q8_0") -> dict[str, Any]:
    """GGUF file of the base model with the job's adapter merged in (lab/recipes/mlx_export.py); then into Ollama."""
    import shlex
    job = _job(job_id)
    adapter = _adapter(job)
    if outtype not in ("q8_0", "f16"):
        raise ValueError("outtype is q8_0 or f16")
    run = runs.start("mlx_export", f"--model {job['model']} --adapter {shlex.quote(adapter)} --name {job_id} --outtype {outtype}", "local", origin="posttrain-export")
    js = _jobs()
    for j in js:
        if j["id"] == job_id:
            j["export_run"] = run["id"]
    _path().write_text(json.dumps(js, indent=1))
    return run


# compare chat: one local mlx_lm server loads the model (and adapter) each request asks for, one at a time;
# it stops after 10 idle minutes so the memory comes back
CHAT_PORT = int(os.environ.get("CORTEX_MLX_CHAT_PORT", "8790"))
_chat: dict[str, Any] = {"proc": None, "last": 0.0, "lock": None}


def _chat_server() -> str:
    import shlex
    import subprocess
    import threading
    import urllib.request
    if _chat["lock"] is None:
        _chat["lock"] = threading.Lock()
    def up() -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{CHAT_PORT}/v1/models", timeout=2)
            return True
        except OSError:
            return False
    with _chat["lock"]:
        _chat["last"] = time.time()
        p = _chat["proc"]
        if (p is None or p.poll() is not None) and up():  # an earlier Cortex process started it; nothing would stop it
            subprocess.run(["pkill", "-f", f"mlx_lm server --host 127.0.0.1 --port {CHAT_PORT}"])
            for _ in range(20):
                if not up():
                    break
                time.sleep(0.5)
        if p is None or p.poll() is not None:
            py = shlex.split(os.environ.get("CORTEX_MLX_CHAT_PYTHON") or "uv run --python 3.12 --with mlx-lm==0.32.0 python")
            log = open(_path().parent / "chat_server.log", "ab")
            _chat["proc"] = subprocess.Popen(py + ["-m", "mlx_lm", "server", "--host", "127.0.0.1", "--port", str(CHAT_PORT),
                                                  "--chat-template-args", '{"enable_thinking": false}', "--max-tokens", "600"],
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

            def reap() -> None:
                while _chat["proc"] is not None and _chat["proc"].poll() is None:
                    time.sleep(30)
                    if time.time() - _chat["last"] > 600:
                        _chat["proc"].terminate()
            threading.Thread(target=reap, daemon=True).start()
        for _ in range(240):
            if up():
                return f"http://127.0.0.1:{CHAT_PORT}"
            if _chat["proc"].poll() is not None:
                raise ValueError("the local model server stopped; see posttrain/chat_server.log in the vault")
            time.sleep(0.5)
        raise ValueError("the local model server did not start in 2 minutes")


def chat_targets() -> list[dict[str, Any]]:
    """What compare chat can talk to: every Mac base model, and every finished Mac job's trained model."""
    out = [{"id": m["id"], "label": f"{m['label']} (base)", "model": m["id"], "adapter": None} for m in MODELS if m["engine"] == "mlx"]
    pr = {v["job"] for v in promoted().values()}
    for j in jobs():
        a = str((j.get("result") or {}).get("adapter") or "")
        if j["status"] == "done" and a.endswith(".safetensors") and Path(a).exists():
            name = next((m["label"] for m in MODELS if m["id"] == j["model"]), j["model"].split("/")[-1])
            ok = ((j.get("result") or {}).get("verdict") or {}).get("promote")
            out.append({"id": f"job:{j['id']}", "label": f"{name} + {j['method'].upper()} on {j['task'].split('/')[-1][:24]} · {j['id']}" + (" · in use" if j["id"] in pr else "" if ok else " · checks failed"),
                        "model": j["model"], "adapter": str(Path(a).parent), "job": j["id"]})
    return out


def chat(target: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    import urllib.request
    t = next((x for x in chat_targets() if x["id"] == target), None)
    if not t:
        raise ValueError("unknown model to chat with")
    msgs = [{"role": m["role"], "content": str(m["content"])[:8000]} for m in messages[-20:] if m.get("role") in ("user", "assistant", "system")]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("the last message must be yours")
    base = _chat_server()
    body = {"model": t["model"], "messages": msgs, "max_tokens": 600, "temperature": 0.2}
    if t["adapter"]:
        body["adapters"] = t["adapter"]
    t0 = time.time()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    _chat["last"] = time.time()
    txt = r["choices"][0]["message"].get("content") or ""
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    return {"reply": txt.strip(), "seconds": round(time.time() - t0, 1), "target": t["label"]}
