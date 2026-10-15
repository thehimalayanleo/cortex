#!/usr/bin/env python3
"""Train the arm planner: LoRA on simulator-verified programs for NVIDIA Nemotron 3 Nano 4B, then serve it.

Runs on the GPU box (ssh executor). The training code is armft/armft.py next to this file (ARM-FT-1 / ARM-SAFE-1 in
the Lab Notebook); it needs a venv with the Mamba kernels (ARM_TRAIN_PYTHON, default ~/armft/venv/bin/python).
Steps: stop the planner server to free GPU memory, train one seed, evaluate on held-out scenes 31-40 (planning,
AdvBench refusal, unsafe arm requests), point the server at the new adapter and restart it. Results stay in
~/armft/results (outside the synced recipes folder).

    python arm_train.py --seed 6 [--safety] [--max-steps 3]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import metric, result, status  # noqa: E402

HERE = Path(__file__).resolve().parent / "armft"
HOME = Path.home() / "armft"
PY = os.path.expanduser(os.environ.get("ARM_TRAIN_PYTHON", "~/armft/venv/bin/python"))


def pids(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split() if int(p) != os.getpid()]


def restart_server():
    subprocess.Popen(f"cd {HOME} && setsid nohup {PY} serve_student.py > serve_student.log 2>&1 < /dev/null &", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--safety", action="store_true", help="add the ARM-SAFE-1 refusal examples")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    if pids("armft.py run") or pids("prune1.py"):
        raise SystemExit("The 5090 is busy with another training job. Try again when it finishes.")
    # GPU check before pausing anything: training needs about 15 GB; the planner server's memory comes back when it pauses
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    apps = [(int(a), int(b)) for a, b in (l.split(",") for l in q.strip().splitlines() if l.strip())]
    server = set(pids("python serve_student.py"))
    free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    free += sum(m for p, m in apps if p in server)
    others = [p for p, _ in apps if p not in server]
    if free < 15000 or len(others) >= 2:
        raise SystemExit(f"The 5090 is busy: {free // 1000} GB free after pausing the planner, {len(others)} other GPU jobs. Training needs about 15 GB and at most one other job. Try again later.")
    tag = ("appsafe" if a.safety else "app") + (f"smoke{a.max_steps}" if a.max_steps else "")
    HOME.mkdir(exist_ok=True)
    for f in ("armft.py", "serve_student.py", "arm_world.py"):  # the box runs exactly the synced code
        shutil.copy(HERE / f, HOME / f)
    for p in pids("python serve_student.py"):
        os.kill(p, 15)
    status(phase="train", msg=f"seed {a.seed}, safety {'on' if a.safety else 'off'}; planner server paused")
    cmd = [PY, "armft.py", "run", "--seed", str(a.seed), "--tag", tag] + (["--safety"] if a.safety else []) + (["--max-steps", str(a.max_steps)] if a.max_steps else [])
    env = {**os.environ, "ARMFT_OUT": str(HOME / "results"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, cwd=HOME, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith('{"step"'):
                d = json.loads(line)
                metric(step=d["step"], loss=d["loss"])
            elif "Error" in line or "Traceback" in line:
                print(line, flush=True)
        if p.wait():
            raise SystemExit(f"training failed (exit {p.returncode})")
        run = json.loads((HOME / "results" / f"run_{a.seed}_{tag}.json").read_text())
        base = json.loads((HOME / "results" / "base.json").read_text())
        (HOME / "current_adapter").write_text(str(HOME / "results" / f"adapter_{a.seed}_{tag}"))
    finally:
        restart_server()
    keys = ("plan_success", "plan_valid", "advbench_refusal", "unsafe_arm_refusal")
    result(seed=a.seed, safety=a.safety, steps=run["steps"], examples=run["examples"], seconds=round(time.time() - t0),
           tuned={k: run.get(k) for k in keys + ("unsafe_arm_safe_refusal", "benign_people_refusal")},
           base={k: base.get(k) for k in keys}, per_family=run["per_family"], adapter=f"adapter_{a.seed}_{tag}")


if __name__ == "__main__":
    main()
