#!/usr/bin/env python3
"""Compress a post-trained model safely: one-shot ALPS pruning, with the model's own refusals in the calibration set.

Runs on the GPU box (ssh executor) with the PRUNE-1/PRUNE-2 code (armft/prune1.py). Safety on = PRUNE-2's recipe:
32 of the 128 calibration examples are harmful requests followed by the dense model's own replies; it kept refusal at
0.93-0.98 at 60% sparsity where plain ALPS fell to 0.00-0.18, with the skill unchanged (Lab Notebook PRUNE-2).
Reports skill and refusal before (dense) and after, and the same verdict as post-training.

    python compress.py --sparsity 0.6 [--no-safety] [--seed 0]
Note: unstructured sparsity sets weights to zero; it shrinks a sparse export but does not speed up dense inference.
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
STUDENT = Path.home() / "prune1" / "student"   # Qwen3-1.7B post-trained on the arm task (PRUNE-1 setup)
OUT = Path.home() / "prune1" / "results"


def pids(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split() if int(p) != os.getpid()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.6)
    ap.add_argument("--no-safety", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not STUDENT.exists():
        raise SystemExit("No post-trained student on the 5090 yet (expected ~/prune1/student).")
    if pids("armft.py run") or pids("prune1.py"):
        raise SystemExit("The 5090 is busy with another training job. Try again when it finishes.")
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    apps = [(int(x), int(y)) for x, y in (l.split(",") for l in q.strip().splitlines() if l.strip())]
    server = set(pids("python serve_student.py"))
    free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    free += sum(m for p, m in apps if p in server)
    if free < 12000 or len([p for p, _ in apps if p not in server]) >= 2:
        raise SystemExit(f"The 5090 is busy: {free // 1000} GB free after pausing the planner. Compression needs about 12 GB. Try again later.")
    for f in ("armft.py", "prune1.py", "arm_world.py"):
        shutil.copy(HERE / f, HOME / f)
    for p in server:
        os.kill(p, 15)
    safe = not a.no_safety
    status(phase="compress", msg=f"ALPS {int(a.sparsity * 100)}% sparsity, safety calibration {'on' if safe else 'off'}; planner paused")
    tag = "compress_safe" if safe else "compress_plain"
    cmd = [PY, "prune1.py", str(STUDENT), "alps", str(a.sparsity), str(a.seed), "--tag", tag] + (["--refusal-mix", "32"] if safe else [])
    env = {**os.environ, "PRUNE1_OUT": str(OUT), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, cwd=HOME, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if line.startswith('{"layer"'):
                d = json.loads(line)
                metric(step=d["layer"], seconds=d["s"])
            elif "Error" in line or "Traceback" in line:
                print(line.rstrip(), flush=True)
        if p.wait():
            raise SystemExit("compression failed")
    finally:
        subprocess.Popen(f"cd {HOME} && setsid nohup {PY} serve_student.py > serve_student.log 2>&1 < /dev/null &", shell=True)
    after = json.loads((OUT / f"{tag}_alps_s{a.sparsity}_c{a.seed}.json").read_text())
    dense_f = OUT / "full_dense_s0.0_c0.json"
    dense = json.loads(dense_f.read_text()) if dense_f.exists() else {}
    keys = ("plan_success", "advbench_refusal", "unsafe_arm_safe_refusal", "benign_refusal")
    result(mode="compress", sparsity=a.sparsity, safety=safe, seconds=round(time.time() - t0),
           before={k: dense.get(k) for k in keys}, after={k: after.get(k) for k in keys}, model="Qwen3-1.7B arm planner")


if __name__ == "__main__":
    main()
