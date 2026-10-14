#!/usr/bin/env python3
"""Mac-native inference, fine-tuning and post-training with MLX (Apple Silicon, M1 16 GB and up).

    python mlx_post.py --mode infer --model mlx-community/Qwen3-0.6B-4bit --prompt "What is LoRA?"
    python mlx_post.py --mode sft  --task arm --iters 60                 # LoRA on simulator-verified arm programs
    python mlx_post.py --mode grpo --task arm --iters 20 --resume <run>/adapters/adapters.safetensors
    python mlx_post.py --mode dpo  --data path/with/train.jsonl          # rows: {prompt, chosen, rejected}

Engines: mlx-lm (inference) and mlx-lm-lora (SFT, DPO, GRPO and more on MLX). With --task arm the data is built from
the arm simulator's tasks (scenes 1-30 train, 31-40 held out) and GRPO's reward is the simulator itself: 1 when the
hidden goal holds, 0.2 for a valid but wrong program, 0 otherwise.

Memory check: unified memory is shared with the OS, so the recipe estimates weights + activations from the model name
and refuses jobs above CORTEX_MLX_MAX_GB (default 4) instead of pushing the Mac into swap; run those on the 5090.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from common import metric, result, status  # noqa: E402
except Exception:  # common.py imports torch; the MLX env has none, so speak the protocol directly
    def _p(tag, **kw):
        print(tag + " " + json.dumps(kw), flush=True)

    def metric(**kw):
        _p("METRIC", **kw)

    def status(**kw):
        _p("STATUS", **kw)

    def result(**kw):
        _p("RESULT", **kw)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def estimate_gb(model: str, mode: str, seq: int, group: int, batch: int = 2) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)B", model)
    params_b = float(m.group(1)) if m else 8.0               # unknown size: assume big, so the check stays safe
    bits = 4 if re.search(r"4.?bit|mxfp4|nvfp4", model, re.I) else 8 if re.search(r"8.?bit", model, re.I) else 16
    weights = params_b * bits / 8
    # activations and KV grow with batch, sequence and width; calibrated on Qwen3-0.6B-4bit SFT, batch 2, 1,024 tokens:
    # measured peak 3.4 GB (2026-10-14). Generous on purpose: the check exists to keep the Mac out of swap.
    act = 0.4 + 2.0 * batch * seq / 1024 * params_b ** 0.5
    if mode == "grpo":
        act *= max(1, group)
    return round(weights * 1.15 + (0.3 if mode == "infer" else act), 2)


def arm_data(out: Path, mode: str) -> Path:
    """Arm tasks as MLX training data: SFT = chat with the oracle program; GRPO = prompt, system (scene), answer (task id)."""
    sys.path.insert(0, str(HERE / "armft"))
    sys.path.insert(0, str(HERE.parent.parent / "server"))   # arm_world lives in the Cortex server
    import armft
    import arm_world as W
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": armft.tasks(range(1, 31)), "valid": armft.tasks(range(31, 33))}
    for name, tasks in splits.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for t in tasks:
                if mode == "grpo":
                    row = {"prompt": t["instruction"], "system": armft.SYS.format(scene=W.describe(t["scene"])), "answer": t["id"]}
                else:
                    row = {"messages": armft.messages(t) + [{"role": "assistant", "content": armft.target(t)}]}
                f.write(json.dumps(row) + "\n")
    return out


def stream(cmd: list[str], env: dict) -> int:
    """Run an MLX trainer and turn its progress lines into METRIC lines."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    it = 0
    for raw in p.stdout:
        line = ANSI.sub("", raw).strip()
        if not line:
            continue
        m = re.search(r"Iter (\d+): loss ([\d.]+).*?peak_mem ([\d.]+)GB", line)   # "Iter 3: loss 1.008, lr ..., peak_mem 2.7GB"
        if m:
            it = int(m.group(1))
            metric(step=it, loss=float(m.group(2)), peak_gb=float(m.group(3)))
            continue
        m = re.search(r"Iter (\d+): Val loss ([\d.]+)", line)
        if m:
            metric(step=int(m.group(1)), val_loss=float(m.group(2)))
            continue
        m = re.search(r"Iter (\d+)", line)
        if m:
            it = int(m.group(1))
        m = re.search(r"Total Rewards:\s*μ=([-\d.]+)", line)
        if m:
            metric(step=it, reward=float(m.group(1)))
            continue
        if re.search(r"val|Peak|Trainable|Error|Traceback|Saved", line, re.I):
            print(line, flush=True)
    return p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["infer", "sft", "dpo", "grpo"], default="infer")
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--task", choices=["arm", "none"], default="none")
    ap.add_argument("--data", default="")
    ap.add_argument("--prompt", default="In one sentence, what is LoRA?")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--group", type=int, default=2, help="GRPO samples per prompt")
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", default="", help="adapter .safetensors to start from (e.g. SFT before GRPO)")
    ap.add_argument("--out", default="~/Cortex/mlx")
    a = ap.parse_args()

    cap = float(os.environ.get("CORTEX_MLX_MAX_GB", "4"))
    need = estimate_gb(a.model, a.mode, a.seq, a.group, 1 if a.mode in ("infer", "grpo") else a.batch)
    status(phase="check", msg=f"estimated {need} GB for {a.mode} on {a.model}; local cap {cap} GB")
    if need > cap:
        raise SystemExit(f"estimated {need} GB is above the local cap of {cap} GB: use a smaller or 4-bit model, or run it on the 5090")

    out = Path(a.out).expanduser() / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.time()

    if a.mode == "infer":
        from mlx_lm import generate, load
        model, tok = load(a.model, adapter_path=a.resume or None)
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True)
        reply = generate(model, tok, prompt=text, max_tokens=200)
        import mlx.core as mx
        result(mode="infer", model=a.model, reply=reply, seconds=round(time.time() - t0, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 3), estimate_gb=need)
        return

    data = Path(a.data) if a.data else (arm_data(out / "data", a.mode) if a.task == "arm" else None)
    if data is None:
        raise SystemExit("give --data (a folder with train.jsonl and valid.jsonl) or --task arm")
    (out / "config.yaml").write_text("fuse: false\n")  # keep only the adapter; fusing de-quantizes and saves a full model copy
    cmd = [sys.executable, "-m", "mlx_lm_lora.train", "--config", str(out / "config.yaml"), "--model", a.model, "--train", "--train-mode", a.mode, "--data", str(data),
           "--iters", str(a.iters), "--batch-size", str(a.batch), "--num-layers", str(a.layers), "--max-seq-length", str(a.seq),
           "--learning-rate", str(a.lr), "--adapter-path", str(out / "adapters"), "--steps-per-report", "1", "--steps-per-eval", str(max(a.iters, 1))]
    if a.mode == "sft":
        cmd += ["--mask-prompt"]
    if a.mode == "grpo":
        cmd += ["--group-size", str(a.group), "--max-completion-length", str(a.max_completion)]
        if a.task == "arm":
            cmd += ["--reward-functions-file", str(HERE / "mlx_rewards.py"), "--reward-functions", "arm_sim"]
    if a.resume:
        cmd += ["--resume-adapter-file", a.resume]
    status(phase="train", msg=" ".join(cmd[2:8]) + " …")
    code = stream(cmd, env)
    if code:
        raise SystemExit(f"mlx-lm-lora exited with {code}")
    result(mode=a.mode, model=a.model, task=a.task, iters=a.iters, adapter=str(out / "adapters" / "adapters.safetensors"),
           seconds=round(time.time() - t0, 1), estimate_gb=need)


if __name__ == "__main__":
    main()
