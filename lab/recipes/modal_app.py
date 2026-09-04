"""Run any lab recipe on a Modal GPU and keep its --out artifacts in a Modal volume.

    modal run lab/recipes/modal_app.py --recipe pretrain_nano --args "--steps 500"
    modal run lab/recipes/modal_app.py --recipe sft_lora --args "--max-samples 2000 --steps 200" --gpu H100
    modal run lab/recipes/modal_app.py --recipe kernel_bench --args "--seqs 1024,4096,8192"

What it does
  * builds an image with torch, TRL, Unsloth, datasets, sentence-transformers,
    lm-eval, tiktoken, peft and Pillow; the recipes directory is added to the
    image, so the remote function sees the same files you edited locally
  * runs `python /root/recipes/<recipe>.py <args> --out /vol/<recipe>` inside the
    container, streaming stdout (METRIC / STATUS / RESULT lines included) back to
    your terminal line by line
  * commits the volume at the end so checkpoints and JSONL artifacts persist;
    fetch them with `modal volume get cortex-lab-out <recipe>/... .`

Notes
  * --gpu picks the card (H100 default; A10G or L4 are cheaper for the small
    recipes); --timeout is in seconds
  * Hugging Face gated models: create a Modal secret named `huggingface` with
    HF_TOKEN and it is attached automatically if it exists
  * the image is large because of Unsloth and lm-eval; the first build takes a
    while and is cached afterwards
"""
from __future__ import annotations

import os
import subprocess
import sys

import modal

RECIPES_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_NAME = "cortex-lab-out"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch", "numpy", "tiktoken", "datasets", "transformers", "accelerate", "peft", "trl",
        "sentence-transformers", "lm-eval", "Pillow", "einops", "bitsandbytes",
    )
    .pip_install("unsloth")
    .add_local_dir(RECIPES_DIR, remote_path="/root/recipes", ignore=["out", "__pycache__", "*.pt", "*.npy"])
)

app = modal.App("cortex-lab")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _secrets():
    try:
        return [modal.Secret.from_name("huggingface")]
    except Exception:
        return []


@app.function(image=image, gpu="H100", timeout=6 * 60 * 60, volumes={"/vol": volume}, secrets=_secrets())
def run_recipe(recipe: str, args: str) -> int:
    """Run one recipe as a subprocess and stream its stdout back. Returns the exit code."""
    script = f"/root/recipes/{recipe}.py"
    if not os.path.exists(script):
        print(f"no such recipe: {script}", flush=True)
        return 2
    out_dir = f"/vol/{recipe}"
    cmd = [sys.executable, "-u", script] + args.split() + ["--out", out_dir]
    print("running: " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME="/vol/hf_cache")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd="/root/recipes")
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    volume.commit()
    print(f"exit code {code}; artifacts under volume {VOLUME_NAME}:{out_dir}", flush=True)
    return code


@app.local_entrypoint()
def main(recipe: str = "pretrain_nano", args: str = "", gpu: str = "H100", timeout: int = 6 * 60 * 60):
    fn = run_recipe
    if gpu != "H100" or timeout != 6 * 60 * 60:
        fn = run_recipe.with_options(gpu=gpu, timeout=timeout)
    code = fn.remote(recipe, args)
    if code != 0:
        raise SystemExit(code)
