"""NVIDIA Nemotron 3 Nano (30B-A3B, FP8) behind an OpenAI-compatible endpoint on one Modal H100.

The arm planner uses this until a Nebius Token Factory key is set. Serving follows the model card: vLLM with the
model's own reasoning parser, FP8 KV cache. The endpoint requires the key in the Modal secret `cortex-nemotron`
(VLLM_API_KEY). It scales to zero after 10 idle minutes; the first call after that waits for a cold start.

    modal deploy serve/nemotron_modal.py        # prints the URL; requests go to <url>/v1
"""
import subprocess

import modal

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
PARSER = f"https://huggingface.co/{MODEL}/resolve/main/nano_v3_reasoning_parser.py"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl")
    .uv_pip_install("vllm==0.31.0", "huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "MAX_JOBS": "16"})
    .run_commands(f"curl -sfL {PARSER} -o /root/nano_v3_reasoning_parser.py")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
fi_cache = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)  # the FP8 MoE kernel JIT-builds once
app = modal.App("cortex-nemotron")


@app.function(image=image, gpu="H100", cpu=16.0, memory=64 * 1024, timeout=60 * 60, scaledown_window=10 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/root/.cache/flashinfer": fi_cache},
              secrets=[modal.Secret.from_name("cortex-nemotron")])
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    subprocess.Popen(
        f"vllm serve {MODEL} --served-model-name {MODEL} --host 0.0.0.0 --port 8000 --max-num-seqs 8 "
        "--max-model-len 32768 --trust-remote-code --kv-cache-dtype fp8 "
        "--reasoning-parser-plugin /root/nano_v3_reasoning_parser.py --reasoning-parser nano_v3 "
        "--api-key $VLLM_API_KEY",
        shell=True,
    )
