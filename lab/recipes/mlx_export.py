#!/usr/bin/env python3
"""Export a post-trained Mac model as one GGUF file (llama.cpp, Ollama, LM Studio), and load it into Ollama.

Steps: fuse the LoRA adapter into the model and dequantize (mlx_lm fuse), convert to GGUF with llama.cpp's
converter at a pinned release (fetched once to ~/.cache/cortex), delete the fused copy, then, when Ollama runs on
this Mac, create an Ollama model from the file and ask it one question as a check.

    python mlx_export.py --model mlx-community/Qwen3-0.6B-4bit --adapter .../adapters.safetensors --name job-id [--outtype q8_0]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlx_post import result, status  # noqa: E402  (same protocol helpers, no torch needed)

LLAMA_TAG = "b11313"  # 2026-10-01
LLAMA = Path.home() / ".cache" / "cortex" / f"llama.cpp-{LLAMA_TAG}"
# the converter's own pins (requirements/requirements-convert_hf_to_gguf.txt at that tag)
CONVERT_ENV = ["uv", "run", "--quiet", "--python", "3.12", "--with", "torch==2.11.0", "--with", "numpy~=2.2.6", "--with", "sentencepiece>=0.1.98,<0.3.0",
               "--with", "transformers==4.57.6", "--with", "protobuf>=4.21.0,<5.0.0", "--with", str(LLAMA / "gguf-py"), "python"]


def fetch_converter() -> None:
    if (LLAMA / "convert_hf_to_gguf.py").exists():
        return
    status(phase="setup", msg=f"fetching the llama.cpp converter ({LLAMA_TAG}, once)")
    LLAMA.parent.mkdir(parents=True, exist_ok=True)
    tgz = LLAMA.parent / f"llama.cpp-{LLAMA_TAG}.tar.gz"
    urllib.request.urlretrieve(f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{LLAMA_TAG}", tgz)
    LLAMA.mkdir(exist_ok=True)
    top = f"llama.cpp-{LLAMA_TAG}"
    subprocess.run(["tar", "-xzf", str(tgz), "-C", str(LLAMA), "--strip-components=1", f"{top}/convert_hf_to_gguf.py", f"{top}/conversion", f"{top}/gguf-py"], check=True)
    tgz.unlink()


def run(cmd: list[str], what: str, cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
        raise SystemExit(f"{what} failed: " + " | ".join(tail)[-600:])


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
        return shutil.which("ollama") is not None
    except OSError:
        return False


def ask_ollama(tag: str, prompt: str) -> str:
    body = {"model": tag, "stream": False, "messages": [{"role": "user", "content": prompt}], "options": {"num_predict": 200}}
    for extra in ({"think": False}, {}):  # models without a thinking template reject "think"
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body | extra).encode(), headers={"content-type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"]
        except urllib.error.HTTPError:
            continue
    return "(Ollama gave no reply)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="adapters.safetensors from a post-training job")
    ap.add_argument("--name", required=True, help="job id; names the file and the Ollama model")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "f16", "bf16"])
    ap.add_argument("--prompt", default="Tom has 3 apples and buys 4 more. How many apples does he have?")
    a = ap.parse_args()
    t0 = time.time()
    out = Path.home() / "Cortex" / "exports" / a.name
    out.mkdir(parents=True, exist_ok=True)
    fused = out / "fused"
    fetch_converter()

    status(phase="fuse", msg="merging the adapter into the model")
    shutil.rmtree(fused, ignore_errors=True)
    run([sys.executable, "-m", "mlx_lm", "fuse", "--model", a.model, "--adapter-path", str(Path(a.adapter).parent), "--save-path", str(fused), "--dequantize"], "fuse")

    # fuse re-saves the tokenizer with transformers 5 (extra_special_tokens as a list), which the converter's pinned
    # transformers 4.57 cannot read; the model's own tokenizer files are unchanged by training, so put them back
    import fnmatch
    from huggingface_hub import snapshot_download
    keep = ["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json", "chat_template*", "*.model"]
    src = Path(snapshot_download(a.model, allow_patterns=keep))
    for f in src.iterdir():  # the cached snapshot also holds the quantized weights and config: copy tokenizer files only
        if f.is_file() and any(fnmatch.fnmatch(f.name, p) for p in keep):
            shutil.copy(f, fused / f.name)

    status(phase="convert", msg=f"writing GGUF ({a.outtype})")
    gguf = out / f"{a.name}-{a.outtype}.gguf"
    run(CONVERT_ENV + [str(LLAMA / "convert_hf_to_gguf.py"), str(fused), "--outfile", str(gguf), "--outtype", a.outtype], "GGUF conversion", cwd=LLAMA)
    shutil.rmtree(fused, ignore_errors=True)
    info = {"gguf": str(gguf), "gb": round(gguf.stat().st_size / 1e9, 3), "outtype": a.outtype, "converter": f"llama.cpp {LLAMA_TAG}"}

    if ollama_up():
        status(phase="ollama", msg="loading the file into Ollama and asking one question")
        tag = "cortex-" + "".join(c if c.isalnum() else "-" for c in a.name.lower())
        (out / "Modelfile").write_text(f"FROM ./{gguf.name}\n")
        run(["ollama", "create", tag, "-f", "Modelfile"], "ollama create", cwd=out)
        info.update(ollama=tag, ollama_prompt=a.prompt, ollama_reply=ask_ollama(tag, a.prompt)[:600])
    else:
        info["ollama"] = None
    result(mode="export", model=a.model, adapter=a.adapter, seconds=round(time.time() - t0, 1), **info)


if __name__ == "__main__":
    main()
