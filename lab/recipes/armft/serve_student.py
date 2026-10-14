"""OpenAI-compatible endpoint for the fine-tuned arm planner: Nemotron 3 Nano 4B + a LoRA from armft.py (ARM-FT-1 best seed by default).
Listens on the Tailscale address only, behind the key in ~/nemotron/api.key. Stop with: pkill -f serve_student.py"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
CURRENT = Path.home() / "armft/current_adapter"  # written by the arm_train recipe; default is ARM-FT-1's best seed
ADAPTER = Path(CURRENT.read_text().strip()) if CURRENT.exists() else Path.home() / "armft/results/adapter_4_full"
NAME = "nemotron-3-nano-4b-arm-lora"
print("adapter", ADAPTER, flush=True)
KEY = (Path.home() / "nemotron/api.key").read_text().strip()
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}), str(ADAPTER)).eval()
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _auth(self):
        if self.headers.get("authorization") != f"Bearer {KEY}":
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if self._auth():
            self._send(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})

    def do_POST(self):
        if not self._auth():
            return
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = tok(text, return_tensors="pt", add_special_tokens=False).to(0)
        with lock, torch.no_grad():
            g = model.generate(**x, max_new_tokens=min(int(req.get("max_tokens", 700)), 900), do_sample=False)
        out = tok.decode(g[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        n_in, n_out = x["input_ids"].shape[1], g.shape[1] - x["input_ids"].shape[1]
        self._send(200, {"id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "model": NAME,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}})

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("100.86.175.54", 8002), H).serve_forever()
