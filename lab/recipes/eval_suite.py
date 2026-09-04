"""Evaluate a model: lm-eval-harness tasks plus a custom generative exact-match eval (Lab 09: evaluation).

What this teaches
  * two kinds of eval. Log-likelihood tasks (most of lm-eval-harness) score
    fixed choices by their probability; generative tasks let the model write
    an answer and compare it with a reference. The second is what you care
    about for a tuned model and it is the one people get wrong: decoding
    settings, answer extraction and normalization all change the number.
  * exact match with SQuAD-style normalization (lowercase, strip punctuation
    and articles, collapse whitespace), and why the bootstrap confidence
    interval matters: with n items, the standard error of an accuracy p is
    about sqrt(p (1 - p) / n), so 100 items cannot separate 0.72 from 0.78.
  * where an LLM judge would plug in (a stub with a clear TODO; the recipe
    never fabricates judge scores).

How to run
  smoke (CPU, offline): train the minimal GPT for --steps on synthetic
  arithmetic, then run the generative exact-match eval on held-out arithmetic
  prompts and print accuracy with a bootstrap 95% CI:
    python lab/recipes/eval_suite.py --smoke --steps 300
  real (RTX 5090):
    python lab/recipes/eval_suite.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks gsm8k,hellaswag --limit 200
    python lab/recipes/eval_suite.py --model out/sft_lora/adapter --custom-jsonl my_eval.jsonl
  --custom-jsonl rows are {"prompt": ..., "answer": ...}; add --chat to wrap the
  prompt in the model's chat template.
  needs: pip install lm-eval transformers   (peft if --model is an adapter directory)
"""
from __future__ import annotations

import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("eval_suite", __doc__.split("\n")[0])
    p.add_argument("--model", default=None, help="HF model id, local model dir, or a peft adapter dir")
    p.add_argument("--tasks", default=None, help="comma-separated lm-eval task names")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="lm-eval: items per task")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--custom-jsonl", default=None, help='rows of {"prompt", "answer"}')
    p.add_argument("--chat", action="store_true", help="wrap custom prompts in the chat template")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--judge", choices=["none", "stub"], default="none", help="LLM judge hook (stub only, see llm_judge)")
    p.add_argument("--ckpt", default=None, help="smoke: evaluate this pretrain_nano/sft checkpoint instead of training one")
    return p


# --------------------------------------------------------------------------- scoring


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def first_line(s: str) -> str:
    """Generative answers are compared on their first non-empty line, so trailing chatter does not count."""
    for line in s.splitlines():
        if line.strip():
            return line.strip()
    return s.strip()


def llm_judge(prompt: str, gold: str, pred: str):
    """Hook for an LLM-as-judge score in [0, 1] or None when no judge is configured.

    TODO: call a judge model here (for example a local vLLM server) with a rubric
    that asks whether `pred` answers `prompt` as well as `gold`, and return a float.
    Until that exists this returns None and the judge column is reported as
    "not implemented"; nothing is fabricated.
    """
    return None


def custom_eval(rows: list[dict], generate_fn, judge: bool, seed: int, log_examples: int = 5) -> dict:
    """rows: [{prompt, answer}], generate_fn: list[str] -> list[str]. Returns the summary dict."""
    preds = generate_fn([r["prompt"] for r in rows])
    ems = [exact_match(first_line(p), r["answer"]) for p, r in zip(preds, rows)]
    for r, p in list(zip(rows, preds))[:log_examples]:
        C.log(f"  prompt={r['prompt']!r} pred={first_line(p)!r} gold={r['answer']!r}")
    mean, lo, hi = C.bootstrap_ci(ems, seed=seed)
    out = {"n": len(rows), "exact_match": mean, "ci95_lo": lo, "ci95_hi": hi}
    if judge:
        scores = [llm_judge(r["prompt"], r["answer"], p) for r, p in zip(rows, preds)]
        valid = [s for s in scores if s is not None]
        out["judge"] = sum(valid) / len(valid) if valid else "not implemented"
    return out


# --------------------------------------------------------------------------- smoke


def smoke(args):
    device = C.pick_device(args.device)
    lines = C.arithmetic_lines(400, seed=3)
    held, train = lines[:80], lines[80:]
    if args.ckpt:
        model, tok, _ = C.load_checkpoint(args.ckpt, device)
    else:
        tok = C.CharTokenizer()
        model = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=64)).to(device)
        ids = torch.tensor(tok.encode("\n".join(train) + "\n"), dtype=torch.long)
        opt = C.make_adamw(model, 3e-3)
        gen = torch.Generator().manual_seed(args.seed)
        C.status("train", f"{args.steps} steps on {len(train)} arithmetic lines")
        for step in range(args.steps):
            lr = C.lr_at(step, args.steps, 3e-3, 10, 0.1)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = C.random_windows(ids, 32, 48, gen)
            loss = C.lm_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            C.metric(step, loss=loss.item(), lr=lr)
    rows = [{"prompt": line.split("=")[0] + "= ", "answer": line.split("=")[1].strip()} for line in held]
    C.write_jsonl(os.path.join(args.out, "smoke_eval.jsonl"), rows)

    def generate_fn(prompts):
        ids, mask = C.left_pad([tok.encode(p) for p in prompts], tok.pad_id)
        out = C.generate(model, ids.to(device), 8, greedy=True, eos_id=tok.eos_id, attn_mask=mask.to(device))
        return [tok.decode(row[ids.shape[1]:].tolist()) for row in out]

    C.status("eval", f"exact match on {len(rows)} held-out arithmetic prompts")
    summary = custom_eval(rows, generate_fn, args.judge == "stub", args.seed)
    C.log(f"exact match {summary['exact_match']:.3f}  95% bootstrap CI [{summary['ci95_lo']:.3f}, {summary['ci95_hi']:.3f}]  n={summary['n']}")
    C.status("done", "")
    C.result(**{f"custom_{k}": v for k, v in summary.items()}, steps=args.steps)


# --------------------------------------------------------------------------- real


def load_hf(model_path: str, device):
    transformers = C.require("transformers")
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if os.path.isfile(adapter_cfg):
        peft = C.require("peft")
        import json

        base = json.load(open(adapter_cfg))["base_model_name_or_path"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        model = peft.PeftModel.from_pretrained(model, model_path).merge_and_unload()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer


def real(args):
    if not args.model:
        raise SystemExit("real mode needs --model")
    device = C.pick_device(args.device)
    summary = {}
    if args.tasks:
        lm_eval = C.require("lm_eval", "lm-eval")
        C.status("lm_eval", args.tasks)
        model_args = f"pretrained={args.model},dtype={'bfloat16' if device.type == 'cuda' else 'float32'}"
        if os.path.isfile(os.path.join(args.model, "adapter_config.json")):
            import json

            base = json.load(open(os.path.join(args.model, "adapter_config.json")))["base_model_name_or_path"]
            model_args = f"pretrained={base},peft={args.model},dtype={'bfloat16' if device.type == 'cuda' else 'float32'}"
        res = lm_eval.simple_evaluate(model="hf", model_args=model_args, tasks=args.tasks.split(","), num_fewshot=args.num_fewshot,
                                      limit=args.limit, batch_size=args.batch_size)
        for i, (task, metrics) in enumerate(res["results"].items()):
            nums = {k.replace(",", "_"): v for k, v in metrics.items() if isinstance(v, (int, float))}
            C.metric(i, task=task, **nums)
            summary[task] = nums
            C.log(f"{task}: {nums}")
    if args.custom_jsonl:
        rows = C.read_jsonl(args.custom_jsonl)
        model, tokenizer = load_hf(args.model, device)

        @torch.no_grad()
        def generate_fn(prompts):
            outs = []
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i: i + args.batch_size]
                if args.chat:
                    batch = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch]
                enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                C.metric(len(outs), generated=len(outs))
            return outs

        C.status("custom", f"{len(rows)} generative items")
        summary["custom"] = custom_eval(rows, generate_fn, args.judge == "stub", args.seed)
        C.log(f"custom: {summary['custom']}")
    C.status("done", "")
    flat = {f"{t}_{k}": v for t, m in summary.items() for k, v in m.items()}
    C.result(model=args.model, **flat)


def main():
    args = build_parser().parse_args()
    if args.steps is None:
        args.steps = 300
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if args.smoke else real)(args)


if __name__ == "__main__":
    main()
