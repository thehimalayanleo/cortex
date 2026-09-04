"""Ask a model for Lean 4 proofs and let the Lean compiler grade them (Lab 14: verifiable reasoning).

What this teaches
  * the cleanest reward there is: a proof either compiles or it does not. No
    judge, no rubric, no answer extraction beyond finding the tactic block.
  * the mechanics of driving Lean from Python: write `theorem <name> <statement> := by <tactics>`
    into a scratch file, run `lake env lean file.lean` inside a Lake project
    (so Mathlib and the project's toolchain are on the path), read the exit
    code and the error lines.
  * pass@1 with an interval. One sample per theorem at the chosen temperature;
    the fraction that compiles is pass@1 and the Wilson interval says how
    little a handful of theorems can tell you.

How to run
  smoke (no model): compiles a hand-written correct proof and a hand-written
  wrong proof for each built-in theorem and checks that Lean accepts the
  first and rejects the second. Uses `lake env lean` inside --lean-project if
  given, else plain `lean` on PATH (the built-in theorems need only core Lean),
  else prints a message and exits 0:
    python lab/recipes/lean_eval.py --smoke
    python lab/recipes/lean_eval.py --smoke --lean-project ~/github-repos/sfp/lean-sfp
  real (RTX 5090):
    python lab/recipes/lean_eval.py --lean-project ~/github-repos/sfp/lean-sfp --theorems my_theorems.jsonl --model deepseek-ai/DeepSeek-Prover-V2-7B
  --theorems rows: {"name": ..., "statement": "(n : Nat) : n + 0 = n"} (the part after the name).
  Run `lake build` (or at least `lake env true`) once inside the project before
  the first eval: on a fresh project `lake env` first downloads the toolchain
  and clones Mathlib, and that setup output would be counted as a failure.
  needs: pip install transformers   (plus a Lean 4 toolchain via elan, and a Lake project with Mathlib for non-core statements)

The default model is DeepSeek-Prover-V2-7B, a Lean 4 prover released by DeepSeek;
any causal LM works with --model, it will just prove less.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402

BUILTIN = [
    {"name": "add_zero_nat", "statement": "(n : Nat) : n + 0 = n", "good": "simp", "bad": "exact rfl.symm.symm.trans (by decide)"},
    {"name": "and_intro_pq", "statement": "(p q : Prop) (hp : p) (hq : q) : p ∧ q", "good": "exact ⟨hp, hq⟩", "bad": "exact hp"},
    {"name": "add_comm_nat", "statement": "(a b : Nat) : a + b = b + a", "good": "omega", "bad": "rfl"},
]


def build_parser():
    p = C.base_parser("lean_eval", __doc__.split("\n")[0])
    p.add_argument("--theorems", default=None, help="JSONL of {name, statement}; built-in examples when omitted")
    p.add_argument("--lean-project", default=None, help="Lake project dir; `lake env lean` runs there")
    p.add_argument("--import-mathlib", action="store_true", help="prefix the scratch file with `import Mathlib`")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-Prover-V2-7B")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--timeout", type=int, default=300, help="seconds per Lean check")
    return p


# --------------------------------------------------------------------------- Lean


def lean_command(project: str | None) -> list[str] | None:
    if project:
        if not shutil.which("lake"):
            return None
        return ["lake", "env", "lean"]
    if shutil.which("lean"):
        return ["lean"]
    return None


def check_proof(name: str, statement: str, tactics: str, project: str | None, import_mathlib: bool, timeout: int) -> tuple[bool, str]:
    """Write the theorem to a scratch .lean file and compile it. Returns (ok, compiler output)."""
    cmd = lean_command(project)
    if cmd is None:
        raise RuntimeError("no Lean toolchain found")
    header = "import Mathlib\n\n" if import_mathlib else ""
    src = f"{header}theorem {name} {statement} := by\n  " + tactics.strip().replace("\n", "\n  ") + "\n"
    scratch_dir = os.path.join(project, ".cortex_scratch") if project else tempfile.mkdtemp(prefix="lean_eval_")
    os.makedirs(scratch_dir, exist_ok=True)
    path = os.path.join(scratch_dir, f"{re.sub(r'[^A-Za-z0-9_]', '_', name)}.lean")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        r = subprocess.run(cmd + [path], cwd=project or scratch_dir, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        ok = r.returncode == 0 and "error" not in out.lower() and "sorry" not in tactics
        return ok, out
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


# --------------------------------------------------------------------------- model


def extract_tactics(text: str) -> str:
    """Take the tactic block the model wrote after `:= by`, stopping at a code fence or a new declaration."""
    text = text.split("```")[0]
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(theorem|lemma|example|def|#)", line) and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip() or "sorry"


def load_model(name: str, device):
    transformers = C.require("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).to(device).eval()
    return model, tok


@torch.no_grad()
def propose(model, tok, name, statement, import_mathlib, temperature, max_new, device) -> str:
    header = "import Mathlib\n\n" if import_mathlib else ""
    prompt = f"Complete the following Lean 4 code:\n\n```lean4\n{header}theorem {name} {statement} := by\n"
    enc = tok(prompt, return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0, temperature=temperature or None,
                         pad_token_id=tok.eos_token_id)
    return extract_tactics(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


def main():
    args = build_parser().parse_args()
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    theorems = C.read_jsonl(args.theorems) if args.theorems else BUILTIN
    if lean_command(args.lean_project) is None:
        C.log("no Lean toolchain: install elan (https://github.com/leanprover/elan) or pass --lean-project with lake on PATH")
        C.status("done", "skipped: no lean")
        C.result(skipped="no lean toolchain", n_theorems=len(theorems))
        return
    C.log(f"lean command: {' '.join(lean_command(args.lean_project))} (project={args.lean_project or 'none, core Lean only'})")

    if args.smoke:
        C.status("check", "hand-written good and bad proofs")
        good_ok = bad_rejected = 0
        rows = []
        for i, t in enumerate(theorems):
            g, gout = check_proof(t["name"], t["statement"], t["good"], args.lean_project, args.import_mathlib, args.timeout)
            b, bout = check_proof(t["name"] + "_bad", t["statement"], t["bad"], args.lean_project, args.import_mathlib, args.timeout)
            good_ok += g
            bad_rejected += not b
            C.log(f"  {t['name']:<14} good proof {'compiles' if g else 'FAILED: ' + gout[:120]!r} | bad proof {'rejected' if not b else 'ACCEPTED (unexpected)'}")
            C.metric(i + 1, good_ok=good_ok, bad_rejected=bad_rejected)
            rows.append({"name": t["name"], "good_ok": g, "bad_rejected": not b, "good_output": gout, "bad_output": bout})
        C.write_jsonl(os.path.join(args.out, "smoke_results.jsonl"), rows)
        C.status("done", "")
        C.result(n_theorems=len(theorems), good_compiled=good_ok, bad_rejected=bad_rejected, all_as_expected=(good_ok == len(theorems) == bad_rejected))
        return

    device = C.pick_device(args.device)
    C.status("load", args.model)
    model, tok = load_model(args.model, device)
    C.status("prove", f"{len(theorems)} theorems, temperature {args.temperature}")
    passed, rows = 0, []
    for i, t in enumerate(theorems):
        tactics = propose(model, tok, t["name"], t["statement"], args.import_mathlib, args.temperature, args.max_new, device)
        ok, out = check_proof(t["name"], t["statement"], tactics, args.lean_project, args.import_mathlib, args.timeout)
        passed += ok
        C.log(f"  {t['name']:<20} {'PASS' if ok else 'fail'}  proof={tactics[:80]!r}")
        C.metric(i + 1, passed=passed, pass_at_1_running=passed / (i + 1))
        rows.append({"name": t["name"], "statement": t["statement"], "proof": tactics, "ok": ok, "lean_output": out[:2000]})
    p, lo, hi = C.wilson_interval(passed, len(theorems))
    path = os.path.join(args.out, "proofs.jsonl")
    C.write_jsonl(path, rows)
    C.status("done", f"wrote {path}")
    C.result(model=args.model, n_theorems=len(theorems), passed=passed, pass_at_1=p, ci95_lo=lo, ci95_hi=hi, proofs=path)


if __name__ == "__main__":
    main()
