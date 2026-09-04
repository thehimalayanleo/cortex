"""Build the corpus for the training pie: a mixed pretraining text, a reasoning-only text for mid-training, SFT pairs, DPO pairs, and a held-out reasoning eval, all from your own Traces plus a synthetic reasoning set with verifiable answers (Lab 21).

What this teaches
  * a corpus is a list of sources with a token count each, not a folder of files;
    the RESULT line carries `sources: {name: tokens}` so the Lab can draw the pie
  * the synthetic reasoning set: two-step arithmetic, transitive comparisons and
    modus ponens / modus tollens chains, each with a short reasoning chain and a
    gold answer a program can check (grpo_reason.py and eval_suite.py use it)
  * one line format shared by every later stage:
        q: <question>
        a: <chain> answer: <gold>
    so pretraining sees the format, SFT masks everything before `a: `, RL scores
    what comes after `answer:`, and eval extracts the same span
  * where your own data enters: Traces records with prompt/response become SFT
    rows, chosen/rejected become DPO rows, plain notes become pretraining text

How to run
  smoke (offline, seconds): synthetic sources only unless --traces-jsonl is given
    python lab/recipes/data_prep.py --smoke --out out/data_prep
  real: more reasoning items, GPT-2 token counts, and (optionally) a hub text set
    python lab/recipes/data_prep.py --out out/data_prep --n-reason 20000 --tokenizer gpt2 \
        --traces-jsonl out/traces_all.jsonl --hf-dataset roneneldan/TinyStories --max-samples 20000
  --traces-jsonl is the "all" export of the collector (one JSON record per line);
  the pipeline runner writes it for you. --vault reads traces/*/traces.jsonl
  straight from a vault folder instead.

Outputs under --out (paths are echoed in RESULT):
  corpus.txt          the pretraining mixture, one document per paragraph
  reason.txt          reasoning lines only (domain b for midtrain.py)
  sft.jsonl           {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "source"}
  dpo.jsonl           {"prompt", "chosen", "rejected", "source"}
  reason_train.jsonl  {"question", "prompt", "chain", "answer", "kind"}
  reason_heldout.jsonl  the same shape; never in corpus.txt, sft.jsonl, or reason.txt
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

NAMES = ["ann", "bob", "cat", "dan", "eve", "fay", "gus", "ida", "jon", "kim"]
RULES = [("it rains", "the road is wet"), ("the sun is out", "the grass is dry"), ("the bell rings", "the class ends"),
         ("the oven is on", "the room is warm"), ("the tide is high", "the rocks are hidden"), ("the shop is open", "the light is on")]
ADJ = [("taller", "tallest"), ("older", "oldest"), ("faster", "fastest"), ("heavier", "heaviest")]


def build_parser():
    p = C.base_parser("data_prep", __doc__.split("\n")[0])
    p.add_argument("--vault", default=None, help="a vault folder: reads traces/*/traces.jsonl from it")
    p.add_argument("--traces-jsonl", default=None, help="an 'all' export of the collector (one record per line)")
    p.add_argument("--n-reason", type=int, default=None, help="synthetic reasoning items (smoke default 600, real 6000)")
    p.add_argument("--heldout-frac", type=float, default=0.15)
    p.add_argument("--tokenizer", choices=["char", "gpt2"], default=None, help="how tokens are counted (smoke: char)")
    p.add_argument("--hf-dataset", default=None, help="real only: a hub text dataset to add as a 'stories' source")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-trace-chars", type=int, default=200000)
    p.add_argument("--max-item-chars", type=int, default=2000, help="longest trace text kept as one document")
    return p


# --------------------------------------------------------------------------- synthetic reasoning with verifiable answers


def prompt_of(question: str) -> str:
    """The exact prompt string every later stage feeds the model."""
    return f"q: {question}\na: "


def make_reasoning(n: int, seed: int) -> list[dict]:
    """n items cycling through three kinds. Every item has a chain and a gold answer a program can check."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 3
        if kind == 0:  # two-step arithmetic
            a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 9)
            op1, op2 = rng.choice(["+", "-"]), rng.choice(["+", "-", "*"])
            mid = a + b if op1 == "+" else a - b
            if op2 == "+":
                ans = mid + c
            elif op2 == "-":
                ans = mid - c
            else:
                ans = mid * c
            q = f"what is {a} {op1} {b} {op2} {c}"
            chain = f"{a} {op1} {b} = {mid}. {mid} {op2} {c} = {ans}."
            out.append({"kind": "arith", "question": q, "chain": chain, "answer": str(ans)})
        elif kind == 1:  # transitive comparison: x > y, y > z, who is the most?
            x, y, z = rng.sample(NAMES, 3)
            cmp, sup = rng.choice(ADJ)
            if rng.random() < 0.5:
                q = f"{x} is {cmp} than {y}. {y} is {cmp} than {z}. who is the {sup}"
                chain = f"{x} is {cmp} than {y} and {y} is {cmp} than {z}, so {x} is {cmp} than {z}."
                out.append({"kind": "chain", "question": q, "chain": chain, "answer": x})
            else:
                q = f"{y} is {cmp} than {z}. {x} is {cmp} than {y}. who is the {sup}"
                chain = f"{x} is {cmp} than {y} and {y} is {cmp} than {z}, so {x} is {cmp} than {z}."
                out.append({"kind": "chain", "question": q, "chain": chain, "answer": x})
        else:  # modus ponens (yes) or modus tollens (no)
            p, c = rng.choice(RULES)
            if rng.random() < 0.5:
                q = f"if {p} then {c}. {p}. is it true that {c}"
                chain = f"{p}, and the rule says {c} follows."
                out.append({"kind": "logic", "question": q, "chain": chain, "answer": "yes"})
            else:
                q = f"if {p} then {c}. it is false that {c}. is it true that {p}"
                chain = f"if {p} were true then {c} would be true, but it is false."
                out.append({"kind": "logic", "question": q, "chain": chain, "answer": "no"})
    for r in out:
        r["prompt"] = prompt_of(r["question"])
    return out


def reasoning_line(r: dict) -> str:
    return f"q: {r['question']}\na: {r['chain']} answer: {r['answer']}\n"


# --------------------------------------------------------------------------- your traces


def read_traces(vault: str | None, traces_jsonl: str | None) -> list[dict]:
    rows: list[dict] = []
    files = []
    if traces_jsonl and os.path.isfile(traces_jsonl):
        files.append(traces_jsonl)
    if vault:
        files += sorted(glob.glob(os.path.join(os.path.expanduser(vault), "traces", "*", "traces.jsonl")))
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def split_traces(rows: list[dict], max_item: int, max_total: int) -> tuple[list[str], list[dict], list[dict]]:
    """Return (documents for pretraining, sft rows, dpo rows) from collector records."""
    docs, sft, dpo = [], [], []
    used = 0
    for r in rows:
        prompt, response = r.get("prompt"), r.get("response")
        chosen, rejected = r.get("chosen"), r.get("rejected")
        if prompt and response:
            sft.append({"messages": [{"role": "user", "content": str(prompt)[:max_item]}, {"role": "assistant", "content": str(response)[:max_item]}], "source": "traces"})
        if chosen and rejected:
            dpo.append({"prompt": str(prompt or r.get("content") or "")[:max_item], "chosen": str(chosen)[:max_item], "rejected": str(rejected)[:max_item], "source": "traces"})
        for key in ("content", "response", "chosen"):
            t = r.get(key)
            if isinstance(t, str) and len(t.strip()) >= 20 and used < max_total:
                t = t.strip()[:max_item]
                docs.append(t)
                used += len(t)
    return docs, sft, dpo


# --------------------------------------------------------------------------- main


def main():
    args = build_parser().parse_args()
    if args.n_reason is None:
        args.n_reason = 600 if args.smoke else 6000
    if args.tokenizer is None:
        args.tokenizer = "char" if args.smoke else "gpt2"
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    C.status("reason", f"{args.n_reason} synthetic reasoning items")
    items = make_reasoning(args.n_reason, args.seed)
    rng.shuffle(items)
    n_held = max(1, int(args.heldout_frac * len(items)))
    heldout, train = items[:n_held], items[n_held:]
    by_kind = {}
    for r in train:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1

    C.status("traces", "reading the collector" if (args.vault or args.traces_jsonl) else "no traces given; synthetic only")
    trace_rows = read_traces(args.vault, args.traces_jsonl)
    trace_docs, trace_sft, trace_dpo = split_traces(trace_rows, args.max_item_chars, args.max_trace_chars)
    C.log(f"traces: {len(trace_rows)} records -> {len(trace_docs)} documents, {len(trace_sft)} sft rows, {len(trace_dpo)} dpo rows")

    # the mixture: stories (synthetic, plus a hub set outside smoke), topic sentences, plain arithmetic, reasoning lines, your traces
    sources: dict[str, list[str]] = {}
    sources["stories"] = [s.strip() for s in C.STORIES.split("\n") if s.strip()]
    if args.hf_dataset and not args.smoke:
        datasets = C.require("datasets")
        C.status("data", f"loading {args.hf_dataset} (max {args.max_samples} rows)")
        ds = datasets.load_dataset(args.hf_dataset, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= args.max_samples:
                break
            t = str(row[args.text_field]).strip()
            if t:
                sources["stories"].append(t)
    sources["topics"] = [s + "." for v in C.TOPICS.values() for s in v]
    sources["arithmetic"] = C.arithmetic_lines(len(train) // 2 + 50, seed=args.seed + 7)
    sources["reasoning"] = [reasoning_line(r) for r in train]
    sources["traces"] = trace_docs

    if args.tokenizer == "gpt2":
        try:
            tok = C.TiktokenWrapper("gpt2")
        except SystemExit:
            C.log("tiktoken is not installed; counting characters instead")
            tok = C.CharTokenizer()
            args.tokenizer = "char"
    else:
        tok = None
    count = (lambda t: len(tok.encode(t))) if tok is not None else len  # noqa: E731
    C.status("write", f"writing {args.out}")
    tokens = {k: sum(count(t) for t in v) for k, v in sources.items()}
    docs = [t for v in sources.values() for t in v]
    rng.shuffle(docs)
    corpus = "\n\n".join(t.strip() for t in docs) + "\n"
    with open(os.path.join(args.out, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write(corpus)
    with open(os.path.join(args.out, "reason.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sources["reasoning"]))

    sft_rows = [{"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": f"{r['chain']} answer: {r['answer']}"}], "source": "synthetic"} for r in train]
    sft_rows += trace_sft
    rng.shuffle(sft_rows)
    dpo_rows = list(trace_dpo)
    if not dpo_rows:  # the synthetic preference pairs from the browser lab, so dpo.py has something to run on
        dpo_rows = [{"prompt": p, "chosen": c, "rejected": r, "source": "synthetic"} for p, c, r in C.PREFS]
    paths = {
        "corpus": os.path.join(args.out, "corpus.txt"),
        "reason_text": os.path.join(args.out, "reason.txt"),
        "sft": os.path.join(args.out, "sft.jsonl"),
        "dpo": os.path.join(args.out, "dpo.jsonl"),
        "reason_train": os.path.join(args.out, "reason_train.jsonl"),
        "reason_heldout": os.path.join(args.out, "reason_heldout.jsonl"),
    }
    C.write_jsonl(paths["sft"], sft_rows)
    C.write_jsonl(paths["dpo"], dpo_rows)
    C.write_jsonl(paths["reason_train"], train)
    C.write_jsonl(paths["reason_heldout"], heldout)
    total = sum(tokens.values())
    for i, (k, v) in enumerate(tokens.items()):
        C.metric(i, tokens=v, frac=v / max(1, total))
        C.log(f"  {k:<11} {v:>9,} {args.tokenizer} tokens  ({v / max(1, total):.1%})  {len(sources[k])} docs")
    C.log(f"reasoning train {len(train)} (by kind {by_kind}), held-out {len(heldout)}; sft {len(sft_rows)} rows, dpo {len(dpo_rows)} rows")
    C.status("done", f"{total:,} tokens in {len(docs)} documents")
    C.result(sources=tokens, total_tokens=total, tokenizer=args.tokenizer, n_docs=len(docs), n_sft=len(sft_rows), n_dpo=len(dpo_rows),
             n_reason_train=len(train), n_reason_heldout=len(heldout), n_traces=len(trace_rows), **paths)


if __name__ == "__main__":
    main()
