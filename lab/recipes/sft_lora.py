"""Supervised fine-tuning with LoRA and an answer-only loss (Lab 04: the SFT loop).

What this teaches
  * the loss mask. In SFT you show the model a prompt and a response but only
    penalize its predictions on the response tokens; prompt positions get the
    label -100 and cross_entropy ignores them. The recipe prints the fraction
    of tokens that carry a gradient so you can see how much of a batch is
    actually training the model.
  * LoRA: the base weights are frozen and each targeted matrix W gets a
    trainable low-rank update B A with B initialized to zero, so the model
    starts exactly at the base model and the adapter is the only thing saved.
  * the library path: TRL's SFTTrainer on prompt/completion rows with
    completion_only_loss, Unsloth's FastLanguageModel for the fast kernels
    when it is importable, plain peft otherwise.

How to run
  smoke (CPU, offline): fine-tunes the minimal GPT from common.py on the
  synthetic Q/A pairs with a hand-written answer-only mask, then greedy-decodes
  every question and reports exact match:
    python lab/recipes/sft_lora.py --smoke --steps 300
    python lab/recipes/sft_lora.py --smoke --steps 300 --ckpt out/pretrain_nano/ckpt.pt
    python lab/recipes/sft_lora.py --smoke --steps 300 --no-mask     # see what dropping the mask does
  the training pie (Lab 21): --ckpt selects this nano path even without --smoke, and
  --pairs-jsonl replaces the built-in Q/A with data_prep.py's sft.jsonl (or any
  {"messages": [...]} / {"prompt", "response"} rows); a held-out slice is kept for exact match:
    python lab/recipes/sft_lora.py --ckpt out/midtrain/ckpt.pt --pairs-jsonl out/data_prep/sft.jsonl --steps 300 --max-new 64
  real (RTX 5090): LoRA on a small instruct model with the NVIDIA agentic SFT set
    python lab/recipes/sft_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset nvidia/Nemotron-SFT-Agentic-v2 --max-samples 5000 --steps 300
  needs: pip install transformers trl peft datasets   (and optionally unsloth)

Dataset rows are read from --messages-field (default "messages"; a
"conversations" column in ShareGPT form with from/value keys is converted).
The last assistant turn becomes the completion, everything before it is
rendered with the chat template as the prompt.
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("sft_lora", __doc__.split("\n")[0])
    p.add_argument("--ckpt", default=None, help="smoke: start from a pretrain_nano.py checkpoint")
    p.add_argument("--no-mask", action="store_true", help="smoke: train on prompt tokens too (to see why the mask matters)")
    p.add_argument("--pairs-jsonl", default=None, help="nano path: rows of {messages} or {prompt, response} instead of the built-in Q/A")
    p.add_argument("--heldout-frac", type=float, default=0.1, help="nano path with --pairs-jsonl: fraction kept for exact match")
    p.add_argument("--max-new", type=int, default=24, help="nano path: tokens decoded per question at eval")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="nvidia/Nemotron-SFT-Agentic-v2")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--messages-field", default="messages")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-unsloth", action="store_true", help="use plain transformers + peft even if unsloth is installed")
    return p


# --------------------------------------------------------------------------- smoke: mask by hand


def build_example(tok: C.CharTokenizer, q: str, a: str, mask: bool) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels). Labels are -100 on the prompt when mask is True.

    Layout: "q: <question>\na: <answer><eos>". The label for position t is the
    token at t+1, so the shift is applied here once and the model is trained
    with plain cross-entropy on (ids[:-1], labels[1:]).
    """
    prompt = tok.encode(f"q: {q}\na: ")
    answer = tok.encode(a, add_eos=True)
    ids = prompt + answer
    labels = ([-100] * len(prompt) if mask else list(prompt)) + answer
    return ids, labels


def load_pairs(path: str) -> list[tuple[str, str]]:
    """(question, answer) pairs from {messages}, {prompt, response}, or {question, answer} rows."""
    pairs = []
    for r in C.read_jsonl(path):
        msgs = r.get("messages")
        if msgs and len(msgs) >= 2 and msgs[-1].get("role") == "assistant":
            pairs.append((str(msgs[-2].get("content", "")), str(msgs[-1].get("content", ""))))
        elif r.get("prompt") and r.get("response"):
            pairs.append((str(r["prompt"]), str(r["response"])))
        elif r.get("question") and r.get("answer"):
            pairs.append((str(r["question"]), str(r["answer"])))
    return pairs


def final_answer(s: str) -> str:
    """What exact match compares: the text after the last 'answer:' when the reply has one, else the whole reply."""
    s = s.strip()
    if "answer:" in s:
        s = s.rsplit("answer:", 1)[1]
    return s.strip().splitlines()[0].strip() if s.strip() else ""


def smoke(args):
    device = C.pick_device(args.device)
    tok = C.CharTokenizer()
    if args.ckpt:
        model, tok, _ = C.load_checkpoint(args.ckpt, device)
        C.log(f"starting from {args.ckpt}")
    else:
        model = C.GPT(C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=96)).to(device)
        C.log("no --ckpt: training the minimal GPT from scratch on the Q/A pairs")
    if args.pairs_jsonl:
        pairs = load_pairs(args.pairs_jsonl)
        if len(pairs) < 2:
            raise SystemExit(f"{args.pairs_jsonl}: need at least 2 usable rows, found {len(pairs)}")
        n_held = max(1, int(args.heldout_frac * len(pairs)))
        held_pairs, train_pairs = pairs[:n_held], pairs[n_held:]
        C.log(f"{args.pairs_jsonl}: {len(train_pairs)} train pairs, {len(held_pairs)} held out")
    else:
        train_pairs = C.QA[:16]
        held_pairs = C.QA[16:]
    examples = [build_example(tok, q, a, not args.no_mask) for q, a in train_pairs]
    max_len = model.cfg.seq_len + 1  # ids[:-1] must fit the context; longer examples are cut at the end
    n_cut = sum(1 for e in examples if len(e[0]) > max_len)
    if n_cut:
        C.log(f"warning: {n_cut} of {len(examples)} examples are longer than seq_len {model.cfg.seq_len} and lose their tail (pretrain with a longer --seq-len)")
    ids, mask = C.pad_batch([e[0] for e in examples], tok.pad_id, max_len)
    labels, _ = C.pad_batch([e[1] for e in examples], -100, max_len)
    labels[~mask] = -100
    x, y = ids[:, :-1].to(device), labels[:, 1:].to(device)
    supervised = int((y != -100).sum())
    real_tokens = int(mask[:, 1:].sum())
    frac = supervised / real_tokens
    C.log(f"supervised tokens: {supervised} of {real_tokens} non-pad targets = {frac:.3f} (mask={'off' if args.no_mask else 'on'})")

    opt = C.make_adamw(model, args.lr, 0.0)
    C.status("train", f"{args.steps} steps of answer-only SFT on {len(train_pairs)} pairs")
    model.train()
    gen = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        lr = C.lr_at(step, args.steps, args.lr, 10, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        if x.shape[0] > args.batch:  # minibatches once the pair set is bigger than one batch
            ix = torch.randint(0, x.shape[0], (args.batch,), generator=gen)
            loss = C.lm_loss(model(x[ix]), y[ix])
        else:
            loss = C.lm_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        C.metric(step, loss=loss.item(), lr=lr, supervised_frac=frac)

    def exact_match(pairs, log_n=8):
        hits = 0
        for i, (q, a) in enumerate(pairs):
            p = torch.tensor([tok.encode(f"q: {q}\na: ")], device=device)
            out = C.generate(model, p, args.max_new, greedy=True, eos_id=tok.eos_id)
            pred = tok.decode(out[0, p.shape[1]:].tolist())
            hits += int(final_answer(pred) == final_answer(a))
            if i < log_n:
                C.log(f"  q={q!r} pred={pred!r} gold={a!r}")
        return hits / max(1, len(pairs))

    C.status("eval", "greedy decoding")
    em_train = exact_match(train_pairs[:64])
    em_held = exact_match(held_pairs)
    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), model, tok, args.steps)
    C.status("done", f"saved {path}")
    C.result(final_loss=loss.item(), supervised_frac=frac, exact_match_train=em_train, exact_match_heldout=em_held,
             n_train=len(train_pairs), n_heldout=len(held_pairs), steps=args.steps, checkpoint=path)


# --------------------------------------------------------------------------- real: TRL + LoRA


def to_messages(row: dict, field: str) -> list[dict] | None:
    conv = row.get(field) or row.get("conversations")
    if not conv:
        return None
    out = []
    for m in conv:
        role = m.get("role") or m.get("from")
        content = m.get("content") if "content" in m else m.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    return out


def real(args):
    transformers = C.require("transformers")
    trl = C.require("trl")
    datasets = C.require("datasets")
    device = C.pick_device(args.device)
    target_modules = args.target_modules.split(",")

    use_unsloth = False
    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel  # noqa: F401

            use_unsloth = True
        except Exception as e:  # unsloth raises non-ImportError types on unsupported GPUs
            C.log(f"unsloth not usable ({type(e).__name__}: {e}); using transformers + peft")
    C.status("load", f"model={args.model} unsloth={use_unsloth}")
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_len, load_in_4bit=args.load_in_4bit)
        model = FastLanguageModel.get_peft_model(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=target_modules, use_gradient_checkpointing="unsloth")
        peft_config = None
    else:
        peft = C.require("peft")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        kw = dict(torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        if args.load_in_4bit:
            kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, **kw)
        peft_config = peft.LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                      target_modules=target_modules, task_type="CAUSAL_LM")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    C.status("data", f"{args.dataset} split={args.split} max={args.max_samples}")
    ds = datasets.load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    rows, skipped = [], 0
    for i, row in enumerate(ds):
        if len(rows) >= args.max_samples:
            break
        msgs = to_messages(row, args.messages_field)
        if not msgs or msgs[-1]["role"] != "assistant" or len(msgs) < 2:
            skipped += 1
            if i == 0 and not msgs:
                raise SystemExit(f"no '{args.messages_field}' or 'conversations' column; columns are {list(row.keys())}. "
                                 f"pass --messages-field")
            continue
        tools = row.get("tools")
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                               **({"tools": tools} if tools else {}))
        rows.append({"prompt": prompt, "completion": msgs[-1]["content"] + tokenizer.eos_token})
    C.log(f"{len(rows)} prompt/completion rows, {skipped} skipped")
    if not rows:
        raise SystemExit("no usable rows")
    # the same mask statistic the smoke path prints, on a sample of rows
    sup, tot = 0, 0
    for r in rows[:200]:
        p = len(tokenizer(r["prompt"])["input_ids"])
        c = len(tokenizer(r["completion"])["input_ids"])
        n = min(p + c, args.max_len)
        sup += max(0, n - p)
        tot += n
    frac = sup / max(1, tot)
    C.log(f"supervised token fraction over first {min(200, len(rows))} rows at max_len {args.max_len}: {frac:.3f}")
    train_ds = datasets.Dataset.from_list(rows)

    sig = inspect.signature(trl.SFTConfig.__init__).parameters
    cfg_kw = dict(output_dir=os.path.join(args.out, "trainer"), max_steps=args.steps, per_device_train_batch_size=args.batch,
                  gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
                  logging_steps=1, save_strategy="no", report_to="none", bf16=device.type == "cuda", seed=args.seed)
    cfg_kw["max_length" if "max_length" in sig else "max_seq_length"] = args.max_len
    if "completion_only_loss" in sig:
        cfg_kw["completion_only_loss"] = True
    else:
        C.log("this TRL version has no completion_only_loss flag; prompt/completion rows are masked by default in older versions")
    tr_sig = inspect.signature(trl.SFTTrainer.__init__).parameters
    tr_kw = dict(model=model, args=trl.SFTConfig(**cfg_kw), train_dataset=train_ds, callbacks=[C.make_metric_callback()])
    tr_kw["processing_class" if "processing_class" in tr_sig else "tokenizer"] = tokenizer
    if peft_config is not None:
        tr_kw["peft_config"] = peft_config
    trainer = trl.SFTTrainer(**tr_kw)
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in trainer.model.parameters())
    C.log(f"trainable params {n_train:,} of {n_all:,} ({n_train / n_all:.4%})")
    out = trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    C.status("done", f"adapter saved to {adapter}")
    C.result(train_loss=out.training_loss, steps=out.global_step, supervised_frac=frac, trainable_params=n_train,
             adapter=adapter, model=args.model, dataset=args.dataset)


def main():
    args = build_parser().parse_args()
    nano = args.smoke or bool(args.ckpt)  # a lab checkpoint can only be loaded by the nano path
    d = dict(steps=300, lr=3e-3, batch=16) if nano else dict(steps=300, lr=2e-4, batch=4)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if nano else real)(args)


if __name__ == "__main__":
    main()
