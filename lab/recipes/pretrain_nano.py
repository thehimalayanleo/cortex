"""Pretrain the minimal GPT from common.py (Lab 02: pretraining a decoder).

What this teaches
  * the whole pretraining loop in one file: tokenize, sample windows, forward,
    cross-entropy on the next token, AdamW with weight decay on matrices only,
    warmup then cosine (or warmup-stable-decay) learning rate, gradient clipping,
    bf16 autocast on cuda, periodic held-out loss and sample generation
  * what tokens/s and FLOPs/token mean for your hardware
  * `--loop T`: apply the same block stack T times with input injection (a
    recurrent-depth transformer). Parameters stay fixed, compute per token
    scales with T, so you can compare "params" against "FLOPs" as the thing
    that buys loss.

How to run
  smoke (CPU, offline, char-level on the synthetic corpus, about a minute):
    python lab/recipes/pretrain_nano.py --smoke --steps 200
  real (RTX 5090, TinyStories with the GPT-2 BPE, a few minutes for 2000 steps):
    python lab/recipes/pretrain_nano.py --steps 2000 --n-layer 6 --d-model 384 --n-head 6 --seq-len 256 --batch 32
  real from curate.py shards (train.bin / val.bin of uint16 GPT-2 ids):
    python lab/recipes/pretrain_nano.py --data-dir out/curate --steps 2000
  recurrent depth comparison at matched parameters:
    python lab/recipes/pretrain_nano.py --steps 2000 --n-layer 2 --loop 3

Outputs
  METRIC lines with loss, lr, tokens/s (and val_loss at eval steps),
  a checkpoint at <out>/ckpt.pt that midtrain.py, sft_lora.py, dpo.py,
  grpo_tool.py and eval_suite.py can load, and a RESULT line.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("pretrain_nano", __doc__.split("\n")[0])
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--loop", type=int, default=1, help="recurrent depth: apply the block stack this many times")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--schedule", choices=["cosine", "wsd", "constant"], default="cosine")
    p.add_argument("--cooldown-frac", type=float, default=0.2, help="for --schedule wsd")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--sample-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", choices=["gpt2", "char"], default="gpt2", help="real mode tokenizer")
    p.add_argument("--data-dir", default=None, help="directory with train.bin and val.bin (uint16 GPT-2 ids) from curate.py")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (cuda)")
    return p


def load_tokens(args, device):
    """Return (train_ids, val_ids, tokenizer) as 1-D long tensors on cpu."""
    if args.smoke:
        tok = C.CharTokenizer()
        text = C.synthetic_text("all")
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n_val = max(args.seq_len + 2, int(0.1 * ids.numel()))
        return ids[:-n_val], ids[-n_val:], tok
    if args.data_dir:
        import numpy as np

        tr = torch.from_numpy(np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16).astype(np.int64))
        va = torch.from_numpy(np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16).astype(np.int64))
        return tr, va, C.TiktokenWrapper("gpt2")
    datasets = C.require("datasets")
    C.status("data", f"loading {args.dataset} (max {args.max_samples} rows)")
    ds = datasets.load_dataset(args.dataset, split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= args.max_samples:
            break
        texts.append(row[args.text_field])
    if args.tokenizer == "char":
        tok = C.CharTokenizer(sorted(set("".join(texts))))
    else:
        tok = C.TiktokenWrapper("gpt2")
    ids = []
    for t in texts:
        ids.extend(tok.encode(t, add_eos=True))
    ids = torch.tensor(ids, dtype=torch.long)
    n_val = max(args.seq_len + 2, int(0.02 * ids.numel()))
    C.log(f"tokenized {len(texts)} docs into {ids.numel()} tokens; val = last {n_val}")
    return ids[:-n_val], ids[-n_val:], tok


@torch.no_grad()
def eval_loss(model, data, args, device, gen):
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = C.random_windows(data, args.batch, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        with C.autocast_ctx(device):
            losses.append(C.lm_loss(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = build_parser().parse_args()
    smoke = args.smoke
    d = dict(n_layer=2, d_model=64, n_head=4, seq_len=64, batch=16, lr=3e-3, warmup=20, steps=200, eval_every=25, sample_every=100) if smoke else \
        dict(n_layer=6, d_model=384, n_head=6, seq_len=256, batch=32, lr=6e-4, warmup=100, steps=2000, eval_every=100, sample_every=500)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    device = C.pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    train_ids, val_ids, tok = load_tokens(args, device)
    cfg = C.GPTConfig(vocab_size=tok.vocab_size, n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head,
                      seq_len=args.seq_len, loop=args.loop)
    model = C.GPT(cfg).to(device)
    n_params = model.num_params()
    fpt = model.flops_per_token()
    C.log(f"device={device} params={n_params:,} (non-embedding {model.num_params(True):,}) loop={args.loop} "
          f"train_tokens={train_ids.numel():,} flops/token(train)={fpt:.3e}")
    if args.compile and device.type == "cuda":
        model = torch.compile(model)

    opt = C.make_adamw(model, args.lr, args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)
    C.status("train", f"{args.steps} steps, batch {args.batch} x {args.seq_len} tokens, schedule {args.schedule}")
    model.train()
    t_last = time.perf_counter()
    tokens_since = 0
    last_loss = float("nan")
    val = float("nan")
    for step in range(args.steps):
        lr = C.lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_ratio, args.schedule, args.cooldown_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = C.random_windows(train_ids, args.batch, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        with C.autocast_ctx(device):
            loss = C.lm_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = loss.item()
        tokens_since += x.numel()
        fields = {}
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            val = eval_loss(model, val_ids, args, device, eval_gen)
            fields["val_loss"] = val
        if step % args.log_every == 0 or fields or step == args.steps - 1:
            now = time.perf_counter()
            tok_s = tokens_since / max(1e-9, now - t_last)
            t_last, tokens_since = now, 0
            C.metric(step, loss=last_loss, lr=lr, grad_norm=float(gnorm), tokens_per_s=tok_s,
                     tflops=tok_s * fpt / 1e12, **fields)
        if args.sample_every and (step + 1) % args.sample_every == 0:
            prompt = "the cat" if smoke else "Once upon a time"
            ids = torch.tensor([tok.encode(prompt)], device=device)
            out = C.generate(model, ids, 60, temperature=0.8, top_k=40, eos_id=tok.eos_id)
            C.log(f"sample@{step + 1}: {tok.decode(out[0].tolist())!r}")
            model.train()

    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), getattr(model, "_orig_mod", model), tok, args.steps,
                             extra={"args": vars(args)})
    C.status("done", f"saved {path}")
    C.result(train_loss=last_loss, val_loss=val, params=n_params, flops_per_token=fpt, loop=args.loop,
             steps=args.steps, checkpoint=path)


if __name__ == "__main__":
    main()
