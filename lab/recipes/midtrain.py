"""Mid-training: continue a pretrained checkpoint on a two-domain mixture with a cooldown (Lab 03).

What this teaches
  * a data mixture as sampling weights: each sequence in a batch comes from
    domain a with probability w_a and from domain b with probability w_b
  * the warmup-stable-decay tail: the learning rate is held at --lr and then
    cooled linearly to --min-lr-ratio times --lr over the last --cooldown-frac
    of the run; the held-out losses usually drop sharply during the cooldown
  * forgetting made visible: held-out loss on BOTH domains is logged at every
    eval, so a lopsided mix shows up as one curve going down while the other
    goes up

How to run
  smoke (domain a = synthetic stories, domain b = synthetic arithmetic; if no
  --ckpt is given a short pretraining on domain a runs first so there is
  something to forget):
    python lab/recipes/midtrain.py --smoke --steps 200 --mix a=0.7,b=0.3
  real, continuing from a pretrain_nano.py checkpoint (its tokenizer is reused):
    python lab/recipes/midtrain.py --ckpt out/pretrain_nano/ckpt.pt \
        --domain-a roneneldan/TinyStories:text --domain-b wikimedia/wikipedia:20231101.simple:text \
        --mix a=0.7,b=0.3 --steps 1000 --cooldown-frac 0.3

Domain spec for real mode: "dataset_name[:config]:text_field" (the config is
optional); the split is "train" and --max-samples rows are read per domain.
Plain text files override either domain in any mode (the training pie, Lab 21):
    python lab/recipes/midtrain.py --smoke --ckpt out/pretrain/ckpt.pt \
        --text-a out/data_prep/corpus.txt --text-b out/data_prep/reason.txt --mix a=0.4,b=0.6
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("midtrain", __doc__.split("\n")[0])
    p.add_argument("--ckpt", default=None, help="pretrain_nano.py checkpoint; smoke mode makes one if omitted")
    p.add_argument("--mix", default="a=0.7,b=0.3", help="sampling weights per domain, normalized")
    p.add_argument("--domain-a", default="roneneldan/TinyStories:text")
    p.add_argument("--domain-b", default="wikimedia/wikipedia:20231101.simple:text")
    p.add_argument("--text-a", default=None, help="a text file for domain a (overrides --domain-a and the smoke stories)")
    p.add_argument("--text-b", default=None, help="a text file for domain b (overrides --domain-b and the smoke arithmetic)")
    p.add_argument("--lr", type=float, default=None, help="stable-phase learning rate")
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=0, help="short re-warmup when starting from a cooled checkpoint")
    p.add_argument("--cooldown-frac", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--pre-steps", type=int, default=150, help="smoke only: pretraining steps on domain a when --ckpt is absent")
    return p


def parse_mix(s: str) -> dict[str, float]:
    w = {}
    for part in s.split(","):
        k, v = part.split("=")
        w[k.strip()] = float(v)
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def load_domain(spec: str, tok, max_samples: int) -> torch.Tensor:
    datasets = C.require("datasets")
    parts = spec.split(":")
    name, field = parts[0], parts[-1]
    config = parts[1] if len(parts) == 3 else None
    C.status("data", f"loading {name} config={config} field={field} max={max_samples}")
    ds = datasets.load_dataset(name, config, split="train", streaming=True) if config else \
        datasets.load_dataset(name, split="train", streaming=True)
    ids = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        ids.extend(tok.encode(row[field], add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def split(ids: torch.Tensor, seq_len: int, frac: float = 0.1):
    n_val = max(seq_len + 2, int(frac * ids.numel()))
    return ids[:-n_val], ids[-n_val:]


@torch.no_grad()
def held_out(model, data, batch, seq_len, n, device, gen):
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = C.random_windows(data, batch, seq_len, gen)
        with C.autocast_ctx(device):
            tot += C.lm_loss(model(x.to(device)), y.to(device)).item()
    model.train()
    return tot / n


def smoke_pretrain(tok, device, steps, seed):
    """A short pretraining on domain a only, so mid-training starts from a model that knows stories."""
    cfg = C.GPTConfig(vocab_size=tok.vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=64)
    model = C.GPT(cfg).to(device)
    ids = torch.tensor(tok.encode(C.synthetic_text("stories")), dtype=torch.long)
    tr, _ = split(ids, 64)
    opt = C.make_adamw(model, 3e-3)
    gen = torch.Generator().manual_seed(seed + 100)
    C.status("pretrain", f"no --ckpt given: {steps} steps on domain a first")
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = C.lr_at(step, steps, 3e-3, 10, 1.0, "constant")
        x, y = C.random_windows(tr, 16, 64, gen)
        loss = C.lm_loss(model(x.to(device)), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    C.log(f"pretrain finished at loss {loss.item():.3f}")
    return model


def main():
    args = build_parser().parse_args()
    d = dict(steps=200, batch=16, lr=2e-3, eval_every=20) if args.smoke else dict(steps=1000, batch=32, lr=3e-4, eval_every=50)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    device = C.pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    mix = parse_mix(args.mix)
    assert set(mix) == {"a", "b"}, "--mix must name domains a and b"

    if args.ckpt:
        model, tok, ck = C.load_checkpoint(args.ckpt, device)
        C.log(f"loaded {args.ckpt} (step {ck['step']}), params={model.num_params():,}")
    elif args.smoke:
        tok = C.CharTokenizer()
        model = smoke_pretrain(tok, device, args.pre_steps, args.seed)
    else:
        raise SystemExit("real mode needs --ckpt from pretrain_nano.py")
    seq_len = model.cfg.seq_len

    def from_file(path):
        return torch.tensor(tok.encode(open(path, encoding="utf-8", errors="replace").read()), dtype=torch.long)

    if args.smoke:
        a_ids = from_file(args.text_a) if args.text_a else torch.tensor(tok.encode(C.synthetic_text("stories")), dtype=torch.long)
        b_ids = from_file(args.text_b) if args.text_b else torch.tensor(tok.encode("\n".join(C.arithmetic_lines(400, seed=11)) + "\n"), dtype=torch.long)
    else:
        a_ids = from_file(args.text_a) if args.text_a else load_domain(args.domain_a, tok, args.max_samples)
        b_ids = from_file(args.text_b) if args.text_b else load_domain(args.domain_b, tok, args.max_samples)
    a_tr, a_va = split(a_ids, seq_len)
    b_tr, b_va = split(b_ids, seq_len)
    C.log(f"domain a: {a_tr.numel():,} train tokens, domain b: {b_tr.numel():,}; mix={mix}")

    opt = C.make_adamw(model, args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)
    ev = lambda data: held_out(model, data, args.batch, seq_len, args.eval_batches, device, eval_gen)  # noqa: E731
    la0, lb0 = ev(a_va), ev(b_va)
    C.metric(0, val_a=la0, val_b=lb0, lr=0.0, frac_a=0.0)
    C.log(f"before mid-training: held-out a={la0:.3f} b={lb0:.3f}")
    C.status("train", f"{args.steps} steps, cooldown over the last {args.cooldown_frac:.0%}")
    model.train()
    la, lb = la0, lb0
    for step in range(args.steps):
        lr = C.lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_ratio, "wsd", args.cooldown_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        # per-sequence domain draw: Bernoulli(w_a) for each row of the batch
        n_a = int(torch.rand(args.batch, generator=gen).lt(mix["a"]).sum())
        xs, ys = [], []
        if n_a:
            x, y = C.random_windows(a_tr, n_a, seq_len, gen)
            xs.append(x), ys.append(y)
        if args.batch - n_a:
            x, y = C.random_windows(b_tr, args.batch - n_a, seq_len, gen)
            xs.append(x), ys.append(y)
        x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
        with C.autocast_ctx(device):
            loss = C.lm_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        fields = {}
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            la, lb = ev(a_va), ev(b_va)
            fields.update(val_a=la, val_b=lb)
        C.metric(step + 1, loss=loss.item(), lr=lr, frac_a=n_a / args.batch, **fields)

    path = C.save_checkpoint(os.path.join(args.out, "ckpt.pt"), model, tok, args.steps, extra={"args": vars(args)})
    C.status("done", f"saved {path}")
    C.result(val_a_before=la0, val_b_before=lb0, val_a_after=la, val_b_after=lb,
             forgetting_a=la - la0, gain_b=lb0 - lb, mix_a=mix["a"], checkpoint=path)


if __name__ == "__main__":
    main()
