"""AdamW versus Muon on the minimal GPT, same data, same schedule (Lab 12: optimizers).

What this teaches
  * Muon: for every 2-D weight matrix, take the momentum-averaged gradient G,
    replace it by an approximate orthogonalization U V^T (the polar factor of
    G), and step along that; everything that is not a matrix (embeddings, norm
    gains, the tied lm_head) is left to AdamW. The orthogonalization is done
    with five Newton-Schulz iterations of the quintic
        X <- a X + b (X X^T) X + c (X X^T)^2 X
    on the normalized matrix, which needs only matmuls and so runs in bf16.
  * an honest A/B: both optimizers see the exact same batches in the same
    order (a seeded generator that is reset per run), the same warmup and
    cosine schedule, and the same clipping; the only difference is the update.

How to run
  smoke:  python lab/recipes/optim_bench.py --smoke --steps 100
  real:   python lab/recipes/optim_bench.py --steps 1500 --n-layer 6 --d-model 384 --n-head 6 --seq-len 256 --batch 32

Every METRIC line carries an "opt" field ("adamw" or "muon") so the two loss
curves can be drawn on one chart. The Muon learning rate is separate
(--muon-lr) because its update has unit spectral norm per matrix and so lives
on a different scale from AdamW's.

Reference: Keller Jordan's Muon (2024), https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import common as C  # noqa: E402


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization of G (the U V^T of its SVD) by Newton-Schulz iteration.

    The coefficients (3.4445, -4.7750, 2.0315) are the ones from Jordan's Muon
    implementation; they trade exactness for a fast climb of small singular
    values toward one, which is all the optimizer needs.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum (Nesterov) + Newton-Schulz orthogonalized update for 2-D parameters."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                # scale so the update RMS is comparable across shapes (Jordan's max(1, rows/cols)^0.5 factor)
                u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])


def build_optimizers(model: C.GPT, name: str, args):
    if name == "adamw":
        return [C.make_adamw(model, args.lr, args.weight_decay)]
    # Muon for hidden matrices only: embeddings (tied to lm_head) and norm gains stay with AdamW
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "tok_emb" not in n and "lm_head" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "tok_emb" not in n and "lm_head" not in n)]
    return [Muon(hidden, lr=args.muon_lr, momentum=0.95, weight_decay=args.weight_decay),
            torch.optim.AdamW(rest, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)]


def run(name: str, args, train_ids, val_ids, tok, device):
    C.set_seed(args.seed)  # identical init for both runs
    cfg = C.GPTConfig(vocab_size=tok.vocab_size, n_layer=args.n_layer, d_model=args.d_model, n_head=args.n_head, seq_len=args.seq_len)
    model = C.GPT(cfg).to(device)
    opts = build_optimizers(model, name, args)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    gen = torch.Generator().manual_seed(args.seed + 7)      # identical batches for both runs
    eval_gen = torch.Generator().manual_seed(args.seed + 8)
    C.status("train", f"opt={name} steps={args.steps}")
    model.train()
    t0 = time.perf_counter()
    val = float("nan")
    for step in range(args.steps):
        mult = C.lr_at(step, args.steps, 1.0, args.warmup, args.min_lr_ratio, "cosine")
        for o, base in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, base):
                g["lr"] = b * mult
        x, y = C.random_windows(train_ids, args.batch, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        with C.autocast_ctx(device):
            loss = C.lm_loss(model(x), y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        fields = {}
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vs = []
                for _ in range(args.eval_batches):
                    vx, vy = C.random_windows(val_ids, args.batch, args.seq_len, eval_gen)
                    vs.append(C.lm_loss(model(vx.to(device)), vy.to(device)).item())
            val = sum(vs) / len(vs)
            fields["val_loss"] = val
            model.train()
        C.metric(step, opt=name, loss=loss.item(), grad_norm=float(gn), lr_mult=mult, **fields)
    return {"final_loss": loss.item(), "val_loss": val, "seconds": time.perf_counter() - t0}


def main():
    p = C.base_parser("optim_bench", __doc__.split("\n")[0])
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="AdamW learning rate (also used for Muon's AdamW group)")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--opts", default="adamw,muon")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--text-field", default="text")
    args = p.parse_args()
    d = dict(n_layer=2, d_model=64, n_head=4, seq_len=64, batch=16, lr=3e-3, warmup=10, steps=100, eval_every=25) if args.smoke else \
        dict(n_layer=6, d_model=384, n_head=6, seq_len=256, batch=32, lr=6e-4, warmup=100, steps=1500, eval_every=100)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    device = C.pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        tok = C.CharTokenizer()
        ids = torch.tensor(tok.encode(C.synthetic_text("all")), dtype=torch.long)
    else:
        datasets = C.require("datasets")
        tok = C.TiktokenWrapper("gpt2")
        ds = datasets.load_dataset(args.dataset, split="train", streaming=True)
        ids = []
        for i, row in enumerate(ds):
            if i >= args.max_samples:
                break
            ids.extend(tok.encode(row[args.text_field], add_eos=True))
        ids = torch.tensor(ids, dtype=torch.long)
    n_val = max(args.seq_len + 2, int(0.05 * ids.numel()))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    results = {}
    for name in args.opts.split(","):
        results[name] = run(name.strip(), args, train_ids, val_ids, tok, device)
        C.log(f"{name}: {results[name]}")
    C.status("done", "both runs finished")
    C.result(**{f"{k}_{m}": v for k, r in results.items() for m, v in r.items()}, steps=args.steps)


if __name__ == "__main__":
    main()
