"""A mixture-of-experts MLP inside the minimal GPT, trained on a two-domain mixture (Lab 11: architecture).

What this teaches
  * the MoE layer. A router r(x) = softmax(W_r x) over E experts picks the
    top-k; the token goes through those k SwiGLU experts and the outputs are
    combined with the (renormalized) router weights. Every token touches k
    experts, so the active parameters per token are far fewer than the total.
  * why a router needs a balance loss. Without it, one expert wins early,
    gets all the gradient, and wins harder (routing collapse). The Switch
    Transformer auxiliary loss is
        L_balance = E * sum_e f_e * P_e
    with f_e the fraction of tokens routed to expert e and P_e the mean router
    probability of e; it is minimized when both are uniform (1/E) and it is
    differentiable through P_e. The optional router z-loss
    mean(logsumexp(logits)^2) keeps the router logits from growing.
  * how to see specialization: at each eval the recipe prints a domain x
    expert usage matrix (fraction of domain-a tokens and of domain-b tokens
    that each expert receives at the last MoE layer). Stories and arithmetic
    are different enough that you can watch experts take sides, or not.

How to run
  smoke (CPU, offline; domain a = synthetic stories, domain b = synthetic arithmetic):
    python lab/recipes/moe_nano.py --smoke --steps 300 --experts 4 --top-k 1
    python lab/recipes/moe_nano.py --smoke --steps 300 --experts 4 --top-k 2 --balance-coef 0
  real (RTX 5090), two Hugging Face text datasets as in midtrain.py:
    python lab/recipes/moe_nano.py --steps 2000 --experts 8 --top-k 2 --domain-a roneneldan/TinyStories:text --domain-b wikimedia/wikipedia:20231101.simple:text
  needs (real): pip install datasets tiktoken

METRIC fields: loss (total), lm_loss, balance_loss, z_loss, max_expert_share, load_e<i> (last MoE layer),
and at eval steps val_a, val_b plus usage_a_e<i> / usage_b_e<i>. RESULT reports total and active parameters.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402
from midtrain import load_domain, parse_mix, split  # noqa: E402


class MoE(nn.Module):
    """Top-k routed mixture of SwiGLU experts. Stores its auxiliary losses in self.aux after each forward."""

    def __init__(self, d: int, n_expert: int, top_k: int, hidden: int):
        super().__init__()
        self.E, self.k = n_expert, top_k
        self.router = nn.Linear(d, n_expert, bias=False)
        self.experts = nn.ModuleList([C.SwiGLU(d, hidden) for _ in range(n_expert)])
        self.aux = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat).float()
        probs = F.softmax(logits, -1)
        top_p, top_i = probs.topk(self.k, -1)                       # (N, k)
        gate = top_p / top_p.sum(-1, keepdim=True) if self.k > 1 else top_p
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, gate[rows, slot, None].to(flat.dtype) * expert(flat[rows]))
        N = flat.shape[0]
        f = torch.bincount(top_i.flatten(), minlength=self.E).float() / (N * self.k)   # fraction of assignments per expert
        P = probs.mean(0)                                                              # mean router prob per expert
        self.aux = {"balance": self.E * (f * P).sum(), "z": torch.logsumexp(logits, -1).pow(2).mean(), "load": f.detach()}
        return out.view(shape)


def build_moe_gpt(tok_vocab, n_layer, d_model, n_head, seq_len, n_expert, top_k, expert_hidden, moe_every):
    model = C.GPT(C.GPTConfig(vocab_size=tok_vocab, n_layer=n_layer, d_model=d_model, n_head=n_head, seq_len=seq_len))
    moe_layers = []
    for i, blk in enumerate(model.blocks):
        if (i + 1) % moe_every == 0:
            blk.mlp = MoE(d_model, n_expert, top_k, expert_hidden)
            moe_layers.append(blk.mlp)
    for m in moe_layers:                                        # init the fresh experts like the rest of the model
        m.apply(model._init)
    return model, moe_layers


def param_counts(model, moe_layers, top_k):
    total = sum(p.numel() for p in model.parameters())
    all_expert = sum(p.numel() for m in moe_layers for p in m.experts.parameters())
    active_expert = sum(top_k * sum(p.numel() for p in m.experts[0].parameters()) for m in moe_layers)
    return total, total - all_expert + active_expert


def aux_losses(moe_layers):
    bal = torch.stack([m.aux["balance"] for m in moe_layers]).mean()
    z = torch.stack([m.aux["z"] for m in moe_layers]).mean()
    return bal, z


@torch.no_grad()
def eval_domain(model, moe_layers, data, batch, seq_len, n, device, gen):
    """Held-out loss and the expert usage vector (last MoE layer) on one domain."""
    model.eval()
    tot, usage = 0.0, torch.zeros(moe_layers[-1].E)
    for _ in range(n):
        x, y = C.random_windows(data, batch, seq_len, gen)
        with C.autocast_ctx(device):
            tot += C.lm_loss(model(x.to(device)), y.to(device)).item()
        usage += moe_layers[-1].aux["load"].cpu()
    model.train()
    return tot / n, usage / n


def main():
    p = C.base_parser("moe_nano", __doc__.split("\n")[0])
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--expert-hidden", type=int, default=None, help="hidden width of each expert (default 2 * d_model)")
    p.add_argument("--moe-every", type=int, default=1, help="replace the MLP in every n-th layer")
    p.add_argument("--balance-coef", type=float, default=0.01)
    p.add_argument("--z-coef", type=float, default=0.0)
    p.add_argument("--mix", default="a=0.5,b=0.5")
    p.add_argument("--domain-a", default="roneneldan/TinyStories:text")
    p.add_argument("--domain-b", default="wikimedia/wikipedia:20231101.simple:text")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=6)
    args = p.parse_args()
    d = dict(n_layer=2, d_model=64, n_head=4, seq_len=64, batch=16, lr=3e-3, warmup=20, steps=300, eval_every=50) if args.smoke else \
        dict(n_layer=6, d_model=384, n_head=6, seq_len=256, batch=32, lr=6e-4, warmup=100, steps=2000, eval_every=100)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.expert_hidden is None:
        args.expert_hidden = 2 * args.d_model
    C.set_seed(args.seed)
    device = C.pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    mix = parse_mix(args.mix)

    if args.smoke:
        tok = C.CharTokenizer()
        a_ids = torch.tensor(tok.encode(C.synthetic_text("stories")), dtype=torch.long)
        b_ids = torch.tensor(tok.encode("\n".join(C.arithmetic_lines(400, seed=11)) + "\n"), dtype=torch.long)
    else:
        tok = C.TiktokenWrapper("gpt2")
        a_ids = load_domain(args.domain_a, tok, args.max_samples)
        b_ids = load_domain(args.domain_b, tok, args.max_samples)
    a_tr, a_va = split(a_ids, args.seq_len)
    b_tr, b_va = split(b_ids, args.seq_len)

    model, moe_layers = build_moe_gpt(tok.vocab_size, args.n_layer, args.d_model, args.n_head, args.seq_len, args.experts, args.top_k,
                                      args.expert_hidden, args.moe_every)
    model = model.to(device)
    total, active = param_counts(model, moe_layers, args.top_k)
    C.log(f"experts={args.experts} top_k={args.top_k} expert_hidden={args.expert_hidden} MoE layers={len(moe_layers)}: "
          f"total params {total:,}, active per token {active:,} ({active / total:.1%})")
    opt = C.make_adamw(model, args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)
    E = args.experts

    def evaluate(step):
        la, ua = eval_domain(model, moe_layers, a_va, args.batch, args.seq_len, args.eval_batches, device, eval_gen)
        lb, ub = eval_domain(model, moe_layers, b_va, args.batch, args.seq_len, args.eval_batches, device, eval_gen)
        C.log(f"step {step}: val_a {la:.3f} val_b {lb:.3f}; expert usage (last MoE layer)")
        C.log("           " + " ".join(f"e{i:<6}" for i in range(E)))
        C.log("  domain a " + " ".join(f"{u:.3f}  " for u in ua.tolist()))
        C.log("  domain b " + " ".join(f"{u:.3f}  " for u in ub.tolist()))
        fields = {"val_a": la, "val_b": lb}
        fields.update({f"usage_a_e{i}": ua[i].item() for i in range(E)})
        fields.update({f"usage_b_e{i}": ub[i].item() for i in range(E)})
        return la, lb, ua, ub, fields

    la, lb, ua, ub, f0 = evaluate(0)
    C.metric(0, **f0)
    C.status("train", f"{args.steps} steps on mix {mix}, balance_coef={args.balance_coef}, z_coef={args.z_coef}")
    model.train()
    for step in range(1, args.steps + 1):
        lr = C.lr_at(step - 1, args.steps, args.lr, args.warmup, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        n_a = int(torch.rand(args.batch, generator=gen).lt(mix["a"]).sum())
        xs, ys = [], []
        if n_a:
            x, y = C.random_windows(a_tr, n_a, args.seq_len, gen)
            xs.append(x), ys.append(y)
        if args.batch - n_a:
            x, y = C.random_windows(b_tr, args.batch - n_a, args.seq_len, gen)
            xs.append(x), ys.append(y)
        x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
        with C.autocast_ctx(device):
            lm = C.lm_loss(model(x), y)
        bal, z = aux_losses(moe_layers)
        loss = lm + args.balance_coef * bal + args.z_coef * z
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        load = moe_layers[-1].aux["load"]
        fields = dict(loss=loss.item(), lm_loss=lm.item(), balance_loss=bal.item(), z_loss=z.item(), max_expert_share=load.max().item(), lr=lr)
        fields.update({f"load_e{i}": load[i].item() for i in range(E)})
        if step % args.eval_every == 0 or step == args.steps:
            la, lb, ua, ub, fe = evaluate(step)
            fields.update(fe)
        C.metric(step, **fields)
    path = os.path.join(args.out, "moe_ckpt.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "tokenizer": tok.to_dict()}, path)
    C.status("done", f"saved {path}")
    C.result(total_params=total, active_params=active, active_frac=active / total, experts=E, top_k=args.top_k, val_a=la, val_b=lb,
             usage_a=[round(u, 4) for u in ua.tolist()], usage_b=[round(u, 4) for u in ub.tolist()], final_lm_loss=lm.item(),
             final_balance_loss=bal.item(), checkpoint=path)


if __name__ == "__main__":
    main()
