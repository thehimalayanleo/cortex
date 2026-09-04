"""Train an embedding model with InfoNCE and in-batch negatives (Lab 07: encoders and embeddings).

What this teaches
  * the contrastive objective. A batch holds N (query, positive) pairs. Encode
    both sides, normalize, take the N x N similarity matrix S_ij = q_i . p_j / tau.
    The loss is cross-entropy with the diagonal as the label,
        L = -(1/N) sum_i log( exp(S_ii) / sum_j exp(S_ij) )
    so every other positive in the batch is a negative for free (in-batch
    negatives). The symmetric version adds the same loss with the roles of
    query and positive swapped. Larger batches mean harder softmaxes.
  * mean pooling over a bidirectional encoder (the GPT block from common.py
    with causal=False) and why the padding mask must enter the pooling.
  * how to measure an embedding model without labels: recall@1 on held-out
    pairs (does the query's nearest neighbour in the corpus equal its positive).

How to run
  smoke (CPU, offline): a small bidirectional encoder at character level on
  synthetic pairs (a sentence with words dropped -> the full sentence):
    python lab/recipes/embed_contrastive.py --smoke --steps 200
  smoke on pairs from your own vault (see embed_vault.py --smoke):
    python lab/recipes/embed_contrastive.py --smoke --pairs-jsonl out/embed_vault/pairs.jsonl
  real (RTX 5090): sentence-transformers fine-tune of nomic-embed-text-v1.5 (or
  a small BGE) on {"query", "positive"} rows written by embed_vault.py, with
  MultipleNegativesRankingLoss (that is InfoNCE with in-batch negatives) and an
  optional Matryoshka wrapper so truncated embeddings stay usable:
    python lab/recipes/embed_contrastive.py --pairs-jsonl out/embed_vault/pairs.jsonl --model nomic-ai/nomic-embed-text-v1.5 --matryoshka 768,512,256,128,64
  needs: pip install sentence-transformers datasets   (nomic needs einops and trust_remote_code)
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402


def build_parser():
    p = C.base_parser("embed_contrastive", __doc__.split("\n")[0])
    p.add_argument("--pairs-jsonl", default=None, help='rows of {"query": ..., "positive": ...}')
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.05, help="temperature (smoke)")
    p.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--matryoshka", default=None, help="comma-separated dims, e.g. 768,512,256,128,64")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    return p


# --------------------------------------------------------------------------- smoke


def synthetic_pairs(seed: int) -> list[dict]:
    """query = the sentence with about a third of its words removed; positive = the full sentence."""
    rng = random.Random(seed)
    sentences = [s for v in C.TOPICS.values() for s in v] + C.story_sentences()
    pairs = []
    for s in sentences:
        words = s.split()
        keep = [w for w in words if rng.random() > 0.35] or words[:2]
        pairs.append({"query": " ".join(keep), "positive": s})
    return pairs


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__()
        self.gpt = C.GPT(C.GPTConfig(vocab_size=vocab_size, n_layer=2, d_model=64, n_head=4, seq_len=seq_len, causal=False))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.gpt.hidden(ids, attn_mask=mask)
        m = mask[..., None].float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)     # mean over real tokens only
        return F.normalize(pooled, dim=-1)


def info_nce(q: torch.Tensor, p: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE. Returns (loss, in-batch accuracy of the query->positive direction)."""
    s = q @ p.T / tau
    labels = torch.arange(q.shape[0], device=q.device)
    loss = 0.5 * (F.cross_entropy(s, labels) + F.cross_entropy(s.T, labels))
    return loss, (s.argmax(1) == labels).float().mean()


def encode_texts(enc, tok, texts, device, seq_len):
    ids, mask = C.pad_batch([tok.encode(t)[:seq_len] for t in texts], tok.pad_id)
    return enc(ids.to(device), mask.to(device))


@torch.no_grad()
def recall_at_1(enc, tok, pairs, corpus, device, seq_len) -> float:
    """For each held-out query, is its nearest neighbour among ALL positives its own positive."""
    enc.eval()
    q = encode_texts(enc, tok, [p["query"] for p in pairs], device, seq_len)
    d = encode_texts(enc, tok, corpus, device, seq_len)
    nn = (q @ d.T).argmax(1).tolist()
    hits = sum(corpus[j] == p["positive"] for j, p in zip(nn, pairs))
    enc.train()
    return hits / len(pairs)


def smoke(args):
    device = C.pick_device(args.device)
    pairs = C.read_jsonl(args.pairs_jsonl) if args.pairs_jsonl else synthetic_pairs(args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_held = max(4, int(args.heldout_frac * len(pairs)))
    held, train = pairs[:n_held], pairs[n_held:]
    corpus = sorted({p["positive"] for p in pairs})
    seq_len = 96
    chars = sorted(set("".join(p["query"] + p["positive"] for p in pairs)))
    tok = C.CharTokenizer(chars)
    enc = Encoder(tok.vocab_size, seq_len).to(device)
    C.log(f"{len(train)} train pairs, {len(held)} held-out, corpus of {len(corpus)} documents; chance recall@1 = {1 / len(corpus):.3f}")
    r0 = recall_at_1(enc, tok, held, corpus, device, seq_len)
    C.metric(0, recall_at_1=r0)
    opt = C.make_adamw(enc, args.lr, 0.01)
    C.status("train", f"{args.steps} steps of InfoNCE, batch {args.batch}, tau {args.tau}")
    for step in range(1, args.steps + 1):
        lr = C.lr_at(step, args.steps, args.lr, 10, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rng.sample(train, min(args.batch, len(train)))
        q = encode_texts(enc, tok, [b["query"] for b in batch], device, seq_len)
        p = encode_texts(enc, tok, [b["positive"] for b in batch], device, seq_len)
        loss, acc = info_nce(q, p, args.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        fields = dict(loss=loss.item(), in_batch_acc=acc, lr=lr)
        if step % 25 == 0 or step == args.steps:
            fields["recall_at_1"] = recall_at_1(enc, tok, held, corpus, device, seq_len)
        C.metric(step, **fields)
    r1 = recall_at_1(enc, tok, held, corpus, device, seq_len)
    path = os.path.join(args.out, "encoder.pt")
    torch.save({"model": enc.state_dict(), "tokenizer": tok.to_dict(), "seq_len": seq_len}, path)
    C.status("done", f"saved {path}")
    C.result(recall_at_1_before=r0, recall_at_1_after=r1, chance=1 / len(corpus), n_train=len(train), n_heldout=len(held),
             final_loss=loss.item(), checkpoint=path)


# --------------------------------------------------------------------------- real: sentence-transformers


def real(args):
    if not args.pairs_jsonl:
        raise SystemExit("real mode needs --pairs-jsonl (make one with embed_vault.py)")
    st = C.require("sentence_transformers", "sentence-transformers")
    datasets = C.require("datasets")
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    pairs = C.read_jsonl(args.pairs_jsonl)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    is_nomic = "nomic" in args.model.lower()
    qp, dp = ("search_query: ", "search_document: ") if is_nomic else ("", "")
    n_held = max(8, int(args.heldout_frac * len(pairs)))
    held, train = pairs[:n_held], pairs[n_held:]
    C.status("load", args.model)
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = args.max_len
    train_ds = datasets.Dataset.from_list([{"anchor": qp + p["query"], "positive": dp + p["positive"]} for p in train])
    loss = losses.MultipleNegativesRankingLoss(model)
    if args.matryoshka:
        dims = [int(d) for d in args.matryoshka.split(",")]
        loss = losses.MatryoshkaLoss(model, loss, matryoshka_dims=dims)
    corpus = {str(i): dp + t for i, t in enumerate(sorted({p["positive"] for p in pairs}))}
    inv = {v: k for k, v in corpus.items()}
    queries = {str(i): qp + p["query"] for i, p in enumerate(held)}
    relevant = {str(i): {inv[dp + p["positive"]]} for i, p in enumerate(held)}
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, name="heldout", accuracy_at_k=[1, 5], precision_recall_at_k=[1, 5],
                                              map_at_k=[10], ndcg_at_k=[10], mrr_at_k=[10])
    before = evaluator(model)
    C.log(f"before: {before}")
    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(args.out, "trainer"), num_train_epochs=args.epochs, max_steps=args.steps if args.steps else -1,
        per_device_train_batch_size=args.batch, learning_rate=args.lr, warmup_ratio=0.1, bf16=torch.cuda.is_available(),
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=1, save_strategy="no", report_to="none", seed=args.seed)
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator,
                                         callbacks=[C.make_metric_callback()])
    trainer.train()
    after = evaluator(model)
    C.log(f"after: {after}")
    path = os.path.join(args.out, "model")
    model.save(path)
    C.status("done", f"saved {path}")
    key = next((k for k in after if k.endswith("accuracy@1")), None)
    C.result(recall_at_1_before=before.get(key) if key else None, recall_at_1_after=after.get(key) if key else None,
             n_train=len(train), n_heldout=len(held), model=args.model, matryoshka=args.matryoshka, checkpoint=path)


def main():
    args = build_parser().parse_args()
    d = dict(steps=200, batch=32, lr=2e-3) if args.smoke else dict(steps=None, batch=32, lr=2e-5)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    (smoke if args.smoke else real)(args)


if __name__ == "__main__":
    main()
