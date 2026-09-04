"""Embed the reader's own vault and build training pairs for embed_contrastive.py (Lab 07/08).

What this teaches
  * the vault as a corpus: notes/*.md with YAML frontmatter, and
    library/<id>/meta.json ({title, authors, takeaway, topics}) next to
    library/<id>/text.txt holding the extracted paper text
  * chunking (about --chunk-chars characters, cut at paragraph boundaries),
    batched embedding with a sentence-transformers model, and what
    chunks-per-second your card sustains
  * where supervision for a retrieval model comes from when there are no
    labels: pairs the vault already implies,
        title      -> the paper's first chunk (its abstract, usually)
        takeaway   -> the chunk that shares the most key terms with it
        note title -> the note's first chunk
    written as {"query", "positive"} rows that embed_contrastive.py consumes
  * a sanity check that costs nothing: for a few random chunks, print the
    nearest neighbour's title and see whether it makes sense

How to run
  smoke (offline, seconds): a tiny fake vault is written under --out and
  embedded with a hashed bag-of-words encoder (no model download):
    python lab/recipes/embed_vault.py --smoke
  real (RTX 5090 or CPU):
    python lab/recipes/embed_vault.py --vault ~/Cortex --out out/embed_vault
    python lab/recipes/embed_vault.py --vault ~/Cortex --model BAAI/bge-small-en-v1.5
  needs (real): pip install sentence-transformers numpy   (nomic also needs einops)

Outputs under --out: embeddings.npy (float32, one row per chunk, L2-normalized),
chunks.jsonl ({doc, kind, title, text, chunk_index}), pairs.jsonl ({query, positive}).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402

STOP = set("the a an of and or to in on for with that this is are was were be by as at from it its we our their they which "
           "into than then these those such can may will not but if via".split())


def build_parser():
    p = C.base_parser("embed_vault", __doc__.split("\n")[0])
    p.add_argument("--vault", default=os.path.expanduser("~/Cortex"))
    p.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    p.add_argument("--fallback-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--chunk-chars", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-chunks-per-doc", type=int, default=64)
    return p


# --------------------------------------------------------------------------- reading the vault


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal YAML frontmatter reader: `key: value` lines between two --- fences."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, m.group(2)


def chunk_text(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(para) > chunk_chars:                       # a single huge paragraph: hard cut
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(para), chunk_chars):
                chunks.append(para[i: i + chunk_chars])
            continue
        if len(cur) + len(para) + 2 > chunk_chars and cur:
            chunks.append(cur)
            cur = para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur:
        chunks.append(cur)
    return chunks[:max_chunks]


def read_vault(vault: str, chunk_chars: int, max_chunks: int) -> tuple[list[dict], list[dict], int, int]:
    """Return (chunks, pairs, n_notes, n_papers)."""
    chunks, pairs = [], []
    n_notes = n_papers = 0
    for path in sorted(glob.glob(os.path.join(vault, "notes", "*.md"))):
        meta, body = parse_frontmatter(open(path, encoding="utf-8", errors="replace").read())
        title = meta.get("title") or os.path.splitext(os.path.basename(path))[0].replace("-", " ")
        cs = chunk_text(body, chunk_chars, max_chunks)
        if not cs:
            continue
        n_notes += 1
        doc = "notes/" + os.path.basename(path)
        for i, c in enumerate(cs):
            chunks.append({"doc": doc, "kind": "note", "title": title, "text": c, "chunk_index": i})
        pairs.append({"query": title, "positive": cs[0]})
    for d in sorted(glob.glob(os.path.join(vault, "library", "*"))):
        meta_path, text_path = os.path.join(d, "meta.json"), os.path.join(d, "text.txt")
        if not (os.path.isfile(meta_path) and os.path.isfile(text_path)):
            continue
        try:
            meta = json.load(open(meta_path))
        except json.JSONDecodeError:
            continue
        title = meta.get("title") or os.path.basename(d)
        cs = chunk_text(open(text_path, encoding="utf-8", errors="replace").read(), chunk_chars, max_chunks)
        if not cs:
            continue
        n_papers += 1
        doc = "library/" + os.path.basename(d)
        for i, c in enumerate(cs):
            chunks.append({"doc": doc, "kind": "paper", "title": title, "text": c, "chunk_index": i})
        pairs.append({"query": title, "positive": cs[0]})
        takeaway = meta.get("takeaway")
        if takeaway:
            terms = {w for w in re.findall(r"[a-z]{5,}", takeaway.lower()) if w not in STOP}
            if terms:
                best = max(cs, key=lambda c: len(terms & set(re.findall(r"[a-z]{5,}", c.lower()))))
                pairs.append({"query": takeaway, "positive": best})
    return chunks, pairs, n_notes, n_papers


# --------------------------------------------------------------------------- embedders


class HashedBagOfWords:
    """Offline stand-in for a real encoder: hash each word into one of `dim` buckets, count, L2-normalize."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts, batch_size=64, **kw) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


def load_encoder(name: str, fallback: str):
    st = C.require("sentence_transformers", "sentence-transformers")
    try:
        model = st.SentenceTransformer(name, trust_remote_code=True)
        return model, name
    except Exception as e:
        C.log(f"could not load {name} ({type(e).__name__}: {e}); falling back to {fallback}")
        return st.SentenceTransformer(fallback), fallback


def embed_all(encoder, texts: list[str], batch: int, prefix: str) -> np.ndarray:
    rows, t0, done = [], time.perf_counter(), 0
    for i in range(0, len(texts), batch):
        rows.append(np.asarray(encoder.encode([prefix + t for t in texts[i: i + batch]], batch_size=batch, normalize_embeddings=True)))
        done += len(rows[-1])
        C.metric(done, chunks_embedded=done, chunks_per_s=done / max(1e-9, time.perf_counter() - t0))
    return np.concatenate(rows).astype(np.float32)


# --------------------------------------------------------------------------- smoke vault


def write_fake_vault(root: str) -> str:
    notes = os.path.join(root, "notes")
    lib = os.path.join(root, "library")
    os.makedirs(notes, exist_ok=True)
    weather, cooking, machines = (C.TOPICS[k] for k in ("weather", "cooking", "machines"))
    for name, title, sents in (("weather-log", "A weather log", weather), ("kitchen-notes", "Kitchen notes", cooking),
                               ("shop-diary", "Workshop diary", machines)):
        body = ". ".join(sents[:7]) + ".\n\n" + ". ".join(sents[7:]) + "."
        with open(os.path.join(notes, name + ".md"), "w") as f:
            f.write(f'---\ntitle: "{title}"\nkind: permanent\n---\n{body}\n')
    stories = C.STORIES.split("\n")
    papers = [("p1", "Small animals and lost objects", "a fox and a frog recover a red ball from a river", stories[:5]),
              ("p2", "Machines, gardens and the night sky", "a small robot keeps a fallen star in a jar by the window", stories[5:])]
    for pid, title, takeaway, body in papers:
        d = os.path.join(lib, pid)
        os.makedirs(d, exist_ok=True)
        json.dump({"title": title, "authors": ["synthetic"], "takeaway": takeaway, "topics": ["synthetic"]}, open(os.path.join(d, "meta.json"), "w"))
        with open(os.path.join(d, "text.txt"), "w") as f:
            f.write("\n\n".join(body) + "\n")
    return root


def main():
    args = build_parser().parse_args()
    C.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.smoke:
        args.vault = write_fake_vault(os.path.join(args.out, "fake_vault"))
        args.chunk_chars = min(args.chunk_chars, 300)
        C.log(f"smoke: wrote a fake vault at {args.vault}")
    C.status("read", f"reading {args.vault}")
    chunks, pairs, n_notes, n_papers = read_vault(args.vault, args.chunk_chars, args.max_chunks_per_doc)
    if not chunks:
        raise SystemExit(f"no notes/*.md or library/*/text.txt found under {args.vault}")
    C.log(f"{n_notes} notes, {n_papers} papers, {len(chunks)} chunks, {len(pairs)} pairs")

    if args.smoke:
        encoder, model_name, prefix = HashedBagOfWords(), "hashed-bag-of-words", ""
    else:
        encoder, model_name = load_encoder(args.model, args.fallback_model)
        prefix = "search_document: " if "nomic" in model_name.lower() else ""
    C.status("embed", f"{len(chunks)} chunks with {model_name}")
    emb = embed_all(encoder, [c["text"] for c in chunks], args.batch, prefix)

    np.save(os.path.join(args.out, "embeddings.npy"), emb)
    C.write_jsonl(os.path.join(args.out, "chunks.jsonl"), chunks)
    C.write_jsonl(os.path.join(args.out, "pairs.jsonl"), pairs)

    # nearest-neighbour sanity check: does the closest other chunk come from a related document?
    C.status("check", "nearest neighbours of 5 random chunks")
    rng = random.Random(args.seed)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    same_doc = 0
    picks = rng.sample(range(len(chunks)), min(5, len(chunks)))
    for i in picks:
        j = int(sims[i].argmax())
        same_doc += int(chunks[i]["doc"] == chunks[j]["doc"])
        C.log(f"  [{chunks[i]['title']}] chunk {chunks[i]['chunk_index']} -> top-1: [{chunks[j]['title']}] chunk {chunks[j]['chunk_index']} (cos {sims[i, j]:.3f})")
    C.status("done", f"wrote {args.out}")
    C.result(n_notes=n_notes, n_papers=n_papers, n_chunks=len(chunks), n_pairs=len(pairs), dim=int(emb.shape[1]), model=model_name,
             nn_same_doc_frac=same_doc / len(picks), embeddings=os.path.join(args.out, "embeddings.npy"),
             chunks=os.path.join(args.out, "chunks.jsonl"), pairs=os.path.join(args.out, "pairs.jsonl"))


if __name__ == "__main__":
    main()
