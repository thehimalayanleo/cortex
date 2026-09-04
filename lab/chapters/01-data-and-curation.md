---
title: "Lab 01: Data, tokenization, and curation"
kind: permanent
topics: [lab]
chapter: 1
station: data
recipe: recipes/curate.py
reading_time: 50 min
---

## What you will be able to do

- Train a byte-level BPE tokenizer from scratch, explain what each merge does, and compute bytes per token and bits per byte so that two models with different vocabularies can be compared on one scale.
- Turn a pile of documents into packed token shards with document boundaries the model can learn from, and say what packing costs and what it saves.
- Remove exact and near duplicates with MinHash and locality-sensitive hashing, choosing the band parameters from a target Jaccard threshold instead of guessing.
- Choose mixture weights for a multi-source corpus with the repetition constraint in view, and use embeddings plus k-means to see and fix an unbalanced corpus.
- Decontaminate training data against the evaluations you plan to report, and state precisely what was removed.

## The idea in one paragraph

A language model never reads text; it reads a stream of integers, and everything about that stream is a decision you make before the first gradient step. Tokenization decides what an integer stands for (a byte, a word piece, a whole common word). Packing decides how the stream is cut into rectangles the GPU can process. Curation decides which documents get into the stream at all, how many times each one appears, and in what proportion the sources are mixed. Every one of those decisions moves the final loss, and most are cheap to get right before a run and expensive to fix after it starts, because the tokenized cache is the root of the dependency graph that every later step hangs from.

## The math

### Tokens and byte-pair encoding

A document is a byte string $s \in \{0,\dots,255\}^*$. A tokenizer is a pair of functions, an encoder $\mathrm{enc}: \{0,\dots,255\}^* \to \{0,\dots,V-1\}^*$ and a decoder $\mathrm{dec}$ with $\mathrm{dec}(\mathrm{enc}(s)) = s$ for every $s$. $V$ is the vocabulary size. The model outputs a categorical distribution over $V$ symbols at each position, so $V$ sets the width of the output softmax and of the embedding table, each of size $V \times d_{\text{model}}$.

The simplest lossless tokenizer is the identity on bytes, $V = 256$. Its problem is length: a 4 KB document becomes 4,096 positions, and attention cost grows with the square of that. The opposite extreme, one id per whole word, has unbounded $V$ and no way to represent a word it has not seen. Byte-pair encoding (BPE) sits between the two. Start with the 256 byte symbols. Count every adjacent pair of symbols in the training text, merge the most frequent pair into a new symbol, and repeat. After $M$ merges the vocabulary has $V = 256 + M$ symbols plus any special tokens such as end-of-text, and every byte string still has an encoding, because the base bytes never leave the vocabulary. Encoding a new string applies the merges in the order they were learned. Frequent strings such as " the" collapse to one id; rare strings fall apart into pieces, down to single bytes if necessary. Production tokenizers add a pre-tokenization regex that splits text into words, numbers, and punctuation before merging, so that a merge never crosses a word boundary; that is why "the" and " the" are different ids and why digits are usually split into short groups.

The quantity that tells you what a tokenizer bought you is the compression ratio $\rho$, in bytes per token, measured on held-out text from the distribution you will train on:

$$\rho = \frac{\text{bytes}}{\text{tokens}}.$$

The ratio depends on the domain, so measure it per domain: English prose compresses well, code and non-Latin scripts and numbers compress worse under a vocabulary trained mostly on English.

The ratio also fixes how to compare models that use different tokenizers. A model's loss is measured per token, $L_{\text{tok}}$ nats. Two models with different vocabularies make different numbers of predictions for the same text, so their per-token losses are not comparable. What is comparable is the total code length: the text costs $\sum_t -\ln p(x_t \mid x_{<t})$ nats however it is chopped. Divide by bytes rather than tokens and convert nats to bits:

$$\text{BPB} = \frac{L_{\text{tok}}}{\rho \ln 2}.$$

Worked example: a model with $L_{\text{tok}} = 2.4$ nats at $\rho = 4.0$ has $\text{BPB} = 2.4 / (4.0 \times 0.693) = 0.87$. A character-level model with $L_{\text{tok}} = 1.2$ nats and $\rho = 1$ has $\text{BPB} = 1.73$. The first model is better even though its per-token loss is twice as large. One more consequence: at a fixed context of $T$ tokens, a tokenizer with larger $\rho$ sees $\rho T$ bytes of history. The lab's data station shows this at the character level: "the" is three ids and three positions there, and one id in Marin's cache.

### Packing

Documents have varying lengths; a GPU wants a $B \times T$ rectangle. Padding each document to $T$ wastes a fraction $1 - \mathbb{E}[\min(\ell, T)] / T$ of every batch, where $\ell$ is document length in tokens. For web text, whose median length is far below a 4,096-token window, the waste is most of the batch. Packing instead concatenates all documents into one stream with an end-of-text id between them and cuts the stream into consecutive windows of $T$ tokens. Waste is zero. The price is that a window can hold the tail of one document and the head of the next, and the tokens of the second document attend to the first. The model learns that the end-of-text id resets context, and this works in practice; a block-diagonal attention mask (each document attends only to itself) is a cheap correction that Llama 3 reports using for its long-context stages. Whatever you choose, the end-of-text id must be present. Without it the model has no signal that the topic changed and spends capacity trying to predict across boundaries.

Each window yields $T$ training examples at once, because the target is the input shifted left by one and the causal mask makes position $t$ a prediction of $x_{t+1}$ from $x_{\le t}$. This is why decoders are data-efficient per token compared with models that predict one label per sequence.

### The pipeline as a dependency graph

Marin writes each stage as a step with declared inputs and outputs and runs the graph in dependency order, like a Makefile: download, then tokenize into a `TokenizedCache`, then train, then evaluate. The `tokenized(...)` call quoted in the data station produces a cache; every `train_lm` call that names that cache depends on it and never re-tokenizes. Two properties follow. Any change to a tokenizer or a filter invalidates every downstream step, which is correct and expensive, so curation decisions are made once and versioned. And a mixture is a dictionary from caches to weights, so adding a source is adding a node, not rewriting the loader. Lab 02 reads the training node line by line.

### Quality filtering

A quality filter is a per-document decision $q(s) \in \{0, 1\}$ or score $q(s) \in [0, 1]$. Heuristic filters check measurable properties: mean word length within a range, fraction of lines ending in punctuation, fraction of alphabetic characters, ratio of symbols to words, presence of common stop words, absence of repeated lines and repeated $n$-grams. These are the rules published with Gopher and reused by later web corpora. Model-based filters train a classifier to separate a reference set (encyclopedia, books, textbook-like text) from random crawl, or score documents by the perplexity of a small language model trained on the reference. The classifier gives $q(s) = P(\text{reference} \mid s)$ and you keep documents with $q(s) > \tau$.

Both are selection by proxy, and the proxy leaks. A perplexity filter built on an encyclopedia keeps encyclopedia prose and discards dialogue, code, and tables that a model needs. The only real test of a filter is the one in Lab 02: train two small models at equal token budgets, one on filtered and one on unfiltered data, and compare held-out loss on the distribution you care about. Treat $\tau$ as a hyperparameter of that experiment.

### Exact and near duplicates

Duplicates hurt twice. They waste tokens, and repeated documents get memorized: the loss on them collapses while the loss on fresh text does not move, and the model reproduces them verbatim at sampling time. Exact duplicates are easy: normalize whitespace and case, hash the result, keep one document per hash.

Near duplicates (boilerplate, syndicated articles, versions of a page) need a similarity. Represent a document by its set of word $n$-grams, called shingles, $S(s)$, with $n$ around 5 for text. The Jaccard similarity of two documents is

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}.$$

Computing $J$ for every pair is quadratic in corpus size. MinHash makes it linear. Take a random permutation $\pi$ of the universe of shingles and define $h_\pi(A) = \min_{a \in A} \pi(a)$. Then

$$P\left[h_\pi(A) = h_\pi(B)\right] = J(A, B).$$

Proof: because $\pi$ is a uniformly random permutation, the minimum of $\pi$ over $A \cup B$ is attained at a uniformly random element of $A \cup B$. The two minima agree exactly when that element lies in $A \cap B$, which happens with probability $|A \cap B| / |A \cup B|$. With $k$ independent permutations the fraction of agreeing coordinates, $\hat J$, is an unbiased estimator of $J$ with variance $J(1 - J) / k$; at $k = 128$ and $J = 0.5$ the standard error is $\sqrt{0.25 / 128} = 0.044$. In practice a permutation is replaced by a hash function with a random salt.

The signatures still have to be compared pairwise, so add locality-sensitive hashing. Split the $k$ coordinates into $b$ bands of $r$ rows each, $k = br$. Two documents are candidates if all $r$ rows agree in at least one band. The probability that one band agrees is $J^r$, so the probability that at least one of $b$ bands agrees is

$$P(\text{candidate}) = 1 - \left(1 - J^r\right)^b.$$

This is an S-curve in $J$ with its steepest point near $J^* \approx (1/b)^{1/r}$. Worked example with $b = 20$, $r = 5$: $J^* = 0.05^{0.2} = 0.55$. At $J = 0.8$ the candidate probability is $1 - (1 - 0.328)^{20} = 1 - 0.672^{20} \approx 0.9996$. At $J = 0.3$ it is $1 - (1 - 0.00243)^{20} \approx 1 - e^{-0.0486} = 0.047$. Documents that share 80 percent of their shingles are almost always caught, documents that share 30 percent are almost always ignored, and the cost is one hash-table insert per band per document instead of a comparison per pair. Candidates go into a union-find structure; keep one document per connected component. Lee et al. (2022) measured that this kind of deduplication improves held-out loss at equal tokens and cuts verbatim memorization by an order of magnitude.

### Mixtures and how to pick weights

A corpus built from $K$ sources with token counts $D_1, \dots, D_K$ is trained on by sampling source $i$ with probability $w_i$, $\sum_i w_i = 1$. The training distribution is $p_{\text{mix}} = \sum_i w_i p_i$, and the expected training loss decomposes as

$$L_{\text{mix}}(\theta) = \sum_i w_i L_i(\theta),$$

where $L_i$ is the expected loss on source $i$. Held-out loss on any single source is therefore an objective the model was never asked to minimize alone; mixing is a choice of what to trade against what.

The first constraint on $w$ is repetition. With a budget of $D$ training tokens, source $i$ is seen for

$$e_i = \frac{w_i D}{D_i}$$

epochs. Muennighoff et al. (2023) measured that repeating data for up to about four epochs costs little relative to fresh data, and that beyond that the return per repeated token falls off quickly. So $w_i \lesssim 4 D_i / D$ is a ceiling, and the sum of those ceilings tells you whether the corpus is large enough for the budget at all.

Within the feasible set, weights are chosen by evaluation. Proportional sampling, $w_i = D_i / \sum_j D_j$, over-represents whatever is abundant (crawl) and under-represents what is scarce and valuable (books, code, math). Upweighting scarce high-quality sources by factors of two to five is common and defensible if you measure per-source held-out loss for each candidate. DoReMi (Xie et al., 2023) replaces the guesswork with a proxy model: train a small reference, then train a second small proxy whose sampling weights are pushed toward the sources where it lags the reference most (a minimax over excess loss), and reuse the resulting $w$ for the full run. Its caveat is that the weights are optimized at proxy scale for held-out loss, not for downstream tasks. Lab 03 returns to weights in the setting where they change during a run.

### Balancing with embeddings and clusters

Source labels are coarse. Inside a crawl there are regions (SEO spam, product listings, forum threads, scientific text) whose proportions you cannot read off a URL. Embed every document with an encoder such as nomic-embed-text-v1.5, run k-means with $K$ centroids in the embedding space, and count documents per cluster, $n_1, \dots, n_K$. Lab 08 builds a toy version of exactly this on the cluster station. Two uses follow. Inspect the largest clusters and the clusters with the lowest mean quality score, and decide whether to drop them. Then resample so that no cluster exceeds a cap: keep document $s$ in cluster $c(s)$ with probability $\min(1, n_{\max} / n_{c(s)})$, which flattens the head of the distribution without touching the tail. Semantic deduplication (Abbas et al., 2023) uses the same embeddings: within a cluster, pairs with cosine similarity above a threshold are treated as duplicates that MinHash missed because they are paraphrases rather than copies. Set the threshold by reading pairs at several similarities; there is no universal value.

### Contamination

An evaluation is worthless if its test items are in the training set. Define contamination at the $n$-gram level: a training document is contaminated by an evaluation example if they share any $n$-gram for an $n$ chosen so that chance overlap is negligible. GPT-3's report used 13-gram overlap; other reports use character windows of about 50. Decontamination removes or trims the training documents that match; the report should state $n$, the evaluation sets scanned, and the number of documents removed. The reverse direction is also useful: flag evaluation items that appear in the training data and report scores with and without them. Neither direction catches paraphrases; the embedding pass above is the tool for that.

## Build it small

Two mechanisms in one file: a byte-level BPE trainer with an encoder, and MinHash signatures compared against exact Jaccard. Plain Python, no dependencies.

```python
# Lab 01, build it small: byte-level BPE and MinHash near-duplicate detection.
import random, collections, hashlib

# ---------- BPE ----------
def bpe_train(text, num_merges):
    words = collections.Counter(text.split(" "))          # word -> count
    seqs = {w: tuple(w.encode("utf-8")) for w in words}    # word -> tuple of byte ids
    merges = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for w, c in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        new_id = 256 + len(merges)
        merges.append(((a, b), new_id))
        for w, s in seqs.items():                          # apply the merge everywhere
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(new_id); i += 2
                else:
                    out.append(s[i]); i += 1
            seqs[w] = tuple(out)
    return merges

def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    for (a, b), new_id in merges:                          # merges apply in training order
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        ids = out
    return ids

corpus = ("the cat sat on the mat . the dog sat on the log . " * 20
          + "a cat and a dog met on the mat . ") * 5
merges = bpe_train(corpus, num_merges=30)
sample = "the cat sat on the log ."
ids = bpe_encode(sample, merges)
print(f"bytes {len(sample.encode())}  tokens {len(ids)}  "
      f"bytes/token {len(sample.encode()) / len(ids):.2f}")
print("first merges:", [(bytes([a]) if a < 256 else a, bytes([b]) if b < 256 else b)
                        for (a, b), _ in merges[:5]])

# ---------- MinHash ----------
def shingles(doc, n=3):
    w = doc.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def minhash(shingle_set, k=128, seed=0):
    rng = random.Random(seed)
    salts = [rng.getrandbits(64) for _ in range(k)]
    sig = []
    for s in salts:
        sig.append(min(int(hashlib.blake2b(f"{s}:{sh}".encode(), digest_size=8).hexdigest(), 16)
                       for sh in shingle_set))
    return sig

def jaccard(a, b):
    return len(a & b) / len(a | b)

d1 = "the quick brown fox jumps over the lazy dog and runs into the forest at night"
d2 = "the quick brown fox jumps over the lazy cat and runs into the forest at night"
d3 = "gradient descent with momentum converges faster on ill conditioned quadratics"
S1, S2, S3 = shingles(d1), shingles(d2), shingles(d3)
m1, m2, m3 = minhash(S1), minhash(S2), minhash(S3)
est = lambda x, y: sum(p == q for p, q in zip(x, y)) / len(x)
print(f"J(d1,d2) exact {jaccard(S1, S2):.3f}  minhash {est(m1, m2):.3f}")
print(f"J(d1,d3) exact {jaccard(S1, S3):.3f}  minhash {est(m1, m3):.3f}")
```

Expected output: `bytes 24  tokens 13  bytes/token 1.85`; the first merges are `at`, `th`, then `th`+`e` into `the`, then `on` and `og`; and the MinHash estimates sit close to the exact Jaccard values, $0.647$ exact against about $0.66$ estimated for the near-duplicate pair and $0.000$ for the unrelated pair. The compression ratio is poor because the corpus is tiny and the merge budget is 30. Note what the third merge did: it built a whole word out of a previous merge, which is how BPE reaches multi-character pieces without ever storing a word list. Change the seed in `minhash` and the estimate moves by about one standard error, $\sqrt{J(1-J)/128}$.

## Build it real

`recipes/curate.py` runs the full pipeline on the 5090 and writes packed shards that `recipes/pretrain_nano.py` (Lab 02) reads directly. It takes a Hugging Face dataset name (`roneneldan/TinyStories` for a fast first run, a FineWeb sample for a realistic one), a text field, and an output directory, and runs the stages in this order, each one writing a JSON line to `curation_report.jsonl` with its input count, output count, and wall time.

The first stage is normalization and heuristic filtering with the Gopher-style rules: `--min_words`, `--max_words`, `--max_symbol_ratio`, `--min_alpha_frac`, and a repeated-line check, with `--quality none` to skip them for the unfiltered control you need in Lab 02. The second is exact deduplication by hash of the normalized text. The third is MinHash with `--shingle 5 --minhash_k 128 --bands 20 --rows 5`, implemented with vectorized universal hashing in NumPy rather than per-shingle Python hashing, and a union-find over LSH candidates. The fourth, optional, is `--embed_model nomic-ai/nomic-embed-text-v1.5 --clusters 64 --cluster_cap 0.05`, which embeds documents with sentence-transformers under the `search_document:` prefix, runs k-means, writes cluster sizes and ten sample documents per cluster to the report, and downsamples any cluster above the cap. The fifth is `--decontam evals.jsonl --ngram 13`, which builds a set of 13-gram hashes from the supplied evaluation items and drops any training document that hits one. The last stage tokenizes with `--tokenizer` (a Hugging Face tokenizer name; the GPT-2 tokenizer keeps every id under $2^{16}$ and lets you write `uint16` shards, and any vocabulary above 65,535 ids, including Llama 3's, forces `--dtype uint32`), appends the end-of-text id after every document, and writes `train.bin` and `val.bin` with `--val_frac 0.005` held out at the document level, never the window level, so no window straddles the split.

Watch three things in the logs. The stage-by-stage document counts: a MinHash stage that removes more than a third of a crawl sample, or less than a percent, means the threshold or the shingle size is wrong for that data. The compression ratio the tokenization stage prints per source. And, for the embedding stage, documents per second after the first minute, because that stage is the only one that is GPU-bound and its throughput sets your wall time.

To estimate that wall time, the encoder is 137M parameters and a forward pass costs about $2 \times 137 \times 10^6$ FLOPs per token, so a 256-token document costs about $7 \times 10^{10}$ FLOPs. If the 5090 sustains $R$ FLOP/s on this workload, throughput is $R / (7 \times 10^{10})$ documents per second. Under the assumption $R = 10^{14}$ (a round figure; the recipe prints the measured rate), that is about 1,400 documents per second and one million documents in about twelve minutes. The MinHash stage runs on the CPU and, for a million documents of a few hundred words, will typically be the slower stage; the recipe shards it across processes with `--workers`.

## How it goes wrong

1. The loss curve is fine but samples contain no line breaks, or code comes out with mangled indentation. The tokenizer was trained on prose and represents runs of spaces and newlines as long byte sequences, so the model rarely sees them and never learns them. Fix: train or choose a tokenizer on the mixture you will use, and check the compression ratio per source before tokenizing the corpus.

2. Generated text drifts from one topic into an unrelated one mid-sentence. There is no end-of-text id between packed documents, so the model learned that topics change without warning. Fix: append the id after every document; optionally use a block-diagonal mask.

3. Training loss is much lower than held-out loss on the same source after the first epoch. Duplicates. The model has memorized repeated documents. Fix: exact deduplication, then MinHash, and re-check the epoch count $e_i$ per source.

4. Deduplication removed a large fraction of a source and downstream loss on that source got worse. The LSH threshold was too low or the shingle size too small, so templated but distinct documents (product pages, legal boilerplate, references) were merged into single components and thrown away. Fix: raise $J^*$ by increasing $r$ or decreasing $b$, and sample pairs near the threshold to read before committing.

5. The quality filter improved loss on encyclopedic text and hurt loss on everything else. The classifier or the reference perplexity model encoded a genre, not quality. Fix: build the reference set from several genres, or keep the filter but cap how much of any one source it may remove, and validate with the equal-token experiment.

6. Benchmarks look strong; a colleague finds the test items in the shards. No decontamination, or decontamination run before a later stage re-introduced documents from a cache. Fix: make decontamination the last filter before tokenization and record the count of removed documents in the report.

7. Token ids wrap around and the model trains on garbage without any error. The vocabulary exceeds 65,535 ids and the shards were written as `uint16`. Fix: check `max(id) < 2**16` before choosing the dtype, and assert it when loading.

8. The corpus is dominated by one cluster after embedding, but the source-level mixture looked balanced. A single source hides a skewed internal distribution (a crawl that is mostly listings). Fix: apply the cluster cap, then re-check the source-level weights, because capping changes the effective $D_i$.

## Measure it

Measure the tokenizer by compression ratio per domain on held-out text, and by the fraction of tokens that are single bytes (a high byte fraction on a domain means the vocabulary does not cover it). Report model quality in bits per byte so the number survives a tokenizer change.

Measure deduplication by the fraction of documents removed at each stage and by verbatim memorization: sample from a trained model with a prefix taken from the training set and count how often the continuation matches the training text for 50 or more tokens. Lee et al. (2022) report that deduplication cuts that rate roughly tenfold; the direction is what matters, and the number for your corpus comes from your run.

Measure the filter and the mixture by the only test that counts: two small models at equal tokens (Lab 02's recipe at its smallest setting), one per variant, compared by held-out loss on each source. A difference of $0.01$ nats per token is at the edge of seed noise at that scale; run two seeds before believing anything smaller than $0.02$.

Measure contamination by the number of evaluation items with any 13-gram match in the training shards, reported alongside the evaluation score. A clean corpus has zero such items for every set you report.

Measure balance by the entropy of the cluster-size distribution, $H = -\sum_c (n_c / n) \ln(n_c / n)$, compared with $\ln K$. A corpus where most mass sits in a handful of clusters has $H$ far below $\ln K$; the cap raises $H$ and you can watch it do so.

## Exercises

1. Train the toy BPE with 300 merges on a paragraph of English and a paragraph of Python of equal byte length, then compute bytes per token for each. Check: the code ratio is lower, and the merges list contains multi-space runs only if the code paragraph is long enough for them to be frequent.

2. Derive the standard error of $\hat J$ at $k = 64$ and $J = 0.9$, then confirm it empirically by re-running the MinHash comparison with 50 seeds. Check: $\sqrt{0.09 / 64} = 0.0375$.

3. For $k = 128$, choose $(b, r)$ so that $J^* \approx 0.7$, then compute the candidate probability at $J = 0.6$ and $J = 0.85$. Check: $b = 16$, $r = 8$ gives $J^* = 0.707$, and the two probabilities are about $0.24$ and $0.99$.

4. You have three sources of 20 B, 5 B, and 1 B tokens and a budget of 40 B. Compute the maximum feasible weight of the smallest source under the four-epoch ceiling, and the weights of a mixture that upweights the 5 B source by three relative to proportional sampling. Check: the smallest source can take at most $w = 0.1$; proportional weights are $(0.769, 0.192, 0.038)$, and tripling the middle one before renormalizing gives $(0.556, 0.417, 0.028)$.

5. Write a 13-gram decontamination pass over the TinyStories validation split against its own training split and report how many training stories hit. Check: the count is not zero, because TinyStories is synthetic and repetitive; decide whether that is contamination or the nature of the data, and write one sentence defending the decision.

6. Embed 10,000 TinyStories documents with nomic-embed-text-v1.5 at 64 Matryoshka dimensions and at 768, run k-means with $K = 32$ on each, and compare the cluster assignments by adjusted mutual information. Check: the two clusterings agree well above chance, which tells you how much of the balancing signal survives the cheap embedding.

## Test yourself

1. A colleague reports that model A has per-token loss 2.1 and model B has 2.6, and concludes A is better. Model A uses a 256-byte vocabulary and model B a 32k BPE with ratio 3.8 bytes per token. Who is right?

<details><summary>Answer</summary>
Convert to bits per byte. A: $2.1 / (1 \times 0.693) = 3.03$ BPB. B: $2.6 / (3.8 \times 0.693) = 0.99$ BPB. B is far better; A's per-token loss is low because each prediction covers a single byte. Per-token loss across different tokenizers is not a comparison.
</details>

2. Prove that MinHash with a hash function that is not a random permutation (for example, one with many collisions) biases $\hat J$, and say in which direction.

<details><summary>Answer</summary>
Collisions map distinct shingles to the same value, so two documents can agree on the minimum without sharing the element that attained it. That inflates agreement, so $\hat J$ is biased upward. With a 64-bit hash over sets of a few thousand shingles the collision probability is negligible; with a 32-bit hash over a large union it is not, and the bias shows up as false near-duplicates. The proof of $P[\text{agree}] = J$ used that the argmin is uniform over the union, which collisions break.
</details>

3. You pack documents with an end-of-text id and no block-diagonal mask. A window contains the last 300 tokens of document 1 and the first 1,748 of document 2. What fraction of the window's training signal for document 2 is contaminated by document 1, and why is this usually tolerable?

<details><summary>Answer</summary>
Every position in document 2 can attend to the 300 tokens of document 1, so in principle all 1,748 targets see foreign context. In practice the model learns that attention across the end-of-text id carries no information about the next token and assigns it near-zero weight, so the effective contamination is small. It stops being tolerable when documents are short relative to $T$ (many boundaries per window) or when the task rewards long-range recall, which is why long-context stages use the mask.
</details>

4. Spot the bug. A dedup script computes shingles as `set(doc.split())` and uses $b = 20$, $r = 5$. On a news corpus it flags 40 percent of articles as near-duplicates.

<details><summary>Answer</summary>
The shingles are single words, not $n$-grams, so any two articles with similar vocabulary have high Jaccard similarity regardless of content. Word-level unigram sets of English articles overlap heavily. Use $n \ge 3$ word shingles (5 is standard), which encode order, and re-check the removal rate.
</details>

5. Estimate the number of 13-gram hashes you must store to decontaminate against an evaluation suite of 20,000 items averaging 60 tokens, and the memory at 8 bytes per hash.

<details><summary>Answer</summary>
Each item has $60 - 13 + 1 = 48$ overlapping 13-grams, so about $20{,}000 \times 48 = 960{,}000$ hashes, under 8 MB at 8 bytes each. Decontamination is cheap; the cost is the scan over the training corpus, which is one pass of hashing every 13-gram of every document and one set lookup each.
</details>

6. Under DoReMi the proxy upweights sources on which it lags the reference. A source that is intrinsically high-entropy (random-looking logs) will always show high loss. Does DoReMi upweight it, and what stops that?

<details><summary>Answer</summary>
No. DoReMi weights by excess loss, the proxy's loss minus the reference model's loss on the same source, not by absolute loss. A high-entropy source has high loss for both models and small excess, so it is not upweighted. This is the reason the reference model exists; without it, the method would chase noise.
</details>

7. You raise the MinHash threshold from $J^* = 0.55$ to $0.85$ and the model's held-out loss on the crawl source improves, but verbatim memorization gets worse. Explain both effects.

<details><summary>Answer</summary>
At the higher threshold, fewer documents are removed, so the corpus is larger and more diverse, and loss at equal tokens improves because there is less repetition of what was kept. But pairs with $J$ between 0.55 and 0.85 (near-copies with edits) now survive, and the model sees their shared spans repeatedly, which is exactly what drives verbatim regurgitation. The two metrics pull in different directions and the threshold is a trade-off to be reported, not a constant.
</details>

8. Your budget is $D = 100$ B tokens. Sources: crawl 500 B, code 30 B, books 8 B. You want $w_{\text{books}} = 0.15$. Is this feasible under the four-epoch ceiling, and what is the consequence if you do it anyway?

<details><summary>Answer</summary>
$e_{\text{books}} = 0.15 \times 100 / 8 = 1.875$ epochs, which is feasible. The trap is the opposite direction: if you wanted $w_{\text{books}} = 0.4$, that is $5$ epochs, over the ceiling, and the tokens beyond about the fourth epoch would return much less than fresh crawl tokens would. People who upweight small high-quality sources by feel often cross this line without computing it.
</details>

9. A filter keeps documents with reference-LM perplexity below a threshold. Show with a one-line argument why this filter necessarily lowers the entropy of the retained corpus relative to the reference model's view, and why that can hurt a model that must later handle the full distribution.

<details><summary>Answer</summary>
Low perplexity under the reference means high probability under the reference, so the retained set is the part of the crawl the reference already predicts well. Its cross-entropy under the reference is lower by construction. A model trained on it is trained on the easy region of the space and sees fewer of the tokens (dialogue, code, tables, rare scripts) that make up the tails, so its loss on the full distribution can be worse even as its loss on the reference-like region improves. The equal-token experiment on the target distribution is the check.
</details>

10. Why does the epoch-level decision (how many times to see a document) belong at curation time and not at training time, given that the training loop could simply stop sampling a source after four epochs?

<details><summary>Answer</summary>
It can, but the schedule interacts with the mixture and the learning-rate schedule: stopping a source late in the run changes the effective mixture during the cooldown, which Lab 03 shows is the most sensitive part of the run. Deciding $w$ and $D$ jointly at curation time keeps the mixture stationary, or makes any non-stationarity a deliberate mid-training choice rather than an accident of a source running out.
</details>

## What will change, what will not

The identity $\text{BPB} = L_{\text{tok}} / (\rho \ln 2)$ is arithmetic, and the fact that a loss is only comparable as a code length per unit of raw data will hold for any tokenizer, including ones that do not exist yet. BPE itself is a particular greedy compressor and may be displaced: byte-level models with learned patching, tokenizer-free architectures, and vocabularies that adapt during training have all been proposed. When that happens the compression ratio and the BPB calculation survive unchanged; only the value of $\rho$ moves.

The MinHash identity $P[\text{agree}] = J$ and the LSH S-curve are theorems about random permutations and will not change. The choice of $b$, $r$, $n$, and the threshold near $0.55$ are conventions tuned to English web text, and they will move as corpora become more multilingual and more synthetic. Embedding-based semantic deduplication is the direction of travel, and the encoder that computes the embeddings will be replaced every year or two.

The decomposition $L_{\text{mix}} = \sum_i w_i L_i$ and the epoch accounting $e_i = w_i D / D_i$ are bookkeeping and will remain. The four-epoch figure is an empirical result at a particular scale and may soften or tighten with better regularization or with synthetic data whose repetition behaves differently. DoReMi is one of several mixture-optimization methods, and the specific method is likely to be replaced; the principle that weights must be validated by a training run at some scale will not be.

The dependency-graph structure of the pipeline (tokenize once, cache, hang everything off the cache) is a good engineering idea independent of Marin, Levanter, or JAX, and it is the part of the tooling most worth copying regardless of what replaces those libraries. Specific file formats, `uint16` shards, and the Gopher rules are the most ephemeral part of this chapter.

## Read next

- "Neural Machine Translation of Rare Words with Subword Units", Sennrich, 2016. The paper that brought BPE to neural text models; the merge procedure is the one you implemented.
- "Language Models are Unsupervised Multitask Learners", Radford, 2019. Introduced byte-level BPE with the pre-tokenization regex that every modern tokenizer descends from.
- "On the Resemblance and Containment of Documents", Broder, 1997. The origin of MinHash and the proof that hash agreement equals Jaccard similarity.
- "Deduplicating Training Data Makes Language Models Better", Lee, 2022. The measurements behind the claim that deduplication improves loss and cuts memorization.
- "Scaling Language Models: Methods, Analysis and Insights from Training Gopher", Rae, 2021. The source of the heuristic quality rules reused by most later corpora.
- "Scaling Data-Constrained Language Models", Muennighoff, 2023. The four-epoch result and the scaling law for repeated data.
- "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining", Xie, 2023. Mixture weights from a proxy model by minimax excess loss.
- "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", Penedo, 2024. A full curation pipeline with ablations at each stage, the closest published analog to what `curate.py` does.
