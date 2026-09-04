---
title: "Lab 07: Encoders and embeddings"
kind: permanent
topics: [lab]
chapter: 7
station: encoder
recipe: recipes/embed_contrastive.py
reading_time: 50 min
---

## What you will be able to do

1. Write down the one matrix that separates a BERT from a GPT, and explain from that matrix why an encoder gives you a representation of the whole input while a decoder gives you the next token.
2. Derive the InfoNCE loss from noise-contrastive estimation, read its gradient, and predict what temperature, batch size, and hard negatives each do to training.
3. Count the parameters of nomic-embed-text-v1.5 layer by layer and arrive at 137M, then recount for a different vocabulary or MLP without looking anything up.
4. Fine-tune a small embedding model on your own paper library with sentence-transformers on the 5090, with Matryoshka dimensions and mined hard negatives, and report recall@k, MRR, and nDCG before and after.
5. Diagnose the common failure modes of contrastive training (collapse, false negatives, missing prefixes, truncation) from the loss curve and the retrieval numbers alone.

## The idea in one paragraph

A decoder transformer is trained so that each position predicts the token after it, and to make that fair each position is only allowed to look backwards. An encoder removes that restriction: every position sees the whole input, and the model is trained to fill in words that were hidden. Filling blanks teaches it about local structure but not about which texts are similar, so a second stage trains it to produce one vector per text such that two texts that belong together (a query and the document that answers it, two windows of the same paper) sit close and everything else sits far. The training signal for that second stage is a classification problem: given a query, pick its partner out of a lineup that includes every other text in the batch. The bigger and harder the lineup, the sharper the vectors. nomic-embed-text-v1.5 is a BERT-sized model built exactly this way, and you will fine-tune something of that shape on your own papers.

## The math

### Encoder versus decoder at the mask level

A transformer layer computes attention as

$$
\mathrm{Attn}(X) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V,
\qquad Q = XW_Q,\; K = XW_K,\; V = XW_V,
$$

where $X \in \mathbb{R}^{T \times d}$ holds the $T$ input positions, $d_h$ is the per-head width, and $M \in \mathbb{R}^{T \times T}$ is an additive mask. The entire architectural difference between a decoder and an encoder lives in $M$:

$$
M^{\text{causal}}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases},
\qquad
M^{\text{full}}_{ij} = 0 \;\text{ for all } i, j
$$

(plus, in both cases, $-\infty$ on padding columns). With the causal mask, the hidden state at position $i$ is a function of tokens $x_{1..i}$ only, so the only position that has read the whole input is the last one, and it was trained to predict what comes next, not to summarize what came before. With the full mask, every position's hidden state is a function of the whole input, so any position (or any average of positions) can be a summary. That is the reason encoders are the natural embedding models and why decoder-based embedders have to resort to last-token pooling or bidirectional patches. The encoder station draws these two masks side by side as a lower-triangular square and a filled square; that picture is the whole of this subsection.

### Masked language modeling

Let $x = (x_1, \dots, x_T)$ be a token sequence. Choose a random subset of positions $\mathcal{M} \subset \{1, \dots, T\}$ with each position included independently with probability $\rho$ (the masking rate). Form the corrupted input $\tilde{x}$ by replacing $x_i$ for $i \in \mathcal{M}$ with a special `[MASK]` token (BERT instead replaces 80 percent with `[MASK]`, 10 percent with a random token, and leaves 10 percent unchanged, so that the model also learns to check unmasked tokens). The loss is the cross-entropy at masked positions only:

$$
\mathcal{L}_{\mathrm{MLM}}(\theta) = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid \tilde{x}),
$$

where $p_\theta(\cdot \mid \tilde{x})$ is a softmax over the vocabulary applied to a prediction head on the encoder's hidden state at position $i$. Two things follow from the formula. First, only a fraction $\rho$ of positions carry gradient per forward pass, so a higher $\rho$ gives more supervision per token of compute; BERT chose $\rho = 0.15$ and nomic-embed chose $\rho = 0.30$, and the later choice is supported by the observation that the model can still reconstruct with less context if it is trained on more data. Second, `[MASK]` never appears downstream, so the pretrained model has seen a distribution of inputs that the embedding model never will; the contrastive stage repairs this.

### Pooling

The encoder returns $H \in \mathbb{R}^{T \times d}$. You need one vector. With a padding mask $m \in \{0,1\}^T$, mean pooling is

$$
e = \frac{\sum_{t=1}^{T} m_t\, h_t}{\sum_{t=1}^{T} m_t},
$$

and the alternatives are the `[CLS]` position ($e = h_1$), max pooling over positions, and last-token pooling for causal models. The vector is then normalized, $\hat{e} = e / \|e\|_2$, so that cosine similarity is an inner product: $\cos(a, b) = \hat{a}^\top \hat{b}$. nomic-embed uses mean pooling. The choice is not cosmetic: mean pooling ties every position to the loss and gives a smoother gradient early in contrastive training, whereas `[CLS]` has to learn to attend to everything from scratch. Whatever you train with is what you must pool with at inference; a mismatch is a silent quality loss (see How it goes wrong).

### From noise-contrastive estimation to InfoNCE

Noise-contrastive estimation (NCE) solves a problem you would otherwise think needs a partition function. You want to fit an unnormalized model $\tilde{p}_\theta(x)$ to data. Draw data samples from $p_{\text{data}}$ and noise samples from a known distribution $q$, with $\nu$ noise samples per data sample, and train a logistic classifier to tell them apart. The Bayes-optimal posterior that a sample $x$ came from the data is

$$
P(\text{data} \mid x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + \nu\, q(x)},
$$

so if you parameterize the classifier's logit as $\log \tilde{p}_\theta(x) - \log(\nu q(x))$ and minimize the logistic loss, the optimum satisfies $\tilde{p}_\theta = p_{\text{data}}$ including its normalization. The trick is that the classifier only ever needs density ratios.

InfoNCE moves from one-versus-noise to one-versus-many. Fix a context $c$ (a query). Draw a set $X = \{x_1, \dots, x_N\}$ where exactly one element, at unknown index $i^\star$, is a positive drawn from $p(x \mid c)$ and the other $N - 1$ are negatives drawn from the marginal $p(x)$. The posterior that index $i$ is the positive is

$$
P(i^\star = i \mid X, c)
= \frac{p(x_i \mid c)\prod_{j \ne i} p(x_j)}{\sum_{k=1}^{N} p(x_k \mid c)\prod_{j \ne k} p(x_j)}
= \frac{p(x_i \mid c)/p(x_i)}{\sum_{k=1}^{N} p(x_k \mid c)/p(x_k)}.
$$

Every factor of $p(x_j)$ that does not involve the candidate cancels, and what remains is a softmax over density ratios. So if you train a scoring function $f(x, c) > 0$ with the categorical cross-entropy

$$
\mathcal{L}_N = -\,\mathbb{E}\left[\log \frac{f(x_{i^\star}, c)}{\sum_{k=1}^{N} f(x_k, c)}\right],
$$

the minimizer is $f(x, c) \propto p(x \mid c)/p(x)$, the pointwise mutual information between text and query, exponentiated. In embedding models the scoring function is $f(x, c) = \exp(s(x, c)/\tau)$ with $s$ the cosine similarity between the two normalized embeddings and $\tau$ a temperature, which gives the loss you will implement:

$$
\mathcal{L}_{\mathrm{InfoNCE}}(q, d^+, \{d_k\}) = -\log \frac{\exp\big(s(q, d^+)/\tau\big)}{\sum_{k=1}^{N} \exp\big(s(q, d_k)/\tau\big)}.
$$

Two consequences are worth deriving rather than memorizing.

The mutual information bound. Plug the optimal $f$ back in and write the denominator as the positive's ratio plus the sum over negatives. Each negative is drawn from $p(x)$, so $\mathbb{E}_{x \sim p(x)}[p(x \mid c)/p(x)] = \int p(x \mid c)\, dx = 1$, and the sum over $N - 1$ negatives concentrates near $N - 1$. Then

$$
\mathcal{L}_N^{\text{opt}} \approx \mathbb{E}\left[\log\left(1 + \frac{p(x^+)}{p(x^+ \mid c)}(N-1)\right)\right]
\ \ge\ \mathbb{E}\left[\log\left(\frac{p(x^+)}{p(x^+ \mid c)}\, N\right)\right]
= \log N - I(x; c),
$$

so $I(x; c) \ge \log N - \mathcal{L}_N$. The loss can never push the estimated mutual information above $\log N$ nats. With $N = 64$ that ceiling is $4.16$; with $N = 16{,}384$ it is $9.70$. This is the first reason batch size matters.

The gradient. Let $p_k = \exp(s_k/\tau)/\sum_j \exp(s_j/\tau)$ be the softmax over candidates. Differentiating the loss with respect to a score gives

$$
\frac{\partial \mathcal{L}}{\partial s_k} = \frac{1}{\tau}\big(p_k - \mathbb{1}[k = i^\star]\big).
$$

Every negative is pushed away in proportion to how much probability the model currently assigns it. Negatives the model already ranks low contribute almost nothing; the ones that are confused with the positive get almost all the gradient. Lowering $\tau$ makes the softmax sharper, so the gradient concentrates on the very hardest negatives and the effective logit range grows: since cosine lies in $[-1, 1]$, the logits span $2/\tau$, which is 40 nats at $\tau = 0.05$. Too low a temperature and a single false negative dominates the batch; too high and everything gets a uniform, uninformative push.

In practice the loss is made symmetric, $\tfrac{1}{2}(\mathcal{L}_{q \to d} + \mathcal{L}_{d \to q})$, so that documents are also trained to find their queries; this reuses the same $B \times B$ similarity matrix and does not add negatives.

### In-batch negatives and why batch size matters

With a batch of $B$ (query, positive) pairs, embed all queries into $Q \in \mathbb{R}^{B \times d}$ and all documents into $D \in \mathbb{R}^{B \times d}$, compute $S = QD^\top/\tau$, and take the cross-entropy of each row against its own index. Every other document in the batch is a free negative, so $N = B$ in the formulas above. Three things scale with $B$: the mutual information ceiling $\log B$, the expected number of hard negatives per query (linear in $B$, since a random text is hard with some fixed small probability), and memory, because you must hold the activations of $2B$ sequences to backpropagate. nomic-embed's contrastive pretraining used batches of 16,384 pairs. The memory problem has a clean fix called gradient caching: run all $2B$ sequences forward without gradient to get the embeddings, compute the loss and its gradient with respect to the embeddings (a $B \times d$ matrix, tiny), then re-run the encoder in small chunks with gradient enabled and backpropagate the cached embedding gradients. The result is bit-for-bit the large-batch gradient at the memory cost of one chunk. sentence-transformers exposes it as `CachedMultipleNegativesRankingLoss`.

The cost that also grows with $B$ is false negatives: two documents in the batch that are actually near-duplicates or answer the same query. The loss then demands that the model separate things that should be together. You will see this in Build it small.

### Hard negatives

In-batch negatives are random, so most are easy. A supervised fine-tuning stage adds negatives chosen to be confusable: for each query, retrieve the top candidates with the current model (or a first-stage model such as BM25, see Lab 08), remove the known positive, and keep a few of the rest as explicit negatives $d^-_{1..h}$. The denominator now sums over $B(1 + h)$ documents. The danger is that a retrieved candidate is actually relevant but unlabeled; the usual guard is a margin rule that discards any candidate whose score exceeds $s(q, d^+) - \delta$ for a small $\delta$, or equivalently keeps only candidates below some fraction of the positive's score.

### Task prefixes

Retrieval is asymmetric: a short question and a long passage that answers it should embed close, but two short questions about the same topic should not necessarily. A single encoder cannot tell which role a text is playing from the text alone, so nomic-embed prepends a role string, one of `search_query: `, `search_document: `, `classification: `, or `clustering: `, during both training and inference. Mathematically it is a learned conditioning input, a few tokens that shift the representation so that the same encoder computes different functions for different roles. Omitting it at inference is the single most common way to make a good embedding model look bad.

### Matryoshka representation learning

You often want vectors shorter than 768 to save index memory, but truncating a normal embedding throws away information that is spread across all dimensions. Matryoshka training makes the first $m$ dimensions a good embedding on their own for a nested set of sizes $\mathcal{S} = \{64, 128, 256, 512, 768\}$:

$$
\mathcal{L}_{\mathrm{MRL}} = \sum_{m \in \mathcal{S}} w_m\, \mathcal{L}_{\mathrm{InfoNCE}}\big(\mathrm{norm}(e_{1:m}), \mathrm{norm}(d_{1:m})\big),
$$

where $e_{1:m}$ is the prefix of the vector, renormalized after truncation, and the weights $w_m$ are usually all one. The model learns to order information by importance. At inference you truncate to $m$ and renormalize; the storage of an index of $n$ vectors is $4nm$ bytes in float32, so going from 768 to 128 dimensions is a 6x reduction, paid for with a retrieval loss you measure rather than guess.

### nomic-embed-text-v1.5, layer by layer

The shape is BERT-base: 12 layers, hidden width $d = 768$, 12 heads of width 64, vocabulary padded to 30,528 tokens, RoPE instead of learned position embeddings, SwiGLU instead of GeLU in the MLP, post-norm LayerNorm, mean pooling, and Matryoshka dimensions from 64 to 768. The MLM stage ran at 2,048 tokens with 30 percent masking; the 8,192-token context comes from scaling the RoPE frequencies at inference rather than from training at that length. The contrastive stage trained on roughly 235 million weakly paired texts, then a supervised stage with mined hard negatives.

Count the parameters. Per layer, with no biases on the QKV projection and none in the MLP:

| Component | Shape | Parameters |
|---|---|---|
| QKV projection | $768 \times 3 \cdot 768$ | 1,769,472 |
| Attention output projection, with bias | $768 \times 768 + 768$ | 590,592 |
| SwiGLU up and gate | $768 \times 2 \cdot 3072$ | 4,718,592 |
| SwiGLU down | $3072 \times 768$ | 2,359,296 |
| Two LayerNorms | $2 \times (768 + 768)$ | 3,072 |
| Total per layer | | 9,441,024 |

Twelve layers give $12 \times 9{,}441{,}024 = 113{,}292{,}288$. Outside the layers: the token embedding $30{,}528 \times 768 = 23{,}445{,}504$, a token-type embedding $2 \times 768 = 1{,}536$, and the embedding LayerNorm $1{,}536$. RoPE adds no parameters; it rotates $q$ and $k$ by position-dependent angles that are computed, not learned. The total is

$$
113{,}292{,}288 + 23{,}445{,}504 + 1{,}536 + 1{,}536 = 136{,}740{,}864 \approx 137\text{M}.
$$

The MLM prediction head (a $768 \times 768$ dense layer, a LayerNorm, a vocabulary bias, and a decoder tied to the token embedding) exists only during pretraining and is not part of the embedding model. Two sanity checks: swapping SwiGLU for BERT's plain GeLU MLP ($768 \times 3072 + 3072 \times 768$ with biases) gives 4,722,432 per layer instead of 7,077,888, and adding back BERT's learned positions ($512 \times 768$) lands you within rounding of the familiar 110M for BERT-base; and the SwiGLU block dominates the layer, at 75 percent of its parameters.

### Retrieval metrics

Let $\mathcal{Q}$ be a set of queries and, for each $q$, $R_q$ the set of relevant documents and $r_1, r_2, \dots$ the ranked list your model returns.

$$
\mathrm{Recall@}k = \frac{1}{|\mathcal{Q}|}\sum_{q} \frac{|\{r_1, \dots, r_k\} \cap R_q|}{|R_q|},
\qquad
\mathrm{MRR} = \frac{1}{|\mathcal{Q}|}\sum_{q} \frac{1}{\mathrm{rank}_q},
$$

where $\mathrm{rank}_q$ is the position of the first relevant document (contribution 0 if none appears). With graded relevance $\mathrm{rel}(r_i) \ge 0$,

$$
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{\mathrm{rel}(r_i)}{\log_2(i + 1)},
\qquad
\mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k},
$$

where IDCG is the DCG of the ideal ordering. Worked example: one relevant document, returned at rank 3. Recall@1 is 0, Recall@5 is 1, MRR is $1/3$, and nDCG@5 is $\frac{1/\log_2 4}{1/\log_2 2} = 0.5$. Recall@k is what an indexer cares about (did the answer make the shortlist), MRR is what a user sees (how far down is the first hit), and nDCG rewards putting several relevant items early in the right order.

## Build it small

The snippet trains a one-layer bidirectional transformer with mean pooling on synthetic documents. Each of 400 documents belongs to one of 4 topics and has its own private words; a training pair is two random windows of the same document, and every other document in the batch is a negative.

```python
# InfoNCE with in-batch negatives on synthetic documents (CPU, about 30 s)
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
V, K, D, L, N, B = 300, 4, 32, 24, 400, 64      # vocab, topics, dim, window, docs, batch
topic_vocab = [torch.randperm(V)[:60] for _ in range(K)]
doc_vocab = [torch.randperm(V)[:6] for _ in range(N)]  # words peculiar to one document
labels = torch.arange(N) % K

def window(i):
    """A random L-token window of document i: topic words, its own words, and noise."""
    r = torch.rand(L)
    tv, dv = topic_vocab[labels[i]], doc_vocab[i]
    return torch.where(r < 0.55, tv[torch.randint(len(tv), (L,))],
           torch.where(r < 0.80, dv[torch.randint(len(dv), (L,))], torch.randint(V, (L,))))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=64,
                                                dropout=0.0, batch_first=True)
    def forward(self, x):                        # x: (B, L); no causal mask: bidirectional
        return F.normalize(self.layer(self.emb(x)).mean(1), dim=-1)   # mean pool, unit norm

def info_nce(q, d, tau=0.05):
    logits = q @ d.T / tau                       # (B, B); diagonal entries are the positives
    y = torch.arange(len(q))
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

@torch.no_grad()
def nn_purity(enc):
    e = enc(torch.stack([window(i) for i in range(N)]))
    s = e @ e.T
    s.fill_diagonal_(-2.0)
    return (labels[s.argmax(1)] == labels).float().mean().item()

enc = Encoder()
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
print(f"step   0  nn-purity {nn_purity(enc):.2f}  (chance {1/K:.2f})")
for step in range(1, 301):
    idx = torch.randint(N, (B,))
    q = enc(torch.stack([window(i) for i in idx]))   # two independent windows of the same doc
    d = enc(torch.stack([window(i) for i in idx]))
    loss = info_nce(q, d)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}  nn-purity {nn_purity(enc):.2f}")
```

Expected output (seed 0, CPU; your numbers will differ slightly by PyTorch version but the shape will not):

```
step   0  nn-purity 0.50  (chance 0.25)
step 100  loss 3.207  nn-purity 0.69
step 200  loss 2.930  nn-purity 0.90
step 300  loss 2.475  nn-purity 0.93
```

Read the two columns against the math. The loss starts near $\log 64 = 4.16$, the uniform-softmax value, and falls, but it does not go anywhere near zero: a batch of 64 documents from 4 topics contains about 16 per topic that share 55 percent of their vocabulary, and the loss is demanding that the model tell them apart. If same-topic documents were indistinguishable, the floor would be $\log 16 = 2.77$ per direction; the loss gets below it only because of the six private words per document. Meanwhile nearest-neighbour purity (does the closest other document share your topic) climbs from an untrained 0.50 to 0.93. The untrained value is already above chance because mean-pooled random embeddings of bags of words are a crude topic detector; contrastive training sharpens it. This is the same picture the encoder station draws after its InfoNCE phase: three topic clouds that overlap after MLM and separate after a few hundred contrastive steps, with the purity number climbing.

## Build it real

The recipe is `recipes/embed_contrastive.py`. It fine-tunes an embedding model on your own vault so that a title, or a sentence from a paper's body, retrieves the right paper.

Data. The script walks the PDFs in your vault, extracts text with PyMuPDF, and takes the title (PDF metadata, falling back to the first non-empty line) and the abstract (the text between the first occurrence of "Abstract" and the first of "Introduction", capped at 400 words). Each paper yields one training pair, `("search_query: " + title, "search_document: " + abstract)`, plus, with `--body-pairs`, a handful of extra pairs whose query is a random sentence from the body. Papers are split 80/10/10 into train, validation, and test by paper, never by pair, so a test title cannot have its own abstract in training. Hard negatives are mined with the base model before training: for each title, retrieve the top `--hard-negatives` abstracts from the training set, drop the positive, and drop any candidate scoring within 0.05 of the positive (the margin rule from The math).

Model and library. The default is `nomic-ai/nomic-embed-text-v1.5` loaded through sentence-transformers with `trust_remote_code=True`; `--model BAAI/bge-small-en-v1.5` gives you a 33M-parameter alternative that trains in a fraction of the time and needs no prefixes (the script drops them when the model name is not nomic). The loss is `CachedMultipleNegativesRankingLoss` (InfoNCE with in-batch negatives and gradient caching; its `scale` argument is $1/\tau$, and the default 20 is $\tau = 0.05$), wrapped in `MatryoshkaLoss` with dimensions 768, 512, 256, 128, 64 unless you pass `--no-matryoshka`. Training uses `SentenceTransformerTrainer` with `batch_sampler=NO_DUPLICATES`, which guarantees no two pairs in a batch come from the same paper. Evaluation uses `InformationRetrievalEvaluator` on the validation split every epoch and on the test split at the end, and additionally embeds the whole vault with the base and the fine-tuned model to report the metrics of the Measure it section.

Arguments: `--vault ~/Cortex` (root to walk), `--model`, `--out runs/embed/<timestamp>`, `--epochs 3`, `--batch 128` (pairs per optimizer step; the cached loss keeps activations for `--mini-batch 32` at a time), `--lr 2e-5`, `--max-seq 512`, `--hard-negatives 3`, `--body-pairs 4`, `--no-matryoshka`, `--eval-only` (skip training, score the base model), and `--seed`.

What to watch in the logs. The training loss at step 0 should sit near $\log(B(1 + h))$, which is $\log(128 \cdot 4) = 6.24$ for the defaults, and the number that matters is the validation recall@10 each epoch; the train loss will keep falling after the recall stops moving, and that is your signal to stop. A loss that starts far below $\log B$ means leakage (a title that appears verbatim inside its abstract, or duplicate PDFs); a loss that never leaves $\log B$ means collapse (see How it goes wrong). The evaluator prints cosine-similarity recall, MRR, and nDCG at 1, 5, and 10.

How long it takes. Training FLOPs are approximately $6 \cdot P \cdot n_{\text{tok}}$ per epoch, where $P$ is the parameter count and $n_{\text{tok}}$ the number of tokens seen. Assume 2,000 papers, 5 pairs each, about 40 tokens per query and 300 per document, and 3 hard negatives per pair: $n_{\text{tok}} \approx 10{,}000 \times (40 + 4 \times 300) = 1.24 \times 10^{7}$ per epoch, so $6 \times 1.37 \times 10^{8} \times 1.24 \times 10^{7} \approx 1.0 \times 10^{16}$ FLOPs per epoch. At an assumed sustained 100 TFLOP/s in bf16 on the 5090, which is a fraction of its peak that a padded, small-batch encoder can realistically reach, that is about 100 seconds per epoch of pure compute; expect a few minutes per epoch once tokenization, gradient caching's second forward pass, and evaluation are included, and under fifteen minutes for the default three epochs. Memory is not the constraint: the 137M model in bf16 with AdamW states is under 2 GB, and the cached loss caps activations at the mini-batch.

## How it goes wrong

Loss pinned at $\log B$ from the first step and never moving. The embeddings have collapsed to a single direction, so every similarity is the same and the softmax is uniform. It is almost always a learning rate an order of magnitude too high for a pretrained encoder, or a forgotten normalization step combined with a temperature that makes the logits enormous. Check the standard deviation of the embeddings across a batch (it should not be near zero), drop the learning rate to $2 \times 10^{-5}$ or lower, and confirm `F.normalize` is applied.

Loss falls to nearly zero within a few hundred steps while validation recall does not improve. The task is trivial because the positive contains the query. Titles copied into abstracts, a chunker that emits overlapping windows as pairs, or duplicate PDFs under different filenames all do this. Deduplicate papers by normalized title, and check that no query string is a substring of its positive.

Loss plateaus high and jitters, recall creeps. False negatives: the batch contains near-duplicates that the loss is trying to separate (the Build it small floor, in the wild). Use a batch sampler that forbids two pairs from the same source in one batch, and deduplicate near-copies of documents with a similarity threshold before training (Lab 08 covers how).

Recall drops sharply when you switch from the evaluator to your own retrieval code. You dropped the task prefixes, or mixed them up, using `search_document: ` for the queries. The prefix is part of the model; encode queries and documents with the same strings you trained with.

Long abstracts and body passages retrieve badly, short ones fine. `max_seq_length` defaulted to something short and your documents were truncated after the first sentences. Set it explicitly to 512 or more and check `model.max_seq_length` after loading; for nomic, the RoPE scaling lets you go to 8,192 if you are willing to pay the quadratic attention cost.

Everything looks fine in training, but a model you exported and reloaded elsewhere scores worse. Pooling mismatch: you loaded the transformer alone and took `[CLS]`, but the model was trained with mean pooling. Load through sentence-transformers so the pooling module travels with the weights, or reproduce the exact pooling.

After adding hard negatives the validation numbers get worse than without them. The mined negatives include relevant documents that were never labeled, so the model is being taught that correct answers are wrong. Apply the margin filter, mine with the base model rather than the model being trained, and mine fewer per query.

Truncated Matryoshka vectors give cosine scores greater than one or a broken ranking. You truncated but did not renormalize. Truncation of a unit vector leaves a vector of norm less than one, and the dot products are no longer cosines; renormalize after slicing, in both the index and the query path.

## Measure it

Evaluate retrieval, not loss. The test split gives you queries (titles, and body sentences if you generated them) with exactly one relevant document each, so report Recall@1, Recall@10, MRR@10, and nDCG@10 for the base model and the fine-tuned model side by side, over the same corpus of all abstracts in the vault (not just the test split; the distractors matter). Because each query has a single relevant document, nDCG@10 and MRR@10 coincide up to the log discount, and recall@k is the fraction of queries whose paper appeared in the top $k$.

What counts as good depends on the query type. Title-to-abstract is easy, and a strong base model will already be high on it; the fine-tuned model should be at or above the base, and a drop means something in How it goes wrong. Body-sentence-to-paper is the informative one, since that is what a real search over your vault looks like, and it is where the fine-tune should show its gain. Whatever the gain, compute a confidence interval (a paired bootstrap over queries, see Lab 09) before believing it; with 200 test queries, a 5-point change in recall@10 is roughly at the edge of what you can distinguish from noise. Then run a small general benchmark, such as a couple of MTEB retrieval tasks, on both models to confirm you did not trade general ability for vault-specific gain; a fine-tune that helps on your papers and hurts elsewhere is fine for a private index and not fine for anything else. Finally, plot recall@10 against Matryoshka dimension (768, 512, 256, 128, 64) for the fine-tuned model; the curve tells you which dimension to store in the index in Lab 08.

## Exercises

1. Derive $\partial \mathcal{L}/\partial s_k$ for the InfoNCE loss and show that the gradients over all candidates sum to zero. Check: the softmax sums to one and the indicator sums to one.
2. Compute the mutual information ceiling $\log N$ in nats and bits for $N = 32$, $64$, $4{,}096$, and $16{,}384$. Check: 3.47/5.00, 4.16/6.00, 8.32/12.00, 9.70/14.00.
3. In the Build it small snippet, change the batch sampler so that every batch contains at most one document per topic (hint: $B$ must then be at most $K$, so raise $K$ to 64 and $N$ to 640 first). Explain why the loss now reaches a much lower value at the same purity. Check: the same-topic false negatives are gone, so the floor of $\log(B/K)$ disappears.
4. Implement gradient caching in the snippet: compute the embeddings without gradient, compute the loss on detached copies that require gradient, then re-encode in chunks of 16 and call `backward` with the cached gradient. Check: `torch.allclose` between the parameter gradients of the cached and direct methods, to $10^{-5}$.
5. Add Matryoshka training to the snippet with dimensions 8, 16, 32 and measure nearest-neighbour purity at each truncation, with and without MRL. Check: without MRL, purity at 8 dimensions is far below the full-width value; with MRL, the gap shrinks.
6. Recount the nomic-embed parameters if the vocabulary were 50,304 tokens and the MLP were a plain GeLU with hidden 3,072 and biases. Answer: token embedding 38,633,472; per layer 1,769,472 + 590,592 + 4,722,432 + 3,072 = 7,085,568; total $12 \times 7{,}085{,}568 + 38{,}633{,}472 + 3{,}072 = 123{,}663{,}360$.

## Test yourself

1. A colleague mean-pools the hidden states of a causal decoder to get a document embedding. Why is this worse than mean-pooling an encoder, even with the same number of parameters?

<details><summary>Answer</summary>
Under the causal mask, the hidden state at position $t$ is a function of tokens $1..t$ only, so the states of early positions were computed with almost no context and the states of late positions with all of it. The mean therefore weights the beginning of the document as heavily as the end while it carries far less information, and none of the positions was trained to summarize; they were trained to predict the next token. Encoder states each see the entire input, so the mean is a mean of full-context summaries. Decoder-based embedders either use the last token, which has seen everything, or remove the causal mask and continue training.
</details>

2. Spot the bug: `logits = q @ d.T; loss = F.cross_entropy(logits, torch.arange(B))` where `q` and `d` come straight out of a linear layer.

<details><summary>Answer</summary>
Nothing is normalized and there is no temperature. The model can lower the loss indefinitely by inflating the norms of the embeddings, since scaling all logits by a constant sharpens the softmax around whatever is already the argmax, and it will do so instead of learning geometry. Normalize both sides to unit norm so that the logits are cosines in $[-1, 1]$, and divide by a temperature to set their range.
</details>

3. You raise the batch from 64 to 16,384. Estimate the memory of the similarity matrix and its gradient in float32, and explain why that is not what actually runs out of memory.

<details><summary>Answer</summary>
$16{,}384^2 \times 4$ bytes is about 1.07 GB for the matrix and the same again for its gradient, which fits. What does not fit is the activation memory of the encoder: 32,768 sequences (queries and documents) of a few hundred tokens through 12 layers, which is thousands of times larger. Gradient caching solves exactly this by never holding more than one chunk's activations while still producing the full-batch gradient.
</details>

4. A senior researcher claims that lowering the temperature makes the loss weight easy negatives more, because it makes the loss "stricter". Is that right?

<details><summary>Answer</summary>
No. The gradient on candidate $k$ is $(p_k - \mathbb{1}[k = i^\star])/\tau$. Lowering $\tau$ sharpens the softmax, so probability mass concentrates on the few highest-scoring negatives and the easy ones get a gradient closer to zero. Lower temperature focuses on the hardest negatives, and if any of those is a false negative, on it. Higher temperature spreads the gradient across all candidates.
</details>

5. The InfoNCE loss on a well-trained model with $B = 64$ has stopped decreasing at about 0.9 nats. What does that tell you about the mutual information between query and document, and what does it not tell you?

<details><summary>Answer</summary>
The bound gives $I(x; c) \ge \log 64 - 0.9 \approx 3.3$ nats. It does not tell you the mutual information is 3.3 nats or anywhere near it; the estimate saturates at $\log N$, so if the true dependence is stronger, the loss simply cannot express it. This is why a loss that has stopped decreasing at small batch is not evidence that the model has learned everything the pairs contain, and why raising the batch can lower the loss further without any change to the model.
</details>

6. Why does the parameter count of nomic-embed-text-v1.5 not change when the context is extended from 2,048 to 8,192 tokens, and by how much would it change if the model used learned absolute position embeddings at 8,192?

<details><summary>Answer</summary>
RoPE rotates queries and keys by angles that are functions of position and a fixed base frequency; extending the context changes the frequency schedule, not any weights. Learned absolute positions would add an $8{,}192 \times 768$ table, which is 6,291,456 parameters, and could not be extended beyond the trained length without new rows.
</details>

7. Spot the bug: to compute recall@10 for title-to-paper retrieval, you embed every paper's abstract into the index and then, for each paper, query with `"search_query: " + title` and check whether that paper is in the top 10. The base model reports recall@10 of 1.00. Should you believe it?

<details><summary>Answer</summary>
Only after checking that the title is not inside the indexed text. Extracted abstracts frequently begin with the title line, and if so the query is a substring of its positive and any model will retrieve it. Strip the title from the abstract text (or index only the text after the word "Abstract") and re-run. A perfect score on the base model is a sign of leakage before it is a sign of a good model.
</details>

8. If you truncate a non-Matryoshka embedding to its first 64 of 768 dimensions and renormalize, what fraction of the variance do you expect to keep, under what assumption, and why is that assumption roughly right for a well-trained contrastive model?

<details><summary>Answer</summary>
If the embedding is approximately isotropic (variance spread evenly across dimensions), the first 64 dimensions carry about $64/768 \approx 8$ percent of the variance, and nearest-neighbour rankings are nearly random. Contrastive training with a uniformity-inducing loss such as InfoNCE pushes embeddings towards using the whole sphere, which makes near-isotropy a reasonable first approximation. Matryoshka training breaks the symmetry on purpose by putting a loss on each prefix so that early dimensions are forced to carry the most useful information.
</details>

9. You fine-tune only on (title, abstract) pairs and then use the model to retrieve papers from paragraph-long queries. Name the distribution shift and a training change that addresses it.

<details><summary>Answer</summary>
Query length and style. The model learned to map short, noun-heavy queries onto abstracts; paragraph queries look like documents, and with task prefixes the encoder has explicitly learned that `search_query: ` texts are short. Add pairs whose queries are body sentences or paragraphs (the recipe's `--body-pairs`), so that the query distribution at training time covers what you will ask at inference.
</details>

10. With 30 percent masking, what fraction of positions receive gradient in one MLM forward pass, and what limits raising the rate to 90 percent?

<details><summary>Answer</summary>
Thirty percent. At 90 percent, almost every position must be reconstructed from a context that is itself almost entirely masked, so the task degenerates towards unigram frequency prediction, and the model learns little about how tokens depend on each other. The rate trades supervision per pass against the amount of intact context each prediction can use.
</details>

## What will change, what will not

The mask will not change. Whatever the block looks like in five years, a model whose positions can all read each other can produce a summary of its input, and a model whose positions can only read backwards cannot without a trick. Every "decoder-based embedding model" is an instance of paying for that trick, and you can evaluate any new architecture's claim by asking which mask it uses and where it pools.

The density-ratio view of contrastive learning will not change. The InfoNCE minimizer is the pointwise mutual information; the loss is bounded by $\log N$; the gradient is proportional to the softmax probability of each negative divided by the temperature. New losses will appear (they already have: margin variants, listwise variants, distillation from a reranker), and each one is worth reading through the same three questions: what is the implied target of the score, what limits it, and which negatives get the gradient.

The metrics will not change. Recall@k, MRR, and nDCG are defined by what a user needs from a ranked list, not by any model, and the discipline of holding out by document and pairing comparisons across the same queries is a statement about statistics, not about embeddings.

What will change is the recipe around those invariants. BERT-shaped encoders with mean pooling were the dominant embedding architecture at the time of writing, and decoder-based embedders that start from a strong language model and remove the causal mask were already competitive; the role prefixes described here were already turning into free-text instructions. sentence-transformers' class names, the `scale=20` default, the gradient-caching implementation, the 16,384 batch, and Matryoshka's specific dimension ladder are all tooling and hyperparameters that will be replaced. Treat the numbers in Build it real as a way to size a run, not as constants.

The parameter count is a skill, not a fact. The 137M figure is specific to one model, but the habit of writing down every matrix and summing them is how you will read the next model card, catch the one that quietly counts embeddings twice, and estimate memory and FLOPs before you launch a job.

## Read next

1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin, 2018. The masked language modeling objective and the full attention mask that this chapter starts from.
2. Representation Learning with Contrastive Predictive Coding, van den Oord, 2018. Introduces InfoNCE and proves the $\log N$ mutual information bound used in The math.
3. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models, Gutmann, 2010. The density-ratio trick that InfoNCE generalizes.
4. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Reimers, 2019. Pooling choices and the siamese setup that sentence-transformers implements.
5. Dense Passage Retrieval for Open-Domain Question Answering, Karpukhin, 2020. In-batch negatives and BM25-mined hard negatives as a practical recipe.
6. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup, Gao, 2021. Gradient caching, which is how a 32 GB card trains at large effective batch.
7. Matryoshka Representation Learning, Kusupati, 2022. The nested-dimension loss and why truncation works after it.
8. Nomic Embed: Training a Reproducible Long Context Text Embedder, Nussbaum, 2024. The full three-stage recipe for the model whose parameters you counted.
