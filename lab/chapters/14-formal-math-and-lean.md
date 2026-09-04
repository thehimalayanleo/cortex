---
title: "Lab 14: Formal mathematics, code verification, and Lean"
kind: permanent
topics: [lab]
chapter: 14
station: none
recipe: recipes/lean_eval.py
reading_time: 50 min
---

# Lab 14: Formal mathematics, code verification, and Lean

## What you will be able to do

1. Read a Lean 4 tactic state, write a proof of a small true statement from scratch with `intro`, `apply`, `exact`, `rw`, `simp`, `induction`, and `calc`, and have Lean accept it with no `sorry`.
2. Find the Mathlib lemma you need by name convention, by `exact?`, and by type-pattern search, and explain why a proof that compiled against last year's Mathlib may not compile today.
3. State precisely what an AI prover's benchmark number means: which statements, which budget, which checker, and which of the four ways a "proof" can pass the checker without proving the theorem you meant.
4. Run a small open prover model on the 5090 against a handful of miniF2F-style statements with real Lean checking, and audit the survivors by hand.
5. Formalize a definition from your own field (the restricted isometry property) and prove your first real theorem about it, with an honest sense of how far that is from formalizing a recovery guarantee.

## The idea in one paragraph

A proof assistant is a programming language whose type checker is a proof checker: a theorem is a type, a proof is a program of that type, and if the program compiles, the theorem is proved, relative to a small trusted kernel and the axioms you declared. Mathematics written this way is slow to produce and cheap to check, which is exactly the shape of problem that language models plus a verifier are good at. You sample many candidate proofs, keep the ones the checker accepts, train on them, and repeat; the checker never lies, so the loop cannot reward-hack the way a learned judge can. The catch is that the checker only verifies the formal statement in front of it. Translating a human theorem into that statement (autoformalization) is where meaning can leak, and a compiled proof of the wrong statement is worth nothing. The same holds for programs: a verified sorting routine is exactly as trustworthy as its specification. Learning Lean is learning to write statements you can stand behind, and then learning how much of the proving can be delegated.

## The math

### Proof assistants and the Curry-Howard idea

In a dependently typed language, types can mention values: `Fin n` is the type of natural numbers below `n`, and `a = b` is a type whose inhabitants are proofs that `a` equals `b`. Curry-Howard is the observation that the rules of intuitionistic logic and the typing rules of such a language are the same rules read twice. An implication $P \to Q$ is a function type: a proof of it is a function that turns any proof of $P$ into a proof of $Q$. A conjunction $P \wedge Q$ is a pair type. A universal $\forall x : A,\ P(x)$ is a dependent function type: a function that, given any $x$, returns a proof of $P(x)$. An existential $\exists x, P(x)$ is a dependent pair, a witness and a proof about it. A proof is therefore a term, checking a proof is type checking, and a false statement is an uninhabited type. Lean 4's kernel is a few thousand lines that check terms against this type theory; everything else (tactics, automation, the whole of Mathlib) produces terms for the kernel to check and is not itself trusted.

### Lean 4 basics

The two ways to write a proof are as a term or by tactics. Here is the same theorem both ways:

```lean
theorem imp_trans (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
  fun hp => hqr (hpq hp)

theorem imp_trans' (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r := by
  intro hp
  apply hqr
  exact hpq hp
```

Read the header as: a theorem named `imp_trans` with hypotheses `p q r` (propositions) and `hpq`, `hqr` (proofs of implications), concluding `p → r`. The term proof is a lambda. The tactic proof, after `by`, is a script that transforms a goal. `intro hp` moves the antecedent of the goal into the context as hypothesis `hp : p`, leaving the goal `r`. `apply hqr` says the goal `r` would follow from `hqr : q → r` if you could prove `q`, so the goal becomes `q`. `exact hpq hp` supplies a term of exactly that type. Tactics are search procedures for terms; `exact` is the degenerate case where you already have the term.

Rewriting and simplification:

```lean
example (a b : Nat) (h : a = b) : a + 0 = b := by
  rw [Nat.add_zero, h]

example (xs : List Nat) : (xs ++ []).length = xs.length := by
  simp
```

`rw [Nat.add_zero]` replaces `a + 0` with `a` using the lemma `Nat.add_zero : n + 0 = n`, left to right; `rw [h]` then replaces `a` with `b`, and the goal `b = b` closes by reflexivity, which `rw` tries automatically. `simp` applies a database of rewrite rules tagged `@[simp]` until nothing changes; here `List.append_nil` fires and the goal closes. `example` is a theorem with no name.

Chains of (in)equalities:

```lean
example (a b c : Nat) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := by
  calc a ≤ b := h1
    _ ≤ c := h2
```

`calc` lets you write the proof the way you would on paper; each step is justified after `:=`, and Lean composes them with transitivity.

Definitions and `#eval`:

```lean
def sumTo : Nat → Nat
  | 0 => 0
  | n + 1 => sumTo n + (n + 1)

#eval sumTo 4    -- 10
```

This is a program (structural recursion on `Nat`, which Lean checks terminates) and also an object you can prove things about. The logical connectives you will meet: `∀ x, P x`, `∃ x, P x`, `P ∧ Q`, `P ∨ Q`, `¬P`, `P → Q`, `P ↔ Q`. Proofs of `∧` and `∃` are built with the anonymous constructor `⟨h1, h2⟩` and taken apart with `h.1`, `h.2`, or `obtain ⟨x, hx⟩ := h`. `constructor` splits a goal that is a conjunction or an iff; `left` and `right` pick a side of a disjunction; `cases h` or `rcases` do case analysis on a hypothesis. `sorry` is a placeholder that makes any goal compile with a warning; it is how you scaffold a proof and how a dishonest prover cheats.

### Reading a tactic state

When you place the cursor inside a proof, the editor shows the state: hypotheses above the line, goal below the turnstile `⊢`. From the induction proof below, at the start of the successor case, the state is:

```
case succ
k : Nat
ih : 2 * sumTo k = k * (k + 1)
⊢ k * (k + 1) + 2 * (k + 1) = (k + 1) * (k + 1 + 1)
```

Everything you know is above the line; the goal is what remains to be shown. Reading states is the skill. Most of the time spent proving is spent looking at a state and asking which hypothesis, or which lemma, turns the goal into a smaller goal.

### Mathlib and how to find a lemma

Mathlib is the community library: over a million lines covering algebra, analysis, topology, probability, linear algebra, and combinatorics, with a single `import Mathlib` bringing all of it in. Its lemma names follow a grammar you can learn to guess. Names describe the statement left to right in lower-case snake case: `add_comm : a + b = b + a`, `mul_le_mul_of_nonneg_right : b ≤ c → 0 ≤ a → b * a ≤ c * a`, `Finset.sum_range_succ : ∑ x ∈ range (n + 1), f x = ∑ x ∈ range n, f x + f n`. A namespace prefix (`Nat.`, `Real.`, `Finset.`) says what the lemma is about. `_of_` separates the conclusion from the hypotheses, `_iff` marks a biconditional, `_left` and `_right` say which argument varies.

When guessing fails, ask Lean. `exact?` searches for a lemma that closes the goal exactly and prints it:

```lean
example (a b : ℕ) (h : a ≤ b) : a ≤ b + 3 := by exact?
-- Try this: exact Nat.le_add_right_of_le h
```

`apply?` finds lemmas that reduce the goal; `simp?` shows which simp lemmas fired so you can replace `simp` with an explicit `simp only [...]`; `rw?` suggests rewrites. Outside the editor, Loogle (loogle.lean-lang.org) searches Mathlib by type pattern, so `Real.sqrt, _ ≤ _` returns every lemma mentioning the square root and an inequality, and the Mathlib documentation site is searchable by name. `#check @Finset.sum_range_id_mul_two` prints a lemma's full statement, including the implicit arguments, and is the fastest way to learn what a name actually says.

### A worked proof from scratch

The claim is that twice the sum $0 + 1 + \dots + n$ equals $n(n+1)$. Informally: induction on $n$; the base case is $0 = 0$; for the step, $2\,\mathrm{sumTo}(k+1) = 2\,\mathrm{sumTo}(k) + 2(k+1) = k(k+1) + 2(k+1) = (k+1)(k+2)$. Here is the same proof in Lean 4 with no Mathlib, verified with Lean 4.33:

```lean
def sumTo : Nat → Nat
  | 0 => 0
  | n + 1 => sumTo n + (n + 1)

theorem two_mul_sumTo (n : Nat) : 2 * sumTo n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    calc 2 * sumTo (k + 1)
        = 2 * sumTo k + 2 * (k + 1) := by rw [sumTo, Nat.mul_add]
      _ = k * (k + 1) + 2 * (k + 1) := by rw [ih]
      _ = (k + 1) * (k + 1 + 1) := by
          simp only [Nat.mul_add, Nat.add_mul, Nat.mul_one, Nat.one_mul]
          omega
```

Line by line. `induction n with` splits into the `zero` and `succ` cases and in the second gives you `k` and the induction hypothesis `ih`. The base case `2 * sumTo 0 = 0 * (0 + 1)` holds by computation, so `rfl` (both sides reduce to `0`) closes it. In the step, the first `calc` line unfolds the definition (`rw [sumTo]` uses the equation `sumTo (k+1) = sumTo k + (k+1)`) and distributes with `Nat.mul_add : a * (b + c) = a * b + a * c`. The second line rewrites with the induction hypothesis. The last line is arithmetic: `simp only` with the distributivity lemmas expands both sides into sums of `k * k`, `k`, and constants, and `omega`, a decision procedure for linear arithmetic over integers and naturals, finishes, treating the nonlinear term `k * k` as an opaque atom that appears on both sides. If you delete the `simp only` line, `omega` fails, because it cannot see through `k * (k + 1)` to the linear structure inside; that is a lesson about what automation does and does not do.

With Mathlib the arithmetic tail is one word, and the sum can be written with the library's `Finset.range`:

```lean
import Mathlib
open Finset

theorem gauss' (n : ℕ) : (∑ i ∈ range (n + 1), i) * 2 = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [sum_range_succ, add_mul, ih]
    ring
```

`ring` proves any identity in a commutative (semi)ring by normalizing both sides. Mathlib already has this theorem as `Finset.sum_range_id_mul_two : (∑ i ∈ range n, i) * 2 = n * (n - 1)`; note the `n - 1`, which in `ℕ` is truncated subtraction, so the library statement is phrased to avoid needing `n ≥ 1`. For an inequality, the tactic to know is `nlinarith`, which is `linarith` (linear arithmetic over ordered fields) after multiplying pairs of hypotheses; the two-variable AM-GM inequality is one line once you hand it the right square:

```lean
theorem amgm2 (a b : ℝ) : a * b ≤ (a ^ 2 + b ^ 2) / 2 := by
  nlinarith [sq_nonneg (a - b)]
```

Both compiled against the Mathlib of 2026-09-03.

### Autoformalization and why it is hard

Autoformalization is translating an informal statement into a formal one. The difficulty is that natural-language mathematics leaves out what the reader will fill in, and Lean fills in nothing. Concretely: types. "Let $n$ be a number" is `ℕ`, `ℤ`, or `ℝ`, and the same proof strategy may be right for one and wrong for another. Partial operations are total in Lean with conventions: `n - 1` in `ℕ` is `0` when `n = 0`; `a / 0 = 0` in every field, so `a / a = 1` is false; `Real.sqrt x = 0` for negative `x`. A faithful translation must add the hypotheses the informal statement assumed. Quantifier scope: "for every $\varepsilon$ there is $N$" versus "there is $N$ for every $\varepsilon$" is the difference between convergence and boundedness. Implicit domains: "the function is increasing" on which set. And the failure that matters most: a statement with contradictory hypotheses, or a hypothesis that makes the conclusion trivial, is provable, so a misformalized statement can be easier than the original, and a prover that scores well on it has learned nothing. Every autoformalization pipeline therefore needs a separate check on statement fidelity (a human, or a model asked to translate back and compare, or a test that the statement is not vacuously true), because the Lean checker cannot judge meaning.

### Proof search with language models

Write the prover as a policy $\pi_\theta(\text{proof} \mid \text{statement})$ and the checker as a reward $r(\text{statement}, \text{proof}) \in \{0, 1\}$, equal to 1 when Lean compiles the file with no errors and no `sorry`. Expert iteration is:

1. Sample $k$ proofs per statement from $\pi_\theta$.
2. Check each with Lean; keep the pairs with $r = 1$ (deduplicated).
3. Fine-tune $\pi_\theta$ on the kept pairs (supervised, next-token).
4. Repeat, usually growing the statement set.

This is reinforcement learning with a binary reward where the policy-improvement step is imitation of your own successes. It works because verification is exact, so noise in step 2 is zero, and because the expected number of successes per statement scales with $k$: a problem solved once in ten thousand samples becomes training data and next round it is solved in a hundred. The systems differ in how they structure the search. Whole-proof generation samples a complete proof as one string and checks it; it is simple and batches well through vLLM. Tactic-level search treats the tactic state as the environment, samples one tactic, applies it through a Lean interaction layer, and searches over the resulting tree (best-first, or Monte Carlo tree search with a learned value on states); it is slower per sample but recovers from a wrong step. Retrieval adds the top Mathlib lemmas for the current state to the prompt.

Systems you will see cited, described only as far as I am confident. LeanDojo (Yang, 2023) is the interaction toolkit that lets Python drive Lean tactic by tactic and extract a dataset of states and tactics from Mathlib, and ReProver is its retrieval-augmented tactic model. Lean Copilot runs a model inside Lean as tactics (`suggest_tactics`, `search_proof`) so a human and a model share a proof. DeepSeek-Prover (V1 2024, V1.5 with reinforcement learning from Lean feedback and tree search, V2 2025 with subgoal decomposition) is the open whole-proof family most people run locally; the 7B variants fit on your card. AlphaProof (DeepMind, 2024) reached silver-medal level on 2024 IMO problems in Lean, with problems formalized by a separate model and a large reinforcement learning loop; its weights are not public. Goedel-Prover and Kimina-Prover are 2025 open provers trained with the same loop at larger scale.

### Benchmarks and what a score means

miniF2F (Zheng, Han, Polu, 2021) is 488 competition problems (AMC, AIME, IMO, and textbook-level algebra and number theory) formalized in several proof assistants, split into 244 validation and 244 test statements; it has been the headline number for AI provers since GPT-f. PutnamBench (2024) is several hundred Putnam competition problems in Lean 4 (with Isabelle and Coq versions for a subset), harder and less saturated. ProofNet targets undergraduate textbook statements and is used for autoformalization as well as proving.

A score is a pass rate at a budget: pass@$k$ is the fraction of statements for which at least one of $k$ samples compiles, and headline numbers are reported at budgets from 1 to many thousands of samples, sometimes with tree search that has no clean $k$. Two numbers are comparable only at equal budgets, on the same benchmark version (miniF2F statements have been corrected over time), against the same Mathlib version, and with the same checker rules. Early systems solved around a quarter of miniF2F-test; by 2025 leading open provers report well over half at modest budgets and above 80 percent at large ones, and the frontier moved to PutnamBench, where scores were still low. Contamination is real: competition problems and their formalizations are on the internet, and a model that has seen the test set's proofs will score well without generalizing; PutnamBench's statements were written specifically to avoid this. When you read a number, ask for the budget, the version, and whether the survivors were audited for statement fidelity.

### Code verification

The same machinery verifies programs. A specification says what a function must do; the verifier checks that the code does it for all inputs, not for a test set. Dafny is the friendliest entry point: an imperative language with preconditions (`requires`), postconditions (`ensures`), loop invariants, and termination measures, checked automatically by an SMT solver (Z3). A method that returns the maximum of a nonempty array, fully specified:

```dafny
method Max(a: array<int>) returns (m: int)
  requires a.Length > 0
  ensures forall j :: 0 <= j < a.Length ==> a[j] <= m
  ensures exists j :: 0 <= j < a.Length && a[j] == m
{
  m := a[0];
  var i := 1;
  while i < a.Length
    invariant 1 <= i <= a.Length
    invariant forall j :: 0 <= j < i ==> a[j] <= m
    invariant exists j :: 0 <= j < i && a[j] == m
  {
    if a[i] > m { m := a[i]; }
    i := i + 1;
  }
}
```

The loop invariant is the inductive hypothesis of a proof about the loop: it must hold on entry, be preserved by one iteration, and together with the negated loop condition (`i == a.Length`) imply the postcondition. Finding invariants is the creative step, and it is the step language models are good at proposing, because a wrong invariant is caught immediately and the model can be asked again with the error. Verus does the same for Rust with a specification language embedded in the source, aimed at systems code. In Lean, a program is a definition and its specification is a theorem about it; `sumTo` above is a verified program in that sense, with the theorem as its postcondition. Lean makes you write the proof; Dafny tries to find it, and fails silently into "cannot prove" when the invariant is missing.

What a model can do today: propose invariants and postconditions for small functions, translate a natural-language contract into a `requires`/`ensures` pair, and iterate on verifier errors. What it cannot do: be trusted to write the specification. The specification is the ground truth; if the model writes `ensures true`, the code verifies and means nothing. If it writes a postcondition that only mentions the return value's type, the same. A verified program is a proof that the code matches the spec, and the spec must be read by a human who understands the intent. This is the code-verification version of the autoformalization problem, and it is why every serious pipeline separates spec authorship from proof search.

### A first project for a sparse-recovery and privacy background

The restricted isometry property of order $s$ with constant $\delta$ says that for every $s$-sparse $x$,

$$
(1 - \delta)\|x\|_2^2 \le \|Ax\|_2^2 \le (1 + \delta)\|x\|_2^2 .
$$

The first theorem anyone proves about it is uniqueness: if $A$ has RIP of order $2s$ with $\delta < 1$ and $x, y$ are $s$-sparse with $Ax = Ay$, then $x = y$, because $x - y$ is $2s$-sparse and the lower bound gives $(1 - \delta)\|x - y\|^2 \le \|A(x - y)\|^2 = 0$. Here it is in Lean 4 with Mathlib, compiled on 2026-09-03. Vectors are functions `Fin n → ℝ`, norms are written as explicit sums of squares to stay elementary, and sparsity is the cardinality of the support:

```lean
import Mathlib
open Finset

variable {m n : ℕ}

def IsSparse (s : ℕ) (x : Fin n → ℝ) : Prop :=
  (univ.filter fun i => x i ≠ 0).card ≤ s

def HasRIP (A : Matrix (Fin m) (Fin n) ℝ) (s : ℕ) (δ : ℝ) : Prop :=
  ∀ x : Fin n → ℝ, IsSparse s x →
    (1 - δ) * ∑ i, x i ^ 2 ≤ ∑ j, (A.mulVec x j) ^ 2 ∧
    ∑ j, (A.mulVec x j) ^ 2 ≤ (1 + δ) * ∑ i, x i ^ 2

lemma isSparse_sub {s : ℕ} {x y : Fin n → ℝ} (hx : IsSparse s x) (hy : IsSparse s y) :
    IsSparse (2 * s) (x - y) := by
  unfold IsSparse at *
  calc (univ.filter fun i => (x - y) i ≠ 0).card
      ≤ ((univ.filter fun i => x i ≠ 0) ∪ (univ.filter fun i => y i ≠ 0)).card := by
        apply card_le_card
        intro i hi
        simp only [mem_filter, mem_univ, true_and, mem_union, Pi.sub_apply] at hi ⊢
        by_contra h
        rcases not_or.1 h with ⟨hx0, hy0⟩
        exact hi (by rw [not_not.1 hx0, not_not.1 hy0, sub_zero])
    _ ≤ (univ.filter fun i => x i ≠ 0).card + (univ.filter fun i => y i ≠ 0).card :=
        card_union_le _ _
    _ ≤ s + s := add_le_add hx hy
    _ = 2 * s := by ring

theorem rip_unique (A : Matrix (Fin m) (Fin n) ℝ) (s : ℕ) (δ : ℝ) (hδ : δ < 1)
    (hA : HasRIP A (2 * s) δ) (x y : Fin n → ℝ) (hx : IsSparse s x) (hy : IsSparse s y)
    (hxy : A.mulVec x = A.mulVec y) : x = y := by
  have hlow := (hA (x - y) (isSparse_sub hx hy)).1
  have hAz : A.mulVec (x - y) = 0 := by rw [Matrix.mulVec_sub, hxy, sub_self]
  rw [hAz] at hlow
  simp only [Pi.zero_apply, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    sum_const_zero] at hlow
  have hpos : 0 < 1 - δ := by linarith
  have hnn : 0 ≤ ∑ i, (x - y) i ^ 2 := sum_nonneg fun i _ => sq_nonneg _
  have h0 : ∑ i, (x - y) i ^ 2 = 0 := by
    apply le_antisymm _ hnn
    by_contra h
    have := mul_pos hpos (not_le.1 h)
    linarith
  have hzero : ∀ i, (x - y) i ^ 2 = 0 :=
    fun i => (sum_eq_zero_iff_of_nonneg fun i _ => sq_nonneg _).1 h0 i (mem_univ i)
  funext i
  have hi := (pow_eq_zero_iff two_ne_zero).1 (hzero i)
  simpa [sub_eq_zero] using hi
```

The structure mirrors the paper proof: the support of a difference is inside the union of supports (`card_le_card` plus `card_union_le`), the measurement of the difference is zero (`Matrix.mulVec_sub`), a nonnegative sum bounded above by zero is zero, a zero sum of squares has every term zero, and a zero square has a zero base. Everything else is bookkeeping about how Lean represents subtraction of functions (`Pi.sub_apply`) and zero powers. Expect a first attempt at this to take a full day; the mathematics is ten seconds and the bookkeeping is the day, and that ratio improves with practice but never reaches one.

Where this leads, in increasing difficulty. The null space property and its equivalence to uniform $\ell_1$ recovery is a few pages of real analysis with finite sums and is a reasonable second project. The Candès bound ($\delta_{2s} < \sqrt{2} - 1$ implies exact $\ell_1$ recovery) needs operator norms, the decomposition of a vector into blocks of decreasing magnitude, and several inequalities chained carefully; Mathlib has the norms and inner products, and this is weeks, not days, for someone who has done the first two. Random matrices satisfying RIP with high probability need concentration inequalities that Mathlib has only partially; that is a research-level formalization. On the privacy side, differential privacy has been formalized in Lean 4: SampCert (Amazon, 2024) is a verified implementation of discrete Gaussian sampling and the DP properties of the mechanisms built on it, so the definitions exist and a first project could be to prove the composition theorem for pure DP from them. Pick the uniqueness theorem first; it teaches the representation choices (functions versus `EuclideanSpace`, sums versus norms) that determine whether the next theorem is tractable.

## Build it small

The mechanism is the loop: generate, check, keep, retrain. The snippet below runs it with the smallest possible "model", a bag of tactic weights, against eight statements provable in core Lean 4 (no Mathlib), using the actual Lean binary as the verifier. Requires `lean` on the path (install with `elan`). It takes about 40 seconds because every check is a process launch.

```python
import random, subprocess, tempfile, collections

random.seed(0)
STATEMENTS = [
    "theorem t1 (a b : Nat) : a + b = b + a",
    "theorem t2 (n : Nat) (h : n < 5) : n < 10",
    "theorem t3 (p q : Prop) (hp : p) (hq : q) : p ∧ q",
    "theorem t4 : (List.range 10).length = 10",
    "theorem t5 (xs : List Nat) : (xs ++ []).length = xs.length",
    "theorem t6 (p q : Prop) (h : p → q) (hp : p) : q",
    "theorem t7 (a b : Nat) (h : a ≤ b) : a ≤ b + 3",
    "theorem t8 (p : Prop) (hp : p) : p ∨ False",
]
TACTICS = ["omega", "simp", "decide", "rfl", "exact ⟨hp, hq⟩", "exact h hp",
           "constructor <;> assumption", "left; exact hp", "trivial", "assumption"]


def lean_accepts(src):
    """The verifier. Reward 1 iff Lean compiles the file with no error and no sorry."""
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(src)
    r = subprocess.run(["lean", f.name], capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    return r.returncode == 0 and "sorry" not in out and "axiom" not in src


policy = collections.Counter({t: 1.0 for t in TACTICS})   # the "model": a bag of tactic weights
proofs = {}
for rnd in range(5):
    for s in STATEMENTS:
        if s in proofs:
            continue
        cands = random.choices(list(policy), weights=list(policy.values()), k=3)  # generate
        for t in cands:
            if lean_accepts(f"{s} := by\n  {t}\n"):                                 # check
                proofs[s] = t                                                        # keep
                policy[t] += 3.0                                                     # train
                break
    print(f"round {rnd}: solved {len(proofs)}/{len(STATEMENTS)}   "
          f"top tactics: {[t for t, _ in policy.most_common(3)]}")
for s, t in proofs.items():
    print(f"  {s.split()[1]}: by {t}")
```

Output from one run (Lean 4.33; the exact trajectory depends on the seed):

```
round 0: solved 3/8   top tactics: ['exact ⟨hp, hq⟩', 'left; exact hp', 'trivial']
round 1: solved 3/8   top tactics: ['exact ⟨hp, hq⟩', 'left; exact hp', 'trivial']
round 2: solved 6/8   top tactics: ['omega', 'simp', 'exact ⟨hp, hq⟩']
round 3: solved 7/8   top tactics: ['omega', 'simp', 'exact ⟨hp, hq⟩']
round 4: solved 7/8   top tactics: ['omega', 'simp', 'exact ⟨hp, hq⟩']
  t3: by exact ⟨hp, hq⟩
  t4: by trivial
  t8: by left; exact hp
  t1: by omega
  t5: by simp
  t7: by omega
  t2: by omega
```

Two things to see. The policy learned that `omega` and `simp` are broadly useful, which is what real provers learn too, and it learned it only from verified successes. And `t6`, which needs `exact h hp`, was never solved: the policy's mass concentrated on the tactics that had already paid off, so the one it needed was sampled too rarely. That is the exploration problem of expert iteration in miniature, and the reason real systems keep the sampling temperature up, allocate more samples to unsolved problems, and grow the statement set so that the easy wins do not dominate the training data.

## Build it real

`recipes/lean_eval.py` runs an open prover on the 5090 against a JSONL file of statements and checks each candidate with Lean. What you need in place first:

A Lake project with Mathlib built. `lake new leanlab math` then `lake exe cache get` downloads prebuilt Mathlib (several gigabytes; do it once). Pin the Mathlib version close to the model's training date; the model card says which. Lemma names drift (the `∑ i in` syntax became `∑ i ∈`, and the old form now warns), and a proof that used a renamed lemma fails for reasons that have nothing to do with reasoning.

A model. `deepseek-ai/DeepSeek-Prover-V1.5-RL` (7B) is the standard first choice; the 7B DeepSeek-Prover-V2 and Goedel-Prover variants use the same whole-proof format. In bf16 the weights are about 14 GB, leaving room for vLLM's KV cache at `gpu_memory_utilization=0.85`.

A statements file, one JSON object per line with `name`, `header` (the imports and `open` lines), and `statement` (the theorem up to and including `:= by`). Five miniF2F-style examples ship with the recipe, all checked to be true and provable, including:

```lean
theorem mod_example (n : ℕ) (h₀ : n % 5 = 3) : (2 * n) % 5 = 1 := by
theorem linear_example (x : ℝ) (h : 2 * x + 3 = 11) : x = 4 := by
theorem amgm2 (a b : ℝ) : a * b ≤ (a ^ 2 + b ^ 2) / 2 := by
```

Arguments: `--model` (Hugging Face id), `--statements` (JSONL path), `--n` samples per statement (default 32), `--temperature` (default 1.0; provers are sampled hot), `--max-tokens` (default 1024), `--project` (path to the Lake project), `--timeout` seconds per Lean check (default 120), `--workers` parallel Lean checks (default 4), and `--out` for the results JSONL, which records every sample, its compile output, and the verdict. The prompt is the header plus the statement, exactly as the model card shows; the model continues after `:= by`. The recipe stops the generation at the closing code fence and strips everything after it.

The checker is the part to understand, so here is its core:

```python
import re, subprocess, pathlib

PROJECT = pathlib.Path("~/leanlab").expanduser()   # Lake project with Mathlib built


def check(header: str, statement: str, proof: str, timeout: int = 120) -> tuple[bool, str]:
    """Compile header + statement + proof in the Mathlib project. Returns (ok, reason)."""
    src = f"{header}\n\n{statement}\n{proof}\n"
    path = PROJECT / "Scratch.lean"                 # use one file per worker in parallel
    path.write_text(src)
    try:
        r = subprocess.run(["lake", "env", "lean", str(path)], cwd=PROJECT,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = r.stdout + r.stderr
    if r.returncode != 0 or "error" in out:
        return False, out[:2000]
    if "sorry" in out:                              # Lean warns: declaration uses `sorry`
        return False, "sorry"
    if re.search(r"\b(axiom|native_decide|admit)\b", proof):
        return False, "forbidden construct"
    if statement.strip() not in src:                # the model must not alter the statement
        return False, "statement altered"
    return True, ""
```

Three checks beyond "it compiled": no `sorry` warning, no constructs that bypass the kernel (`axiom` introduces an assumption; `native_decide` trusts the compiler rather than the kernel), and the statement is verbatim what you asked for. Prover models will, given the chance, restate the theorem with an extra hypothesis and prove that.

What to watch in the logs. vLLM's throughput line: 32 samples for 5 statements at roughly 600 tokens each is under 100k tokens, a few minutes at most. Then the Lean checks: the first `lake env lean` on a Mathlib-importing file is slow while the oleans page in (a minute or more on a cold cache), and later ones take a few seconds each; with 160 candidates and 4 workers expect 10 to 20 minutes. If checks take much longer, it is usually a proof that sent `simp` or `nlinarith` into a long search, and the timeout is doing its job. For a bigger evaluation, replace the per-file process with the Lean REPL (leanprover-community/repl), which keeps Mathlib loaded and checks a snippet in well under a second.

Then audit. For every statement that passed, read one accepted proof. Check that it proves your statement, that it did not reach the goal through a contradiction in the hypotheses, and that any `have` it introduced was itself proved. Only after that is the pass rate a number you can quote.

## How it goes wrong

The proof compiles and proves nothing. `sorry` inside a `have`, an `axiom` declared above the theorem, or a `native_decide` that evaluates a large computation outside the kernel. Symptom: a suspiciously high pass rate, or proofs of statements you know to be hard that are three lines long. Fix: the three extra checks in the checker, and a grep of the whole generated file, not just the proof body.

The statement drifted. The model's output includes a restated theorem with a changed type (`ℕ` for `ℝ`), an added hypothesis, or a weakened conclusion, and it compiles. Symptom: the accepted proof's first line differs from your statement. Fix: verbatim statement comparison, and prompt so the model continues after `:= by` rather than restating.

Mathlib version mismatch. Symptom: many failures with `unknown identifier` or `unknown constant` on names that look plausible. Cause: the lemma was renamed or moved since the model's training data. Fix: pin Mathlib near the model's date, or accept the loss and report it; do not patch proofs by hand and count them.

Vacuous statements. A misformalized statement with an impossible hypothesis (`h : x < 0` for `x : ℕ`) is provable by `omega` in one step. Symptom: an easy proof of a hard-looking problem. Fix: an audit step that tries to prove `False` from the hypotheses alone; if that succeeds, the statement is wrong.

Timeouts mistaken for failures. `simp` or `nlinarith` can run for minutes on a bad hint set; Lean also has a heartbeat limit that raises an error (deterministic timeout) before your wall-clock timeout. Symptom: results that change when you change the timeout. Fix: fix the budget in advance, report it with the score, and count timeouts separately from wrong proofs.

Exploration collapse in expert iteration. After a few rounds the model solves the same easy problems with the same tactics and the training set is dominated by them. Symptom: training-set pass rate rises while validation pass rate is flat; proofs get shorter and more uniform. Fix: sample more per unsolved statement, keep temperature high for generation, deduplicate proofs per statement before training, and add new statements each round.

The spec is wrong. In code verification, the postcondition is too weak (`ensures true`, or a condition on the type but not the value), or the precondition is so strong that no realistic input satisfies it. Symptom: verification succeeds on the first try. Fix: a human reads the spec against the intent; try to verify a deliberately wrong implementation and confirm the verifier rejects it.

Editor state confusion. Lean's editor mode re-elaborates as you type; a proof that looks complete may have an error higher in the file that made everything after it unchecked. Symptom: no red squiggle under a proof you are sure is incomplete. Fix: check the file with `lake env lean` from the command line before believing the editor.

## Measure it

For a prover, report pass@$k$ at a stated $k$ and stated temperature, the Mathlib version, the checker rules, the timeout, and the number of survivors that passed a fidelity audit. On the five recipe statements a 7B open prover at 32 samples should solve most of them; they are easy by design. On miniF2F-test, published 7B provers with whole-proof sampling report on the order of half the problems at budgets of a few dozen samples and more at thousands; if you reproduce a published figure within a few points at the same budget and Mathlib version, your harness is right, and if you exceed it substantially, look for one of the four cheats before celebrating. Also report the compile rate (the fraction of samples that compile at all, including duplicates), which tracks syntax fluency and Mathlib-version fit separately from reasoning, and the timeout rate.

For your own proving, the metric is time to a compiled proof with the theorem statement fixed in advance; write the statement, commit it, and only then prove it, so that difficulty does not quietly reshape the theorem. For a formalization project, count the definitions that other people could build on, not the theorems; a good `IsSparse` and `HasRIP` are worth more than a clever proof.

For code verification, the number that matters is whether a deliberately introduced bug is caught: a verified function whose spec cannot distinguish it from a wrong one has a spec problem, and the test is to break the implementation and rerun.

## Exercises

1. Prove `theorem add_sq_nat (a b : ℕ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2` with Mathlib's `ring`, then again without `ring` using only `Nat.pow_two`, `Nat.add_mul`, `Nat.mul_add`, `Nat.mul_comm`, and `omega`. Check: both compile. If you leave out the commutativity rewrite, `omega` fails and its counterexample lists `a * b` and `b * a` as separate atoms, which is the lesson: `omega` does not know multiplication commutes, and `ring` does. (The name `add_sq` is taken by Mathlib.)

2. Prove that `sumTo n = n * (n + 1) / 2` in core Lean from `two_mul_sumTo`. Check: you need `Nat.mul_div_cancel_left` or `omega` after the rewrite; the truncated division is exact here because the numerator is even, and you must say why.

3. Write `theorem t6` from the toy loop with a proof, then extend the loop so each round allocates its samples in proportion to the number of rounds a statement has gone unsolved. Check: `t6` is solved within 5 rounds in most seeds.

4. Formalize the null space property of order $s$ for a matrix (for every nonzero $v$ in the kernel and every index set $S$ of size at most $s$, $\|v_S\|_1 < \|v_{S^c}\|_1$) and prove that it implies `rip_unique`'s conclusion for $\ell_1$ minimization in the form: if $x$ is $s$-sparse and $Az = Ax$ with $\|z\|_1 \le \|x\|_1$ then $z = x$. Check: this is the standard one-page proof; expect it to take a day and expect the difficulty to be in `Finset.sum` manipulations over complements.

5. Run `lean_eval.py` twice with the same seed and confirm the results are identical; then run with `--temperature 0.3` and `--temperature 1.2` at `--n 32` and compare pass rates and compile rates. Check: the low temperature usually has the higher compile rate and the lower pass rate.

6. Write the Dafny `Max` method's postconditions in Lean as a theorem about your own `listMax : List Int → Int`, and prove it by induction on the list. Check: you will need a nonempty hypothesis and will discover why Dafny's `requires a.Length > 0` was there.

## Test yourself

1. This compiles. What does it prove, and what should an evaluation harness do about it?

```lean
theorem hard_looking (x : ℕ) (h : x < 0) : x = 42 := by omega
```

<details><summary>Answer</summary>
It proves an implication with a false antecedent: no natural number is below zero, so `omega` derives a contradiction from `h` and the goal follows. The statement is vacuous, and a prover that solves it has learned nothing about 42. A harness should test each statement's hypotheses for inconsistency (try to prove `False` from them) and flag any statement where that succeeds as misformalized rather than solved. This is the most common way autoformalized benchmarks inflate.
</details>

2. Why does this fail, and what is the corrected statement?

```lean
theorem pred_succ (n : ℕ) : n - 1 + 1 = n := by omega
```

<details><summary>Answer</summary>
Subtraction on `ℕ` is truncated: `0 - 1 = 0`, so at `n = 0` the left side is `1` and the statement is false; `omega` reports it cannot prove the goal because there is a counterexample. Corrected: add `(h : 1 ≤ n)` or `(h : 0 < n)`, or state it over `ℤ`. Informal mathematics would never write the original because the reader assumes `n ≥ 1`; Lean assumes nothing.
</details>

3. Spot the bug in the proof attempt, and say what `omega` treats as an atom.

```lean
theorem two_mul_sumTo (n : Nat) : 2 * sumTo n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih => omega
```

<details><summary>Answer</summary>
`omega` fails in the successor case for two reasons. It does not unfold `sumTo`, so `sumTo (k + 1)` is an opaque atom unrelated to `sumTo k` and the hypothesis `ih` is useless. And even after unfolding, the goal contains `k * (k + 1)` and `(k + 1) * (k + 2)`, which `omega` treats as two unrelated atoms because it only understands multiplication by literals; it cannot see that both contain `k * k`. The fix in the chapter is to unfold, rewrite with `ih`, distribute with `Nat.mul_add` and `Nat.add_mul` so the only nonlinear atom is `k * k` on both sides, and then call `omega`. In Mathlib, `ring` does the last step directly.
</details>

4. A prover is asked for `theorem amgm2 (a b : ℝ) : a * b ≤ (a ^ 2 + b ^ 2) / 2 := by` and returns the following; it compiles and the checker marks it as passed. What is wrong?

```lean
theorem amgm2 (a b : ℝ) (hab : a = b) : a * b ≤ (a ^ 2 + b ^ 2) / 2 := by
  nlinarith [sq_nonneg (a - b)]
```

<details><summary>Answer</summary>
The model restated the theorem with an extra hypothesis `hab : a = b`. The proof does not even use it, so here the strengthening is harmless, but the harness cannot know that: a model that learns it may add hypotheses will add the one that makes the next problem trivial (`hab : a = 0`, say). The checker passed it because it only checked compilation. The harness must compare the statement verbatim with what it asked for (or, better, feed only the statement and let the model continue after `:= by`, then check that the file still contains the statement unchanged). A related trick is changing the type: the same statement over `ℕ` has floor division and truncated subtraction and is a different theorem; with this proof it happens not to compile, because `nlinarith` cannot reason about `ℕ` division, but a model will find a proof of the wrong theorem if one exists.
</details>

5. Estimate the wall-clock cost of evaluating a prover on 244 miniF2F-test statements at 64 samples each, (a) with one `lake env lean` process per check at 8 seconds each, sequentially, and (b) with a Lean REPL that checks in 1 second, with 16 parallel workers. State your assumptions.

<details><summary>Answer</summary>
$244 \times 64 = 15{,}616$ checks. (a) $15{,}616 \times 8 \text{ s} \approx 125{,}000$ s, about 35 hours. (b) $15{,}616 \times 1 / 16 \approx 976$ s, about 16 minutes. Assumptions: every sample is checked (no deduplication and no early stopping on the first success per statement, either of which cuts the count substantially), checks do not contend for memory (16 Mathlib-loaded REPLs need tens of GB of RAM), and no check hits the timeout, which would add up to the timeout per hit. The lesson is that the checker, not the GPU, is the bottleneck of a proving evaluation, and the engineering goes there.
</details>

6. System A reports pass@1 of 30 percent on miniF2F-test; system B reports pass@32 of 55 percent. Which is the better prover?

<details><summary>Answer</summary>
You cannot tell. pass@32 is bounded below by pass@1 and for a stochastic prover is typically much higher, so a 30 percent pass@1 system could easily exceed 55 percent at 32 samples. You would also need the same benchmark version, the same Mathlib version, the same checker rules, and the same timeout. If system A used greedy decoding, its pass@1 is a different quantity again (a single deterministic sample) from the expectation that pass@1 denotes for a sampled prover. Ask for a pass@$k$ curve at matched settings.
</details>

7. A colleague argues: "Lean accepted the proof, so the theorem is true; formal verification removes the need to review it." Give the three assumptions hiding in "true" and the one hiding in "the theorem".

<details><summary>Answer</summary>
"True" assumes the kernel is correct (a small trusted base, but not zero), that no `axiom` beyond Lean's standard three (propositional extensionality, quotient soundness, choice) was added, and that nothing bypassed the kernel (`native_decide` trusts the compiler and the C toolchain; `sorry` is a warning, not an error). "The theorem" assumes the formal statement means what the informal one meant: types, totalized operations, quantifier scope, and non-vacuous hypotheses. The review that verification removes is the review of the proof steps; the review of the statement is more important than before, because it is now the only thing that can be wrong.
</details>

8. In the toy loop, `t6` was never solved because the policy's mass concentrated elsewhere. Write the expected number of samples until the first success for `t6` as a function of its tactic's policy weight, and say what breaks in a real expert-iteration run if the same effect dominates.

<details><summary>Answer</summary>
With $w$ the weight of `exact h hp` and $W$ the total weight, each sample succeeds with probability $w / W$, so the expected number of samples to first success is $W / w$; as rewards accrue to other tactics, $W$ grows while $w$ stays at 1, and the expectation grows linearly with the accumulated reward. In a real run the equivalent is a model whose distribution sharpens toward the proof styles that already succeeded; the training set becomes dominated by easy problems solved in one way, the validation pass rate on harder problems stalls, and, because filtering keeps only successes, there is no gradient toward the unsolved problems at all. Fixes: keep sampling temperature high, allocate the sample budget preferentially to unsolved statements, deduplicate proofs per statement, and keep adding new statements so the easy ones are a shrinking fraction.
</details>

9. In the Dafny `Max` method, delete the invariant `exists j :: 0 <= j < i && a[j] == m`. Which postcondition fails and why, and does the code become incorrect?

<details><summary>Answer</summary>
The second postcondition (`m` is attained by some element) fails: at loop exit Dafny knows `m` is an upper bound of the array from the first invariant but has no fact tying `m` to any element, so it cannot prove existence. The code is unchanged and still correct; verification fails because the inductive hypothesis is too weak to carry the fact through the loop. This is the general shape of verifier failures: "cannot prove" means the invariant does not capture what the loop maintains, not that the loop is wrong, and it is the failure a language model is most useful at repairing, since the verifier says exactly which obligation is unmet.
</details>

10. You want to formalize "for a Gaussian matrix with $m \ge C s \log(n/s)$ rows, RIP of order $s$ holds with probability at least $1 - e^{-cm}$". Name the three pieces of mathematics that must exist in Mathlib for this to be a one-month project rather than a one-year one, and say which you would check for first.

<details><summary>Answer</summary>
A concentration inequality for the norm of a Gaussian vector under a fixed matrix (a Johnson-Lindenstrauss style bound), a covering-number or net argument for the unit sphere in $\mathbb{R}^s$ with the union bound over nets, and the union bound over the $\binom{n}{s}$ supports with the entropy estimate $\log \binom{n}{s} \le s \log(en/s)$. Check the concentration inequality first: Mathlib has sub-Gaussian machinery and some Chernoff-type bounds, but whether the exact form you need exists decides everything. The net argument is elementary but long; the binomial bound is a good self-contained first lemma. If the concentration result is missing, the honest plan is to state it as a hypothesis, prove RIP conditional on it, and leave the probabilistic input as a separate project.
</details>

## What will change, what will not

Curry-Howard and the kernel will not change. The correspondence between proofs and programs is a theorem about logic, and every proof assistant that matters (Lean, Coq, Isabelle, Agda) is built on it. The idea that trust reduces to a small kernel plus stated axioms, and that everything else is untrusted automation, is the design principle that makes machine-generated proofs acceptable at all, and it is why a proof from a language model is worth the same as a proof from a human once it checks. Tactic names, `simp` lemma sets, and Mathlib's naming grammar will drift, and Lean 4 itself changes syntax between versions; expect to relearn the surface every couple of years.

The verifier-in-the-loop training recipe will persist, and its failure modes with it. Generate, check, keep, retrain is the algorithm whenever a cheap exact checker exists, and the checker's exactness is the reason the loop does not collapse into reward hacking. Exploration collapse, contamination, and the budget-dependence of pass@$k$ are structural and will be present in whatever replaces the current systems. Which model family leads (DeepSeek-Prover, Goedel, Kimina, or a closed lab system) will change within a year of this writing, and the benchmark that matters will move from miniF2F to PutnamBench to whatever is unsaturated; the questions to ask about a score will not.

Autoformalization is the part with the most room to change and the part where the invariant is most important. Models will get much better at producing formal statements; the need for a fidelity check independent of the checker will not go away, because the checker cannot see intent. The same is true for specifications in code verification: automation will write more invariants and more postconditions, and the human's job will narrow to reading specs, which is the job that cannot be delegated.

The representation choices you make when formalizing your own field are durable in a different sense: they persist in your project and determine what you can prove next. Vectors as functions versus `EuclideanSpace`, sparsity as a support cardinality versus a predicate on index sets, norms as sums versus instances. Choose to match Mathlib's existing definitions where they exist, because the lemmas are there; otherwise choose the representation that makes the next theorem's statement short. That advice will hold in any library.

The tooling around Lean (the editor integration, the REPL, LeanDojo, Lean Copilot, the cache server) will be replaced and rebuilt several times. The one habit worth keeping regardless is to check from the command line what the editor claims, and to keep statements in version control before proofs are attempted.

## Read next

1. "Theorem Proving in Lean 4", Avigad, de Moura, Kong, and Ullrich (online book). The reference for the language and the tactic system; read the chapters on propositions, quantifiers, and tactics first.
2. "Mathematics in Lean", Avigad and Massot (online book). Learn Mathlib by doing; the exercises on real analysis and finite sums map directly onto the RIP project.
3. "miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics", Zheng, 2021. What the benchmark contains and how it was built; the source for what a score there means.
4. "LeanDojo: Theorem Proving with Retrieval-Augmented Language Models", Yang, 2023. The interaction and data-extraction toolkit and the retrieval-augmented tactic prover; the paper to read to understand tactic-level search.
5. "DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search", Xin, 2024. A complete description of the generate, check, retrain loop with reinforcement learning at scale, and the model the recipe runs.
6. "Autoformalization with Large Language Models", Wu, 2022. The first demonstration that language models can translate competition statements to Isabelle, and an honest accounting of how often they get it wrong.
7. "PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition", Tsoukalas, 2024. The harder benchmark and the case for contamination-resistant statements.
8. "Dafny: An Automatic Program Verifier for Functional Correctness", Leino, 2010. The design of specifications, invariants, and automated proof for imperative programs.
