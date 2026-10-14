# Devpost draft: Nebius x NVIDIA Global AI Hackathon, Physical AI track

Status: draft. Items marked TODO need the Token Factory key and a rerun before submission.

## Name

Cortex Physical: a desk arm that proves its plans before it moves

## Tagline

A small Nemotron planner, fine-tuned on plans a simulator verified, that refuses unsafe requests for the right reason.

## What it does

Cortex is a local research lab app. Its Physical tab drives an SO-ARM101 desk arm (or simulated servos). You type a
task ("put the red block in the bin"). An NVIDIA Nemotron model writes a short skill program over the objects on the
table and declares what the table should look like afterwards. A deterministic simulator then compiles the program with
inverse kinematics and replays it. It checks joint limits, reach, collisions, grasps and drops, and whether the
declared goal actually holds. Only a verified plan reaches the arm. A rejected plan goes back to the model with the
simulator's errors.

The planner is NVIDIA Nemotron 3 Nano 4B, fine-tuned in the app. One button trains a LoRA on programs the simulator
verified, scores it on held-out tables and on refusal checks, and switches the arm to the new adapter.

## Why

Language models are good at deciding what should happen on a table and bad at joint angles. Fine-tuning fixes the
format but can quietly remove a model's ability to say no. On a robot arm that is the failure that matters. We wanted
every claim in the demo to be checked by something that is not a model.

## Results (held-out tables, scene seeds 31-40, 65 tasks, no repair; 5 training seeds)

| | plans that work | refuses unsafe arm requests (right reason) | refuses harmful chat requests |
|---|---|---|---|
| Nemotron 3 Nano 4B, base | 1.5% | 0% | 99% |
| + LoRA on 200 verified programs | 72-95% (mean 85%) | 0-85%, reason always "out of reach" | 99% |
| + 56 safety refusals in the mix | 91-95% (mean 93.5%) | 100% in every seed | 99% |

Over-refusal on ordinary tasks that mention people: 0-10%. The out-of-reach task is refused in every seed. Both
experiments were registered with pass rules before running (ARM-FT-1, ARM-SAFE-1) and both passed. Caveat: the 20
held-out unsafe requests and the training templates were written by the same author, so this shows generalization to
new wording, not to an independent red team.

## How we used NVIDIA models and Nebius

- NVIDIA Nemotron 3 Nano 4B (open weights) is the planner. LoRA on one RTX 5090 with NVIDIA's Mamba kernels,
  about 100 s per run; served from the same GPU.
- TODO (Token Factory): Nemotron 3 Super on Nebius Token Factory plans with simulator repair, and its verified plans
  become new training data; Nemotron 3 Ultra is the escalation tier on the last repair. The code path exists
  (server/arm.py, Token Factory key card in the UI); it has not been run with a key yet. Rerun the benchmark with the
  key and fill in the numbers.
- TODO: say where Token Factory accelerated the workflow, once measured.

## Built with

Python, FastAPI, React, TypeScript, PyTorch, PEFT, transformers, mamba-ssm, vLLM (Modal path), MLX (Mac path),
NVIDIA Nemotron 3 Nano 4B, Nebius Token Factory (TODO), Tailscale.

## What was built during the submission period

The Physical tab started on 2026-09-04 as a joint-space planner with simulated servos. Since then: the tabletop
simulator and verifier, the plan-verify-repair loop, the benchmark, the Token Factory and self-hosted providers, the
fine-tuned Nemotron planner and its training button, the safety training, and a Mac-native (MLX) path for small jobs.

## Feedback on Nebius and NVIDIA tools

- Nemotron 3 Nano 4B with PEFT needs mamba-ssm and causal-conv1d kernels; without them transformers falls back to a
  pure PyTorch scan that ran out of memory at 512 tokens on a 32 GB card. Prebuilt wheels exist for torch 2.10 / CUDA 12,
  not for newer torch. A note on the model card would save time.
- PEFT refuses LoRA on the Mamba `out_proj` and `conv1d` modules; the model card could list the supported target modules.
- TODO: Token Factory feedback after use.

## Demo video

out/demo/physical-demo.mp4 (app modules; no hardware arm footage yet). TODO: re-record the narration's Token Factory
line once it is actually used, and upload to YouTube.
