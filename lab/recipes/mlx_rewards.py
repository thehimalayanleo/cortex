"""GRPO reward functions for mlx_post.py (loaded by mlx-lm-lora with --reward-functions-file).

arm_sim: the arm simulator is the verifier. Each row's `answer` is a task id; the task (scene + hidden goal) is rebuilt
and the completion is judged exactly like the arm benchmark: 1.0 when the hidden goal holds, 0.2 for a program the
simulator accepts but that misses the goal, 0.0 otherwise.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "armft"))
sys.path.insert(0, str(HERE.parent.parent / "server"))
import armft  # noqa: E402
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function  # noqa: E402

TASKS = {t["id"]: t for t in armft.tasks(range(1, 41))}


@register_reward_function()
def arm_sim(prompts, completions, answer, types=None):
    out = []
    for c, tid in zip(completions, answer):
        try:
            ok, valid, _, _ = armft.judge(TASKS[tid], c)
        except Exception:  # a malformed program is a zero, never a crash
            ok = valid = False
        out.append(1.0 if ok else (0.2 if valid else 0.0))
    return out
