"""GRPO reward for mlx_post.py (loaded by mlx-lm-lora with --reward-functions-file): the task's own verifier.

The task is named by the PT_TASK environment variable (set by mlx_post.py); `answer` carries what the verifier needs
(the arm task id, or the gsm8k final number). A malformed completion scores zero, never crashes training.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_tasks  # noqa: E402
from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function  # noqa: E402

TASK = pt_tasks.get(os.environ.get("PT_TASK", "arm"))


@register_reward_function()
def task_reward(prompts, completions, answer, types=None):
    out = []
    for c, ans in zip(completions, answer):
        try:
            out.append(float(TASK.score(ans, c)))
        except Exception:
            out.append(0.0)
    return out
