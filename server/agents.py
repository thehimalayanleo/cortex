"""Hand a task to a coding agent CLI (Codex, OpenCode, Claude Code) running inside the vault."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterator

from . import vault

AGENTS = {
    "codex": {"bin": "codex", "argv": lambda task: ["codex", "exec", "--full-auto", "-C", str(vault.VAULT), task]},
    "opencode": {"bin": "opencode", "argv": lambda task: ["opencode", "run", task]},
    "claude": {"bin": "claude", "argv": lambda task: ["claude", "-p", task, "--permission-mode", "acceptEdits"]},
}


def available() -> list[dict]:
    out = []
    for aid, a in AGENTS.items():
        path = shutil.which(a["bin"])
        ver = ""
        if path:
            try:
                ver = subprocess.run([a["bin"], "--version"], capture_output=True, text=True, timeout=15).stdout.strip()[:40]
            except Exception:
                ver = "?"
        out.append({"id": aid, "available": bool(path), "version": ver})
    return out


def run(agent: str, task: str, timeout: int = 900) -> Iterator[str]:
    """Yield output lines. Runs with cwd = the vault so file tools act on the brain."""
    a = AGENTS.get(agent)
    if not a or not shutil.which(a["bin"]):
        yield f"[cortex] agent '{agent}' is not installed"
        return
    env = {**os.environ, "CORTEX_VAULT": str(vault.VAULT)}
    proc = subprocess.Popen(a["argv"](task), cwd=str(vault.VAULT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    yield f"[cortex] {agent} started in {vault.VAULT}"
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill(); yield "[cortex] timed out"
    finally:
        if proc.poll() is None:
            proc.kill()
    yield f"[cortex] {agent} exited with code {proc.returncode}"
    vault.rebuild_index()


def run_capture(agent: str, task: str, max_chars: int = 8000, timeout: int = 600) -> str:
    buf: list[str] = []
    for line in run(agent, task, timeout=timeout):
        buf.append(line)
        if sum(len(l) for l in buf) > max_chars:
            buf.append("[cortex] output truncated"); break
    return "\n".join(buf)
