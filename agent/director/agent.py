"""The Director: a Google ADK agent (Gemini) that runs the Cortex Studio.

It plans shots from a logline, renders takes on the GPU box through Cortex, reads the critics' verdicts, decides
keep or reshoot, and reports through Grafana (the partner track): render telemetry is pushed to Grafana Cloud by
the Cortex server, and the agent queries it back through the hosted Grafana Cloud MCP.

Run locally:
    pip install google-adk
    export GOOGLE_API_KEY=...            # or GOOGLE_GENAI_USE_VERTEXAI=1 GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=us-central1
    export CORTEX_URL=http://127.0.0.1:8788
    export GRAFANA_URL=https://<stack>.grafana.net   # optional; enables the Grafana MCP tools
    cd cortex/agent && adk web            # then pick "director"

Deploy: `adk deploy agent_engine` (Vertex AI Agent Engine) or `adk deploy cloud_run`; see README.md next to this file.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from google.adk import Agent

try:  # the Grafana MCP toolset is optional: without GRAFANA_URL the director still runs the studio
    from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
except Exception:  # pragma: no cover
    McpToolset = None  # type: ignore
    StreamableHTTPConnectionParams = None  # type: ignore

CORTEX = os.environ.get("CORTEX_URL", "http://127.0.0.1:8788").rstrip("/")
MODEL = os.environ.get("DIRECTOR_MODEL", "gemini-2.5-flash")


def _call(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    req = urllib.request.Request(CORTEX + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "cortex-director/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"null")


# ---------------------------------------------------------------- Cortex studio tools (function tools)

def studio_board() -> dict:
    """The Studio board: logline, shots with status, each shot's takes with critic scores and verdicts, keyframe assets."""
    return _call("GET", "/api/studio")


def plan_shots(logline: str, n: int = 4) -> dict:
    """Turn a logline into n planned shots (title, image-to-video prompt, notes) on the board."""
    return _call("POST", "/api/studio/plan", {"logline": logline, "n": n})


def set_shot(shot_id: str, prompt: str | None = None, keyframe: str | None = None, frames: int | None = None, size: str | None = None, status: str | None = None, director_note: str | None = None) -> dict:
    """Edit a shot: prompt, keyframe (asset name or a path on the GPU box), frames, size, status (planned|rendering|rendered|approved|reshoot), director_note."""
    patch = {k: v for k, v in {"prompt": prompt, "keyframe": keyframe, "frames": frames, "size": size, "status": status, "director_note": director_note}.items() if v is not None}
    return _call("PUT", f"/api/studio/shots/{shot_id}", patch)


def render_shot(shot_id: str, smoke: bool = False) -> dict:
    """Render a take on the GPU box (about a minute for 49 frames); smoke=True runs the offline test brick. Returns the run."""
    return _call("POST", f"/api/studio/shots/{shot_id}/render", {"smoke": smoke})


def read_run(run_id: str) -> dict:
    """Status, metrics, result and log tail of a run (a take)."""
    r = _call("GET", f"/api/lab/runs/{run_id}?tail=40")
    m = r.get("metrics", [])
    return {k: r.get(k) for k in ("id", "recipe", "status", "started", "ended", "result")} | {"metrics_tail": m[-10:], "log": r.get("log", [])}


def refresh_shot(shot_id: str) -> dict:
    """After a take finishes: fetch its clip, contact sheet and scores, and set the shot status from the critics' verdict."""
    return _call("POST", f"/api/studio/shots/{shot_id}/refresh")


def gpu_status() -> dict:
    """Is the GPU box reachable and ready (GPU, torch, Tailscale link, busy)?"""
    return _call("GET", "/api/lab/gpu")


def telemetry_status() -> dict:
    """Whether render telemetry is flowing to Grafana Cloud (metrics, logs), and how much was sent."""
    return _call("GET", "/api/telemetry")


TOOLS: list[Any] = [studio_board, plan_shots, set_shot, render_shot, read_run, refresh_shot, gpu_status, telemetry_status]

if McpToolset is not None and os.environ.get("GRAFANA_URL"):
    TOOLS.append(McpToolset(connection_params=StreamableHTTPConnectionParams(url=os.environ.get("GRAFANA_MCP_URL", "https://mcp.grafana.com/mcp"), headers={"X-Grafana-URL": os.environ["GRAFANA_URL"]})))

INSTRUCTION = """You are the Director of a small animated film studio that runs inside Cortex.
Your crew: a render farm (one RTX 5090 reached through Cortex), three critics (identity, flicker, and a VLM judge on keyframes), and Grafana, where every take's metrics land as the series cortex_run{recipe="cinema_render"} with fields identity, flicker, identity_mean, identity_min, flicker_mean, gen_s.

How you work:
1. Read the board first (studio_board). Never plan shots that already exist.
2. Given a logline, plan 3 to 6 shots with concrete image-to-video prompts: subject, action, camera, light, mood; keep character and setting continuous across shots.
3. Before rendering, check gpu_status. Render one shot at a time (render_shot), then poll read_run until status is done or failed, then refresh_shot to get the verdict.
4. Decide: verdict keep -> status rendered; reshoot -> change one thing in the prompt (less motion, a static camera, or a clearer subject) and render again, at most 3 takes per shot. Say what you changed and why.
5. When Grafana tools are available, use them to answer questions like "which shot regressed" or "how did identity drift over today's takes": query the cortex_run metrics, and cite the numbers.
6. Report like a working director: short, concrete, with the numbers (identity mean and min, flicker mean, seconds) and the next action. Never invent scores; read them.
"""

root_agent = Agent(name="director", model=MODEL, description="Plans, renders, critiques, and reshoots shots in the Cortex Studio; reports through Grafana.", instruction=INSTRUCTION, tools=TOOLS)
