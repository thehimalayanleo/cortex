# Cortex Studio: an agentic director for the Agentic Cinema hackathon

Track: **Grafana Labs**. Submission for https://agentic-cinema.devpost.com (deadline 2026-09-09 14:00 PDT).

## Inspiration

Generative video is cheap to start and expensive to finish. A single image-to-video call is one command, but a *shot* is a
loop: render, look, decide whether the character still looks like the character, whether the frames flicker, and reshoot
with a changed prompt. That loop is what a director does with a crew, and it is what nobody has time to do by hand across
forty shots. Cortex already had the crew: a GPU box with Wan 2.2 for motion, an identity critic (DINOv2/CLIP against a hero
image), a flicker critic, and a scene assembler. What was missing was the director.

## What it does

The **Studio** is a section of the Cortex Lab. A logline becomes a shot list; each shot has a keyframe (a hero image or a
generated set), a prompt, and *takes*. A take is one render on the RTX 5090 over Tailscale (or a smoke brick on any
laptop), scored per frame by the critics, and given a verdict: keep if identity never drops below 85 percent of the first
frame and flicker never spikes past three times its mean, otherwise reshoot. Characters have hero sets and LoRAs built the
same way; scenes are ordered shots assembled with ffmpeg.

The **Director** is a Google ADK agent (Gemini) whose tools are the Studio's API: read the board, plan shots, set a
prompt or keyframe, render, read the run, refresh the take, approve or reshoot, plan and assemble a scene. It runs the
loop above on its own and explains each decision in the critics' numbers.

**Grafana** is the report. Every take's per-frame metrics (identity, flicker, generation seconds) are pushed by the Cortex
server to Grafana Cloud as `cortex_run_*` series with `run`, `shot`, `recipe` labels, and every log line to Loki. The
dashboard in `grafana/` shows identity and flicker per take, the keep/reshoot table, and the render log. The Director
queries the same stack back through the **hosted Grafana Cloud MCP** (`https://mcp.grafana.com/mcp`, `X-Grafana-URL`), so
"which shots reshot most this week" is a question the agent answers from Grafana, not from memory.

## How it is built

- Agent: `agent/director/agent.py`, `google-adk`, `Agent(model="gemini-2.5-flash", tools=[function tools over the Cortex API, McpToolset(StreamableHTTPConnectionParams(url="https://mcp.grafana.com/mcp", headers={"X-Grafana-URL": ...}))])`. Runs with `adk web`, deploys with `agent/deploy.sh cloud_run` (Cloud Build + Cloud Run) or `agent_engine` (Vertex AI Agent Engine). `agent/director/Dockerfile` is the same service as a plain container.
- Studio server: `server/studio.py` (shots, takes, characters, scenes), `server/telemetry.py` (Grafana Cloud Influx push + Loki push), `lab/recipes/cinema_render.py` (Wan 2.2 TI2V-5B two-phase decode on the 5090, critics, verdict), `cinema_character.py` (hero, hero set, LoRA).
- Web: `web/src/views/StudioView.tsx`, Lab tab "Studio"; the same actions are chat tools and WebMCP tools.
- Chapter: Lab 20, "Agentic cinema", teaches the loop (critics, verdict rule, why identity is the lever).

## Google Cloud and partner usage

| Requirement | Where |
|---|---|
| Gemini | `agent/director/agent.py` picks `gemini-2.5-flash` when `GOOGLE_API_KEY` or Vertex env is set |
| Agent Builder / ADK | the agent is an ADK `Agent`; `adk web`, `adk deploy cloud_run`, `adk deploy agent_engine` |
| Hosted on Google Cloud | Cloud Run service `cortex-director` from `agent/deploy.sh` |
| Grafana at runtime | server pushes metrics/logs to Grafana Cloud; agent reads through the Grafana Cloud MCP |

## Challenges

- The GPU is shared with research jobs. The renderer refuses a real take when another process holds more than 6 GB unless forced, and retries once after an out-of-memory kill; the smoke brick keeps the whole loop testable without the GPU.
- Wan 2.2 at 5B needs a two-phase decode on 32 GB (latents, free the denoiser and text encoder, tiled VAE); identity lock at 0.90 to the hero across frames was the milestone that made verdicts meaningful.
- `$HOME` in environment forwarded over SSH is not expanded by the remote shell in the way a recipe expects; the recipe expands it.

## What is next

Real characters with LoRA identity across every shot of a scene; the Director choosing reshoot prompts from the critics' curves rather than only from the verdict; a Grafana alert that pages the Director when a scene's identity floor drops.

---

## Demo video script (3 minutes)

0:00 Cortex open on the Studio tab: the board with the logline "a kaiju of black coral rises from a storm and hesitates", five shots, takes with contact sheets.
0:20 `adk web`, pick "director". Prompt: "Read the board. Render 'Okuun rises' on the 5090, wait for the critics, and tell me keep or reshoot with the numbers."
0:40 Agent calls studio_board → render_shot → read_run (log streams: two-phase decode, identity per frame) → refresh_shot. It reports identity_min, flicker_max, verdict.
1:30 Switch to Grafana Cloud: the Cortex Studio dashboard updates with the take's identity and flicker curves and the log lines. Point at the keep/reshoot table.
1:50 Back to the agent: "Which shot reshot most today, and what did its flicker look like?" It answers through the Grafana MCP tools (query the series), not from its context.
2:20 "Plan a ten-second filler scene of the shoreline for post-production, render what is missing with the smoke brick, and assemble it." Agent plans, renders, assembles; the scene mp4 plays.
2:50 Cloud Run URL of the deployed agent in the browser; closing line.

## Submission checklist

- [ ] `GOOGLE_API_KEY` (aistudio.google.com) or a GCP project with Vertex AI; hackathon credits form.
- [ ] Grafana Cloud free stack: stack URL, metrics instance id + token, Loki id + token; accept the Cloud MCP terms; a service-account token for `grafana/push_dashboard.sh`.
- [ ] `export GRAFANA_METRICS_URL GRAFANA_METRICS_ID GRAFANA_LOKI_URL GRAFANA_LOKI_ID GRAFANA_TOKEN GRAFANA_URL` on the Cortex server, restart, render one take, see it on the dashboard.
- [ ] `./agent/deploy.sh cloud_run` with `CORTEX_URL` set to the public Cortex; paste the service URL into Devpost.
- [ ] Record the demo (script above), upload unlisted to YouTube.
- [ ] Devpost: this file's sections, repo link (MIT), Grafana Labs track, video URL.
