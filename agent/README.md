# Cortex agents

`director/` is a Google ADK agent (Gemini) that runs the Cortex Studio: plan shots from a logline, render takes on the
GPU box through Cortex, read the critics, keep or reshoot, and report through Grafana (the Grafana Cloud MCP).

## Run locally

```bash
pip install google-adk
export GOOGLE_API_KEY=...                 # Gemini API key; or GOOGLE_GENAI_USE_VERTEXAI=1 GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=us-central1
export CORTEX_URL=http://127.0.0.1:8788   # a running Cortex (./run.sh)
export GRAFANA_URL=https://<stack>.grafana.net   # optional: adds the Grafana Cloud MCP tools
cd cortex/agent
adk web                                   # open the URL it prints, pick "director"
```

Try: "Read the board, plan four shots for 'a kaiju of black coral rises from a storm and hesitates', render the first with the smoke brick, and tell me the verdict."

## Without a Google key (development)

Unset `GOOGLE_API_KEY` and the director runs on Cortex's own provider (OpenCode Go, glm-5.3) through LiteLLM:
`pip install google-adk litellm`, then `adk web` as above. Every tool works the same; only the model behind the
planning changes. For the hackathon submission the model must be Gemini: set the key (free at aistudio.google.com,
no card) or the Vertex variables, and the agent picks Gemini automatically.

## Deploy

Vertex AI Agent Engine: `adk deploy agent_engine --project <id> --region us-central1 --staging_bucket gs://<bucket> director`.
Cloud Run: `adk deploy cloud_run --project <id> --region us-central1 director`. Set the same env vars on the service.

## Telemetry (Grafana Labs track)

Cortex pushes every take's METRIC lines to Grafana Cloud when these are set on the Cortex server:
`GRAFANA_METRICS_URL`, `GRAFANA_METRICS_ID`, `GRAFANA_LOKI_URL`, `GRAFANA_LOKI_ID`, `GRAFANA_TOKEN`, `GRAFANA_URL`.
Series: `cortex_run{recipe="cinema_render", run=..., shot=...}` with fields identity, flicker, identity_mean, identity_min, flicker_mean, gen_s.
