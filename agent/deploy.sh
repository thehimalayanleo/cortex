#!/usr/bin/env bash
# Deploy the Director agent to Google Cloud. Two paths; both need `gcloud auth login` and a project with billing
# (the hackathon's $100 credits). Set the env vars below before running.
#
#   ./agent/deploy.sh cloud_run      # Cloud Run service (fast, public URL, the demo path)
#   ./agent/deploy.sh agent_engine   # Vertex AI Agent Engine (managed sessions)
#
# Required:  GOOGLE_CLOUD_PROJECT, CORTEX_URL (the public Cortex, e.g. https://cortex.aftersave.app)
# Gemini:    GOOGLE_API_KEY (AI Studio key)  or  GOOGLE_GENAI_USE_VERTEXAI=1 (uses the project's Vertex AI)
# Grafana:   GRAFANA_URL (https://<stack>.grafana.net) to enable the Grafana Cloud MCP tools in the agent
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-cloud_run}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
: "${CORTEX_URL:?set CORTEX_URL to the public Cortex server}"
ENV_VARS="CORTEX_URL=${CORTEX_URL}"
[ -n "${GOOGLE_API_KEY:-}" ] && ENV_VARS="${ENV_VARS},GOOGLE_API_KEY=${GOOGLE_API_KEY}"
[ -n "${GOOGLE_GENAI_USE_VERTEXAI:-}" ] && ENV_VARS="${ENV_VARS},GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION=${REGION}"
[ -n "${GRAFANA_URL:-}" ] && ENV_VARS="${ENV_VARS},GRAFANA_URL=${GRAFANA_URL}"
ADK="${ADK:-adk}"
command -v "$ADK" >/dev/null || ADK="uv run --python 3.11 --with google-adk --with litellm adk"
case "$MODE" in
  cloud_run)
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
    # adk builds the container with Cloud Build and deploys it; --with_ui keeps the dev UI for the demo
    $ADK deploy cloud_run --project "$GOOGLE_CLOUD_PROJECT" --region "$REGION" --service_name cortex-director --with_ui director
    gcloud run services update cortex-director --project "$GOOGLE_CLOUD_PROJECT" --region "$REGION" --update-env-vars "$ENV_VARS" --allow-unauthenticated
    gcloud run services describe cortex-director --project "$GOOGLE_CLOUD_PROJECT" --region "$REGION" --format 'value(status.url)'
    ;;
  agent_engine)
    : "${STAGING_BUCKET:?set STAGING_BUCKET=gs://<bucket> in the project}"
    gcloud services enable aiplatform.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
    $ADK deploy agent_engine --project "$GOOGLE_CLOUD_PROJECT" --region "$REGION" --staging_bucket "$STAGING_BUCKET" --display_name cortex-director director
    ;;
  *) echo "usage: $0 cloud_run|agent_engine"; exit 2;;
esac
