#!/usr/bin/env bash
# Create or update the Cortex Studio dashboard in a Grafana Cloud stack.
#   GRAFANA_URL=https://<stack>.grafana.net GRAFANA_SA_TOKEN=glsa_... ./grafana/push_dashboard.sh
# The token is a service-account token with the Editor role (Administration → Service accounts). Metrics arrive from
# the Cortex server's telemetry push (GRAFANA_METRICS_URL/ID + GRAFANA_TOKEN) as series cortex_run_<field>{run,recipe,executor,shot}.
set -euo pipefail
cd "$(dirname "$0")"
: "${GRAFANA_URL:?}" ; : "${GRAFANA_SA_TOKEN:?}"
curl -sS -X POST "${GRAFANA_URL%/}/api/dashboards/db" -H "Authorization: Bearer ${GRAFANA_SA_TOKEN}" -H "Content-Type: application/json" \
  --data-binary @cortex-studio-dashboard.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), (sys.argv[1].rstrip("/")+d["url"]) if "url" in d else d)' "${GRAFANA_URL}"
