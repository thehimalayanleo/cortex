# The Director agent as a container: ADK API server on :8080 (Cloud Run's default port).
# Build:  docker build -t cortex-director agent/director
# Run:    docker run -p 8080:8080 -e GOOGLE_API_KEY=... -e CORTEX_URL=https://cortex.example -e GRAFANA_URL=https://<stack>.grafana.net cortex-director
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app/director
ENV PORT=8080 PYTHONUNBUFFERED=1
# `adk api_server` serves /run, /run_sse, /list-apps and the dev UI when --with_ui is given
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT} --with_ui /app"]
