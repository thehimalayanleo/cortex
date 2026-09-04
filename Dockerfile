# Cortex: build the web app, then serve it with the FastAPI server. Demo mode seeds a public sample vault.
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/pnpm-lock.yaml* ./
RUN corepack enable && corepack prepare pnpm@9 --activate && pnpm install --frozen-lockfile || pnpm install
COPY web/ ./
RUN pnpm build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/ server/
COPY scripts/ scripts/
COPY lab/ lab/
COPY --from=web /web/dist web/dist
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" openai pypdf pyyaml python-multipart
ENV CORTEX_VAULT=/data/cortex CORTEX_DEMO=1 PORT=8788
EXPOSE 8788
CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
