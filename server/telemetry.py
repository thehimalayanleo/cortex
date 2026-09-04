"""Telemetry to Grafana Cloud: every METRIC line a run emits becomes a metric, every log line becomes a Loki entry.

Configure with environment variables (all optional; nothing is sent when they are absent):
  GRAFANA_METRICS_URL   e.g. https://prometheus-prod-XX-prod-us-central-0.grafana.net/api/v1/push/influx/write
  GRAFANA_METRICS_ID    the Prometheus/Mimir instance id (basic-auth user)
  GRAFANA_LOKI_URL      e.g. https://logs-prod-XXX.grafana.net/loki/api/v1/push
  GRAFANA_LOKI_ID       the Loki instance id
  GRAFANA_TOKEN         a Cloud Access Policy token with metrics:write and logs:write
  GRAFANA_URL           https://<stack>.grafana.net (used by the agents' Grafana MCP connection and for links)

Pushes are batched in a background thread so a slow network never slows a run.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
import urllib.request
from typing import Any

_q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=10000)
_started = False
_lock = threading.Lock()
_stats = {"metrics_sent": 0, "logs_sent": 0, "errors": 0, "last_error": None}


def config() -> dict[str, Any]:
    e = os.environ.get
    return {
        "metrics_url": e("GRAFANA_METRICS_URL", ""), "metrics_id": e("GRAFANA_METRICS_ID", ""),
        "loki_url": e("GRAFANA_LOKI_URL", ""), "loki_id": e("GRAFANA_LOKI_ID", ""),
        "token": e("GRAFANA_TOKEN", ""), "grafana_url": e("GRAFANA_URL", ""),
    }


def enabled() -> dict[str, bool]:
    c = config()
    return {"metrics": bool(c["metrics_url"] and c["metrics_id"] and c["token"]), "logs": bool(c["loki_url"] and c["loki_id"] and c["token"]), "mcp": bool(c["grafana_url"])}


def status() -> dict[str, Any]:
    return {**enabled(), "grafana_url": config()["grafana_url"] or None, "queued": _q.qsize(), **_stats}


def _ensure_worker() -> None:
    global _started
    with _lock:
        if _started:
            return
        _started = True
        threading.Thread(target=_worker, daemon=True, name="grafana-push").start()


def _post(url: str, user: str, token: str, body: bytes, content_type: str) -> None:
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode())
    with urllib.request.urlopen(req, timeout=15) as r:
        r.read()


def _worker() -> None:
    while True:
        kind, item = _q.get()
        batch = [(kind, item)]
        t_end = time.time() + 1.0
        while time.time() < t_end and len(batch) < 200:
            try:
                batch.append(_q.get(timeout=0.2))
            except queue.Empty:
                break
        c = config()
        lines = [i for k, i in batch if k == "metric"]
        logs = [i for k, i in batch if k == "log"]
        try:
            if lines and enabled()["metrics"]:
                _post(c["metrics_url"], c["metrics_id"], c["token"], "\n".join(lines).encode(), "text/plain")
                _stats["metrics_sent"] += len(lines)
            if logs and enabled()["logs"]:
                streams: dict[str, list] = {}
                for lab, ts, line in logs:
                    streams.setdefault(json.dumps(lab, sort_keys=True), []).append([str(ts), line])
                payload = {"streams": [{"stream": json.loads(k), "values": v} for k, v in streams.items()]}
                _post(c["loki_url"], c["loki_id"], c["token"], json.dumps(payload).encode(), "application/json")
                _stats["logs_sent"] += len(logs)
        except Exception as e:
            _stats["errors"] += 1
            _stats["last_error"] = str(e)[:200]


_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _tag(v: Any) -> str:
    return _TAG_RE.sub("_", str(v))[:60] or "none"


def push_metric(run: dict[str, Any], row: dict[str, Any]) -> None:
    """One METRIC row -> Influx line protocol: cortex_run,run=…,recipe=…,executor=…,shot=… loss=…,reward=… ts_ns."""
    if not enabled()["metrics"]:
        return
    tags = {"run": run.get("id", ""), "recipe": run.get("recipe", ""), "executor": run.get("executor", ""), "origin": run.get("origin", "ui")}
    for k in ("shot", "phase", "opt"):
        if k in row and not isinstance(row[k], (int, float)):
            tags[k] = row[k]
    fields = {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "t"}
    if not fields:
        return
    ts = int(float(row.get("t", time.time())) * 1e9)
    line = "cortex_run," + ",".join(f"{k}={_tag(v)}" for k, v in tags.items()) + " " + ",".join(f"{k}={v}" for k, v in fields.items()) + f" {ts}"
    _ensure_worker()
    try:
        _q.put_nowait(("metric", line))
    except queue.Full:
        pass


def push_log(run: dict[str, Any], line: str) -> None:
    if not enabled()["logs"]:
        return
    labels = {"app": "cortex", "run": run.get("id", ""), "recipe": run.get("recipe", ""), "executor": run.get("executor", "")}
    _ensure_worker()
    try:
        _q.put_nowait(("log", (labels, int(time.time() * 1e9), line[:4000])))
    except queue.Full:
        pass
