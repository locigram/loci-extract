#!/usr/bin/env sh
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
TIMEOUT_KEEP_ALIVE="${TIMEOUT_KEEP_ALIVE:-30}"

exec uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WEB_CONCURRENCY" \
  --log-level "$LOG_LEVEL" \
  --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE"
