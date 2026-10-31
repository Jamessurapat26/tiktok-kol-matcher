#!/usr/bin/env bash
# Start the API on :8000 and the Vite dev server on :5173. Ctrl-C stops both.
# Usage: ./start.sh [--demo]
#   --demo  force demo mode (mocks only, no keys, no Apify cost)
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
  "") demo=0 ;;
  --demo) demo=1 ;;
  *)
    echo "Usage: ./start.sh [--demo]" >&2
    exit 2
    ;;
esac

for tool in uv pnpm; do
  if ! command -v "$tool" >/dev/null; then
    echo "$tool is not installed (see Local development in README.md)." >&2
    exit 1
  fi
done

# Vite would quietly move to another port, so fail here instead.
for port in 8000 5173; do
  holder="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1)"
  if [ -n "$holder" ]; then
    echo "Port $port is already in use by PID $holder:" >&2
    ps -o command= -p "$holder" >&2 || true
    echo "Press Ctrl-C in its terminal, or run: kill $holder" >&2
    exit 1
  fi
done

# The API does not load .env itself.
if [ -f "$ROOT/.env" ]; then
  set -a
  . "$ROOT/.env"
  set +a
fi
if [ "$demo" = 1 ]; then
  export DEMO_MODE=1
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  (cd "$ROOT/frontend" && pnpm install)
fi

# Each job gets its own process group, so stopping it also stops the
# servers that uv and pnpm spawn.
set -m
pids=""
cleanup() {
  trap - EXIT INT TERM
  for pid in $pids; do
    kill -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# uv run syncs the venv but keeps extras such as embed, unlike uv sync.
(cd "$ROOT/backend" && exec uv run uvicorn app.api.main:app --port 8000) &
pids="$pids $!"
(cd "$ROOT/frontend" && exec pnpm dev) &
pids="$pids $!"

echo "App: http://localhost:5173  API: http://localhost:8000  (Ctrl-C to stop)"

# If either server dies, stop the other one too.
while true; do
  for pid in $pids; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A server exited; stopping." >&2
      exit 1
    fi
  done
  sleep 1
done
