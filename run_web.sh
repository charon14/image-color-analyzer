#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "$SCRIPT_DIR"
exec python3 color_analysis.py --web --host "$HOST" --port "$PORT"
