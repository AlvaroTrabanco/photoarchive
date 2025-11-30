#!/usr/bin/env bash
set -euo pipefail

PORT_MAIN=8000
PORT_FALLBACK=8001
ROOT_PHOTOS="/Users/alvarotrabanco/Photography/les_mios_semeyes"
OUT_DIR="out"
SITE_DIR="$OUT_DIR/site"
CSV_PATH="$OUT_DIR/archive.csv"
JSON_PATH="$SITE_DIR/catalog.json"
THUMBS_DIR="$SITE_DIR/thumbs"
JOBS="${JOBS:-0}"
HTML_MODE="${HTML_MODE:-force}"

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd -P)"
cd "$SCRIPT_DIR"

echo "== Rebuilding catalog =="
python3 xmp_archive_harvester.py \
  --root "$ROOT_PHOTOS" \
  --csv "$CSV_PATH" \
  --json "$JSON_PATH" \
  --site "$SITE_DIR" \
  --thumbs "$THUMBS_DIR" \
  --thumb-size 512 \
  --select auto \
  --log INFO \
  --jobs "$JOBS" \
  --html "$HTML_MODE"

echo "== Build OK =="
ls -l "$SITE_DIR/index.html" "$SITE_DIR/catalog.json" || true

# Pick a free port
PORT="$PORT_MAIN"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT in use; switching to $PORT_FALLBACK"
  PORT="$PORT_FALLBACK"
fi

export PORT ROOT_PHOTOS SITE_DIR
URL="http://localhost:$PORT"
( sleep 1; command -v open >/dev/null 2>&1 && open "$URL" ) || true

echo "== Serving $SITE_DIR at $URL =="
python3 "$SCRIPT_DIR/serve_with_actions.py"