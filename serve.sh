#!/usr/bin/env bash
set -euo pipefail

# --- config ---
PORT_MAIN=8000
PORT_FALLBACK=8001
ROOT_PHOTOS="/Users/alvarotrabanco/Photography/les_mios_semeyes"
OUT_DIR="out"
SITE_DIR="$OUT_DIR/site"
CSV_PATH="$OUT_DIR/archive.csv"
JSON_PATH="$SITE_DIR/catalog.json"   # catalog.json will be inside site
THUMBS_DIR="$SITE_DIR/thumbs"

# Control parsing parallelism:
JOBS="${JOBS:-0}"

# Control whether index.html is rewritten by the Python build:
#   force = always rewrite
#   auto  = write only if missing (default)
#   skip  = never write index.html
HTML_MODE="${HTML_MODE:-force}"

# ⭐ Save absolute path to the project directory (where serve.sh lives)
SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd -P)"

# Move to this script's directory (project root)
cd "$SCRIPT_DIR"

echo "== Rebuilding catalog =="
echo "   Using jobs: $JOBS (0 = auto)"
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

# cd into the site so relative links (catalog.json, thumbs/) work
cd "$SITE_DIR"

# pick a free port
PORT="$PORT_MAIN"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT in use; switching to $PORT_FALLBACK"
  PORT="$PORT_FALLBACK"
fi

URL="http://localhost:$PORT"
echo "== Serving $PWD at $URL =="
( sleep 1; open "$URL" ) >/dev/null 2>&1 &

# ⭐ Pass PORT and run the server from the project root (not from out/site)
export PORT
python3 "$SCRIPT_DIR/serve_with_actions.py"