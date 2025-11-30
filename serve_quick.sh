#!/usr/bin/env bash
set -euo pipefail

PORT_MAIN="${PORT_MAIN:-8000}"
PORT_FALLBACK="${PORT_FALLBACK:-8001}"
ROOT_PHOTOS="/Users/alvarotrabanco/Photography/les_mios_semeyes"
SITE_DIR="${SITE_DIR:-out/site}"

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd -P)"
cd "$SCRIPT_DIR"

if [[ ! -d "$SITE_DIR" ]]; then
  echo "❌ $SITE_DIR doesn't exist. Run ./serve.sh once to build."
  exit 1
fi
if [[ ! -f "$SITE_DIR/catalog.json" ]]; then
  echo "❌ Missing $SITE_DIR/catalog.json. Run ./serve.sh."
  exit 1
fi

python3 - <<'PY'
from pathlib import Path
from xmp_archive_harvester import write_index_html
site = Path("out/site").resolve()
write_index_html(site)
print("✔ Rewrote index.html (catalog & thumbs reused).")
PY

PORT="$PORT_MAIN"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT in use; switching to $PORT_FALLBACK"
  PORT="$PORT_FALLBACK"
fi

export PORT ROOT_PHOTOS SITE_DIR
URL="http://localhost:$PORT"
( sleep 1; command -v open >/dev/null 2>&1 && open "$URL" ) || true

echo "== Quick serve =="
python3 "$SCRIPT_DIR/serve_with_actions.py"