#!/usr/bin/env bash
set -euo pipefail

# --- Config (override with env vars if needed) ---
PORT_MAIN="${PORT_MAIN:-8000}"
PORT_FALLBACK="${PORT_FALLBACK:-8001}"
SITE_DIR="${SITE_DIR:-out/site}"
PY_SCRIPT="${PY_SCRIPT:-xmp_archive_harvester.py}"

cd "$(dirname "$0")"

# Sanity checks: reuse existing data
if [[ ! -d "$SITE_DIR" ]]; then
  echo "❌ $SITE_DIR doesn't exist. Run ./serve.sh once to build."
  exit 1
fi
if [[ ! -f "$SITE_DIR/catalog.json" ]]; then
  echo "❌ Missing $SITE_DIR/catalog.json. Run ./serve.sh to generate it."
  exit 1
fi

echo "== Rebuilding index.html only (reusing catalog & thumbs) =="

# Call Python script to regenerate just index.html from existing catalog
# Call Python to regenerate only index.html from your current template
python3 - <<'PY'
from pathlib import Path
import json
from xmp_archive_harvester import build_static_site, write_index_html  # adjust import to your path/module name

site = Path("out/site").resolve()
cat = site / "catalog.json"
if not cat.exists() or cat.stat().st_size == 0:
    raise SystemExit("No catalog.json to reuse. Run the full build once.")
# Do not rewrite catalog, just refresh index.html from current template:
write_index_html(site)
print("✔ Rewrote index.html (catalog & thumbs reused).")
PY

# Pick a free port
PORT="$PORT_MAIN"
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT in use; switching to $PORT_FALLBACK"
  PORT="$PORT_FALLBACK"
fi

URL="http://localhost:$PORT"
echo "== Quick serve (no re-parse) =="
echo "Serving $SITE_DIR at $URL"
( sleep 1; command -v open >/dev/null 2>&1 && open "$URL" ) || true

# Serve with no-cache headers so you always see fresh HTML
python3 - "$SITE_DIR" "$PORT" <<'PY'
import http.server, socketserver, sys, functools
root = sys.argv[1]
port = int(sys.argv[2])

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

Handler = functools.partial(NoCacheHandler, directory=root)

with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
    print(f"→ Serving {root} at http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
PY