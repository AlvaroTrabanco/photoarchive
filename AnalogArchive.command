#!/usr/bin/env bash
# Launches the Analog Archive viewer with full rebuild
# Double-clickable from Finder

# Always start in this script’s directory
cd "$(dirname "$0")"

# Force rebuild and serve
PORT=8000
URL="http://localhost:$PORT"

echo "Starting Analog Archive…"
echo "Opening $URL"

# Launch browser in the background (macOS-safe)
( sleep 2; open "$URL" ) >/dev/null 2>&1 &

# Run the main server (full rebuild)
exec ./serve.sh