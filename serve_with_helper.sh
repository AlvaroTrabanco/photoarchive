#!/usr/bin/env bash
set -e
# Start the helper in the background
python3 filehelper.py &
HELPER_PID=$!
# Run the normal server
./serve.sh
# When you stop serve.sh, kill the helper too
kill $HELPER_PID 2>/dev/null || true