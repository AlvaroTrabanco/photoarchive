#!/usr/bin/env python3
from __future__ import annotations
from flask import Flask, request, jsonify
from pathlib import Path
import subprocess, os

# CHANGE THIS to your archive root to avoid arbitrary file access:
ALLOWED_ROOT = Path("/Users/alvarotrabanco/Photography").resolve()

app = Flask(__name__)

def _safe(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not str(p).startswith(str(ALLOWED_ROOT)):
        raise ValueError("Path outside allowed root")
    return p

@app.post("/api/open")
def open_file():
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path", "")
    try:
        p = _safe(path)
        if not p.exists():
            return jsonify(ok=False, error="Path not found"), 404
        # open the file with default app
        subprocess.run(["open", p.as_posix()], check=False)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400

@app.post("/api/reveal")
def reveal_in_finder():
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path", "")
    try:
        p = _safe(path)
        # -R reveals the file in Finder (or the folder if a dir)
        target = p if p.is_file() else p
        subprocess.run(["open", "-R", target.as_posix()], check=False)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400

# optional: basic ping
@app.get("/api/ping")
def ping():
    return jsonify(ok=True)

if __name__ == "__main__":
    # only bind to localhost
    app.run(host="127.0.0.1", port=8787, debug=False)