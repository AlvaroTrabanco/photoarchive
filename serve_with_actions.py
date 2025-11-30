#!/usr/bin/env python3
import http.server, socketserver, urllib.parse, json, os, sys, subprocess, pathlib, functools

ROOT = os.environ.get("ROOT_PHOTOS") or ""
SITE_DIR = os.environ.get("SITE_DIR") or os.getcwd()
PORT = int(os.environ.get("PORT") or "8000")

ROOT_PATH = pathlib.Path(ROOT).resolve() if ROOT else None

def _ok_headers(handler):
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")

def _bad_request(handler, msg="bad request"):
    handler.send_response(400)
    _ok_headers(handler)
    handler.end_headers()
    handler.wfile.write(json.dumps({"ok": False, "error": msg}).encode("utf-8"))

def _json_ok(handler, payload):
    handler.send_response(200)
    _ok_headers(handler)
    handler.end_headers()
    handler.wfile.write(json.dumps({"ok": True, **payload}).encode("utf-8"))

def _inside_root(p: pathlib.Path) -> bool:
    if not ROOT_PATH:
        return True
    try:
        return ROOT_PATH in p.resolve().parents or p.resolve() == ROOT_PATH
    except Exception:
        return False

def do_reveal(abs_path: str):
    p = pathlib.Path(abs_path).expanduser().resolve()
    if not _inside_root(p):
        raise RuntimeError("Path outside ROOT_PHOTOS")
    if not p.exists():
        raise FileNotFoundError(str(p))
    # Reveal file (or enclosing folder if you passed a folder)
    # macOS: use AppleScript reveal for best UX
    script = f'''
    tell application "Finder"
        activate
        try
            reveal POSIX file "{p}"
        on error
            reveal POSIX file "{p.parent}"
        end try
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=False)

def do_open(abs_path: str):
    p = pathlib.Path(abs_path).expanduser().resolve()
    if not _inside_root(p):
        raise RuntimeError("Path outside ROOT_PHOTOS")
    if not p.exists():
        raise FileNotFoundError(str(p))
    subprocess.run(["open", p.as_posix()], check=False)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # add no-cache headers on all responses
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self):
        # We only support /action endpoints
        if self.path.split("?")[0] != "/action":
            return super().do_POST()

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return _bad_request(self, "invalid JSON")

        cmd = (payload.get("cmd") or "").lower()
        path = payload.get("path") or ""
        if not cmd or not path:
            return _bad_request(self, "missing cmd or path")

        try:
            if cmd == "reveal":
                do_reveal(path)
                return _json_ok(self, {"action": "reveal", "path": path})
            elif cmd == "open":
                do_open(path)
                return _json_ok(self, {"action": "open", "path": path})
            else:
                return _bad_request(self, "unknown cmd")
        except FileNotFoundError:
            return _bad_request(self, f"not found: {path}")
        except RuntimeError as e:
            return _bad_request(self, str(e))
        except Exception as e:
            return _bad_request(self, f"failed: {e}")

# Serve files out of SITE_DIR
Handler = functools.partial(Handler, directory=SITE_DIR)

with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"→ Serving {SITE_DIR} at http://127.0.0.1:{PORT}")
    print(f"   Root for actions: {ROOT_PATH or '(no root check)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass