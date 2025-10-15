#!/usr/bin/env python3
import os, sys, urllib.parse, subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # serves this folder (index.html, catalog.json, thumbs)
PHOTOS_ROOT = Path(os.environ.get("PHOTOS_ROOT", "/Users/alvarotrabanco/Photography/les_mios_semeyes")).resolve()

class Handler(SimpleHTTPRequestHandler):


    def end_headers(self):
    # disable caching for EVERYTHING we serve
    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    self.send_header("Pragma", "no-cache")
    self.send_header("Expires", "0")
    super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/open" or parsed.path == "/reveal":
            qs = urllib.parse.parse_qs(parsed.query)
            raw = (qs.get("path") or [""])[0]
            if not raw:
                self.send_error(400, "missing ?path=")
                return
            # Normalize & guard: path must be inside PHOTOS_ROOT
            p = Path(urllib.parse.unquote(raw)).resolve()
            try:
                p.relative_to(PHOTOS_ROOT)
            except ValueError:
                self.send_error(403, "path outside PHOTOS_ROOT")
                return

            try:
                if parsed.path == "/open":
                    # open Lightroom Classic pointed at the containing folder
                    target = p if p.is_dir() else p.parent
                    subprocess.run(
                        ["open", "-a", "Adobe Lightroom Classic", str(target)],
                        check=False
                    )
                else:  # /reveal
                    # reveal the exact file in Finder
                    subprocess.run(["open", "-R", str(p)], check=False)

                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                self.send_error(500, f"{e}")
            return

        # Static files
        super().do_GET()

    def translate_path(self, path):
        # Normal static file handling from ROOT
        path = urllib.parse.urlsplit(path).path
        return str((ROOT / path.lstrip("/")).resolve())
        
if __name__ == "__main__":
    os.chdir(ROOT)
    port = int(os.environ.get("PORT", "8000"))
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Serving {ROOT} on http://localhost:{port}")
        print(f"Photos root: {PHOTOS_ROOT}")
        httpd.serve_forever()