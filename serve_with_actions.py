#!/usr/bin/env python3
import os
import urllib.parse
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    """Serve static files plus two extra routes: /open and /reveal"""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/open":
            return self.handle_open(parsed)
        if parsed.path == "/reveal":
            return self.handle_reveal(parsed)
        return super().do_GET()

    def _ok(self, msg="OK"):
        data = msg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bad(self, code=400, msg="Bad Request"):
        data = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_open(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        path = qs.get("path", [""])[0]
        if not path:
            return self._bad(400, "Missing path")
        # Try Lightroom Classic; fallback to default opener
        try:
            subprocess.run(["open", "-a", "Adobe Lightroom Classic", path], check=False)
        except Exception:
            subprocess.run(["open", path], check=False)
        return self._ok("opened")

    def handle_reveal(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        path = qs.get("path", [""])[0]
        if not path:
            return self._bad(400, "Missing path")
        # Use macOS reveal (no AppleScript, no f-strings weirdness)
        subprocess.run(["open", "-R", path], check=False)
        return self._ok("revealed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    httpd = ThreadingHTTPServer(("localhost", port), Handler)
    sa = httpd.socket.getsockname()
    print(f"Serving on http://{sa[0]}:{sa[1]} (cwd={os.getcwd()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass