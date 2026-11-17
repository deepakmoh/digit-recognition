"""Local web app: draw a number in the browser and see it recognized.

    python app.py            # then open http://127.0.0.1:8000
    python app.py --open     # also opens the page in your browser
    python app.py 8765       # use another port
"""
import base64
import io
import json
import sys
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from recognizer import Recognizer

STATIC = Path(__file__).with_name("static")
recognizer = Recognizer()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, b'{"app": "digit-recognizer"}', "application/json")
        if self.path not in ("/", "/index.html"):
            return self.send_error(404)
        body = (STATIC / "index.html").read_bytes()
        self._send(200, body, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/recognize":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            data_url = json.loads(self.rfile.read(length))["image"]
            png = base64.b64decode(data_url.split(",", 1)[1])
            result = recognizer.recognize(Image.open(io.BytesIO(png)))
            self._send(200, json.dumps(result).encode(), "application/json")
        except Exception as e:  # report bad input to the page instead of crashing
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def already_running(port: int) -> bool:
    """True if this app is already serving on the port (e.g. launched twice)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
            return b"digit-recognizer" in r.read()
    except Exception:
        return False


def main():
    args = sys.argv[1:]
    open_browser = "--open" in args
    ports = [int(a) for a in args if a.isdigit()] or list(range(8000, 8011))

    for port in ports:
        url = f"http://127.0.0.1:{port}"
        if already_running(port):
            print(f"Already running at {url}")
            if open_browser:
                webbrowser.open(url)
            return
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            continue  # port taken by something else; try the next one
        print(f"Digit recognizer running at {url}")
        print("Keep this window open while you use it. Close it (or press Ctrl+C) to stop.")
        if open_browser:
            threading.Timer(0.5, webbrowser.open, [url]).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return
    sys.exit(f"No free port among {ports[0]}-{ports[-1]}.")


if __name__ == "__main__":
    main()
