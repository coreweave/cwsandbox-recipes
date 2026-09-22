"""Tiny loopback-only HTTP service for the reverse and round-trip recipes."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import urlopen


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status = 200
        if self.path == "/":
            body = b"hello from sandbox\n"
        elif self.path == "/roundtrip":
            try:
                with urlopen("http://127.0.0.1:18080/", timeout=10) as response:
                    body = b"hello from sandbox\nupstream: " + response.read()
            except (URLError, TimeoutError):
                status, body = 502, b"CKS upstream is unavailable\n"
        else:
            status, body = 404, b"not found\n"
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8081), Handler).serve_forever()
