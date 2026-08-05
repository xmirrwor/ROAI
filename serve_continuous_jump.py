"""Serve the continuous-jump validation video through an SSH tunnel."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent


class ViewerHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/":
            self.path = "/continuous_jump_viewer.html"
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    handler = partial(ViewerHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
