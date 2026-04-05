"""Simple HTTP server for nervx visualization."""

from __future__ import annotations

import http.server
import sys
import webbrowser
from functools import partial


def serve_viz(directory: str, port: int = 8741, open_browser: bool = True) -> None:
    """Start a local HTTP server to view the nervx visualization.

    Args:
        directory: Path to the directory containing index.html and nervx-viz.json.
        port: Port number (default 8741).
        open_browser: Whether to auto-open the browser.
    """
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    handler.extensions_map = {
        ".html": "text/html",
        ".js": "application/javascript",
        ".json": "application/json",
        ".css": "text/css",
        "": "application/octet-stream",
    }

    server = http.server.HTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}"

    print(f"nervx viz running at {url}", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        server.server_close()
