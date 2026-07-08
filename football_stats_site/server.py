"""Dev server entry point: `python -m football_stats_site.server [port]`."""
from __future__ import annotations

import sys
from wsgiref.simple_server import make_server

from .app import create_app


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app = create_app()
    with make_server("127.0.0.1", port, app) as httpd:
        print(f"football-stats-site API serving on http://127.0.0.1:{port}")
        print("Try: /api/health /api/teams /api/standings /api/fixtures /api/results /api/matches")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
