"""Dev server entry point: `python -m football_stats_site.server [port]`.

Serves both the JSON API (/api/*) and the static frontend (everything
else) from a single process via CombinedApp -- see web.py.
"""
from __future__ import annotations

import sys
from wsgiref.simple_server import make_server

from .web import create_combined_app


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app = create_combined_app()
    with make_server("127.0.0.1", port, app) as httpd:
        print(f"football-stats-site serving on http://127.0.0.1:{port}")
        print(f"  Frontend: http://127.0.0.1:{port}/")
        print("  API:      /api/health /api/teams /api/standings /api/fixtures")
        print("            /api/results /api/matches /api/live /api/top-scorers")
        print("            /api/players /api/search /api/teams/<name>")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
