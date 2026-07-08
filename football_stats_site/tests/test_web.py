import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.data import generate_season
from football_stats_site.tests.wsgi_client import get
from football_stats_site.web import CombinedApp


@pytest.fixture
def static_dir():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "index.html"), "w") as f:
            f.write("<html>frontend</html>")
        yield d


def test_api_paths_are_delegated_to_the_json_api(static_dir):
    app = CombinedApp(static_dir=static_dir)
    resp = get(app, "/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_non_api_paths_serve_the_static_frontend(static_dir):
    app = CombinedApp(static_dir=static_dir)
    resp = get(app, "/")
    assert resp.status_code == 200
    assert resp.body == b"<html>frontend</html>"


def test_frontend_can_reach_a_real_api_route_with_data(static_dir):
    from football_stats_site.app import create_app

    api = create_app(generate_season(cutoff_matchday=2))
    app = CombinedApp(api_app=api, static_dir=static_dir)
    resp = get(app, "/api/standings")
    assert resp.status_code == 200
    assert resp.json()["standings"]
