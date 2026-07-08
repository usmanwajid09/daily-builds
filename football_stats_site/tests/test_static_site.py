import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from football_stats_site.static_site import StaticFileApp
from football_stats_site.tests.wsgi_client import get, post


@pytest.fixture
def static_dir():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "index.html"), "w") as f:
            f.write("<html>home</html>")
        with open(os.path.join(d, "styles.css"), "w") as f:
            f.write("body { color: red; }")
        os.mkdir(os.path.join(d, "sub"))
        with open(os.path.join(d, "sub", "index.html"), "w") as f:
            f.write("<html>sub</html>")
        yield d


def test_serves_index_at_root(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/")
    assert resp.status_code == 200
    assert resp.body == b"<html>home</html>"
    assert resp.headers["Content-Type"].startswith("text/html")


def test_serves_named_asset_with_correct_content_type(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/styles.css")
    assert resp.status_code == 200
    assert resp.body == b"body { color: red; }"
    assert resp.headers["Content-Type"].startswith("text/css")


def test_directory_request_serves_its_index(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/sub/")
    assert resp.status_code == 200
    assert resp.body == b"<html>sub</html>"


def test_missing_asset_with_extension_is_404(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/missing.js")
    assert resp.status_code == 404


def test_unknown_extensionless_path_falls_back_to_index_spa_style(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/team/River%20Athletic")
    assert resp.status_code == 200
    assert resp.body == b"<html>home</html>"


def test_path_traversal_is_blocked(static_dir):
    app = StaticFileApp(static_dir)
    resp = get(app, "/../../../etc/passwd")
    # Either blocked outright (falls back to SPA index since no dotted
    # extension survives normalization) or 404 -- never actually reads a
    # file outside static_dir.
    assert resp.status_code in (200, 404)
    assert b"root:" not in resp.body


def test_write_method_is_405(static_dir):
    app = StaticFileApp(static_dir)
    resp = post(app, "/")
    assert resp.status_code == 405
