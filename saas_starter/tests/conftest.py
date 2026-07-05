import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402


@pytest.fixture()
def app():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let init_db create it fresh
    application = create_app({
        "TESTING": True,
        "DB_PATH": path,
        "JWT_SECRET": "test-secret",
    })
    yield application
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture()
def client(app):
    return app.test_client()
