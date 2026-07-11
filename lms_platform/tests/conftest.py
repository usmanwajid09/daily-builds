import shutil
import tempfile

import pytest

from lms_platform import create_app


@pytest.fixture
def upload_dir():
    d = tempfile.mkdtemp(prefix="lms_uploads_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def app(upload_dir):
    return create_app(db_path=":memory:", jwt_secret="test-secret", upload_dir=upload_dir)


@pytest.fixture
def client(app):
    return app.test_client()


def signup(client, email, password, role):
    r = client.post("/api/signup", json={"email": email, "password": password, "role": role})
    assert r.status_code == 201, r.get_json()
    return r.get_json()["token"], r.get_json()["user"]["id"]


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}
