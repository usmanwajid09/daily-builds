"""Tests for file_upload_api."""
import io

import pytest

from file_upload_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _upload(client, content=b"hello, daily-builds!\n", filename="greeting.txt"):
    return client.post(
        "/files",
        data={"file": (io.BytesIO(content), filename)},
        content_type="multipart/form-data",
    )


def test_upload_file(client):
    resp = _upload(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["filename"] == "greeting.txt"
    assert data["size"] == 21
    assert data["mimetype"] == "text/plain"
    assert "uploaded_at" in data
    assert "content" not in data


def test_list_files(client):
    _upload(client)
    _upload(client, content=b"a,b\n1,2\n", filename="data.csv")
    listed = client.get("/files").get_json()
    assert len(listed) == 2
    assert [f["filename"] for f in listed] == ["greeting.txt", "data.csv"]
    assert all("content" not in f for f in listed)


def test_download_file(client):
    file_id = _upload(client).get_json()["id"]
    resp = client.get(f"/files/{file_id}")
    assert resp.status_code == 200
    assert resp.data == b"hello, daily-builds!\n"
    assert "greeting.txt" in resp.headers["Content-Disposition"]


def test_download_missing_file_404(client):
    resp = client.get("/files/99")
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"]


def test_upload_rejects_bad_extension(client):
    resp = _upload(client, filename="malware.exe")
    assert resp.status_code == 400
    assert "extension not allowed" in resp.get_json()["error"]


def test_upload_rejects_no_extension(client):
    resp = _upload(client, filename="README")
    assert resp.status_code == 400


def test_upload_rejects_missing_file_part(client):
    resp = client.post("/files", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "file" in resp.get_json()["error"]


def test_upload_rejects_empty_filename(client):
    resp = client.post(
        "/files",
        data={"file": (io.BytesIO(b"x"), "")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_upload_rejects_oversized_file(client):
    resp = _upload(client, content=b"x" * (1024 * 1024 + 1), filename="big.txt")
    assert resp.status_code == 413
    assert "too large" in resp.get_json()["error"]


def test_upload_allows_each_allowed_extension(client):
    for filename in ("a.md", "b.csv", "c.pdf", "d.png", "e.jpg", "f.jpeg", "g.gif"):
        resp = _upload(client, content=b"data", filename=filename)
        assert resp.status_code == 201, filename
    assert len(client.get("/files").get_json()) == 7


def test_delete_file(client):
    file_id = _upload(client).get_json()["id"]
    resp = client.delete(f"/files/{file_id}")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] == file_id
    assert client.get("/files").get_json() == []
    assert client.get(f"/files/{file_id}").status_code == 404


def test_delete_missing_file_404(client):
    assert client.delete("/files/99").status_code == 404
