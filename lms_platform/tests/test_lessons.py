import io

from lms_platform.tests.conftest import auth_header, signup


def _instructor_with_course(client, title="Course"):
    token, _ = signup(client, "prof@example.com", "password123", "instructor")
    course = client.post("/api/courses", json={"title": title}, headers=auth_header(token)).get_json()
    return token, course


def test_create_text_lesson(client):
    token, course = _instructor_with_course(client)
    r = client.post(
        f"/api/courses/{course['id']}/lessons",
        json={"title": "L1", "content_type": "text", "content": "body text"},
        headers=auth_header(token),
    )
    assert r.status_code == 201
    body = r.get_json()
    assert body["content_type"] == "text"
    assert body["order_index"] == 1


def test_lesson_order_index_auto_increments(client):
    token, course = _instructor_with_course(client)
    l1 = client.post(f"/api/courses/{course['id']}/lessons",
                      json={"title": "L1", "content_type": "text", "content": "a"},
                      headers=auth_header(token)).get_json()
    l2 = client.post(f"/api/courses/{course['id']}/lessons",
                      json={"title": "L2", "content_type": "text", "content": "b"},
                      headers=auth_header(token)).get_json()
    assert l1["order_index"] == 1
    assert l2["order_index"] == 2


def test_create_lesson_rejects_duplicate_order_index(client):
    token, course = _instructor_with_course(client)
    client.post(f"/api/courses/{course['id']}/lessons",
                json={"title": "L1", "content_type": "text", "content": "a", "order_index": 5},
                headers=auth_header(token))
    r = client.post(f"/api/courses/{course['id']}/lessons",
                     json={"title": "L2", "content_type": "text", "content": "b", "order_index": 5},
                     headers=auth_header(token))
    assert r.status_code == 500 or r.status_code >= 400  # sqlite IntegrityError surfaces as a server error


def test_create_lesson_requires_content_for_text(client):
    token, course = _instructor_with_course(client)
    r = client.post(f"/api/courses/{course['id']}/lessons",
                     json={"title": "L1", "content_type": "text", "content": "  "},
                     headers=auth_header(token))
    assert r.status_code == 400


def test_create_lesson_cannot_set_content_type_file_directly(client):
    token, course = _instructor_with_course(client)
    r = client.post(f"/api/courses/{course['id']}/lessons",
                     json={"title": "L1", "content_type": "file", "content": "fake/path.pdf"},
                     headers=auth_header(token))
    assert r.status_code == 400


def test_non_owner_cannot_create_lesson(client):
    token1, course = _instructor_with_course(client, "Course A")
    token2, _ = signup(client, "other@example.com", "password123", "instructor")
    r = client.post(f"/api/courses/{course['id']}/lessons",
                     json={"title": "L1", "content_type": "text", "content": "a"},
                     headers=auth_header(token2))
    assert r.status_code == 404


def test_student_cannot_create_lesson(client):
    token, course = _instructor_with_course(client)
    stud_token, _ = signup(client, "stud@example.com", "password123", "student")
    r = client.post(f"/api/courses/{course['id']}/lessons",
                     json={"title": "L1", "content_type": "text", "content": "a"},
                     headers=auth_header(stud_token))
    assert r.status_code == 403


def test_list_lessons_ordered(client):
    token, course = _instructor_with_course(client)
    client.post(f"/api/courses/{course['id']}/lessons",
                json={"title": "Second", "content_type": "text", "content": "b", "order_index": 2},
                headers=auth_header(token))
    client.post(f"/api/courses/{course['id']}/lessons",
                json={"title": "First", "content_type": "text", "content": "a", "order_index": 1},
                headers=auth_header(token))
    r = client.get(f"/api/courses/{course['id']}/lessons", headers=auth_header(token))
    titles = [l["title"] for l in r.get_json()["lessons"]]
    assert titles == ["First", "Second"]


def test_update_lesson_title(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "Old", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.patch(f"/api/lessons/{lesson['id']}", json={"title": "New"}, headers=auth_header(token))
    assert r.status_code == 200
    assert r.get_json()["title"] == "New"


def test_delete_lesson(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "Doomed", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.delete(f"/api/lessons/{lesson['id']}", headers=auth_header(token))
    assert r.status_code == 200
    assert client.get(f"/api/lessons/{lesson['id']}", headers=auth_header(token)).status_code == 404


def test_upload_sets_content_type_file(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(b"%PDF-1.4 fake"), "notes.pdf")},
        content_type="multipart/form-data",
        headers=auth_header(token),
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["content_type"] == "file"
    assert body["content"].endswith("notes.pdf")


def test_upload_rejects_disallowed_extension(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(b"#!/bin/sh\necho hi"), "script.sh")},
        content_type="multipart/form-data",
        headers=auth_header(token),
    )
    assert r.status_code == 400


def test_upload_rejects_empty_file(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(b""), "empty.txt")},
        content_type="multipart/form-data",
        headers=auth_header(token),
    )
    assert r.status_code == 400


def test_upload_rejects_oversized_file(client):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    big = b"x" * (11 * 1024 * 1024)  # 11 MB, over the 10 MB per-file cap
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(big), "big.txt")},
        content_type="multipart/form-data",
        headers=auth_header(token),
    )
    assert r.status_code == 413


def test_upload_actually_writes_file_to_disk(client, app, upload_dir):
    token, course = _instructor_with_course(client)
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token)).get_json()
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(b"hello file bytes"), "notes.txt")},
        content_type="multipart/form-data",
        headers=auth_header(token),
    )
    stored_name = r.get_json()["content"]
    import os
    dest = os.path.join(upload_dir, stored_name)
    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == b"hello file bytes"


def test_non_owner_cannot_upload(client):
    token1, course = _instructor_with_course(client, "Course A")
    token2, _ = signup(client, "other@example.com", "password123", "instructor")
    lesson = client.post(f"/api/courses/{course['id']}/lessons",
                          json={"title": "L1", "content_type": "text", "content": "a"},
                          headers=auth_header(token1)).get_json()
    r = client.post(
        f"/api/lessons/{lesson['id']}/upload",
        data={"file": (io.BytesIO(b"data"), "notes.txt")},
        content_type="multipart/form-data",
        headers=auth_header(token2),
    )
    assert r.status_code == 404
