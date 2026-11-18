"""Tests for the blog_api Flask app, using Flask's test client."""
import pytest

from blog_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _post(client, title="Hello World", content="Body text", **extra):
    payload = {"title": title, "content": content}
    payload.update(extra)
    return client.post("/posts", json=payload)


def test_list_starts_empty(client):
    assert client.get("/posts").get_json() == []


def test_create_post_returns_201_and_slug(client):
    resp = _post(client, "Hello, Flask!", "First post", tags=["Flask", "Python"])
    assert resp.status_code == 201
    assert resp.get_json() == {
        "slug": "hello-flask",
        "title": "Hello, Flask!",
        "content": "First post",
        "tags": ["flask", "python"],
        "status": "draft",
    }


def test_create_defaults_to_draft(client):
    post = _post(client).get_json()
    assert post["status"] == "draft"


def test_create_with_published_status(client):
    post = _post(client, status="published").get_json()
    assert post["status"] == "published"


def test_create_validation_errors(client):
    assert _post(client, "", "body").status_code == 400
    assert _post(client, "  ", "body").status_code == 400
    assert _post(client, "Title", "").status_code == 400
    assert _post(client, "Title", 42).status_code == 400
    assert _post(client, "Title", "body", status="live").status_code == 400
    assert _post(client, "Title", "body", tags="python").status_code == 400
    assert _post(client, "Title", "body", tags=["ok", ""]).status_code == 400
    assert _post(client, "!!!", "no letters in title").status_code == 400


def test_duplicate_titles_get_unique_slugs(client):
    first = _post(client, "Same Title", "one").get_json()
    second = _post(client, "Same Title", "two").get_json()
    third = _post(client, "Same Title", "three").get_json()
    assert (first["slug"], second["slug"], third["slug"]) == (
        "same-title", "same-title-2", "same-title-3",
    )


def test_slug_normalisation(client):
    post = _post(client, "  A  --  B & C!!  ", "body").get_json()
    assert post["slug"] == "a-b-c"


def test_get_post_and_404(client):
    _post(client, "Hello World", "body")
    assert client.get("/posts/hello-world").get_json()["title"] == "Hello World"
    assert client.get("/posts/no-such-post").status_code == 404


def test_status_filter_and_bad_status(client):
    _post(client, "Draft One", "x")
    _post(client, "Live One", "x", status="published")
    assert [p["slug"] for p in client.get(
        "/posts", query_string={"status": "published"}).get_json()] == ["live-one"]
    assert [p["slug"] for p in client.get(
        "/posts", query_string={"status": "draft"}).get_json()] == ["draft-one"]
    assert client.get("/posts", query_string={"status": "bogus"}).status_code == 400


def test_tag_filter_is_case_insensitive(client):
    _post(client, "A", "x", tags=["Python"])
    _post(client, "B", "x", tags=["Flask"])
    assert [p["slug"] for p in client.get(
        "/posts", query_string={"tag": "PYTHON"}).get_json()] == ["a"]
    assert client.get("/posts", query_string={"tag": "missing"}).get_json() == []


def test_search_matches_title_or_content(client):
    _post(client, "Flask Tips", "use create_app")
    _post(client, "Other", "mentions PYTEST here")
    assert [p["slug"] for p in client.get(
        "/posts", query_string={"search": "flask"}).get_json()] == ["flask-tips"]
    assert [p["slug"] for p in client.get(
        "/posts", query_string={"search": "pytest"}).get_json()] == ["other"]
    assert client.get("/posts", query_string={"search": "zzz"}).get_json() == []


def test_publish_and_unpublish(client):
    _post(client, "Hello World", "body")
    published = client.post("/posts/hello-world/publish").get_json()
    assert published["status"] == "published"
    draft = client.post("/posts/hello-world/unpublish").get_json()
    assert draft["status"] == "draft"
    assert client.post("/posts/nope/publish").status_code == 404
    assert client.post("/posts/nope/unpublish").status_code == 404


def test_update_post(client):
    _post(client, "Old Title", "old body", tags=["a"])
    updated = client.put(
        "/posts/old-title",
        json={"title": "New Title", "content": "new body", "tags": ["b"],
              "status": "published"},
    ).get_json()
    assert updated["title"] == "New Title"
    assert updated["content"] == "new body"
    assert updated["tags"] == ["b"]
    assert updated["status"] == "published"
    # Slug is stable even though the title changed.
    assert updated["slug"] == "old-title"
    assert client.get("/posts/old-title").get_json()["title"] == "New Title"


def test_update_validation(client):
    _post(client, "Hello World", "body")
    assert client.put("/posts/hello-world", json={}).status_code == 400
    assert client.put("/posts/hello-world", json={"title": " "}).status_code == 400
    assert client.put("/posts/hello-world", json={"status": "live"}).status_code == 400
    assert client.put("/posts/nope", json={"title": "x"}).status_code == 404


def test_delete_post(client):
    _post(client, "Hello World", "body")
    assert client.delete("/posts/hello-world").get_json() == {
        "deleted": "hello-world"
    }
    assert client.get("/posts/hello-world").status_code == 404
    assert client.delete("/posts/hello-world").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
