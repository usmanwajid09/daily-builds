"""Tests for the knowledge_base_api Flask app, using Flask's test client."""
import pytest

from knowledge_base_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _article(client, title="Getting started with Flask", **extra):
    payload = {
        "title": title,
        "body": "Flask is a lightweight Python web framework.",
        "category": "Guides",
        "tags": ["flask", "python"],
    }
    payload.update(extra)
    return client.post("/articles", json=payload)


@pytest.fixture
def article(client):
    return _article(client).get_json()["id"]


def _seed(client):
    _article(client, title="Flask testing with pytest",
             body="Use Flask's test client to exercise routes.",
             category="Guides", tags=["flask", "testing", "python"])
    _article(client, title="Deploying a Python API",
             body="Ship your Python API with gunicorn.",
             category="Deployment", tags=["python", "deployment"])
    return client


def test_list_articles_starts_empty(client):
    assert client.get("/articles").get_json() == []


def test_create_article_returns_201(client):
    resp = _article(client, tags=["Flask", "PYTHON", "flask"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    # Tags are lower-cased and de-duplicated, in first-seen order.
    assert body["tags"] == ["flask", "python"]


def test_create_article_validation_400(client):
    assert _article(client, title="").status_code == 400
    assert _article(client, body="  ").status_code == 400
    assert _article(client, category=42).status_code == 400
    assert _article(client, tags="flask").status_code == 400
    assert _article(client, tags=["ok", ""]).status_code == 400


def test_create_article_duplicate_title_409(client, article):
    assert _article(client, title="GETTING STARTED WITH FLASK").status_code == 409


def test_get_article_404(client):
    assert client.get("/articles/99").status_code == 404


def test_update_article(client, article):
    resp = client.put(
        f"/articles/{article}",
        json={"category": "Tutorials", "tags": ["flask", "beginner"]},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["category"] == "Tutorials"
    assert body["tags"] == ["flask", "beginner"]
    assert client.put(f"/articles/{article}", json={}).status_code == 400


def test_update_article_duplicate_title_409(client, article):
    _article(client, title="Another title")
    assert client.put(
        f"/articles/{article}", json={"title": "another title"}
    ).status_code == 409


def test_delete_article(client, article):
    assert client.delete(f"/articles/{article}").status_code == 200
    assert client.get(f"/articles/{article}").status_code == 404


def test_list_filters(client, article):
    _seed(client)
    assert len(client.get("/articles").get_json()) == 3
    guides = client.get("/articles?category=guides").get_json()
    assert len(guides) == 2
    flask_tag = client.get("/articles?tag=FLASK").get_json()
    assert len(flask_tag) == 2
    assert client.get("/articles?tag=deployment").get_json()[0]["category"] == "Deployment"


def test_search_title_matches_first(client, article):
    _seed(client)
    results = client.get("/search?q=flask").get_json()
    assert [r["id"] for r in results] == [1, 2]
    assert results[0]["score"] >= results[1]["score"]
    assert "flask" in results[0]["snippet"].lower()


def test_search_body_only_match(client, article):
    _seed(client)
    results = client.get("/search?q=gunicorn").get_json()
    assert [r["title"] for r in results] == ["Deploying a Python API"]
    assert "gunicorn" in results[0]["snippet"].lower()


def test_search_no_match_and_missing_query(client, article):
    assert client.get("/search?q=zookeeper").get_json() == []
    assert client.get("/search").status_code == 400
    assert client.get("/search?q=   ").status_code == 400


def test_related_articles(client, article):
    _seed(client)
    related = client.get(f"/articles/{article}/related").get_json()
    # Article 2 shares flask+python (2 tags), article 3 shares python (1).
    assert [r["id"] for r in related] == [2, 3]
    assert related[0]["shared_tags"] == ["flask", "python"]
    assert related[1]["shared_tags"] == ["python"]


def test_related_none_when_no_shared_tags(client, article):
    _article(client, title="Unrelated", tags=["cooking"])
    assert client.get(f"/articles/{article}/related").get_json() == []


def test_related_404(client):
    assert client.get("/articles/99/related").status_code == 404


def test_categories(client, article):
    _seed(client)
    categories = client.get("/categories").get_json()
    assert categories == [
        {"category": "Deployment", "article_count": 1},
        {"category": "Guides", "article_count": 2},
    ]
