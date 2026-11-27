# Knowledge Base API

A small Flask API for a knowledge base: articles with categories and
tags, full-text search over title and body, and related articles by
shared tags. In-memory store — nothing is written to disk, and each
`create_app()` call starts empty.

## Endpoints

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/articles` | Create an article (`title`, `body`, `category`, `tags?`); 409 on duplicate title |
| GET | `/articles` | List articles; `?category=` and `?tag=` filters |
| GET | `/articles/<id>` | Fetch one article |
| PUT | `/articles/<id>` | Update title / body / category / tags; 409 on duplicate title |
| DELETE | `/articles/<id>` | Delete an article |
| GET | `/search?q=...` | Full-text search over title and body; title matches rank first, each result has a `score` and a `snippet` |
| GET | `/articles/<id>/related` | Articles sharing tags, ranked by shared-tag count, each with `shared_tags` |
| GET | `/categories` | Categories with article counts |

Tags are stored lower-cased and de-duplicated. Search is
case-insensitive and matches whole words; a title hit counts double.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python knowledge_base_api/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_knowledge_base_api.py -q
```

## Example output

```
Created article #1: 'Getting started with Flask' (Guides, tags ['flask', 'python', 'web'])
Created article #2: 'Flask testing with pytest' (Guides, tags ['flask', 'testing', 'python'])
Created article #3: 'Deploying a Python API' (Deployment, tags ['python', 'deployment'])
Search 'flask': 2 result(s)
  #1 'Getting started with Flask' (score 3): Flask is a lightweight Python web framework. Install it with pip and create your...
  #2 'Flask testing with pytest' (score 3): Use Flask's test client to exercise your routes. The Flask test client makes it easy...
Related to 'Getting started with Flask': 2 article(s)
  #2 'Flask testing with pytest' (shared tags ['flask', 'python'])
  #3 'Deploying a Python API' (shared tags ['python'])
Categories: Deployment (1), Guides (2)
```
