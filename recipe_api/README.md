# Recipe API

A tiny REST recipe API built with Flask. Recipes live in an in-memory
store (fresh on every app start — nothing is written to disk), so it is
ideal for learning REST patterns and for testing with Flask's test
client.

## Why

Recipes are a classic CRUD shape with nested lists: a recipe has a list
of ingredients and a list of steps. This app shows how to validate
nested JSON, and how a simple query parameter turns a list endpoint
into a search endpoint.

## Usage

```bash
pip install -r requirements.txt
python recipe_api/demo.py
```

The demo creates two recipes, searches by ingredient, updates servings,
shows a validation error, deletes one recipe, and prints every JSON
response along the way.

To run the app for real:

```bash
flask --app "recipe_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/recipes` | List all recipes; `?ingredient=garlic` filters by ingredient |
| POST | `/recipes` | Create a recipe |
| GET | `/recipes/<id>` | Fetch one recipe |
| PUT | `/recipes/<id>` | Update `title`, `ingredients`, `steps` and/or `servings` |
| DELETE | `/recipes/<id>` | Delete a recipe |

A recipe looks like
`{"id": 1, "title": "Garlic Butter Pasta", "ingredients": ["pasta", "garlic"], "steps": ["Boil the pasta."], "servings": 2}`.
Validation errors return `400` with `{"error": "..."}`; unknown ids
and routes return `404` with `{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_recipe_api.py -v
```
