"""Demo: create recipes, search by ingredient, update and delete via test client."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recipe_api.app import create_app


def main():
    client = create_app().test_client()

    pasta = client.post(
        "/recipes",
        json={
            "title": "Garlic Butter Pasta",
            "ingredients": ["pasta", "garlic", "butter", "parsley"],
            "steps": ["Boil the pasta.", "Melt butter with garlic.", "Toss and serve."],
            "servings": 2,
        },
    ).get_json()
    print(f"Created recipe: {pasta}")

    salad = client.post(
        "/recipes",
        json={
            "title": "Tomato Salad",
            "ingredients": ["tomato", "onion", "olive oil"],
            "steps": ["Slice the tomatoes.", "Add onion and oil."],
            "servings": 4,
        },
    ).get_json()
    print(f"Created recipe: {salad}")

    garlic = client.get("/recipes?ingredient=garlic").get_json()
    print(f"Recipes with garlic: {[r['title'] for r in garlic]}")

    updated = client.put(
        f"/recipes/{pasta['id']}", json={"servings": 4}
    ).get_json()
    print(f"Updated servings to {updated['servings']}: {updated}")

    bad = client.post("/recipes", json={"title": "No ingredients"})
    print(f"Invalid recipe rejected: {bad.status_code} {bad.get_json()}")

    deleted = client.delete(f"/recipes/{salad['id']}").get_json()
    print(f"Deleted recipe: {deleted}")
    print(f"Remaining recipes: {client.get('/recipes').get_json()}")


if __name__ == "__main__":
    main()
