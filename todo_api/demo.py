"""Demo: create, list, and complete a todo via Flask's test client."""
from todo_api.app import create_app


def main():
    client = create_app().test_client()

    created = client.post("/todos", json={"title": "Buy milk"}).get_json()
    print(f"Created todo: {created}")
    client.post("/todos", json={"title": "Read a chapter"})

    listed = client.get("/todos").get_json()
    print(f"Listed {len(listed)} todos: {listed}")

    completed = client.put(
        f"/todos/{created['id']}", json={"done": True}
    ).get_json()
    print(f"Completed todo: {completed}")

    final = client.get("/todos").get_json()
    print(f"Todos after completing one: {final}")


if __name__ == "__main__":
    main()
