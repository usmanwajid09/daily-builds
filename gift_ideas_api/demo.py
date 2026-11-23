"""Demo: add a person, gift ideas, buy one, check the budget."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gift_ideas_api.app import create_app


def main():
    client = create_app().test_client()

    person = client.post("/people", json={"name": "Ayesha", "budget": 100}).get_json()
    print(f"Created person: {person}")

    idea1 = client.post(
        f"/people/{person['id']}/ideas",
        json={"title": "Novel", "price": 25, "occasion": "Birthday"},
    ).get_json()
    idea2 = client.post(
        f"/people/{person['id']}/ideas",
        json={"title": "Headphones", "price": 60, "occasion": "Birthday"},
    ).get_json()
    print(f"Added ideas: {idea1['title']} (${idea1['price']}), "
          f"{idea2['title']} (${idea2['price']})")

    client.put(f"/ideas/{idea1['id']}", json={"status": "bought"})
    budget = client.get(f"/people/{person['id']}/budget").get_json()
    print(f"Budget after buying the novel: {budget}")


if __name__ == "__main__":
    main()
