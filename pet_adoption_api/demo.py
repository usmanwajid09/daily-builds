"""Demo: list pets, apply for one, approve the adoption, search by species."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pet_adoption_api.app import create_app


def main():
    client = create_app().test_client()

    buddy = client.post(
        "/pets",
        json={"name": "Buddy", "species": "dog", "breed": "Labrador", "age": 3},
    ).get_json()
    print(
        f"Created pet #{buddy['id']}: {buddy['name']} "
        f"({buddy['species']}, {buddy['breed']}, age {buddy['age']}) — "
        f"status {buddy['status']}"
    )

    whiskers = client.post(
        "/pets",
        json={"name": "Whiskers", "species": "cat", "breed": "Siamese", "age": 2},
    ).get_json()
    print(f"Created pet #{whiskers['id']}: {whiskers['name']} ({whiskers['species']})")

    dogs = client.get("/pets", query_string={"species": "dog"}).get_json()
    print(f"Search species=dog returned {len(dogs)} pet(s): "
          f"{[p['name'] for p in dogs]}")

    application = client.post(
        f"/pets/{buddy['id']}/applications",
        json={
            "applicant_name": "Usman",
            "applicant_email": "usman@example.com",
            "message": "Big garden, works from home.",
        },
    ).get_json()
    print(
        f"Application #{application['id']} for {buddy['name']} "
        f"— status {application['status']}"
    )

    pet_now = client.get(f"/pets/{buddy['id']}").get_json()
    print(f"Pet #{buddy['id']} status after application: {pet_now['status']}")

    approved = client.post(
        f"/applications/{application['id']}/approve"
    ).get_json()
    print(f"Application #{approved['id']} -> {approved['status']}")

    adopted = client.get(f"/pets/{buddy['id']}").get_json()
    print(f"Pet #{adopted['id']} final status: {adopted['status']}")

    available = client.get("/pets", query_string={"status": "available"}).get_json()
    print(f"Still available: {[p['name'] for p in available]}")


if __name__ == "__main__":
    main()
