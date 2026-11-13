"""Demo: generate a password and score a weak + strong password."""
from password_api.app import create_app


def main():
    client = create_app().test_client()

    generated = client.post("/generate", json={"length": 16}).get_json()
    print(f"Generated password: {generated['password']} (length {generated['length']})")

    weak = client.post("/strength", json={"password": "password"}).get_json()
    print(f"Weak password 'password': score {weak['score']}/5, {weak['label']}, "
          f"feedback: {weak['feedback']}")

    strong = client.post(
        "/strength", json={"password": generated["password"]}
    ).get_json()
    print(f"Generated password strength: score {strong['score']}/5, {strong['label']}, "
          f"feedback: {strong['feedback']}")


if __name__ == "__main__":
    main()
