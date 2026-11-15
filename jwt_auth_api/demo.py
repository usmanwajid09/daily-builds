"""Demo: register, log in, and call a protected route with a JWT."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jwt_auth_api.app import create_app


def main():
    client = create_app().test_client()

    registered = client.post(
        "/auth/register", json={"username": "usman", "password": "hunter2secret"}
    ).get_json()
    print(f"Registered: {registered}")

    no_token = client.get("/auth/protected")
    print(f"Protected without token -> {no_token.status_code}: {no_token.get_json()}")

    login = client.post(
        "/auth/login", json={"username": "usman", "password": "hunter2secret"}
    ).get_json()
    token = login["token"]
    print(f"Login token (first 30 chars): {token[:30]}...")

    headers = {"Authorization": f"Bearer {token}"}
    protected = client.get("/auth/protected", headers=headers).get_json()
    print(f"Protected with token: {protected}")
    me = client.get("/auth/me", headers=headers).get_json()
    print(f"Me: {me}")

    wrong = client.post(
        "/auth/login", json={"username": "usman", "password": "wrongpass"}
    )
    print(f"Login with wrong password -> {wrong.status_code}: {wrong.get_json()}")


if __name__ == "__main__":
    main()
