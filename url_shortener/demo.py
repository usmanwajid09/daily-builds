"""Demo: shorten a URL, follow the redirect, print stats."""
from url_shortener.app import create_app


def main():
    client = create_app().test_client()

    shortened = client.post(
        "/shorten", json={"url": "https://example.com/some/long/page"}
    ).get_json()
    print(f"Shortened: {shortened}")

    code = shortened["code"]
    redirect_resp = client.get(f"/{code}", follow_redirects=False)
    print(
        f"Redirect: status {redirect_resp.status_code}, "
        f"Location: {redirect_resp.headers['Location']}"
    )
    # A second visit, to show the click counter moving.
    client.get(f"/{code}", follow_redirects=False)

    stats = client.get(f"/stats/{code}").get_json()
    print(f"Stats after 2 visits: {stats}")


if __name__ == "__main__":
    main()
