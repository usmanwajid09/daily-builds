"""Demo: submit an RSS XML string and list the parsed items."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rss_reader_api.app import create_app

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Daily Dev News</title>
    <link>https://example.com/news</link>
    <description>Short dev headlines.</description>
    <item>
      <title>Flask 3.1 released</title>
      <link>https://example.com/news/flask-31</link>
      <description>Async views and more.</description>
      <pubDate>Mon, 17 Nov 2026 09:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Python 3.14 lands</title>
      <link>https://example.com/news/py314</link>
      <description>Free-threaded by default.</description>
      <pubDate>Tue, 18 Nov 2026 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def main():
    client = create_app().test_client()

    created = client.post(
        "/feeds", json={"name": "dev-news", "xml": SAMPLE_XML}
    ).get_json()
    print(f"Created feed: {created}")

    feeds = client.get("/feeds").get_json()
    print(f"{len(feeds)} feed(s): {[f['name'] for f in feeds]}")

    items = client.get(f"/feeds/{created['id']}/items").get_json()
    print(f"{len(items)} item(s):")
    for item in items:
        print(f"  - {item['title']} ({item['pubDate']}) -> {item['link']}")

    bad = client.post("/feeds", json={"name": "broken", "xml": "<rss><oops>"})
    print(f"Malformed XML rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
