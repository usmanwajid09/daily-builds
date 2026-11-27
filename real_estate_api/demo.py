"""Demo: list properties, search them, favourite one, check averages."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_estate_api.app import create_app


def main():
    client = create_app().test_client()

    listings = []
    for title, city, price, bedrooms, prop_type in (
        ("Sunny 2-bed flat", "Lahore", 8500000, 2, "apartment"),
        ("Family house with garden", "Lahore", 25000000, 4, "house"),
        ("Downtown studio", "Islamabad", 6200000, 1, "studio"),
        ("Seaview condo", "Karachi", 18000000, 3, "condo"),
    ):
        prop = client.post(
            "/properties",
            json={"title": title, "city": city, "price": price,
                  "bedrooms": bedrooms, "type": prop_type},
        ).get_json()
        listings.append(prop)
        print(f"Listed #{prop['id']}: {prop['title']!r} in {prop['city']} "
              f"— PKR {prop['price']:,} ({prop['bedrooms']} bed, "
              f"{prop['type']})")

    lahore = client.get("/properties?city=Lahore").get_json()
    print(f"Search city=Lahore: {len(lahore)} listing(s)")
    affordable = client.get(
        "/properties?max_price=10000000&min_bedrooms=2"
    ).get_json()
    print(f"Search max_price=10,000,000 & min_bedrooms=2: "
          f"{len(affordable)} listing(s) — "
          + ", ".join(p["title"] for p in affordable))

    fav = client.post(
        f"/properties/{listings[1]['id']}/favourite"
    ).get_json()
    print(f"Favourited #{fav['id']}: {fav['title']!r}")
    again = client.post(f"/properties/{listings[1]['id']}/favourite")
    print(f"Favourite again rejected: "
          f"{again.status_code} {again.get_json()}")
    favourites = client.get("/favourites").get_json()
    print(f"Favourites: {len(favourites)} listing(s)")

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total_listings']} listings, "
          f"average price PKR {stats['average_price']:,}, "
          f"{stats['favourite_count']} favourite(s)")
    for city, data in stats["average_price_by_city"].items():
        print(f"  {city}: average PKR {data['average_price']:,} "
              f"({data['count']} listing(s))")


if __name__ == "__main__":
    main()
