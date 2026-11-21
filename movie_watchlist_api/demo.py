"""Demo: build a watchlist, watch films, rate them, show stats."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movie_watchlist_api.app import create_app


def main():
    client = create_app().test_client()

    films = [
        {"title": "Dune: Part Two", "genre": "Sci-Fi", "year": 2024},
        {"title": "The Dark Knight", "genre": "Action", "year": 2008},
        {"title": "Spirited Away", "genre": "Animation", "year": 2001},
        {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014,
         "status": "watched", "rating": 5},
        {"title": "Parasite", "genre": "Thriller", "year": 2019,
         "status": "watching"},
    ]
    for payload in films:
        film = client.post("/movies", json=payload).get_json()
        print(f"Added {film['title']!r} ({film['year']}, {film['genre']}) "
              f"— {film['status']}"
              + (f", rated {film['rating']}/5" if film["rating"] else ""))

    dup = client.post("/movies", json={"title": "dune: part two",
                                       "genre": "Sci-Fi", "year": 2024})
    print(f"Duplicate title refused: {dup.status_code} {dup.get_json()}")

    watched = client.post("/movies/1/watch", json={"rating": 5}).get_json()
    print(f"Watched {watched['title']!r}, rated {watched['rating']}/5")

    sci_fi = client.get("/movies?genre=Sci-Fi").get_json()
    print(f"Sci-Fi films ({len(sci_fi)}): "
          + ", ".join(f["title"] for f in sci_fi))
    to_watch = client.get("/movies?status=to_watch").get_json()
    print(f"Still to watch ({len(to_watch)}): "
          + ", ".join(f["title"] for f in to_watch))

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total']} films, {stats['watched']} watched, "
          f"avg rating {stats['average_rating']}/5 over "
          f"{stats['rated_films']} rated, by genre {stats['by_genre']}")

    bad = client.post("/movies", json={"title": "Unrated", "genre": "Drama",
                                       "year": 2020, "rating": 4})
    print(f"Rating an unwatched film rejected: {bad.status_code} "
          f"{bad.get_json()}")


if __name__ == "__main__":
    main()
