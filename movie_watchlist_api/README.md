# movie_watchlist_api

A small Flask movie watchlist API: films with a genre, year, status
(`to_watch` / `watching` / `watched`) and a 1–5 star rating, kept in
memory (a fresh store per `create_app()` call, nothing written to
disk). Ratings belong to watched films only — rating an unwatched
film is refused, and moving a film back to `to_watch` or `watching`
clears its rating.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/movies` | Add a film: `{"title", "genre", "year", "status"?, "rating"?}` |
| GET | `/movies` | List films; `?genre=`, `?status=` and `?min_rating=` filters |
| GET | `/movies/<id>` | Fetch one film |
| PUT | `/movies/<id>` | Update title, genre, year, status and/or rating |
| DELETE | `/movies/<id>` | Delete a film |
| POST | `/movies/<id>/watch` | Mark watched, optional `{"rating": 1-5}` |
| GET | `/stats` | Totals, counts by status/genre, average rating |

A film is `{"id", "title", "genre", "year", "status", "rating"}`.
One film per title (case-insensitive).

Validation: a missing/empty `title` or `genre`, a year outside
1888–2030, a bad `status`, a rating outside 1–5, or a rating on an
unwatched film returns JSON `400`; an unknown film returns JSON
`404`; a duplicate title returns JSON `409`; marking an already
watched film watched again (with no new rating) returns JSON `409`.

## Run

```bash
python movie_watchlist_api/demo.py
python -m pytest tests/test_movie_watchlist_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/movies -H 'Content-Type: application/json' \
  -d '{"title": "Dune: Part Two", "genre": "Sci-Fi", "year": 2024}'

curl -X POST localhost:5000/movies/1/watch -H 'Content-Type: application/json' \
  -d '{"rating": 5}'

curl 'localhost:5000/movies?status=to_watch'
curl localhost:5000/stats
```
