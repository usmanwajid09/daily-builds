# review_api

A product-review REST API built with Flask and an in-memory store.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/reviews` | Submit a review (`product_id`, `author`, `rating` 1–5, optional `title`/`comment`) |
| GET | `/reviews` | List reviews (`?product_id=...`, `?status=visible\|flagged\|hidden`) |
| GET | `/reviews/<id>` | Fetch one review |
| DELETE | `/reviews/<id>` | Delete a review |
| POST | `/reviews/<id>/helpful` | Add one helpful vote |
| POST | `/reviews/<id>/flag` | Flag a review for moderation |
| POST | `/reviews/<id>/moderate` | Moderate a flagged review (`action`: `approve` or `hide`) |
| GET | `/products/<product_id>/summary` | Average rating, review count and 1–5 star distribution |

## Rules

- Ratings are integers from 1 to 5.
- One review per author per product — a second one returns `409 Conflict`.
- The summary counts every review except hidden ones, so a flagged
  review still counts until a moderator hides it.
- Approving a flagged review makes it visible again; hiding it removes
  it from the summary and from visible listings.

## Run the demo

```bash
python -m review_api.demo
```

## Run the tests

```bash
python -m pytest tests/test_review_api.py -q
```
