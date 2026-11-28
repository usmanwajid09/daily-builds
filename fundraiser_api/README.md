# fundraiser_api

A Flask fundraiser API with an in-memory store (fresh for each app
instance; nothing is written to disk). All responses are JSON.

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/campaigns` | Create a campaign (`{"title": str, "goal_amount": number > 0, "description": str optional}`) — 201 |
| GET | `/campaigns` | List all campaigns, each with `raised`, `progress_percent`, `donor_count` |
| GET | `/campaigns/<id>` | Fetch one campaign — 404 if unknown |
| DELETE | `/campaigns/<id>` | Delete a campaign — 404 if unknown |
| POST | `/campaigns/<id>/donate` | Record a donation (`{"donor_name": str, "amount": number > 0, "message": str optional}`) — 201; 404 if the campaign is unknown |
| GET | `/campaigns/<id>/donations` | List donations for a campaign, newest first — 404 if unknown |
| GET | `/campaigns/<id>/progress` | `campaign_id`, `title`, `goal_amount`, `raised`, `remaining`, `progress_percent`, `donor_count`, `goal_reached` |
| GET | `/campaigns/<id>/top-donors` | Donors aggregated by name, sorted by total donated desc |

Errors are JSON like `{"error": "message"}` with a 400 or 404 status
code. Missing/empty title or a non-positive goal returns 400; a
missing donor name or non-positive amount returns 400.

## Run the demo

From the repo root:

```bash
python fundraiser_api/demo.py
```

## Example output

```
Created campaign 1: 'School Library Fund' with goal $1000
Donation 1: Ali gave $250
Donation 2: Sara gave $400
Donation 3: Ali gave $150
Donation 4: Bilal gave $100
Progress: raised $900 of $1000 (90.0%), $100 remaining, 3 donors, goal reached: False
Top donors: Ali ($400, 2 donations), Sara ($400, 1 donations), Bilal ($100, 1 donations)
Invalid donation rejected: 400 {'error': 'amount must be a number greater than 0'}
```

## Tests

```bash
python -m pytest tests/test_fundraiser_api.py -v
```
