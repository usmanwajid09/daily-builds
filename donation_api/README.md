# donation_api

A Flask API for charity donations: fundraising campaigns, validated
donations, and per-campaign totals and progress toward the goal.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/campaigns` | Create a campaign (`title`, `goal_amount`, `description?`, `organizer?`) |
| GET | `/campaigns` | List campaigns; filter by `?goal_reached=true/false` |
| GET | `/campaigns/<id>` | Fetch one campaign |
| PUT | `/campaigns/<id>` | Update a campaign (goal cannot drop below amount already raised) |
| DELETE | `/campaigns/<id>` | Delete a campaign (only while it has no donations) |
| POST | `/campaigns/<id>/donations` | Donate (`donor_name`, `amount`, `message?`) |
| GET | `/campaigns/<id>/donations` | Donations for one campaign |
| GET | `/campaigns/<id>/summary` | Campaign + `total_raised`, `donation_count`, `donor_count`, `progress_percent`, `amount_remaining`, `goal_reached` |
| GET | `/donations` | All donations |
| GET | `/donations/<id>` | Fetch one donation |

Amounts must be positive numbers and are rounded to two decimals.
Validation errors return 400, missing campaigns/donations return 404, and
illegal state changes (deleting a campaign with donations, lowering a goal
below the amount raised) return 409, all as `{"error": "..."}`.

## How to run

```bash
python donation_api/demo.py
```

Tests:

```bash
python -m pytest donation_api/tests/test_donation_api.py -v
```

## Demo output

```
Created campaign #1: 'School Books for Vehari' — goal 100000.0
Donation #1: Usman gave 25000.0
Donation #2: Hamza gave 40000.0
Donation #3: Usman gave 10000.0
Summary: raised 75000.0 of 100000.0 (75.0%), 3 donations from 2 donors, 25000.0 remaining, goal reached: False
Donation #4: Anonymous gave 25000.0
Summary: raised 100000.0 of 100000.0 (100.0%), goal reached: True
```
