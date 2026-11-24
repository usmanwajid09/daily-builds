# loyalty_api

A loyalty-program REST API built with Flask and an in-memory store.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/members` | Register a member (`name`, `email`; email must be unique) |
| GET | `/members` | List all members |
| GET | `/members/<id>` | Fetch one member |
| DELETE | `/members/<id>` | Remove a member |
| POST | `/members/<id>/earn` | Add points (`points`, positive integer) |
| POST | `/members/<id>/redeem` | Spend points (`points`, positive integer) |
| POST | `/rewards` | Add a reward (`name`, `cost` in points) |
| GET | `/rewards` | List all rewards |
| POST | `/members/<id>/redeem-reward` | Claim a reward (`reward_id`) |
| GET | `/members/<id>/transactions` | A member's points history |
| GET | `/leaderboard` | Members ranked by points, highest first |
| GET | `/tiers` | Tier thresholds |

## Tiers

Tiers are derived from the points balance: **bronze** below 1000 points,
**silver** from 1000 to 4999, **gold** at 5000 or more. Earning or
redeeming points recomputes the tier automatically.

Redeeming more points than the balance holds returns `409 Conflict`.

## Run the demo

```bash
python -m loyalty_api.demo
```

## Run the tests

```bash
python -m pytest tests/test_loyalty_api.py -q
```
