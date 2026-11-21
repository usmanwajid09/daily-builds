# subscription_tracker_api

A small Flask subscription tracker: recurring subscriptions with a
cost, a monthly or yearly billing cycle and a renewal date, kept in
memory (a fresh store per `create_app()` call, nothing written to
disk). Monthly and yearly equivalents are computed for every plan, so
a $48/year plan counts as $4/month and a $9.99/month plan as
$119.88/year.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/subscriptions` | Add one: `{"name", "cost", "billing_cycle": "monthly"\|"yearly", "renewal_date": "YYYY-MM-DD", "category"?}` |
| GET | `/subscriptions` | List; `?category=` and `?cycle=monthly\|yearly` filters |
| GET | `/subscriptions/<id>` | Fetch one subscription |
| PUT | `/subscriptions/<id>` | Update name, cost, billing_cycle, renewal_date and/or category |
| DELETE | `/subscriptions/<id>` | Delete a subscription |
| GET | `/totals` | Count, monthly/yearly totals, monthly breakdown by category |
| GET | `/upcoming` | Renewals due within `?days=` (default 30) from `?as_of=YYYY-MM-DD` (default today) |

A subscription is `{"id", "name", "cost", "billing_cycle",
"renewal_date", "category", "monthly_cost", "yearly_cost"}`. Next
renewals roll the stored renewal date forward by the billing cycle
until it is on or after the reference date, so a renewal date in the
past still reports the correct upcoming date. One subscription per
name (case-insensitive).

Validation: a missing/empty `name`, a cost that is not a positive
number, a billing cycle other than `monthly`/`yearly`, or a renewal
date that is not `YYYY-MM-DD` returns JSON `400`; an unknown
subscription returns JSON `404`; a duplicate name returns JSON `409`.

## Run

```bash
python subscription_tracker_api/demo.py
python -m pytest tests/test_subscription_tracker_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/subscriptions -H 'Content-Type: application/json' \
  -d '{"name": "Netflix", "cost": 9.99, "billing_cycle": "monthly", "renewal_date": "2026-12-01", "category": "streaming"}'

curl localhost:5000/totals
curl 'localhost:5000/upcoming?days=14'
```
