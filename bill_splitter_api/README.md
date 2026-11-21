# bill_splitter_api

A small Flask bill splitter: shared bills where one person (the
payer) paid the whole bill up front and the others reimburse their
share, kept in memory (a fresh store per `create_app()` call, nothing
written to disk). Shares are equal by default — the last participant
takes the rounding remainder so shares sum exactly to the total — or
custom, as long as they sum to the total.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/bills` | Create a bill: `{"title", "total", "payer", "participants": [..], "shares"?}` |
| GET | `/bills` | List bills |
| GET | `/bills/<id>` | Fetch one bill with shares, payments and balances |
| PUT | `/bills/<id>` | Update title, total, payer, participants and/or shares |
| DELETE | `/bills/<id>` | Delete a bill |
| POST | `/bills/<id>/payments` | Record a reimbursement: `{"person", "amount"}` |
| GET | `/bills/<id>/payments` | List payments on a bill |
| GET | `/bills/<id>/balances` | Per-person share, paid and amount still owed |
| GET | `/bills/<id>/settlements` | Minimal settlement plan: who pays how much to the payer |
| GET | `/summary` | Bill count, total amount, total outstanding, settled bills |

A bill is `{"id", "title", "total", "payer", "participants",
"shares", "payments", "balances", "outstanding", "settled"}`. The
payer's own share is their own cost, so they never owe anything;
every other participant's balance is `share - paid`.

Validation: a missing/empty `title` or `payer`, a total that is not
a positive number, fewer than 2 participants, duplicate participant
names, or a payer who is not a participant returns JSON `400`;
custom shares that do not cover exactly the participants or do not
sum to the total return JSON `400`; a payment from a non-participant,
from the payer, or above the person's remaining share returns JSON
`400`; an unknown bill returns JSON `404`.

## Run

```bash
python bill_splitter_api/demo.py
python -m pytest tests/test_bill_splitter_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/bills -H 'Content-Type: application/json' \
  -d '{"title": "Dinner", "total": 100, "payer": "Usman", "participants": ["Usman", "Ali", "Sara", "Bilal"]}'

curl -X POST localhost:5000/bills/1/payments -H 'Content-Type: application/json' \
  -d '{"person": "Ali", "amount": 25}'

curl localhost:5000/bills/1/settlements
curl localhost:5000/summary
```
