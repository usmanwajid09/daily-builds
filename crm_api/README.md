# Mini-CRM API

A small Flask API for a mini customer-relationship manager: contacts
plus a deals pipeline (`lead` → `qualified` → `proposal` → `won` /
`lost`), deal value totals per stage, and a win conversion rate.
In-memory store — nothing is written to disk, and each
`create_app()` call starts empty.

## Endpoints

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/contacts` | Create a contact (`name`, `email`, `company?`, `phone?`); 409 on duplicate email |
| GET | `/contacts` | List contacts |
| GET | `/contacts/<id>` | Fetch one contact with its deals |
| PUT | `/contacts/<id>` | Update a contact; 409 on duplicate email |
| DELETE | `/contacts/<id>` | Delete a contact; 409 while it still has deals |
| POST | `/contacts/<id>/deals` | Add a deal (`title`, `value`, `stage?` defaults to `lead`) |
| GET | `/contacts/<id>/deals` | List a contact's deals |
| GET | `/deals` | List all deals; `?stage=` and `?contact_id=` filters |
| GET | `/deals/<id>` | Fetch one deal |
| PUT | `/deals/<id>` | Update title / value / stage |
| DELETE | `/deals/<id>` | Delete a deal |
| POST | `/deals/<id>/stage` | Move a deal to another stage; `won`/`lost` are terminal (409 to reopen) |
| GET | `/pipeline` | Per-stage counts and value totals, total value, conversion rate |

Conversion rate is `won / (won + lost) * 100`, and is `null` until
at least one deal is closed.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python crm_api/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_crm_api.py -q
```

## Example output

```
Created contact #1: Sara Ahmed (sara@acme.com, Acme Corp)
Created contact #2: Omar Khan (omar@globex.com, Globex)
Added deal #1: 'Website redesign' $5000 (lead)
Added deal #2: 'Mobile app' $12000 (qualified)
Added deal #3: 'Support contract' $3000 (proposal)
Moved 'Support contract' to won
Moved 'Website redesign' to lost
Reopening a won deal rejected: 409 {'error': 'deal is already won; closed deals cannot be reopened'}
Pipeline: 3 deals, total value $20000, conversion rate 50.0%
  lead: 0 deal(s), $0
  lost: 1 deal(s), $5000
  proposal: 0 deal(s), $0
  qualified: 1 deal(s), $12000
  won: 1 deal(s), $3000
```
