# meal_planner_api

A small Flask meal planner API: meals assigned to a day of the week and a
meal type, kept in memory (a fresh store per `create_app()` call, nothing
written to disk). It builds a weekly plan, totals calories per day and per
week, and aggregates ingredients across all planned meals into a shopping
list.

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/meals` | Add a meal: `{"name", "day", "meal_type", "calories", "ingredients"?}` |
| GET | `/meals` | List meals; `?day=mon..sun`, `?meal_type=breakfast\|lunch\|dinner\|snack` |
| GET | `/meals/<id>` | Fetch one meal |
| PUT | `/meals/<id>` | Update name, day, meal_type, calories and/or ingredients |
| DELETE | `/meals/<id>` | Delete a meal |
| GET | `/plan/<day>` | One day's meals plus its total calories |
| GET | `/plan/week` | Every day's meals and totals, plus the week total |
| GET | `/calories` | Total calories; `?day=` for a single day |
| GET | `/shopping-list` | Ingredients aggregated by name + unit across all meals |

A meal is `{"id", "name", "day", "meal_type", "calories",
"ingredients"}`; an ingredient is `{"name", "quantity", "unit"}`.
Shopping-list items are `{"name", "unit", "quantity", "meals"}` where
`meals` counts how many planned meals use the item. Ingredients with the
same name and unit are summed (name matching is case-insensitive).

Validation: missing/empty `name`, a `day` outside `mon`–`sun`, a bad
`meal_type`, negative or non-whole `calories`, or a malformed ingredient
returns JSON `400`; an unknown id returns JSON `404`.

## Run

```bash
python meal_planner_api/demo.py
python -m pytest tests/test_meal_planner_api.py -v
```

## Examples

```bash
curl -X POST localhost:5000/meals -H 'Content-Type: application/json' \
  -d '{"name": "Overnight oats", "day": "mon", "meal_type": "breakfast",
       "calories": 350,
       "ingredients": [{"name": "Oats", "quantity": 50, "unit": "g"}]}'

curl localhost:5000/plan/mon
curl localhost:5000/plan/week
curl localhost:5000/shopping-list
```
