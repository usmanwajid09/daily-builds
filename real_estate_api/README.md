# Property Listings API

A small Flask API for real-estate listings: create/list/search
properties by city, price range, bedrooms and type, save/favourite a
listing, and average price per city. In-memory store — nothing is
written to disk, and each `create_app()` call starts empty.

## Endpoints

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/properties` | Create a listing (`title`, `city`, `price`, `bedrooms`, `bathrooms?`, `type?`, `area_sqft?`); 409 on same title in same city |
| GET | `/properties` | List/search; `?city=`, `?type=`, `?min_price=`, `?max_price=`, `?min_bedrooms=` |
| GET | `/properties/<id>` | Fetch one listing |
| PUT | `/properties/<id>` | Update a listing; 409 on duplicate title + city |
| DELETE | `/properties/<id>` | Delete a listing |
| POST | `/properties/<id>/favourite` | Save as favourite; 409 if already favourited |
| DELETE | `/properties/<id>/favourite` | Remove from favourites; 404 if not favourited |
| GET | `/favourites` | List favourited listings |
| GET | `/stats` | Total listings, overall average price, average price per city (with counts), favourite count |

Property types: `apartment`, `house`, `condo`, `townhouse`, `studio`.

## Run

```bash
cd /home/hatch/workspace/daily-builds
python real_estate_api/demo.py
/home/hatch/workspace/db-venv/bin/python -m pytest tests/test_real_estate_api.py -q
```

## Example output

```
Listed #1: 'Sunny 2-bed flat' in Lahore — PKR 8,500,000 (2 bed, apartment)
Listed #2: 'Family house with garden' in Lahore — PKR 25,000,000 (4 bed, house)
Listed #3: 'Downtown studio' in Islamabad — PKR 6,200,000 (1 bed, studio)
Listed #4: 'Seaview condo' in Karachi — PKR 18,000,000 (3 bed, condo)
Search city=Lahore: 2 listing(s)
Search max_price=10,000,000 & min_bedrooms=2: 1 listing(s) — Sunny 2-bed flat
Favourited #2: 'Family house with garden'
Favourite again rejected: 409 {'error': 'property 2 is already favourited'}
Favourites: 1 listing(s)
Stats: 4 listings, average price PKR 14,425,000.0, 1 favourite(s)
  Islamabad: average PKR 6,200,000.0 (1 listing(s))
  Karachi: average PKR 18,000,000.0 (1 listing(s))
  Lahore: average PKR 16,750,000.0 (2 listing(s))
```
