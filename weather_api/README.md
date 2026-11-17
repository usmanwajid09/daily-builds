# weather_api

Mock weather API: current conditions and a 3-day forecast for a small
set of cities (Lahore, London, Karachi, New York, Tokyo). In-memory
only — nothing is written to disk.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/weather` | Current conditions for all cities |
| GET | `/weather/<city>` | Current conditions for one city |
| GET | `/weather/<city>/forecast` | 3-day forecast for one city |

All endpoints accept `?units=c` (default) or `?units=f` to convert
temperatures. Unknown cities return `404`; an invalid `units` value
returns `400`.

## Run

```bash
python -m pytest tests/test_weather_api.py -q
python weather_api/demo.py
```

To serve it:

```bash
flask --app "weather_api.app:create_app" run
```

## Demo output

```
Cities available: 5
  Lahore: 24°C, Hazy sunshine, humidity 48%, wind 11 km/h
  London: 9°C, Light rain, humidity 84%, wind 19 km/h
  Karachi: 29°C, Sunny, humidity 61%, wind 16 km/h
  New York: 12°C, Partly cloudy, humidity 55%, wind 14 km/h
  Tokyo: 16°C, Clear, humidity 52%, wind 9 km/h
Lahore now: 24°C = 75.2°F, Hazy sunshine
Lahore 3-day forecast:
  2026-11-18: 25°C, Sunny
  2026-11-19: 23°C, Hazy sunshine
  2026-11-20: 22°C, Partly cloudy
Unknown city -> 404: unknown city: Atlantis
Bad units -> 400: units must be 'c' or 'f'
```
