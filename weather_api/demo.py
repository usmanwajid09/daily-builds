"""Demo the weather_api: list cities, fetch Lahore in C and F, forecast, 404."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather_api.app import create_app

client = create_app().test_client()

cities = client.get("/weather").get_json()
print(f"Cities available: {len(cities)}")
for c in cities:
    print(f"  {c['city']}: {c['temp_c']}°{c['temp_unit']}, {c['condition']}, "
          f"humidity {c['humidity']}%, wind {c['wind_kmh']} km/h")

lahore_c = client.get("/weather/Lahore").get_json()
lahore_f = client.get("/weather/Lahore?units=f").get_json()
print(f"Lahore now: {lahore_c['temp_c']}°C = {lahore_f['temp_c']}°F, "
      f"{lahore_c['condition']}")

forecast = client.get("/weather/Lahore/forecast").get_json()["forecast"]
print("Lahore 3-day forecast:")
for day in forecast:
    print(f"  {day['day']}: {day['temp_c']}°C, {day['condition']}")

missing = client.get("/weather/Atlantis")
print(f"Unknown city -> {missing.status_code}: {missing.get_json()['error']}")
bad_units = client.get("/weather/Lahore?units=k")
print(f"Bad units -> {bad_units.status_code}: {bad_units.get_json()['error']}")
