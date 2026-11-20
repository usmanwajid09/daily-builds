"""Demo: plan meals for a few days, show a day plan, week totals and shopping list."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meal_planner_api.app import create_app


def main():
    client = create_app().test_client()

    oats = client.post(
        "/meals",
        json={
            "name": "Overnight oats",
            "day": "mon",
            "meal_type": "breakfast",
            "calories": 350,
            "ingredients": [
                {"name": "Oats", "quantity": 50, "unit": "g"},
                {"name": "Milk", "quantity": 200, "unit": "ml"},
            ],
        },
    ).get_json()
    print(f"Added: {oats['name']} ({oats['day']} {oats['meal_type']}, "
          f"{oats['calories']} kcal)")

    curry = client.post(
        "/meals",
        json={
            "name": "Chicken curry",
            "day": "mon",
            "meal_type": "dinner",
            "calories": 650,
            "ingredients": [
                {"name": "Chicken", "quantity": 200, "unit": "g"},
                {"name": "Rice", "quantity": 150, "unit": "g"},
            ],
        },
    ).get_json()
    print(f"Added: {curry['name']} ({curry['day']} {curry['meal_type']}, "
          f"{curry['calories']} kcal)")

    salad = client.post(
        "/meals",
        json={
            "name": "Greek salad",
            "day": "tue",
            "meal_type": "lunch",
            "calories": 420,
            "ingredients": [
                {"name": "Tomato", "quantity": 2, "unit": "pcs"},
                {"name": "Oats", "quantity": 30, "unit": "g"},
            ],
        },
    ).get_json()
    print(f"Added: {salad['name']} ({salad['day']} {salad['meal_type']}, "
          f"{salad['calories']} kcal)")

    monday = client.get("/plan/mon").get_json()
    print(f"Monday plan: {monday['meal_count']} meals, "
          f"{monday['total_calories']} kcal total")

    week = client.get("/plan/week").get_json()
    print(f"Week total: {week['week_total_calories']} kcal "
          f"({sum(d['meal_count'] for d in week['days'].values())} meals)")

    total = client.get("/calories").get_json()
    print(f"All meals: {total['meal_count']} meals, "
          f"{total['total_calories']} kcal")

    shopping = client.get("/shopping-list").get_json()
    print(f"Shopping list ({shopping['item_count']} items): "
          + ", ".join(
              f"{i['name']} {i['quantity']}{i['unit']} (x{i['meals']} meals)"
              for i in shopping["items"]
          ))

    bad = client.post(
        "/meals",
        json={"name": "Mystery meal", "day": "funday",
              "meal_type": "lunch", "calories": 100},
    )
    print(f"Bad day rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
