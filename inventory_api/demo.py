"""Demo the inventory_api: add items, adjust stock, low-stock alerts."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inventory_api.app import create_app

client = create_app().test_client()

for name, stock, threshold, price in [
    ("USB-C cable", 120, 10, 4.99),
    ("Mechanical keyboard", 8, 10, 79.0),
    ("Webcam", 3, 5, 49.5),
    ("Monitor stand", 25, 5, 29.99),
]:
    resp = client.post("/items", json={
        "name": name, "stock": stock, "threshold": threshold, "price": price})
    item = resp.get_json()
    print(f"Added item {item['id']}: {item['name']} — stock {item['stock']}, "
          f"low_stock={item['low_stock']}")

low = client.get("/items/low-stock").get_json()
print(f"Low-stock items: {len(low)} -> {[i['name'] for i in low]}")

sold = client.patch("/items/1/stock", json={"delta": -15}).get_json()
print(f"Sold 15 USB-C cables: stock now {sold['stock']}")
restocked = client.patch("/items/2/stock", json={"delta": 20}).get_json()
print(f"Restocked keyboards by 20: stock now {restocked['stock']}, "
      f"low_stock={restocked['low_stock']}")

too_many = client.patch("/items/3/stock", json={"delta": -10})
print(f"Remove 10 webcams (only 3 in stock) -> {too_many.status_code}: "
      f"{too_many.get_json()['error']}")
negative = client.post("/items", json={"name": "Bad item", "stock": -1})
print(f"Negative stock -> {negative.status_code}: {negative.get_json()['error']}")
missing = client.get("/items/99")
print(f"Unknown item -> {missing.status_code}: {missing.get_json()['error']}")

total_value = sum(
    (i["price"] or 0) * i["stock"] for i in client.get("/items").get_json())
print(f"Total inventory value: ${total_value:.2f}")
