"""Demo: open a restaurant, build a menu, place an order and walk it to delivered."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from food_delivery_api.app import create_app


def main():
    client = create_app().test_client()

    restaurant = client.post(
        "/restaurants",
        json={"name": "Lahore Biryani House", "cuisine": "Pakistani",
              "delivery_fee": 150},
    ).get_json()
    print(f"Restaurant #{restaurant['id']}: {restaurant['name']} "
          f"({restaurant['cuisine']}, delivery fee Rs {restaurant['delivery_fee']:.0f})")

    biryani = client.post(
        f"/restaurants/{restaurant['id']}/menu",
        json={"name": "Chicken Biryani", "price": 350, "category": "Main"},
    ).get_json()
    raita = client.post(
        f"/restaurants/{restaurant['id']}/menu",
        json={"name": "Raita", "price": 50, "category": "Side"},
    ).get_json()
    drink = client.post(
        f"/restaurants/{restaurant['id']}/menu",
        json={"name": "Mint Margarita", "price": 120, "category": "Drink"},
    ).get_json()
    menu = client.get(f"/restaurants/{restaurant['id']}/menu").get_json()
    print(f"Menu: {len(menu)} items "
          f"({', '.join(i['name'] for i in menu)})")

    order = client.post(
        "/orders",
        json={
            "restaurant_id": restaurant["id"],
            "customer_name": "Usman",
            "items": [
                {"menu_item_id": biryani["id"], "quantity": 2},
                {"menu_item_id": raita["id"], "quantity": 1},
                {"menu_item_id": drink["id"], "quantity": 2},
            ],
        },
    ).get_json()
    print(f"Order #{order['id']} for {order['customer_name']}: "
          f"subtotal Rs {order['subtotal']:.2f} + delivery "
          f"Rs {order['delivery_fee']:.2f} = total Rs {order['total']:.2f}")
    for line in order["items"]:
        print(f"  {line['quantity']}x {line['name']} @ Rs {line['unit_price']:.0f} "
              f"= Rs {line['line_total']:.2f}")

    for step in ("preparing", "out_for_delivery", "delivered"):
        moved = client.post(
            f"/orders/{order['id']}/status", json={"status": step}
        ).get_json()
        print(f"Order #{order['id']} -> {moved['status']}")

    bad_jump = client.post(
        "/orders", json={"restaurant_id": restaurant["id"],
                         "customer_name": "Ali",
                         "items": [{"menu_item_id": 999, "quantity": 1}]},
    )
    print(f"Order with an unknown menu item rejected: {bad_jump.status_code} "
          f"{bad_jump.get_json()}")

    cancelled = client.post(
        "/orders",
        json={"restaurant_id": restaurant["id"], "customer_name": "Sara",
              "items": [{"menu_item_id": biryani["id"], "quantity": 1}]},
    ).get_json()
    client.post(f"/orders/{cancelled['id']}/status", json={"status": "cancelled"})
    deleted = client.delete(f"/orders/{cancelled['id']}")
    print(f"Cancelled order #{cancelled['id']} deleted: "
          f"{deleted.status_code} {deleted.get_json()}")

    delivered = client.get(
        "/orders", query_string={"status": "delivered"}).get_json()
    print(f"Delivered orders: {len(delivered)} "
          f"(total revenue Rs {sum(o['total'] for o in delivered):.2f})")


if __name__ == "__main__":
    main()
