"""Flask food-delivery API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /restaurants`` — create a restaurant (``{\"name\": str,
  \"cuisine\": str?, \"delivery_fee\": number >= 0?}``)
- ``GET /restaurants`` — list restaurants; ``?cuisine=`` filter
- ``GET /restaurants/<id>`` — fetch one restaurant
- ``PUT /restaurants/<id>`` — update name, cuisine and/or delivery_fee
- ``DELETE /restaurants/<id>`` — delete a restaurant (409 if it still
  has menu items)
- ``POST /restaurants/<id>/menu`` — add a menu item (``{\"name\": str,
  \"price\": number > 0, \"category\": str?}``)
- ``GET /restaurants/<id>/menu`` — list a restaurant's menu
- ``GET /restaurants/<id>/menu/<item_id>`` — fetch one menu item
- ``PUT /restaurants/<id>/menu/<item_id>`` — update a menu item
- ``DELETE /restaurants/<id>/menu/<item_id>`` — delete a menu item
  (409 if it appears in a live — non-cancelled — order)
- ``POST /orders`` — place an order (``{\"restaurant_id\": int,
  \"customer_name\": str, \"items\": [{\"menu_item_id\": int,
  \"quantity\": int >= 1}, ...]}``); the total is the items subtotal
  plus the restaurant's delivery fee
- ``GET /orders`` — list orders; ``?status=`` and ``?restaurant_id=``
  filters
- ``GET /orders/<id>`` — fetch one order
- ``POST /orders/<id>/status`` — move an order (``{\"status\": str}``)
- ``DELETE /orders/<id>`` — delete a cancelled order only

Order status flow — only these transitions are allowed::

    placed -> preparing | cancelled
    preparing -> out_for_delivery | cancelled
    out_for_delivery -> delivered
    delivered -> (terminal)
    cancelled -> (terminal)

A restaurant is ``{\"id\": int, \"name\": str, \"cuisine\": str,
\"delivery_fee\": float}``. A menu item is ``{\"id\": int,
\"restaurant_id\": int, \"name\": str, \"price\": float,
\"category\": str}``. An order is ``{\"id\": int, \"restaurant_id\": int,
\"customer_name\": str, \"items\": [{\"menu_item_id\": int, \"name\": str,
\"unit_price\": float, \"quantity\": int, \"line_total\": float}],
\"subtotal\": float, \"delivery_fee\": float, \"total\": float,
\"status\": str}``; item names and prices are snapshotted at order time.
"""
from flask import Flask, jsonify, request

STATUSES = ("placed", "preparing", "out_for_delivery", "delivered", "cancelled")
ALLOWED_TRANSITIONS = {
    "placed": ("preparing", "cancelled"),
    "preparing": ("out_for_delivery", "cancelled"),
    "out_for_delivery": ("delivered",),
    "delivered": (),
    "cancelled": (),
}


def create_app():
    """Return a Flask app with a fresh, empty in-memory store."""
    app = Flask(__name__)
    app.config["RESTAURANTS"] = {}
    app.config["MENU_ITEMS"] = {}
    app.config["ORDERS"] = {}
    app.config["NEXT_RESTAURANT_ID"] = 1
    app.config["NEXT_ITEM_ID"] = 1
    app.config["NEXT_ORDER_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _require_json_object():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, _error("request body must be a JSON object", 400)
        return data, None

    def _non_empty_string(value):
        return isinstance(value, str) and bool(value.strip())

    def _number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _get_restaurant(restaurant_id):
        restaurant = app.config["RESTAURANTS"].get(restaurant_id)
        if restaurant is None:
            return None, _error(f"restaurant {restaurant_id} not found", 404)
        return restaurant, None

    def _get_menu_item(item_id):
        item = app.config["MENU_ITEMS"].get(item_id)
        if item is None:
            return None, _error(f"menu item {item_id} not found", 404)
        return item, None

    # ---- restaurants -------------------------------------------------

    @app.post("/restaurants")
    def create_restaurant():
        data, err = _require_json_object()
        if err is not None:
            return err
        name = data.get("name")
        if not _non_empty_string(name):
            return _error("name is required and must be a non-empty string", 400)
        cuisine = data.get("cuisine", "")
        if not isinstance(cuisine, str):
            return _error("cuisine must be a string", 400)
        fee = data.get("delivery_fee", 0)
        if not _number(fee) or fee < 0:
            return _error("delivery_fee must be a number >= 0", 400)
        rid = app.config["NEXT_RESTAURANT_ID"]
        app.config["NEXT_RESTAURANT_ID"] += 1
        restaurant = {
            "id": rid,
            "name": name.strip(),
            "cuisine": cuisine.strip(),
            "delivery_fee": float(fee),
        }
        app.config["RESTAURANTS"][rid] = restaurant
        return jsonify(restaurant), 201

    @app.get("/restaurants")
    def list_restaurants():
        restaurants = list(app.config["RESTAURANTS"].values())
        cuisine = request.args.get("cuisine")
        if cuisine:
            restaurants = [r for r in restaurants
                           if r["cuisine"].lower() == cuisine.lower()]
        return jsonify(restaurants)

    @app.get("/restaurants/<int:restaurant_id>")
    def get_restaurant(restaurant_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        return jsonify(restaurant)

    @app.put("/restaurants/<int:restaurant_id>")
    def update_restaurant(restaurant_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        data, err = _require_json_object()
        if err is not None:
            return err
        if "name" in data:
            if not _non_empty_string(data["name"]):
                return _error("name must be a non-empty string", 400)
            restaurant["name"] = data["name"].strip()
        if "cuisine" in data:
            if not isinstance(data["cuisine"], str):
                return _error("cuisine must be a string", 400)
            restaurant["cuisine"] = data["cuisine"].strip()
        if "delivery_fee" in data:
            if not _number(data["delivery_fee"]) or data["delivery_fee"] < 0:
                return _error("delivery_fee must be a number >= 0", 400)
            restaurant["delivery_fee"] = float(data["delivery_fee"])
        return jsonify(restaurant)

    @app.delete("/restaurants/<int:restaurant_id>")
    def delete_restaurant(restaurant_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        if any(item["restaurant_id"] == restaurant_id
               for item in app.config["MENU_ITEMS"].values()):
            return _error(
                f"restaurant {restaurant_id} still has menu items; "
                "delete them first", 409)
        del app.config["RESTAURANTS"][restaurant_id]
        return jsonify({"deleted": restaurant_id})

    # ---- menu items ---------------------------------------------------

    @app.post("/restaurants/<int:restaurant_id>/menu")
    def add_menu_item(restaurant_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        data, err = _require_json_object()
        if err is not None:
            return err
        name = data.get("name")
        if not _non_empty_string(name):
            return _error("name is required and must be a non-empty string", 400)
        price = data.get("price")
        if not _number(price) or price <= 0:
            return _error("price is required and must be a number > 0", 400)
        category = data.get("category", "")
        if not isinstance(category, str):
            return _error("category must be a string", 400)
        item_id = app.config["NEXT_ITEM_ID"]
        app.config["NEXT_ITEM_ID"] += 1
        item = {
            "id": item_id,
            "restaurant_id": restaurant["id"],
            "name": name.strip(),
            "price": float(price),
            "category": category.strip(),
        }
        app.config["MENU_ITEMS"][item_id] = item
        return jsonify(item), 201

    @app.get("/restaurants/<int:restaurant_id>/menu")
    def list_menu(restaurant_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        items = [i for i in app.config["MENU_ITEMS"].values()
                 if i["restaurant_id"] == restaurant["id"]]
        category = request.args.get("category")
        if category:
            items = [i for i in items
                     if i["category"].lower() == category.lower()]
        return jsonify(items)

    @app.get("/restaurants/<int:restaurant_id>/menu/<int:item_id>")
    def get_menu_item(restaurant_id, item_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        item, err = _get_menu_item(item_id)
        if err is not None:
            return err
        if item["restaurant_id"] != restaurant["id"]:
            return _error(
                f"menu item {item_id} does not belong to restaurant "
                f"{restaurant_id}", 404)
        return jsonify(item)

    @app.put("/restaurants/<int:restaurant_id>/menu/<int:item_id>")
    def update_menu_item(restaurant_id, item_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        item, err = _get_menu_item(item_id)
        if err is not None:
            return err
        if item["restaurant_id"] != restaurant["id"]:
            return _error(
                f"menu item {item_id} does not belong to restaurant "
                f"{restaurant_id}", 404)
        data, err = _require_json_object()
        if err is not None:
            return err
        if "name" in data:
            if not _non_empty_string(data["name"]):
                return _error("name must be a non-empty string", 400)
            item["name"] = data["name"].strip()
        if "price" in data:
            if not _number(data["price"]) or data["price"] <= 0:
                return _error("price must be a number > 0", 400)
            item["price"] = float(data["price"])
        if "category" in data:
            if not isinstance(data["category"], str):
                return _error("category must be a string", 400)
            item["category"] = data["category"].strip()
        return jsonify(item)

    @app.delete("/restaurants/<int:restaurant_id>/menu/<int:item_id>")
    def delete_menu_item(restaurant_id, item_id):
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        item, err = _get_menu_item(item_id)
        if err is not None:
            return err
        if item["restaurant_id"] != restaurant["id"]:
            return _error(
                f"menu item {item_id} does not belong to restaurant "
                f"{restaurant_id}", 404)
        for order in app.config["ORDERS"].values():
            if order["status"] == "cancelled":
                continue
            if any(line["menu_item_id"] == item_id for line in order["items"]):
                return _error(
                    f"menu item {item_id} is in a live order; "
                    "cancel the order first", 409)
        del app.config["MENU_ITEMS"][item_id]
        return jsonify({"deleted": item_id})

    # ---- orders -------------------------------------------------------

    @app.post("/orders")
    def create_order():
        data, err = _require_json_object()
        if err is not None:
            return err
        restaurant_id = data.get("restaurant_id")
        if not isinstance(restaurant_id, int) or isinstance(restaurant_id, bool):
            return _error("restaurant_id is required and must be an integer", 400)
        restaurant, err = _get_restaurant(restaurant_id)
        if err is not None:
            return err
        customer = data.get("customer_name")
        if not _non_empty_string(customer):
            return _error(
                "customer_name is required and must be a non-empty string", 400)
        items = data.get("items")
        if not isinstance(items, list) or not items:
            return _error("items must be a non-empty list", 400)
        seen = set()
        lines = []
        subtotal = 0.0
        for entry in items:
            if not isinstance(entry, dict):
                return _error("each item must be an object", 400)
            item_id = entry.get("menu_item_id")
            if not isinstance(item_id, int) or isinstance(item_id, bool):
                return _error("each item needs an integer menu_item_id", 400)
            if item_id in seen:
                return _error(
                    f"menu item {item_id} appears more than once; "
                    "combine quantities instead", 400)
            seen.add(item_id)
            quantity = entry.get("quantity")
            if (not isinstance(quantity, int) or isinstance(quantity, bool)
                    or quantity < 1):
                return _error(
                    f"quantity for menu item {item_id} must be an integer >= 1",
                    400)
            item, item_err = _get_menu_item(item_id)
            if item_err is not None:
                return item_err
            if item["restaurant_id"] != restaurant["id"]:
                return _error(
                    f"menu item {item_id} does not belong to restaurant "
                    f"{restaurant_id}", 400)
            line_total = round(item["price"] * quantity, 2)
            subtotal += line_total
            lines.append({
                "menu_item_id": item_id,
                "name": item["name"],
                "unit_price": item["price"],
                "quantity": quantity,
                "line_total": line_total,
            })
        subtotal = round(subtotal, 2)
        total = round(subtotal + restaurant["delivery_fee"], 2)
        order_id = app.config["NEXT_ORDER_ID"]
        app.config["NEXT_ORDER_ID"] += 1
        order = {
            "id": order_id,
            "restaurant_id": restaurant["id"],
            "customer_name": customer.strip(),
            "items": lines,
            "subtotal": subtotal,
            "delivery_fee": restaurant["delivery_fee"],
            "total": total,
            "status": "placed",
        }
        app.config["ORDERS"][order_id] = order
        return jsonify(order), 201

    @app.get("/orders")
    def list_orders():
        orders = list(app.config["ORDERS"].values())
        status = request.args.get("status")
        if status:
            if status not in STATUSES:
                return _error(
                    f"status must be one of {', '.join(STATUSES)}", 400)
            orders = [o for o in orders if o["status"] == status]
        restaurant_id = request.args.get("restaurant_id")
        if restaurant_id:
            try:
                rid = int(restaurant_id)
            except ValueError:
                return _error("restaurant_id must be an integer", 400)
            orders = [o for o in orders if o["restaurant_id"] == rid]
        return jsonify(orders)

    @app.get("/orders/<int:order_id>")
    def get_order(order_id):
        order = app.config["ORDERS"].get(order_id)
        if order is None:
            return _error(f"order {order_id} not found", 404)
        return jsonify(order)

    @app.post("/orders/<int:order_id>/status")
    def move_order(order_id):
        order = app.config["ORDERS"].get(order_id)
        if order is None:
            return _error(f"order {order_id} not found", 404)
        data, err = _require_json_object()
        if err is not None:
            return err
        status = data.get("status")
        if status not in STATUSES:
            return _error(
                f"status must be one of {', '.join(STATUSES)}", 400)
        allowed = ALLOWED_TRANSITIONS[order["status"]]
        if status not in allowed:
            return _error(
                f"cannot move order from {order['status']} to {status}; "
                f"allowed: {', '.join(allowed) or 'none'}", 400)
        order["status"] = status
        return jsonify(order)

    @app.delete("/orders/<int:order_id>")
    def delete_order(order_id):
        order = app.config["ORDERS"].get(order_id)
        if order is None:
            return _error(f"order {order_id} not found", 404)
        if order["status"] != "cancelled":
            return _error(
                "only a cancelled order can be deleted; cancel it first", 409)
        del app.config["ORDERS"][order_id]
        return jsonify({"deleted": order_id})

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
