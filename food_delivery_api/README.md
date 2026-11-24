# food_delivery_api

A small Flask food-delivery API: restaurants with menus, and orders
with a guarded status flow, a delivery fee and totals — kept in memory
(a fresh store per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/restaurants` | Create a restaurant: `{"name", "cuisine"?, "delivery_fee"?}` |
| GET | `/restaurants` | List restaurants; `?cuisine=` filter |
| GET | `/restaurants/<id>` | Fetch one restaurant |
| PUT | `/restaurants/<id>` | Update name, cuisine and/or delivery_fee |
| DELETE | `/restaurants/<id>` | Delete a restaurant (409 if it still has menu items) |
| POST | `/restaurants/<id>/menu` | Add a menu item: `{"name", "price", "category"?}` |
| GET | `/restaurants/<id>/menu` | List a restaurant's menu; `?category=` filter |
| GET | `/restaurants/<id>/menu/<item_id>` | Fetch one menu item |
| PUT | `/restaurants/<id>/menu/<item_id>` | Update a menu item |
| DELETE | `/restaurants/<id>/menu/<item_id>` | Delete a menu item (409 if in a live order) |
| POST | `/orders` | Place an order: `{"restaurant_id", "customer_name", "items": [{"menu_item_id", "quantity"}]}` |
| GET | `/orders` | List orders; `?status=` and `?restaurant_id=` filters |
| GET | `/orders/<id>` | Fetch one order |
| POST | `/orders/<id>/status` | Move an order: `{"status": str}` |
| DELETE | `/orders/<id>` | Delete a cancelled order only |

Order totals: `total = sum(price x quantity) + delivery_fee`.
Item names and prices are snapshotted into the order at placement time.

## Order status workflow

```
placed -> preparing -> out_for_delivery -> delivered
placed -> cancelled
preparing -> cancelled
delivered / cancelled -> (terminal)
```

Any other transition returns JSON `400` naming the allowed moves.

Validation: a missing/empty `name` or `customer_name`, a non-positive
`price`, a negative `delivery_fee`, an empty `items` list, a bad
`quantity`, a duplicated menu item, or an item from another restaurant
returns JSON `400`; an unknown restaurant, menu item or order returns
JSON `404`; deleting a restaurant with menu items, a menu item in a
live order, or a non-cancelled order returns JSON `409`.

## Run

```bash
python food_delivery_api/demo.py
python -m pytest tests/test_food_delivery_api.py -v
```
