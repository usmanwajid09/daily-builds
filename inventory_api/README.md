# inventory_api

Inventory API: track items with stock counts, add or remove stock,
and list low-stock items. In-memory only — nothing is written to disk.
Stock can never go below zero.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/items` | Add an item (`name`, `stock`, `threshold` default 5, `price` optional) |
| GET | `/items` | List all items |
| GET | `/items/<id>` | Fetch one item |
| PATCH | `/items/<id>/stock` | Adjust stock (`{"delta": int}`; negative removes stock) |
| GET | `/items/low-stock` | Items at or below threshold (`?threshold=N` overrides) |
| DELETE | `/items/<id>` | Delete an item |

Creating an item with negative stock, or removing more units than are
in stock, returns `400`; unknown item ids return `404`.

## Run

```bash
python -m pytest tests/test_inventory_api.py -q
python inventory_api/demo.py
```

## Demo output

```
Added item 1: USB-C cable — stock 120, low_stock=False
Added item 2: Mechanical keyboard — stock 8, low_stock=True
Added item 3: Webcam — stock 3, low_stock=True
Added item 4: Monitor stand — stock 25, low_stock=False
Low-stock items: 2 -> ['Mechanical keyboard', 'Webcam']
Sold 15 USB-C cables: stock now 105
Restocked keyboards by 20: stock now 28, low_stock=False
Remove 10 webcams (only 3 in stock) -> 400: cannot remove 10 units: only 3 in stock
Negative stock -> 400: stock is required and must be an integer >= 0
Unknown item -> 404: item 99 not found
Total inventory value: $3634.20
```
