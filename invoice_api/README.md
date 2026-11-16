# Invoice API

A tiny REST invoice API built with Flask. Invoices live in an
in-memory store (fresh on every app start — nothing is written to
disk), so it is ideal for learning REST patterns and for testing with
Flask's test client.

## Why

Invoices are the classic "server computes the money" shape: the client
sends line items and a tax rate, and the server derives the subtotal,
tax and total. This app shows how to validate money inputs and keep
derived values trustworthy by never accepting them from the client.

## Usage

```bash
pip install -r requirements.txt
python invoice_api/demo.py
```

The demo creates two invoices (one with an explicit 10% tax rate, one
with the default), marks the first paid, filters by status, and shows a
validation error for a zero-quantity line item.

To run the app for real:

```bash
flask --app "invoice_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| GET | `/invoices` | List all invoices; `?status=unpaid` or `?status=paid` filters |
| POST | `/invoices` | Create an invoice: `{"customer", "items": [...], "tax_rate"?}` |
| GET | `/invoices/<id>` | Fetch one invoice |
| POST | `/invoices/<id>/pay` | Mark an invoice paid |
| DELETE | `/invoices/<id>` | Delete an invoice |

A line item is
`{"description": "Website design", "quantity": 1, "unit_price": 500.0}`;
the server adds `line_total` and computes `subtotal`, `tax` and
`total` (all rounded to 2 decimals). `tax_rate` defaults to `0.10`
and must be between 0 and 1. Validation errors return `400` with
`{"error": "..."}`; unknown ids and routes return `404` with
`{"error": "..."}`.

## Run tests

```bash
python -m pytest tests/test_invoice_api.py -v
```
