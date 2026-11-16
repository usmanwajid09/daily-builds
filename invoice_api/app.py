"""REST invoice API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /invoices`` — list all invoices (``?status=unpaid`` or
  ``?status=paid`` filters by status)
- ``POST /invoices`` — create an invoice with line items
- ``GET /invoices/<id>`` — fetch one invoice
- ``POST /invoices/<id>/pay`` — mark an invoice paid
- ``DELETE /invoices/<id>`` — delete an invoice

An invoice is ``{"id": int, "customer": str, "items": [...],
"tax_rate": float, "subtotal": float, "tax": float, "total": float,
"status": "unpaid" | "paid"}``. Each line item is
``{"description": str, "quantity": number, "unit_price": number,
"line_total": float}``. Subtotal, tax and total are computed by the
server and rounded to two decimals — the client never supplies them.
"""
from flask import Flask, jsonify, request

DEFAULT_TAX_RATE = 0.10
STATUSES = ("unpaid", "paid")


def create_app():
    """Return a Flask app with a fresh, empty in-memory invoice store."""
    app = Flask(__name__)
    app.config["INVOICES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _validate_items(items):
        """Validate line items. Returns (cleaned, None) or (None, error)."""
        if not isinstance(items, list) or not items:
            return None, _error("items is required and must be a non-empty list", 400)
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                return None, _error("each item must be a JSON object", 400)
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                return None, _error(
                    "each item needs a non-empty string description", 400
                )
            quantity = item.get("quantity")
            if not _is_number(quantity) or quantity <= 0:
                return None, _error("each item quantity must be a number above 0", 400)
            unit_price = item.get("unit_price")
            if not _is_number(unit_price) or unit_price < 0:
                return None, _error(
                    "each item unit_price must be a number of 0 or more", 400
                )
            cleaned.append(
                {
                    "description": description.strip(),
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_total": round(quantity * unit_price, 2),
                }
            )
        return cleaned, None

    def _get_invoice(invoice_id):
        invoice = app.config["INVOICES"].get(invoice_id)
        if invoice is None:
            return None, _error(f"invoice {invoice_id} not found", 404)
        return invoice, None

    @app.get("/invoices")
    def list_invoices():
        invoices = sorted(app.config["INVOICES"].values(), key=lambda i: i["id"])
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error("status must be unpaid or paid", 400)
            invoices = [i for i in invoices if i["status"] == status]
        return jsonify(invoices)

    @app.post("/invoices")
    def create_invoice():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        customer = data.get("customer")
        if not isinstance(customer, str) or not customer.strip():
            return _error("customer is required and must be a non-empty string", 400)
        items, err = _validate_items(data.get("items"))
        if err is not None:
            return err
        tax_rate = data.get("tax_rate", DEFAULT_TAX_RATE)
        if not _is_number(tax_rate) or tax_rate < 0 or tax_rate > 1:
            return _error("tax_rate must be a number between 0 and 1", 400)
        subtotal = round(sum(i["line_total"] for i in items), 2)
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)
        invoice_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        invoice = {
            "id": invoice_id,
            "customer": customer.strip(),
            "items": items,
            "tax_rate": tax_rate,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "status": "unpaid",
        }
        app.config["INVOICES"][invoice_id] = invoice
        return jsonify(invoice), 201

    @app.get("/invoices/<int:invoice_id>")
    def get_invoice(invoice_id):
        invoice, err = _get_invoice(invoice_id)
        if err is not None:
            return err
        return jsonify(invoice)

    @app.post("/invoices/<int:invoice_id>/pay")
    def pay_invoice(invoice_id):
        invoice, err = _get_invoice(invoice_id)
        if err is not None:
            return err
        if invoice["status"] == "paid":
            return _error(f"invoice {invoice_id} is already paid", 400)
        invoice["status"] = "paid"
        return jsonify(invoice)

    @app.delete("/invoices/<int:invoice_id>")
    def delete_invoice(invoice_id):
        _invoice, err = _get_invoice(invoice_id)
        if err is not None:
            return err
        del app.config["INVOICES"][invoice_id]
        return jsonify({"deleted": invoice_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
