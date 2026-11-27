"""Flask property listings API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /properties`` — create a listing (``{\"title\": str,
  \"city\": str, \"price\": number, \"bedrooms\": int,
  \"bathrooms\": int?, \"type\": \"apartment\" | \"house\" |
  \"condo\" | \"townhouse\" | \"studio\"?, \"area_sqft\": number?}``;
  price must be > 0, bedrooms/bathrooms >= 0; a listing with the
  same title in the same city is a 409)
- ``GET /properties`` — list/search listings (``?city=``,
  ``?type=``, ``?min_price=``, ``?max_price=``,
  ``?min_bedrooms=`` filters)
- ``GET /properties/<id>`` — fetch one listing
- ``PUT /properties/<id>`` — update a listing (409 on a duplicate
  title + city)
- ``DELETE /properties/<id>`` — delete a listing
- ``POST /properties/<id>/favourite`` — save a listing as a
  favourite (409 if it is already favourited)
- ``DELETE /properties/<id>/favourite`` — remove a listing from
  favourites (404 if it was not favourited)
- ``GET /favourites`` — list favourited listings
- ``GET /stats`` — total listings, average price overall and per
  city (each with its listing count), and favourite count

A property is ``{\"id\": int, \"title\": str, \"city\": str,
\"price\": number, \"bedrooms\": int, \"bathrooms\": int, \"type\":
str, \"area_sqft\": number | None, \"favourite\": bool}``.
"""
from flask import Flask, jsonify, request

TYPES = ("apartment", "house", "condo", "townhouse", "studio")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory listing store."""
    app = Flask(__name__)
    app.config["PROPERTIES"] = {}
    app.config["NEXT_PROPERTY_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_property(property_id):
        prop = app.config["PROPERTIES"].get(property_id)
        if prop is None:
            return None, _error(f"property {property_id} not found", 404)
        return prop, None

    def _public_property(prop):
        return dict(prop)

    def _duplicate_title_city(title, city, exclude_id=None):
        return any(
            p["id"] != exclude_id
            and p["title"].lower() == title.lower()
            and p["city"].lower() == city.lower()
            for p in app.config["PROPERTIES"].values()
        )

    def _validate_property_body(data, partial=False):
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        for field in ("title", "city"):
            if not partial or field in data:
                value = data.get(field)
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        f"{field} is required and must be a "
                        "non-empty string",
                        400,
                    )
        if "price" in data or not partial:
            price = data.get("price")
            if not _is_number(price) or price <= 0:
                return _error("price is required and must be a number > 0", 400)
        for field in ("bedrooms", "bathrooms"):
            if field in data or (not partial and field == "bedrooms"):
                value = data.get(field)
                if not _is_int(value) or value < 0:
                    return _error(
                        f"{field} is required and must be an integer >= 0",
                        400,
                    )
        if "type" in data and data["type"] not in TYPES:
            return _error(f"type must be one of {', '.join(TYPES)}", 400)
        if "area_sqft" in data:
            area = data["area_sqft"]
            if area is not None and (not _is_number(area) or area <= 0):
                return _error("area_sqft must be a number > 0 or null", 400)
        return None

    @app.post("/properties")
    def create_property():
        data = request.get_json(silent=True)
        err = _validate_property_body(data)
        if err is not None:
            return err
        title = data["title"].strip()
        city = data["city"].strip()
        if _duplicate_title_city(title, city):
            return _error(
                f"a listing titled {title!r} already exists in {city!r}",
                409,
            )
        property_id = app.config["NEXT_PROPERTY_ID"]
        app.config["NEXT_PROPERTY_ID"] += 1
        prop = {
            "id": property_id,
            "title": title,
            "city": city,
            "price": data["price"],
            "bedrooms": data["bedrooms"],
            "bathrooms": data.get("bathrooms", 1),
            "type": data.get("type", "apartment"),
            "area_sqft": data.get("area_sqft"),
            "favourite": False,
        }
        app.config["PROPERTIES"][property_id] = prop
        return jsonify(_public_property(prop)), 201

    @app.get("/properties")
    def list_properties():
        city = request.args.get("city")
        prop_type = request.args.get("type")
        if prop_type is not None and prop_type not in TYPES:
            return _error(f"type filter must be one of {', '.join(TYPES)}", 400)
        min_price = request.args.get("min_price", type=float)
        max_price = request.args.get("max_price", type=float)
        min_bedrooms = request.args.get("min_bedrooms", type=int)
        if min_price is not None and min_price < 0:
            return _error("min_price must be >= 0", 400)
        if max_price is not None and max_price < 0:
            return _error("max_price must be >= 0", 400)
        if min_price is not None and max_price is not None \
                and min_price > max_price:
            return _error("min_price cannot exceed max_price", 400)
        if min_bedrooms is not None and min_bedrooms < 0:
            return _error("min_bedrooms must be >= 0", 400)
        props = sorted(
            app.config["PROPERTIES"].values(), key=lambda p: p["id"]
        )
        if city is not None:
            props = [p for p in props if p["city"].lower() == city.lower()]
        if prop_type is not None:
            props = [p for p in props if p["type"] == prop_type]
        if min_price is not None:
            props = [p for p in props if p["price"] >= min_price]
        if max_price is not None:
            props = [p for p in props if p["price"] <= max_price]
        if min_bedrooms is not None:
            props = [p for p in props if p["bedrooms"] >= min_bedrooms]
        return jsonify([_public_property(p) for p in props])

    @app.get("/properties/<int:property_id>")
    def get_property(property_id):
        prop, err = _get_property(property_id)
        if err is not None:
            return err
        return jsonify(_public_property(prop))

    @app.put("/properties/<int:property_id>")
    def update_property(property_id):
        prop, err = _get_property(property_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_property_body(data, partial=True)
        if err is not None:
            return err
        fields = (
            "title", "city", "price", "bedrooms", "bathrooms", "type",
            "area_sqft",
        )
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, city, price, bedrooms, "
                "bathrooms, type and/or area_sqft",
                400,
            )
        new_title = (
            data["title"].strip() if "title" in data else prop["title"]
        )
        new_city = (
            data["city"].strip() if "city" in data else prop["city"]
        )
        if _duplicate_title_city(new_title, new_city, exclude_id=property_id):
            return _error(
                f"a listing titled {new_title!r} already exists in "
                f"{new_city!r}",
                409,
            )
        for field in fields:
            if field in data:
                value = data[field]
                if field in ("title", "city"):
                    value = value.strip()
                prop[field] = value
        return jsonify(_public_property(prop))

    @app.delete("/properties/<int:property_id>")
    def delete_property(property_id):
        _prop, err = _get_property(property_id)
        if err is not None:
            return err
        del app.config["PROPERTIES"][property_id]
        return jsonify({"deleted": property_id})

    @app.post("/properties/<int:property_id>/favourite")
    def favourite_property(property_id):
        prop, err = _get_property(property_id)
        if err is not None:
            return err
        if prop["favourite"]:
            return _error(
                f"property {property_id} is already favourited", 409
            )
        prop["favourite"] = True
        return jsonify(_public_property(prop))

    @app.delete("/properties/<int:property_id>/favourite")
    def unfavourite_property(property_id):
        prop, err = _get_property(property_id)
        if err is not None:
            return err
        if not prop["favourite"]:
            return _error(
                f"property {property_id} is not favourited", 404
            )
        prop["favourite"] = False
        return jsonify(_public_property(prop))

    @app.get("/favourites")
    def list_favourites():
        favourites = [
            p for p in sorted(
                app.config["PROPERTIES"].values(), key=lambda p: p["id"]
            )
            if p["favourite"]
        ]
        return jsonify([_public_property(p) for p in favourites])

    @app.get("/stats")
    def stats():
        props = list(app.config["PROPERTIES"].values())
        total = len(props)
        by_city = {}
        for prop in props:
            entry = by_city.setdefault(
                prop["city"], {"count": 0, "total_price": 0}
            )
            entry["count"] += 1
            entry["total_price"] += prop["price"]
        average_by_city = {
            city: {
                "count": entry["count"],
                "average_price": round(
                    entry["total_price"] / entry["count"], 2
                ),
            }
            for city, entry in sorted(by_city.items())
        }
        return jsonify(
            {
                "total_listings": total,
                "average_price": (
                    round(sum(p["price"] for p in props) / total, 2)
                    if total else None
                ),
                "average_price_by_city": average_by_city,
                "favourite_count": sum(
                    1 for p in props if p["favourite"]
                ),
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
