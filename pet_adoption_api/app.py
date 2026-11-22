"""Flask pet adoption API.

In-memory store, fresh for each app instance. All responses are JSON.

Pets — ``POST /pets`` (``{"name": str, "species": str, "breed": str?,
"age": int?, "description": str?}``); ``GET /pets`` with ``?species=``,
``?breed=`` and ``?status=`` filters; ``GET /pets/<id>``;
``PUT /pets/<id>``; ``DELETE /pets/<id>`` (only while ``available``).

Adoption applications — ``POST /pets/<id>/applications``
(``{"applicant_name": str, "applicant_email": str, "message": str?}``);
``GET /pets/<id>/applications``; ``GET /applications``;
``POST /applications/<id>/approve``; ``POST /applications/<id>/reject``.

Pet status workflow::

    available -> pending -> adopted

Applying for an available pet moves it to ``pending``. Approving an
application moves the pet to ``adopted`` and rejects every other pending
application for that pet. Rejecting an application moves the pet back to
``available`` when no other pending application remains.
"""
from flask import Flask, jsonify, request

SPECIES = ("dog", "cat", "bird", "rabbit", "fish", "other")
PET_STATUSES = ("available", "pending", "adopted")
APPLICATION_STATUSES = ("pending", "approved", "rejected")


def create_app():
    """Return a Flask app with a fresh, empty in-memory pet store."""
    app = Flask(__name__)
    app.config["PETS"] = {}
    app.config["APPLICATIONS"] = {}
    app.config["NEXT_PET_ID"] = 1
    app.config["NEXT_APPLICATION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_pet(pet_id):
        pet = app.config["PETS"].get(pet_id)
        if pet is None:
            return None, _error(f"pet {pet_id} not found", 404)
        return pet, None

    def _get_application(application_id):
        application = app.config["APPLICATIONS"].get(application_id)
        if application is None:
            return None, _error(f"application {application_id} not found", 404)
        return application, None

    def _validate_pet_payload(data, partial=False):
        if not isinstance(data, dict):
            return None, _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not partial or "name" in data:
            if not isinstance(name, str) or not name.strip():
                return None, _error(
                    "name is required and must be a non-empty string", 400
                )
        species = data.get("species")
        if not partial or "species" in data:
            if species not in SPECIES:
                return None, _error(
                    f"species must be one of {', '.join(SPECIES)}", 400
                )
        breed = data.get("breed")
        if breed is not None and (not isinstance(breed, str) or not breed.strip()):
            return None, _error("breed must be a non-empty string", 400)
        age = data.get("age")
        if age is not None and (
            isinstance(age, bool) or not isinstance(age, int) or age < 0
        ):
            return None, _error("age must be a non-negative integer", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return None, _error("description must be a string", 400)
        return data, None

    @app.post("/pets")
    def create_pet():
        data = request.get_json(silent=True)
        valid, err = _validate_pet_payload(data)
        if err is not None:
            return err
        pet_id = app.config["NEXT_PET_ID"]
        app.config["NEXT_PET_ID"] += 1
        pet = {
            "id": pet_id,
            "name": data["name"].strip(),
            "species": data["species"],
            "breed": data["breed"].strip() if data.get("breed") else None,
            "age": data.get("age"),
            "description": data.get("description", ""),
            "status": "available",
        }
        app.config["PETS"][pet_id] = pet
        return jsonify(pet), 201

    @app.get("/pets")
    def list_pets():
        pets = list(app.config["PETS"].values())
        species = request.args.get("species")
        if species is not None:
            if species not in SPECIES:
                return _error(
                    f"species filter must be one of {', '.join(SPECIES)}", 400
                )
            pets = [p for p in pets if p["species"] == species]
        breed = request.args.get("breed")
        if breed is not None:
            pets = [p for p in pets if (p["breed"] or "").lower() == breed.lower()]
        status = request.args.get("status")
        if status is not None:
            if status not in PET_STATUSES:
                return _error(
                    f"status filter must be one of {', '.join(PET_STATUSES)}", 400
                )
            pets = [p for p in pets if p["status"] == status]
        return jsonify(pets)

    @app.get("/pets/<int:pet_id>")
    def get_pet(pet_id):
        pet, err = _get_pet(pet_id)
        if err is not None:
            return err
        return jsonify(pet)

    @app.put("/pets/<int:pet_id>")
    def update_pet(pet_id):
        pet, err = _get_pet(pet_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        valid, err = _validate_pet_payload(data, partial=True)
        if err is not None:
            return err
        if "name" in data:
            pet["name"] = data["name"].strip()
        if "species" in data:
            pet["species"] = data["species"]
        if "breed" in data:
            pet["breed"] = data["breed"].strip() if data["breed"] else None
        if "age" in data:
            pet["age"] = data["age"]
        if "description" in data:
            pet["description"] = data["description"]
        return jsonify(pet)

    @app.delete("/pets/<int:pet_id>")
    def delete_pet(pet_id):
        pet, err = _get_pet(pet_id)
        if err is not None:
            return err
        if pet["status"] != "available":
            return _error(
                f"cannot delete a pet with status {pet['status']}", 409
            )
        del app.config["PETS"][pet_id]
        return jsonify({"deleted": pet_id})

    @app.post("/pets/<int:pet_id>/applications")
    def apply_for_pet(pet_id):
        pet, err = _get_pet(pet_id)
        if err is not None:
            return err
        if pet["status"] == "adopted":
            return _error("pet has already been adopted", 409)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        applicant_name = data.get("applicant_name")
        if not isinstance(applicant_name, str) or not applicant_name.strip():
            return _error(
                "applicant_name is required and must be a non-empty string", 400
            )
        email = data.get("applicant_email")
        if not isinstance(email, str) or "@" not in email or not email.strip():
            return _error("applicant_email must be a valid email address", 400)
        message = data.get("message", "")
        if not isinstance(message, str):
            return _error("message must be a string", 400)
        for existing in app.config["APPLICATIONS"].values():
            if (
                existing["pet_id"] == pet_id
                and existing["applicant_email"].lower() == email.strip().lower()
                and existing["status"] == "pending"
            ):
                return _error(
                    "this email already has a pending application for this pet",
                    409,
                )
        application_id = app.config["NEXT_APPLICATION_ID"]
        app.config["NEXT_APPLICATION_ID"] += 1
        application = {
            "id": application_id,
            "pet_id": pet_id,
            "applicant_name": applicant_name.strip(),
            "applicant_email": email.strip(),
            "message": message,
            "status": "pending",
        }
        app.config["APPLICATIONS"][application_id] = application
        pet["status"] = "pending"
        return jsonify(application), 201

    @app.get("/pets/<int:pet_id>/applications")
    def list_pet_applications(pet_id):
        pet, err = _get_pet(pet_id)
        if err is not None:
            return err
        applications = [
            a for a in app.config["APPLICATIONS"].values() if a["pet_id"] == pet_id
        ]
        return jsonify(applications)

    @app.get("/applications")
    def list_applications():
        applications = list(app.config["APPLICATIONS"].values())
        status = request.args.get("status")
        if status is not None:
            if status not in APPLICATION_STATUSES:
                return _error(
                    f"status filter must be one of "
                    f"{', '.join(APPLICATION_STATUSES)}",
                    400,
                )
            applications = [a for a in applications if a["status"] == status]
        return jsonify(applications)

    @app.get("/applications/<int:application_id>")
    def get_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        return jsonify(application)

    @app.post("/applications/<int:application_id>/approve")
    def approve_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        if application["status"] != "pending":
            return _error(
                f"cannot approve an application with status "
                f"{application['status']}",
                409,
            )
        pet = app.config["PETS"][application["pet_id"]]
        application["status"] = "approved"
        pet["status"] = "adopted"
        for other in app.config["APPLICATIONS"].values():
            if (
                other["pet_id"] == pet["id"]
                and other["id"] != application_id
                and other["status"] == "pending"
            ):
                other["status"] = "rejected"
        return jsonify(application)

    @app.post("/applications/<int:application_id>/reject")
    def reject_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        if application["status"] != "pending":
            return _error(
                f"cannot reject an application with status "
                f"{application['status']}",
                409,
            )
        application["status"] = "rejected"
        pet = app.config["PETS"][application["pet_id"]]
        still_pending = any(
            a["pet_id"] == pet["id"] and a["status"] == "pending"
            for a in app.config["APPLICATIONS"].values()
        )
        if not still_pending:
            pet["status"] = "available"
        return jsonify(application)

    return app
