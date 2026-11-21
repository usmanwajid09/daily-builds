"""Flask time tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /projects`` — create a project (``{"name": str,
  "hourly_rate": number >= 0 (default 0), "description": str?}``).
  A second project with the same name (case-insensitive) is refused
  with 409.
- ``GET /projects`` — list projects
- ``GET /projects/<id>`` — fetch one project
- ``PUT /projects/<id>`` — update name, hourly_rate and/or
  description
- ``DELETE /projects/<id>`` — delete a project and its entries
- ``POST /projects/<id>/entries/start`` — start a timer
  (``{"description": str?}``); refused with 409 while another entry
  is still running on that project
- ``POST /entries/<id>/stop`` — stop a running timer
  (``{"end_time": "YYYY-MM-DDTHH:MM"?}``, defaults to now); refused
  with 400 if the entry is already stopped
- ``POST /projects/<id>/entries`` — add a finished entry by hand
  (``{"start_time": "YYYY-MM-DDTHH:MM", "end_time": "...",
  "description": str?}``)
- ``GET /projects/<id>/entries`` — list a project's entries
- ``GET /entries/<id>`` — fetch one entry
- ``DELETE /entries/<id>`` — delete an entry
- ``GET /projects/<id>/summary`` — total/completed time, entry
  counts, whether a timer is running, and earnings from completed
  time at the project's hourly rate
- ``GET /summary`` — totals across all projects

A project is ``{"id": int, "name": str, "hourly_rate": float,
"description": str | None}``. An entry is ``{"id": int,
"project_id": int, "description": str | None, "start_time": str,
"end_time": str | None, "duration_minutes": float | None,
"running": bool}``.
"""
from datetime import datetime

from flask import Flask, jsonify, request


def _parse_time(value):
    """Parse an ISO datetime string, or return None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def create_app():
    """Return a Flask app with a fresh, empty in-memory store."""
    app = Flask(__name__)
    app.config["PROJECTS"] = {}
    app.config["ENTRIES"] = {}
    app.config["NEXT_PROJECT_ID"] = 1
    app.config["NEXT_ENTRY_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _public_project(project):
        return {
            "id": project["id"],
            "name": project["name"],
            "hourly_rate": project["hourly_rate"],
            "description": project["description"],
        }

    def _public_entry(entry):
        duration = None
        if entry["end_time"] is not None:
            duration = round(
                (entry["end_time"] - entry["start_time"]).total_seconds()
                / 60, 2
            )
        return {
            "id": entry["id"],
            "project_id": entry["project_id"],
            "description": entry["description"],
            "start_time": entry["start_time"].isoformat(timespec="minutes"),
            "end_time": entry["end_time"].isoformat(timespec="minutes")
            if entry["end_time"] is not None else None,
            "duration_minutes": duration,
            "running": entry["end_time"] is None,
        }

    def _get_project(project_id):
        project = app.config["PROJECTS"].get(project_id)
        if project is None:
            return None, _error(f"project {project_id} not found", 404)
        return project, None

    def _get_entry(entry_id):
        entry = app.config["ENTRIES"].get(entry_id)
        if entry is None:
            return None, _error(f"entry {entry_id} not found", 404)
        return entry, None

    def _running_entry(project_id):
        return next(
            (
                e for e in app.config["ENTRIES"].values()
                if e["project_id"] == project_id and e["end_time"] is None
            ),
            None,
        )

    def _duplicate_name(name, exclude_id=None):
        return any(
            p["name"].lower() == name.lower() and p["id"] != exclude_id
            for p in app.config["PROJECTS"].values()
        )

    @app.post("/projects")
    def create_project():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        rate = data.get("hourly_rate", 0)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                or rate < 0:
            return _error("hourly_rate must be a number >= 0", 400)
        description = data.get("description")
        if description is not None and (not isinstance(description, str)
                                        or not description.strip()):
            return _error("description must be a non-empty string", 400)
        if _duplicate_name(name.strip()):
            return _error(f"a project named {name.strip()!r} already exists", 409)
        project_id = app.config["NEXT_PROJECT_ID"]
        app.config["NEXT_PROJECT_ID"] += 1
        project = {
            "id": project_id,
            "name": name.strip(),
            "hourly_rate": round(float(rate), 2),
            "description": description.strip() if description else None,
        }
        app.config["PROJECTS"][project_id] = project
        return jsonify(_public_project(project)), 201

    @app.get("/projects")
    def list_projects():
        projects = sorted(
            app.config["PROJECTS"].values(), key=lambda p: p["id"]
        )
        return jsonify([_public_project(p) for p in projects])

    @app.get("/projects/<int:project_id>")
    def get_project(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        return jsonify(_public_project(project))

    @app.put("/projects/<int:project_id>")
    def update_project(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("name", "hourly_rate", "description")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide name, hourly_rate and/or "
                "description",
                400,
            )
        if "name" in data:
            if not isinstance(data["name"], str) or not data["name"].strip():
                return _error("name must be a non-empty string", 400)
            if _duplicate_name(data["name"].strip(), exclude_id=project_id):
                return _error(
                    f"a project named {data['name'].strip()!r} already exists",
                    409,
                )
            project["name"] = data["name"].strip()
        if "hourly_rate" in data:
            rate = data["hourly_rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                    or rate < 0:
                return _error("hourly_rate must be a number >= 0", 400)
            project["hourly_rate"] = round(float(rate), 2)
        if "description" in data:
            description = data["description"]
            if description is not None and (
                not isinstance(description, str) or not description.strip()
            ):
                return _error("description must be a non-empty string", 400)
            project["description"] = description.strip() if description else None
        return jsonify(_public_project(project))

    @app.delete("/projects/<int:project_id>")
    def delete_project(project_id):
        _project, err = _get_project(project_id)
        if err is not None:
            return err
        del app.config["PROJECTS"][project_id]
        for entry_id in [
            e["id"] for e in app.config["ENTRIES"].values()
            if e["project_id"] == project_id
        ]:
            del app.config["ENTRIES"][entry_id]
        return jsonify({"deleted": project_id})

    @app.post("/projects/<int:project_id>/entries/start")
    def start_entry(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if _running_entry(project_id) is not None:
            return _error(
                "a timer is already running on this project", 409
            )
        description = data.get("description")
        if description is not None and (not isinstance(description, str)
                                        or not description.strip()):
            return _error("description must be a non-empty string", 400)
        start_time = None
        if "start_time" in data:
            start_time = _parse_time(data["start_time"])
            if start_time is None:
                return _error(
                    "start_time must be an ISO datetime like "
                    "YYYY-MM-DDTHH:MM",
                    400,
                )
        if start_time is None:
            start_time = datetime.now().replace(second=0, microsecond=0)
        entry_id = app.config["NEXT_ENTRY_ID"]
        app.config["NEXT_ENTRY_ID"] += 1
        entry = {
            "id": entry_id,
            "project_id": project_id,
            "description": description.strip() if description else None,
            "start_time": start_time,
            "end_time": None,
        }
        app.config["ENTRIES"][entry_id] = entry
        return jsonify(_public_entry(entry)), 201

    @app.post("/entries/<int:entry_id>/stop")
    def stop_entry(entry_id):
        entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        if entry["end_time"] is not None:
            return _error("entry is already stopped", 400)
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        end_time = None
        if "end_time" in data:
            end_time = _parse_time(data["end_time"])
            if end_time is None:
                return _error(
                    "end_time must be an ISO datetime like YYYY-MM-DDTHH:MM",
                    400,
                )
        if end_time is None:
            end_time = datetime.now().replace(second=0, microsecond=0)
        if end_time <= entry["start_time"]:
            return _error("end_time must be after start_time", 400)
        entry["end_time"] = end_time
        return jsonify(_public_entry(entry))

    @app.post("/projects/<int:project_id>/entries")
    def create_entry(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        start_time = _parse_time(data.get("start_time"))
        if start_time is None:
            return _error(
                "start_time is required and must be an ISO datetime like "
                "YYYY-MM-DDTHH:MM",
                400,
            )
        end_time = _parse_time(data.get("end_time"))
        if end_time is None:
            return _error(
                "end_time is required and must be an ISO datetime like "
                "YYYY-MM-DDTHH:MM",
                400,
            )
        if end_time <= start_time:
            return _error("end_time must be after start_time", 400)
        description = data.get("description")
        if description is not None and (not isinstance(description, str)
                                        or not description.strip()):
            return _error("description must be a non-empty string", 400)
        entry_id = app.config["NEXT_ENTRY_ID"]
        app.config["NEXT_ENTRY_ID"] += 1
        entry = {
            "id": entry_id,
            "project_id": project_id,
            "description": description.strip() if description else None,
            "start_time": start_time,
            "end_time": end_time,
        }
        app.config["ENTRIES"][entry_id] = entry
        return jsonify(_public_entry(entry)), 201

    @app.get("/projects/<int:project_id>/entries")
    def list_entries(project_id):
        _project, err = _get_project(project_id)
        if err is not None:
            return err
        entries = [e for e in app.config["ENTRIES"].values()
                   if e["project_id"] == project_id]
        entries.sort(key=lambda e: e["start_time"])
        return jsonify([_public_entry(e) for e in entries])

    @app.get("/entries/<int:entry_id>")
    def get_entry(entry_id):
        entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        return jsonify(_public_entry(entry))

    @app.delete("/entries/<int:entry_id>")
    def delete_entry(entry_id):
        _entry, err = _get_entry(entry_id)
        if err is not None:
            return err
        del app.config["ENTRIES"][entry_id]
        return jsonify({"deleted": entry_id})

    def _summary_for(project):
        entries = [e for e in app.config["ENTRIES"].values()
                   if e["project_id"] == project["id"]]
        done = [e for e in entries if e["end_time"] is not None]
        total_minutes = round(
            sum(
                (e["end_time"] - e["start_time"]).total_seconds() / 60
                for e in done
            ),
            2,
        )
        return {
            "project_id": project["id"],
            "name": project["name"],
            "total_minutes": total_minutes,
            "total_hours": round(total_minutes / 60, 2),
            "entry_count": len(entries),
            "completed_entries": len(done),
            "running": _running_entry(project["id"]) is not None,
            "earnings": round(total_minutes / 60 * project["hourly_rate"], 2),
        }

    @app.get("/projects/<int:project_id>/summary")
    def project_summary(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        return jsonify(_summary_for(project))

    @app.get("/summary")
    def overall_summary():
        projects = list(app.config["PROJECTS"].values())
        summaries = [_summary_for(p) for p in projects]
        total_minutes = round(sum(s["total_minutes"] for s in summaries), 2)
        return jsonify(
            {
                "project_count": len(projects),
                "total_minutes": total_minutes,
                "total_hours": round(total_minutes / 60, 2),
                "total_earnings": round(
                    sum(s["earnings"] for s in summaries), 2
                ),
                "running_projects": sum(1 for s in summaries if s["running"]),
                "projects": summaries,
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
