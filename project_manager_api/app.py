"""Flask project manager API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /projects`` — create a project (``{\"name\": str,
  \"description\": str?}``; project names are unique, 409 on a
  duplicate)
- ``GET /projects`` — list projects (each with its task counts and
  progress %)
- ``GET /projects/<id>`` — fetch one project (with its tasks)
- ``PUT /projects/<id>`` — rename a project and/or change its
  description (409 on a duplicate name)
- ``DELETE /projects/<id>`` — delete a project and all its tasks
- ``POST /projects/<id>/tasks`` — add a task (``{\"title\": str,
  \"description\": str?, \"assignee\": str?, \"due_date\":
  \"YYYY-MM-DD\"?, \"status\": \"todo\" | \"in_progress\" | \"done\"?}``)
- ``GET /projects/<id>/tasks`` — list a project's tasks
  (``?status=`` filters; each task carries an ``overdue`` flag)
- ``GET /projects/<id>/tasks/<task_id>`` — fetch one task
- ``PUT /projects/<id>/tasks/<task_id>`` — update a task's title,
  description, assignee and/or due date
- ``DELETE /projects/<id>/tasks/<task_id>`` — delete a task
- ``POST /projects/<id>/tasks/<task_id>/status`` — move a task
  through the workflow (``{\"status\": ...}``; allowed moves are
  todo -> in_progress, todo -> done, in_progress -> done,
  in_progress -> todo (reopen), done -> todo (reopen); anything
  else is a 409, an unknown status is a 400)
- ``GET /projects/<id>/progress`` — total/done task counts,
  progress %, overdue count and the overdue task ids

A project is ``{\"id\": int, \"name\": str, \"description\": str,
\"tasks\": [...], \"total_tasks\": int, \"done_tasks\": int,
\"progress_pct\": float}``; a task is ``{\"id\": int, \"title\":
str, \"description\": str, \"assignee\": str | None, \"status\":
str, \"due_date\": str | None, \"overdue\": bool}``. A task is
overdue when it has a due date in the past and is not done.
"""
from datetime import date

from flask import Flask, jsonify, request

STATUSES = ("todo", "in_progress", "done")

# Allowed status moves (the task workflow). Anything not listed here
# is rejected with a 409.
ALLOWED_MOVES = {
    "todo": ("in_progress", "done"),
    "in_progress": ("done", "todo"),
    "done": ("todo",),
}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_due_date(value):
    """Return a ``date`` for a ``YYYY-MM-DD`` string, else None."""
    if not isinstance(value, str):
        return None
    try:
        parts = value.split("-")
        if len(parts) != 3 or len(parts[0]) != 4:
            return None
        year, month, day = (int(p) for p in parts)
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def create_app():
    """Return a Flask app with a fresh, empty in-memory project store."""
    app = Flask(__name__)
    app.config["PROJECTS"] = {}
    app.config["NEXT_PROJECT_ID"] = 1
    app.config["NEXT_TASK_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_project(project_id):
        project = app.config["PROJECTS"].get(project_id)
        if project is None:
            return None, _error(f"project {project_id} not found", 404)
        return project, None

    def _find_task(project, task_id):
        for task in project["tasks"]:
            if task["id"] == task_id:
                return task
        return None

    def _is_overdue(task):
        if task["status"] == "done" or task["due_date"] is None:
            return False
        due = _parse_due_date(task["due_date"])
        return due is not None and due < date.today()

    def _public_task(task):
        return {
            "id": task["id"],
            "title": task["title"],
            "description": task["description"],
            "assignee": task["assignee"],
            "status": task["status"],
            "due_date": task["due_date"],
            "overdue": _is_overdue(task),
        }

    def _progress(project):
        total = len(project["tasks"])
        done = sum(1 for t in project["tasks"] if t["status"] == "done")
        return total, done

    def _public_project(project, with_tasks=True):
        total, done = _progress(project)
        data = {
            "id": project["id"],
            "name": project["name"],
            "description": project["description"],
            "total_tasks": total,
            "done_tasks": done,
            "progress_pct": round(done / total * 100, 1) if total else 0.0,
        }
        if with_tasks:
            data["tasks"] = [_public_task(t) for t in project["tasks"]]
        return data

    def _validate_task_body(data, partial=False):
        """Validate a task payload; return an error response or None."""
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not partial or "title" in data:
            title = data.get("title")
            if not isinstance(title, str) or not title.strip():
                return _error(
                    "title is required and must be a non-empty string", 400
                )
        if "description" in data and not isinstance(data["description"], str):
            return _error("description must be a string", 400)
        if "assignee" in data:
            assignee = data["assignee"]
            if assignee is not None and (
                not isinstance(assignee, str) or not assignee.strip()
            ):
                return _error("assignee must be a non-empty string or null", 400)
        if "due_date" in data:
            due_date = data["due_date"]
            if due_date is not None and _parse_due_date(due_date) is None:
                return _error("due_date must be a YYYY-MM-DD date or null", 400)
        if "status" in data and data["status"] not in STATUSES:
            return _error(
                f"status must be one of {', '.join(STATUSES)}", 400
            )
        return None

    @app.post("/projects")
    def create_project():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        if any(p["name"].lower() == name.strip().lower()
               for p in app.config["PROJECTS"].values()):
            return _error(f"a project named {name.strip()!r} already exists", 409)
        project_id = app.config["NEXT_PROJECT_ID"]
        app.config["NEXT_PROJECT_ID"] += 1
        project = {
            "id": project_id,
            "name": name.strip(),
            "description": description,
            "tasks": [],
        }
        app.config["PROJECTS"][project_id] = project
        return jsonify(_public_project(project)), 201

    @app.get("/projects")
    def list_projects():
        projects = sorted(
            app.config["PROJECTS"].values(), key=lambda p: p["id"]
        )
        return jsonify([_public_project(p, with_tasks=False) for p in projects])

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
        if "name" not in data and "description" not in data:
            return _error(
                "nothing to update: provide name and/or description", 400
            )
        if "name" in data:
            name = data["name"]
            if not isinstance(name, str) or not name.strip():
                return _error("name must be a non-empty string", 400)
            if any(p["id"] != project_id
                   and p["name"].lower() == name.strip().lower()
                   for p in app.config["PROJECTS"].values()):
                return _error(
                    f"a project named {name.strip()!r} already exists", 409
                )
            project["name"] = name.strip()
        if "description" in data:
            if not isinstance(data["description"], str):
                return _error("description must be a string", 400)
            project["description"] = data["description"]
        return jsonify(_public_project(project))

    @app.delete("/projects/<int:project_id>")
    def delete_project(project_id):
        _project, err = _get_project(project_id)
        if err is not None:
            return err
        del app.config["PROJECTS"][project_id]
        return jsonify({"deleted": project_id})

    @app.post("/projects/<int:project_id>/tasks")
    def create_task(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_task_body(data)
        if err is not None:
            return err
        task_id = app.config["NEXT_TASK_ID"]
        app.config["NEXT_TASK_ID"] += 1
        task = {
            "id": task_id,
            "title": data["title"].strip(),
            "description": data.get("description", ""),
            "assignee": (
                data["assignee"].strip()
                if data.get("assignee") else None
            ),
            "status": data.get("status", "todo"),
            "due_date": data.get("due_date"),
        }
        project["tasks"].append(task)
        return jsonify(_public_task(task)), 201

    @app.get("/projects/<int:project_id>/tasks")
    def list_tasks(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        status = request.args.get("status")
        if status is not None and status not in STATUSES:
            return _error(
                f"status filter must be one of {', '.join(STATUSES)}", 400
            )
        tasks = project["tasks"]
        if status is not None:
            tasks = [t for t in tasks if t["status"] == status]
        return jsonify([_public_task(t) for t in tasks])

    @app.get("/projects/<int:project_id>/tasks/<int:task_id>")
    def get_task(project_id, task_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        task = _find_task(project, task_id)
        if task is None:
            return _error(
                f"task {task_id} not found in project {project_id}", 404
            )
        return jsonify(_public_task(task))

    @app.put("/projects/<int:project_id>/tasks/<int:task_id>")
    def update_task(project_id, task_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        task = _find_task(project, task_id)
        if task is None:
            return _error(
                f"task {task_id} not found in project {project_id}", 404
            )
        data = request.get_json(silent=True)
        err = _validate_task_body(data, partial=True)
        if err is not None:
            return err
        if not any(k in data for k in
                   ("title", "description", "assignee", "due_date")):
            return _error(
                "nothing to update: provide title, description, assignee "
                "and/or due_date",
                400,
            )
        if "title" in data:
            task["title"] = data["title"].strip()
        if "description" in data:
            task["description"] = data["description"]
        if "assignee" in data:
            task["assignee"] = (
                data["assignee"].strip() if data["assignee"] else None
            )
        if "due_date" in data:
            task["due_date"] = data["due_date"]
        return jsonify(_public_task(task))

    @app.delete("/projects/<int:project_id>/tasks/<int:task_id>")
    def delete_task(project_id, task_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        task = _find_task(project, task_id)
        if task is None:
            return _error(
                f"task {task_id} not found in project {project_id}", 404
            )
        project["tasks"] = [t for t in project["tasks"] if t["id"] != task_id]
        return jsonify({"deleted": task_id})

    @app.post("/projects/<int:project_id>/tasks/<int:task_id>/status")
    def change_task_status(project_id, task_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        task = _find_task(project, task_id)
        if task is None:
            return _error(
                f"task {task_id} not found in project {project_id}", 404
            )
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        new_status = data.get("status")
        if new_status not in STATUSES:
            return _error(
                f"status must be one of {', '.join(STATUSES)}", 400
            )
        if new_status == task["status"]:
            return _error(
                f"task is already {new_status}", 409
            )
        if new_status not in ALLOWED_MOVES[task["status"]]:
            return _error(
                f"cannot move a task from {task['status']} to "
                f"{new_status}",
                409,
            )
        task["status"] = new_status
        return jsonify(_public_task(task))

    @app.get("/projects/<int:project_id>/progress")
    def project_progress(project_id):
        project, err = _get_project(project_id)
        if err is not None:
            return err
        total, done = _progress(project)
        overdue = [t for t in project["tasks"] if _is_overdue(t)]
        return jsonify(
            {
                "project_id": project_id,
                "total_tasks": total,
                "done_tasks": done,
                "progress_pct": round(done / total * 100, 1) if total else 0.0,
                "overdue_count": len(overdue),
                "overdue_task_ids": [t["id"] for t in overdue],
                "by_status": {
                    s: sum(1 for t in project["tasks"] if t["status"] == s)
                    for s in STATUSES
                },
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
