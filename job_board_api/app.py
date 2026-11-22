"""Flask job board API.

In-memory store, fresh for each app instance. All responses are JSON.

Jobs — ``POST /jobs`` (``{"title": str, "company": str, "location": str,
"job_type": str, "salary_min": number?, "salary_max": number?,
"description": str?}``); ``GET /jobs`` with ``?location=``, ``?type=``,
``?status=`` and ``?company=`` filters; ``GET /jobs/<id>``;
``PUT /jobs/<id>``; ``DELETE /jobs/<id>``;
``POST /jobs/<id>/close``; ``POST /jobs/<id>/reopen``.

Job types: ``full-time``, ``part-time``, ``contract``, ``internship``,
``remote``. New jobs start ``open``; closing a job blocks new
applications, reopening allows them again.

Applications — ``POST /jobs/<id>/applications``
(``{"applicant_name": str, "applicant_email": str, "cover_letter": str?}``);
``GET /jobs/<id>/applications``; ``GET /applications/<id>``;
``POST /applications/<id>/status`` (``{"status": str}``).

Application statuses: ``submitted``, ``reviewed``, ``shortlisted``,
``rejected``, ``hired``. Transitions are forward only::

    submitted -> reviewed | rejected
    reviewed -> shortlisted | rejected
    shortlisted -> hired | rejected
    rejected | hired -> (terminal)

One application per email per job; a second one returns 409.
"""
from flask import Flask, jsonify, request

JOB_TYPES = ("full-time", "part-time", "contract", "internship", "remote")
JOB_STATUSES = ("open", "closed")
APPLICATION_STATUSES = ("submitted", "reviewed", "shortlisted", "rejected", "hired")
ALLOWED_TRANSITIONS = {
    "submitted": ("reviewed", "rejected"),
    "reviewed": ("shortlisted", "rejected"),
    "shortlisted": ("hired", "rejected"),
    "rejected": (),
    "hired": (),
}


def create_app():
    """Return a Flask app with a fresh, empty in-memory job store."""
    app = Flask(__name__)
    app.config["JOBS"] = {}
    app.config["APPLICATIONS"] = {}
    app.config["NEXT_JOB_ID"] = 1
    app.config["NEXT_APPLICATION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_job(job_id):
        job = app.config["JOBS"].get(job_id)
        if job is None:
            return None, _error(f"job {job_id} not found", 404)
        return job, None

    def _get_application(application_id):
        application = app.config["APPLICATIONS"].get(application_id)
        if application is None:
            return None, _error(f"application {application_id} not found", 404)
        return application, None

    def _validate_salary(data, field):
        value = data.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            return _error(f"{field} must be a non-negative number", 400)
        return None

    def _validate_job_payload(data, partial=False):
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        for field in ("title", "company", "location"):
            if not partial or field in data:
                value = data.get(field)
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        f"{field} is required and must be a non-empty string", 400
                    )
        if not partial or "job_type" in data:
            if data.get("job_type") not in JOB_TYPES:
                return _error(
                    f"job_type must be one of {', '.join(JOB_TYPES)}", 400
                )
        for field in ("salary_min", "salary_max"):
            err = _validate_salary(data, field)
            if err is not None:
                return err
        salary_min = data.get("salary_min")
        salary_max = data.get("salary_max")
        if (
            salary_min is not None
            and salary_max is not None
            and salary_max < salary_min
        ):
            return _error("salary_max must not be less than salary_min", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        return None

    @app.post("/jobs")
    def create_job():
        data = request.get_json(silent=True)
        err = _validate_job_payload(data)
        if err is not None:
            return err
        job_id = app.config["NEXT_JOB_ID"]
        app.config["NEXT_JOB_ID"] += 1
        job = {
            "id": job_id,
            "title": data["title"].strip(),
            "company": data["company"].strip(),
            "location": data["location"].strip(),
            "job_type": data["job_type"],
            "salary_min": data.get("salary_min"),
            "salary_max": data.get("salary_max"),
            "description": data.get("description", ""),
            "status": "open",
        }
        app.config["JOBS"][job_id] = job
        return jsonify(job), 201

    @app.get("/jobs")
    def list_jobs():
        jobs = list(app.config["JOBS"].values())
        location = request.args.get("location")
        if location is not None:
            jobs = [j for j in jobs if j["location"].lower() == location.lower()]
        job_type = request.args.get("type")
        if job_type is not None:
            if job_type not in JOB_TYPES:
                return _error(
                    f"type filter must be one of {', '.join(JOB_TYPES)}", 400
                )
            jobs = [j for j in jobs if j["job_type"] == job_type]
        status = request.args.get("status")
        if status is not None:
            if status not in JOB_STATUSES:
                return _error(
                    f"status filter must be one of {', '.join(JOB_STATUSES)}", 400
                )
            jobs = [j for j in jobs if j["status"] == status]
        company = request.args.get("company")
        if company is not None:
            jobs = [j for j in jobs if j["company"].lower() == company.lower()]
        return jsonify(jobs)

    @app.get("/jobs/<int:job_id>")
    def get_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        return jsonify(job)

    @app.put("/jobs/<int:job_id>")
    def update_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_job_payload(data, partial=True)
        if err is not None:
            return err
        for field in ("title", "company", "location"):
            if field in data:
                job[field] = data[field].strip()
        if "job_type" in data:
            job["job_type"] = data["job_type"]
        if "salary_min" in data:
            job["salary_min"] = data["salary_min"]
        if "salary_max" in data:
            job["salary_max"] = data["salary_max"]
        if (
            job["salary_min"] is not None
            and job["salary_max"] is not None
            and job["salary_max"] < job["salary_min"]
        ):
            return _error("salary_max must not be less than salary_min", 400)
        if "description" in data:
            job["description"] = data["description"]
        return jsonify(job)

    @app.delete("/jobs/<int:job_id>")
    def delete_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        del app.config["JOBS"][job_id]
        for application_id in [
            a_id
            for a_id, a in app.config["APPLICATIONS"].items()
            if a["job_id"] == job_id
        ]:
            del app.config["APPLICATIONS"][application_id]
        return jsonify({"deleted": job_id})

    @app.post("/jobs/<int:job_id>/close")
    def close_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        if job["status"] == "closed":
            return _error("job is already closed", 409)
        job["status"] = "closed"
        return jsonify(job)

    @app.post("/jobs/<int:job_id>/reopen")
    def reopen_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        if job["status"] == "open":
            return _error("job is already open", 409)
        job["status"] = "open"
        return jsonify(job)

    @app.post("/jobs/<int:job_id>/applications")
    def apply_for_job(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        if job["status"] == "closed":
            return _error("job is closed and no longer accepts applications", 409)
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
        cover_letter = data.get("cover_letter", "")
        if not isinstance(cover_letter, str):
            return _error("cover_letter must be a string", 400)
        for existing in app.config["APPLICATIONS"].values():
            if (
                existing["job_id"] == job_id
                and existing["applicant_email"].lower() == email.strip().lower()
            ):
                return _error(
                    "this email has already applied for this job", 409
                )
        application_id = app.config["NEXT_APPLICATION_ID"]
        app.config["NEXT_APPLICATION_ID"] += 1
        application = {
            "id": application_id,
            "job_id": job_id,
            "applicant_name": applicant_name.strip(),
            "applicant_email": email.strip(),
            "cover_letter": cover_letter,
            "status": "submitted",
        }
        app.config["APPLICATIONS"][application_id] = application
        return jsonify(application), 201

    @app.get("/jobs/<int:job_id>/applications")
    def list_job_applications(job_id):
        job, err = _get_job(job_id)
        if err is not None:
            return err
        return jsonify(
            [a for a in app.config["APPLICATIONS"].values() if a["job_id"] == job_id]
        )

    @app.get("/applications/<int:application_id>")
    def get_application(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        return jsonify(application)

    @app.post("/applications/<int:application_id>/status")
    def set_application_status(application_id):
        application, err = _get_application(application_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        new_status = data.get("status")
        if new_status not in APPLICATION_STATUSES:
            return _error(
                f"status must be one of {', '.join(APPLICATION_STATUSES)}", 400
            )
        if new_status not in ALLOWED_TRANSITIONS[application["status"]]:
            return _error(
                f"cannot move application from {application['status']} "
                f"to {new_status}",
                409,
            )
        application["status"] = new_status
        return jsonify(application)

    return app
