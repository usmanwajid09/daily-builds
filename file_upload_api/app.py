"""File upload API built with Flask.

Uploaded files are kept in memory only (nothing is written to disk).
All responses are JSON except downloads, which return the raw bytes.
Endpoints:

- ``POST /files`` — upload one file (multipart field ``file``).
  Allowed extensions: txt, md, csv, pdf, png, jpg, jpeg, gif.
  Max size: 1 MB.
- ``GET /files`` — list uploaded files (metadata only)
- ``GET /files/<id>`` — download a file by id
- ``DELETE /files/<id>`` — delete a file
"""
import io
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file

ALLOWED_EXTENSIONS = {"txt", "md", "csv", "pdf", "png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB


def create_app():
    """Return a Flask app with a fresh, empty in-memory file store."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
    app.config["FILES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_file(file_id):
        record = app.config["FILES"].get(file_id)
        if record is None:
            return None, _error(f"file {file_id} not found", 404)
        return record, None

    @app.post("/files")
    def upload_file():
        if "file" not in request.files:
            return _error("no file part in the request (field name: 'file')", 400)
        uploaded = request.files["file"]
        if not uploaded.filename:
            return _error("no file selected", 400)
        filename = os.path.basename(uploaded.filename)
        if "." not in filename:
            return _error("file must have an extension", 400)
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _error(
                "extension not allowed; allowed: "
                + ", ".join(sorted(ALLOWED_EXTENSIONS)),
                400,
            )
        content = uploaded.read()
        if len(content) > MAX_FILE_SIZE:
            return _error("file is too large (max 1 MB)", 413)
        file_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        record = {
            "id": file_id,
            "filename": filename,
            "size": len(content),
            "mimetype": uploaded.mimetype or "application/octet-stream",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        app.config["FILES"][file_id] = record
        return (
            jsonify({k: v for k, v in record.items() if k != "content"}),
            201,
        )

    @app.get("/files")
    def list_files():
        files = sorted(app.config["FILES"].values(), key=lambda f: f["id"])
        return jsonify(
            [{k: v for k, v in f.items() if k != "content"} for f in files]
        )

    @app.get("/files/<int:file_id>")
    def download_file(file_id):
        record, err = _get_file(file_id)
        if err is not None:
            return err
        return send_file(
            io.BytesIO(record["content"]),
            mimetype=record["mimetype"],
            as_attachment=True,
            download_name=record["filename"],
        )

    @app.delete("/files/<int:file_id>")
    def delete_file(file_id):
        _record, err = _get_file(file_id)
        if err is not None:
            return err
        del app.config["FILES"][file_id]
        return jsonify({"deleted": file_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    @app.errorhandler(413)
    def handle_413(exc):
        return _error("file is too large (max 1 MB)", 413)

    return app
