"""Flask music-playlist API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /playlists`` — create a playlist (``{\"name\": str,
  \"description\": str?, \"public\": bool?}``)
- ``GET /playlists`` — list playlists; ``?public=true|false`` filter
- ``GET /playlists/<id>`` — fetch one playlist (includes its tracks
  in order, ``track_count`` and ``total_duration_seconds``)
- ``PUT /playlists/<id>`` — update name, description and/or public
- ``DELETE /playlists/<id>`` — delete a playlist and its tracks
- ``POST /playlists/<id>/tracks`` — add a track (``{\"title\": str,
  \"artist\": str, \"duration_seconds\": int > 0, \"album\": str?}``);
  the same title + artist twice in one playlist is a 409
- ``GET /playlists/<id>/tracks`` — list a playlist's tracks in order
- ``GET /playlists/<id>/tracks/<track_id>`` — fetch one track
- ``PUT /playlists/<id>/tracks/<track_id>`` — update a track, or
  reorder it with ``{\"position\": int}`` (0-based)
- ``DELETE /playlists/<id>/tracks/<track_id>`` — remove a track
- ``GET /playlists/<id>/summary`` — ``{\"playlist_id\", \"track_count\",
  \"total_duration_seconds\", \"total_duration\"}`` where the last is
  a ``"MM:SS"`` / ``"H:MM:SS"`` string

A playlist is ``{\"id\": int, \"name\": str, \"description\": str,
\"public\": bool, \"tracks\": [...], \"track_count\": int,
\"total_duration_seconds\": int}``. A track is ``{\"id\": int,
\"playlist_id\": int, \"title\": str, \"artist\": str,
\"duration_seconds\": int, \"album\": str, \"position\": int}``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory store."""
    app = Flask(__name__)
    app.config["PLAYLISTS"] = {}
    app.config["TRACKS"] = {}
    app.config["NEXT_PLAYLIST_ID"] = 1
    app.config["NEXT_TRACK_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _require_json_object():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, _error("request body must be a JSON object", 400)
        return data, None

    def _non_empty_string(value):
        return isinstance(value, str) and bool(value.strip())

    def _playlist_view(playlist):
        tracks = sorted(
            (t for t in app.config["TRACKS"].values()
             if t["playlist_id"] == playlist["id"]),
            key=lambda t: t["position"],
        )
        view = dict(playlist)
        view["tracks"] = [dict(t) for t in tracks]
        view["track_count"] = len(tracks)
        view["total_duration_seconds"] = sum(
            t["duration_seconds"] for t in tracks)
        return view

    def _get_playlist(playlist_id):
        playlist = app.config["PLAYLISTS"].get(playlist_id)
        if playlist is None:
            return None, _error(f"playlist {playlist_id} not found", 404)
        return playlist, None

    def _get_track(playlist_id, track_id):
        track = app.config["TRACKS"].get(track_id)
        if track is None or track["playlist_id"] != playlist_id:
            return None, _error(
                f"track {track_id} not found in playlist {playlist_id}", 404)
        return track, None

    def _format_duration(seconds):
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _reindex(playlist_id):
        tracks = sorted(
            (t for t in app.config["TRACKS"].values()
             if t["playlist_id"] == playlist_id),
            key=lambda t: t["position"],
        )
        for index, track in enumerate(tracks):
            track["position"] = index

    # ---- playlists ---------------------------------------------------

    @app.post("/playlists")
    def create_playlist():
        data, err = _require_json_object()
        if err is not None:
            return err
        name = data.get("name")
        if not _non_empty_string(name):
            return _error("name is required and must be a non-empty string", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        public = data.get("public", False)
        if not isinstance(public, bool):
            return _error("public must be a boolean", 400)
        pid = app.config["NEXT_PLAYLIST_ID"]
        app.config["NEXT_PLAYLIST_ID"] += 1
        playlist = {
            "id": pid,
            "name": name.strip(),
            "description": description.strip(),
            "public": public,
        }
        app.config["PLAYLISTS"][pid] = playlist
        return jsonify(_playlist_view(playlist)), 201

    @app.get("/playlists")
    def list_playlists():
        playlists = list(app.config["PLAYLISTS"].values())
        public = request.args.get("public")
        if public is not None:
            if public.lower() not in ("true", "false"):
                return _error("public must be true or false", 400)
            wanted = public.lower() == "true"
            playlists = [p for p in playlists if p["public"] is wanted]
        return jsonify([_playlist_view(p) for p in playlists])

    @app.get("/playlists/<int:playlist_id>")
    def get_playlist(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        return jsonify(_playlist_view(playlist))

    @app.put("/playlists/<int:playlist_id>")
    def update_playlist(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        data, err = _require_json_object()
        if err is not None:
            return err
        if "name" in data:
            if not _non_empty_string(data["name"]):
                return _error("name must be a non-empty string", 400)
            playlist["name"] = data["name"].strip()
        if "description" in data:
            if not isinstance(data["description"], str):
                return _error("description must be a string", 400)
            playlist["description"] = data["description"].strip()
        if "public" in data:
            if not isinstance(data["public"], bool):
                return _error("public must be a boolean", 400)
            playlist["public"] = data["public"]
        return jsonify(_playlist_view(playlist))

    @app.delete("/playlists/<int:playlist_id>")
    def delete_playlist(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        for track_id in [t["id"] for t in app.config["TRACKS"].values()
                        if t["playlist_id"] == playlist_id]:
            del app.config["TRACKS"][track_id]
        del app.config["PLAYLISTS"][playlist_id]
        return jsonify({"deleted": playlist_id})

    @app.get("/playlists/<int:playlist_id>/summary")
    def playlist_summary(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        view = _playlist_view(playlist)
        total = view["total_duration_seconds"]
        return jsonify({
            "playlist_id": playlist_id,
            "track_count": view["track_count"],
            "total_duration_seconds": total,
            "total_duration": _format_duration(total),
        })

    # ---- tracks -------------------------------------------------------

    @app.post("/playlists/<int:playlist_id>/tracks")
    def add_track(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        data, err = _require_json_object()
        if err is not None:
            return err
        title = data.get("title")
        if not _non_empty_string(title):
            return _error("title is required and must be a non-empty string", 400)
        artist = data.get("artist")
        if not _non_empty_string(artist):
            return _error(
                "artist is required and must be a non-empty string", 400)
        duration = data.get("duration_seconds")
        if (not isinstance(duration, int) or isinstance(duration, bool)
                or duration <= 0):
            return _error(
                "duration_seconds is required and must be an integer > 0", 400)
        album = data.get("album", "")
        if not isinstance(album, str):
            return _error("album must be a string", 400)
        for existing in app.config["TRACKS"].values():
            if (existing["playlist_id"] == playlist_id
                    and existing["title"].lower() == title.strip().lower()
                    and existing["artist"].lower() == artist.strip().lower()):
                return _error(
                    f"track {title.strip()!r} by {artist.strip()} is already "
                    f"in playlist {playlist_id}", 409)
        position = len([t for t in app.config["TRACKS"].values()
                        if t["playlist_id"] == playlist_id])
        track_id = app.config["NEXT_TRACK_ID"]
        app.config["NEXT_TRACK_ID"] += 1
        track = {
            "id": track_id,
            "playlist_id": playlist_id,
            "title": title.strip(),
            "artist": artist.strip(),
            "duration_seconds": duration,
            "album": album.strip(),
            "position": position,
        }
        app.config["TRACKS"][track_id] = track
        return jsonify(track), 201

    @app.get("/playlists/<int:playlist_id>/tracks")
    def list_tracks(playlist_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        return jsonify(_playlist_view(playlist)["tracks"])

    @app.get("/playlists/<int:playlist_id>/tracks/<int:track_id>")
    def get_track(playlist_id, track_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        track, err = _get_track(playlist_id, track_id)
        if err is not None:
            return err
        return jsonify(track)

    @app.put("/playlists/<int:playlist_id>/tracks/<int:track_id>")
    def update_track(playlist_id, track_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        track, err = _get_track(playlist_id, track_id)
        if err is not None:
            return err
        data, err = _require_json_object()
        if err is not None:
            return err
        if "title" in data:
            if not _non_empty_string(data["title"]):
                return _error("title must be a non-empty string", 400)
            new_title = data["title"].strip()
            for existing in app.config["TRACKS"].values():
                if (existing["id"] != track_id
                        and existing["playlist_id"] == playlist_id
                        and existing["title"].lower() == new_title.lower()
                        and existing["artist"].lower()
                        == data.get("artist", track["artist"]).strip().lower()):
                    return _error(
                        f"track {new_title!r} by "
                        f"{data.get('artist', track['artist'])} is already "
                        f"in playlist {playlist_id}", 409)
            track["title"] = new_title
        if "artist" in data:
            if not _non_empty_string(data["artist"]):
                return _error("artist must be a non-empty string", 400)
            new_artist = data["artist"].strip()
            for existing in app.config["TRACKS"].values():
                if (existing["id"] != track_id
                        and existing["playlist_id"] == playlist_id
                        and existing["title"].lower()
                        == data.get("title", track["title"]).strip().lower()
                        and existing["artist"].lower() == new_artist.lower()):
                    return _error(
                        f"track {data.get('title', track['title'])!r} by "
                        f"{new_artist} is already in playlist {playlist_id}",
                        409)
            track["artist"] = new_artist
        if "duration_seconds" in data:
            duration = data["duration_seconds"]
            if (not isinstance(duration, int) or isinstance(duration, bool)
                    or duration <= 0):
                return _error("duration_seconds must be an integer > 0", 400)
            track["duration_seconds"] = duration
        if "album" in data:
            if not isinstance(data["album"], str):
                return _error("album must be a string", 400)
            track["album"] = data["album"].strip()
        if "position" in data:
            position = data["position"]
            siblings = [t for t in app.config["TRACKS"].values()
                        if t["playlist_id"] == playlist_id]
            if (not isinstance(position, int) or isinstance(position, bool)
                    or not 0 <= position < len(siblings)):
                return _error(
                    f"position must be an integer from 0 to "
                    f"{len(siblings) - 1}", 400)
            ordered = sorted(siblings, key=lambda t: t["position"])
            ordered.remove(track)
            ordered.insert(position, track)
            for index, sibling in enumerate(ordered):
                sibling["position"] = index
        return jsonify(track)

    @app.delete("/playlists/<int:playlist_id>/tracks/<int:track_id>")
    def delete_track(playlist_id, track_id):
        playlist, err = _get_playlist(playlist_id)
        if err is not None:
            return err
        track, err = _get_track(playlist_id, track_id)
        if err is not None:
            return err
        del app.config["TRACKS"][track_id]
        _reindex(playlist_id)
        return jsonify({"deleted": track_id})

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
