# music_playlist_api

A small Flask music-playlist API: playlists with ordered tracks, total
duration, and a public/private flag — kept in memory (a fresh store
per `create_app()` call, nothing written to disk).

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| POST | `/playlists` | Create a playlist: `{"name", "description"?, "public"?}` |
| GET | `/playlists` | List playlists; `?public=true\|false` filter |
| GET | `/playlists/<id>` | Fetch one playlist with its tracks in order, `track_count` and `total_duration_seconds` |
| PUT | `/playlists/<id>` | Update name, description and/or public |
| DELETE | `/playlists/<id>` | Delete a playlist and its tracks |
| POST | `/playlists/<id>/tracks` | Add a track: `{"title", "artist", "duration_seconds", "album"?}` |
| GET | `/playlists/<id>/tracks` | List a playlist's tracks in order |
| GET | `/playlists/<id>/tracks/<track_id>` | Fetch one track |
| PUT | `/playlists/<id>/tracks/<track_id>` | Update a track, or reorder it with `{"position": int}` (0-based) |
| DELETE | `/playlists/<id>/tracks/<track_id>` | Remove a track (remaining tracks re-index) |
| GET | `/playlists/<id>/summary` | `{"playlist_id", "track_count", "total_duration_seconds", "total_duration"}` (`"MM:SS"` or `"H:MM:SS"`) |

Tracks are appended at the end; `PUT .../tracks/<id>` with a
`position` moves a track and re-indexes the rest. Adding (or renaming
to) the same title + artist twice in one playlist returns JSON `409`.

Validation: a missing/empty `name`, `title` or `artist`, a
non-positive or non-integer `duration_seconds`, a non-boolean
`public`, or a `position` outside the playlist returns JSON `400`;
an unknown playlist or track returns JSON `404`.

## Run

```bash
python music_playlist_api/demo.py
python -m pytest tests/test_music_playlist_api.py -v
```
