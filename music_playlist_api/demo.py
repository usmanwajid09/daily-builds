"""Demo: build a playlist, add tracks, reorder one, and read the summary."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_playlist_api.app import create_app


def main():
    client = create_app().test_client()

    playlist = client.post(
        "/playlists",
        json={"name": "Road Trip Mix", "description": "Driving songs",
              "public": True},
    ).get_json()
    print(f"Playlist #{playlist['id']}: {playlist['name']!r} "
          f"(public={playlist['public']})")

    songs = [
        ("Blinding Lights", "The Weeknd", 200, "After Hours"),
        ("Levitating", "Dua Lipa", 203, "Future Nostalgia"),
        ("As It Was", "Harry Styles", 167, "Harry's House"),
        ("Starboy", "The Weeknd", 230, "Starboy"),
    ]
    track_ids = {}
    for title, artist, duration, album in songs:
        track = client.post(
            f"/playlists/{playlist['id']}/tracks",
            json={"title": title, "artist": artist,
                  "duration_seconds": duration, "album": album},
        ).get_json()
        track_ids[title] = track["id"]
        print(f"  Added #{track['id']} {track['title']!r} by {track['artist']} "
              f"({track['duration_seconds']}s) at position {track['position']}")

    summary = client.get(f"/playlists/{playlist['id']}/summary").get_json()
    print(f"Summary: {summary['track_count']} tracks, "
          f"{summary['total_duration_seconds']}s total "
          f"({summary['total_duration']})")

    moved = client.put(
        f"/playlists/{playlist['id']}/tracks/{track_ids['Starboy']}",
        json={"position": 0},
    ).get_json()
    print(f"Moved {moved['title']!r} to position {moved['position']}")
    order = [t["title"] for t in
             client.get(f"/playlists/{playlist['id']}/tracks").get_json()]
    print(f"New order: {order}")

    duplicate = client.post(
        f"/playlists/{playlist['id']}/tracks",
        json={"title": "Levitating", "artist": "Dua Lipa",
              "duration_seconds": 203},
    )
    print(f"Duplicate track rejected: {duplicate.status_code} "
          f"{duplicate.get_json()}")

    removed = client.delete(
        f"/playlists/{playlist['id']}/tracks/{track_ids['As It Was']}")
    print(f"Removed 'As It Was': {removed.status_code} {removed.get_json()}")
    summary = client.get(f"/playlists/{playlist['id']}/summary").get_json()
    print(f"After removal: {summary['track_count']} tracks, "
          f"{summary['total_duration_seconds']}s total "
          f"({summary['total_duration']})")

    private = client.post(
        "/playlists", json={"name": "Secret Jams"}).get_json()
    public_only = client.get(
        "/playlists", query_string={"public": "true"}).get_json()
    print(f"Public playlists: {len(public_only)} "
          f"({[p['name'] for p in public_only]})")


if __name__ == "__main__":
    main()
