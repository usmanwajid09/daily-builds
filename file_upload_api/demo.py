"""Demo: upload, list, download, and delete a file via test_client."""
import io
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_upload_api.app import create_app


def main():
    client = create_app().test_client()

    uploaded = client.post(
        "/files",
        data={"file": (io.BytesIO(b"hello, daily-builds!\n"), "greeting.txt")},
        content_type="multipart/form-data",
    ).get_json()
    print(f"Uploaded: {uploaded}")
    file_id = uploaded["id"]

    rejected = client.post(
        "/files",
        data={"file": (io.BytesIO(b"bad"), "malware.exe")},
        content_type="multipart/form-data",
    )
    print(f"Upload .exe -> {rejected.status_code}: {rejected.get_json()}")

    listed = client.get("/files").get_json()
    print(f"Files listed: {listed}")

    downloaded = client.get(f"/files/{file_id}")
    print(f"Downloaded {downloaded.status_code}: {downloaded.data!r}")

    deleted = client.delete(f"/files/{file_id}").get_json()
    print(f"Deleted: {deleted}")
    print(f"Files left: {client.get('/files').get_json()}")


if __name__ == "__main__":
    main()
