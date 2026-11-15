# file_upload_api

A small Flask file-upload API. Uploaded files are kept in memory only
— nothing is written to disk. Uploads are validated by extension and
size (max 1 MB).

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| POST | `/files` | Upload one file (multipart field `file`). Allowed extensions: txt, md, csv, pdf, png, jpg, jpeg, gif. |
| GET | `/files` | List uploaded files (metadata only, no content). |
| GET | `/files/<id>` | Download a file by id. |
| DELETE | `/files/<id>` | Delete a file. |

Missing files, missing extensions, and disallowed extensions return
`400`; files over 1 MB return `413`; unknown ids return `404`. All
errors are JSON `{"error": ...}`.

## Run

```bash
python file_upload_api/demo.py
python -m pytest tests/test_file_upload_api.py -q
```

## Example

```
Uploaded: {'filename': 'greeting.txt', 'id': 1, 'mimetype': 'text/plain', 'size': 21, 'uploaded_at': '2026-11-15T09:50:00.000000+00:00'}
Upload .exe -> 400: {'error': 'extension not allowed; allowed: csv, gif, jpeg, jpg, md, pdf, png, txt'}
Downloaded 200: b'hello, daily-builds!\n'
```
