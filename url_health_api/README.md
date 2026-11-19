# URL Health API

A tiny URL health-checker API built with Flask. A URL is checked
with the standard library's `urllib` (a HEAD request, with HTTP
error responses still counting as reachable) and the result — status
code, whether it is OK, and response time in milliseconds — is
returned as JSON and kept in an in-memory history (fresh on every
app start — nothing is written to disk).

## Usage

```bash
pip install -r requirements.txt
python -m url_health_api.demo
```

The demo starts a tiny local HTTP server, checks it (200), then
checks an unreachable host (`http://127.0.0.1:1/`) and prints the
clean 502 JSON error, then prints how many successful checks are in
history. No public-internet access is needed.

To run the app for real:

```bash
flask --app "url_health_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/check` | Check a URL: `{"url": str}` |
| GET | `/checks` | List past successful checks |
| GET | `/checks/<id>` | One past check |

A check looks like `{"id": 1, "url": "http://example.com/",
"status": 200, "ok": true, "response_time_ms": 42.1}`. Validation
errors (missing URL, non-http(s) scheme, no host) return `400`;
unreachable hosts return `502` with `{"error": "..."}`; unknown
check ids return `404`.

## Run tests

```bash
python -m pytest tests/test_url_health_api.py -v
```
