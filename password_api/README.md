# Password API

A password utility API built with Flask: cryptographically random
password generation and a password-strength checker. Nothing is ever
stored — both endpoints are pure functions over the request.

## Why

Password generation and strength feedback are small, self-contained
utilities that show up everywhere (signup forms, admin tools, security
audits), and they are a good excuse to use `secrets` instead of
`random`.

## Usage

```bash
pip install -r requirements.txt
python -m password_api.demo
```

The demo generates a 16-character password, scores the weak password
`password`, then scores the generated one. To run the app for real:

```bash
flask --app "password_api.app:create_app" run
```

## API

| Method | Route | Description |
| ------ | ----- | ----------- |
| POST | `/generate` | `{"length"?, "symbols"?, "digits"?, "uppercase"?, "lowercase"?}` → `{"password", "length"}` |
| POST | `/strength` | `{"password"}` → `{"score", "label", "feedback"}` |

Generation defaults to length 16 with all character classes on.
`length` must be an integer from 4 to 128, and at least one character
class must be enabled — otherwise `400` with `{"error": "..."}`. The
strength score is 0–5 (one point each for length ≥ 8, length ≥ 12,
mixed case, a digit, a symbol); labels are Weak / Fair / Good / Strong.

## Run tests

```bash
python -m pytest tests/test_password_api.py -v
```
