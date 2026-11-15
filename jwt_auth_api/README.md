# jwt_auth_api

A small Flask JWT authentication API. Users register with a password
hashed by bcrypt; logging in returns a PyJWT token (HS256, valid for
1 hour) that unlocks protected endpoints. All state lives in memory;
nothing is written to disk.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| POST | `/auth/register` | Register: `{"username": str, "password": str}` (password at least 6 chars). Duplicate usernames return `409`. |
| POST | `/auth/login` | Log in: returns `{"token": "<jwt>"}`. Wrong credentials return `401`. |
| GET | `/auth/protected` | Requires `Authorization: Bearer <token>`. |
| GET | `/auth/me` | Requires a valid token; returns the logged-in username. |

Missing, malformed, or invalid tokens return `401` with a JSON
`{"error": ...}` message; expired tokens return `401` with
`"token has expired"`.

## Run

```bash
python jwt_auth_api/demo.py
python -m pytest tests/test_jwt_auth_api.py -q
```

## Example

```
Registered: {'message': 'registered', 'username': 'usman'}
Protected without token -> 401: {'error': 'missing or malformed Authorization header'}
Login token (first 30 chars): eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...
Protected with token: {'message': 'hello, usman — you reached a protected route'}
```
