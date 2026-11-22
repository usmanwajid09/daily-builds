# pet_adoption_api

A Flask API for a pet adoption shelter: manage pets and adoption
applications, with a status workflow of `available -> pending -> adopted`.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/pets` | Create a pet (`name`, `species`, `breed?`, `age?`, `description?`) |
| GET | `/pets` | List pets; filter by `?species=`, `?breed=`, `?status=` |
| GET | `/pets/<id>` | Fetch one pet |
| PUT | `/pets/<id>` | Update a pet |
| DELETE | `/pets/<id>` | Delete a pet (only while `available`) |
| POST | `/pets/<id>/applications` | Apply to adopt a pet |
| GET | `/pets/<id>/applications` | Applications for one pet |
| GET | `/applications` | All applications; filter by `?status=` |
| GET | `/applications/<id>` | Fetch one application |
| POST | `/applications/<id>/approve` | Approve: pet becomes `adopted`, other pending applications are rejected |
| POST | `/applications/<id>/reject` | Reject: pet returns to `available` when no pending application remains |

Species must be one of `dog`, `cat`, `bird`, `rabbit`, `fish`, `other`.
Applying for an available pet moves it to `pending`. The same email cannot
have two pending applications for the same pet (409). Validation errors
return 400, missing pets/applications return 404, and illegal state changes
return 409, all as `{"error": "..."}`.

## How to run

```bash
python pet_adoption_api/demo.py
```

Tests:

```bash
python -m pytest pet_adoption_api/tests/test_pet_adoption_api.py -v
```

## Demo output

```
Created pet #1: Buddy (dog, Labrador, age 3) — status available
Created pet #2: Whiskers (cat)
Search species=dog returned 1 pet(s): ['Buddy']
Application #1 for Buddy — status pending
Pet #1 status after application: pending
Application #1 -> approved
Pet #1 final status: adopted
Still available: ['Whiskers']
```
