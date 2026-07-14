# fintech-app -- self-review log

## Milestone 1 (2026-07-14): mock data model + categorization engine

**Real bug found and fixed:** `POST /api/accounts` and the category
fields on `POST /api/accounts/<id>/transactions` /
`PATCH /api/transactions/<id>` did not normalize case before
validating against the allowed `account_type` / category sets, while
`currency` right next to them *was* uppercased. A client sending
`{"account_type": "Checking"}` or `{"category": "Dining"}` -- both
reasonable inputs, and the exact casing a naive frontend would send if
it title-cased a dropdown label -- got a spurious 400 even though the
value was valid modulo case. Verified the bug reproduced against the
pre-fix code (`account_type: "Checking"` -> 400, `category: "Dining"`
-> 400), then fixed by `.lower()`-ing both fields before validation in
`account_routes.py` and `transaction_routes.py`, matching the
`currency` field's existing normalize-then-validate pattern right
above it. Added three regression tests
(`test_create_account_normalizes_type_case`,
`test_create_transaction_normalizes_category_case`,
`test_patch_transaction_normalizes_category_case`) that fail against
the pre-fix source and pass after.

**Design decisions worth recording (not bugs):**

- **Balances are always derived, never stored.** `account_balance()`
  sums `transactions.amount` on every read rather than maintaining a
  running balance column. Slightly more query cost, zero risk of the
  stored value drifting out of sync with the transaction log -- the
  same trade-off this repo's other arcs made for similar derived
  values (e.g. `lms_platform`'s `percent_complete`).
- **Manual category is sticky.** `category_is_manual` is set whenever
  a category comes from the caller (either at transaction-creation
  time or via `PATCH`), and `POST /api/demo/recategorize` explicitly
  skips any row with that flag set. This means the categorization
  engine can be rerun after a keyword-rule change without silently
  overwriting a user's manual correction -- verified by
  `test_patch_transaction_recategorize_is_sticky`.
- **`mock_data.py` determinism is scoped to `(seed, today)`, not just
  `seed`.** Documented explicitly in the README after this review
  caught the original wording ("same user always gets the same demo
  data") overstating it -- `POST /api/demo/seed` doesn't pin `today`,
  so the exact transaction list shifts as the "N months back from
  today" window moves. The underlying generator function *is*
  byte-identical for a fixed `(account_type, seed, months, today)`,
  which is what's actually tested
  (`test_generate_mock_transactions_is_deterministic`) and what
  matters for reproducible unit tests.
- **`categorize()` never lets the amount sign override an explicit
  keyword match.** A negative-amount transaction described as a
  "refund" is *not* special-cased to income, because there's no
  "refund" keyword rule -- it falls back to `other`, exactly as the
  fallback logic dictates. This was deliberate (see the docstring and
  `test_refund_description_is_categorized_by_keyword_not_sign`) rather
  than an oversight: adding sign-aware heuristics on top of the
  keyword rules would make the engine's behavior harder to predict for
  a marginal accuracy gain that milestone 1's mock data doesn't even
  exercise.
- **Ownership checks return 404, not 403,** on another user's account
  or transaction (`test_cannot_access_another_users_account`,
  `test_cannot_access_another_users_transaction`), consistent with
  every other multi-tenant arc in this repo.

**Not fixed, intentionally:** `posted_at` accepts any non-empty
string rather than being validated as a real ISO date. Since
`start_date`/`end_date` filtering and the demo-seed sort both do plain
string comparison, a malformed date (e.g. `"01/02/2026"`) would sort
incorrectly rather than error. Manually-entered transactions are the
only path that can introduce this (the mock generator always emits
`date.isoformat()`), and milestone 1's scope is the data model +
categorization engine, not full input hardening. Flagged here for
milestone 2, which will build the analytics dashboard that would
actually surface a bad sort order to a user.
