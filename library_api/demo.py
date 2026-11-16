"""Demo: add books, borrow and return copies, and list overdue loans."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library_api.app import create_app


def main():
    client = create_app().test_client()

    book = client.post(
        "/books",
        json={
            "title": "Dune",
            "author": "Frank Herbert",
            "isbn": "9780441172719",
            "total_copies": 2,
        },
    ).get_json()
    print(f"Added book: {book}")

    loan1 = client.post(
        f"/books/{book['id']}/borrow", json={"borrower": "Ayesha"}
    ).get_json()
    print(f"Borrowed: {loan1}")

    loan2 = client.post(
        f"/books/{book['id']}/borrow",
        json={"borrower": "Bilal", "due_days": 1},
    ).get_json()
    print(f"Borrowed second copy: {loan2}")

    no_copy = client.post(
        f"/books/{book['id']}/borrow", json={"borrower": "Sara"}
    )
    print(f"Third borrow rejected: {no_copy.status_code} {no_copy.get_json()}")

    book_now = client.get(f"/books/{book['id']}").get_json()
    print(f"Book now: {book_now}")

    returned = client.post(
        f"/books/{book['id']}/return", json={"loan_id": loan1["loan_id"]}
    ).get_json()
    print(f"Returned: {returned}")

    print(f"All loans: {client.get('/loans').get_json()}")
    print(f"Overdue loans: {client.get('/loans?overdue=true').get_json()}")


if __name__ == "__main__":
    main()
