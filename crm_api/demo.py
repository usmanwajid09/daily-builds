"""Demo: add contacts and deals, move them through the pipeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crm_api.app import create_app


def main():
    client = create_app().test_client()

    contacts = []
    for name, email, company in (
        ("Sara Ahmed", "sara@acme.com", "Acme Corp"),
        ("Omar Khan", "omar@globex.com", "Globex"),
    ):
        contact = client.post(
            "/contacts",
            json={"name": name, "email": email, "company": company},
        ).get_json()
        contacts.append(contact)
        print(f"Created contact #{contact['id']}: {contact['name']} "
              f"({contact['email']}, {contact['company']})")

    deals = []
    for contact, title, value, stage in (
        (contacts[0], "Website redesign", 5000, "lead"),
        (contacts[0], "Mobile app", 12000, "qualified"),
        (contacts[1], "Support contract", 3000, "proposal"),
    ):
        deal = client.post(
            f"/contacts/{contact['id']}/deals",
            json={"title": title, "value": value, "stage": stage},
        ).get_json()
        deals.append(deal)
        print(f"Added deal #{deal['id']}: {deal['title']!r} "
              f"${deal['value']} ({deal['stage']})")

    won = client.post(
        f"/deals/{deals[2]['id']}/stage", json={"stage": "won"}
    ).get_json()
    print(f"Moved {won['title']!r} to {won['stage']}")
    lost = client.post(
        f"/deals/{deals[0]['id']}/stage", json={"stage": "lost"}
    ).get_json()
    print(f"Moved {lost['title']!r} to {lost['stage']}")

    reopen = client.post(
        f"/deals/{deals[2]['id']}/stage", json={"stage": "proposal"}
    )
    print(f"Reopening a won deal rejected: "
          f"{reopen.status_code} {reopen.get_json()}")

    pipeline = client.get("/pipeline").get_json()
    print(f"Pipeline: {pipeline['total_deals']} deals, "
          f"total value ${pipeline['total_value']}, "
          f"conversion rate {pipeline['conversion_rate']}%")
    for stage, stats in pipeline["by_stage"].items():
        print(f"  {stage}: {stats['count']} deal(s), "
              f"${stats['total_value']}")


if __name__ == "__main__":
    main()
