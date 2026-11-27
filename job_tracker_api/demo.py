"""Demo: track job applications through the hiring pipeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_tracker_api.app import create_app


def main():
    client = create_app().test_client()

    applications = []
    for company, role, salary in (
        ("Acme Corp", "Associate Software Engineer", 90000),
        ("Globex", "AI/ML Engineer", 120000),
        ("Initech", "Web Developer", 80000),
        ("Umbrella", "Backend Developer", 100000),
    ):
        application = client.post(
            "/applications",
            json={"company": company, "role": role, "salary": salary,
                  "applied_date": "2026-11-20"},
        ).get_json()
        applications.append(application)
        print(f"Applied #{application['id']}: {application['role']} "
              f"at {application['company']} "
              f"(PKR {application['salary']:,}/month)")

    for app_id, status, interview_date in (
        (applications[0]["id"], "screening", None),
        (applications[1]["id"], "interview", "2026-12-05"),
        (applications[2]["id"], "offer", None),
        (applications[3]["id"], "rejected", None),
    ):
        payload = {"status": status}
        resp = client.post(
            f"/applications/{app_id}/status", json=payload
        )
        updated = resp.get_json()
        if interview_date:
            client.put(
                f"/applications/{app_id}",
                json={"interview_date": interview_date},
            )
            updated = client.get(f"/applications/{app_id}").get_json()
        print(f"Moved #{app_id} to {updated['status']}"
              + (f", interview {updated['interview_date']}"
                 if updated["interview_date"] else ""))

    blocked = client.post(
        f"/applications/{applications[3]['id']}/status",
        json={"status": "interview"},
    )
    print(f"Moving a rejected application rejected: "
          f"{blocked.status_code} {blocked.get_json()}")

    stats = client.get("/stats").get_json()
    print(f"Stats: {stats['total_applications']} applications, "
          f"by status {stats['by_status']}, "
          f"offer rate {stats['offer_rate']}%, "
          f"interview rate {stats['interview_rate']}%")

    upcoming = client.get("/interviews/upcoming").get_json()
    print(f"Upcoming interviews: {len(upcoming)}")
    for item in upcoming:
        print(f"  {item['interview_date']}: {item['role']} "
              f"at {item['company']}")


if __name__ == "__main__":
    main()
