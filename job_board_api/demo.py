"""Demo: post jobs, apply, review an application, close and reopen a posting."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_board_api.app import create_app


def main():
    client = create_app().test_client()

    backend = client.post(
        "/jobs",
        json={
            "title": "Associate Software Engineer",
            "company": "NovaStack",
            "location": "Lahore",
            "job_type": "full-time",
            "salary_min": 70000,
            "salary_max": 120000,
        },
    ).get_json()
    print(
        f"Posted job #{backend['id']}: {backend['title']} at {backend['company']} "
        f"({backend['location']}, {backend['job_type']}) — {backend['status']}"
    )

    remote = client.post(
        "/jobs",
        json={
            "title": "AI/ML Engineer",
            "company": "TechCorp",
            "location": "Remote",
            "job_type": "remote",
        },
    ).get_json()
    print(f"Posted job #{remote['id']}: {remote['title']} at {remote['company']}")

    lahore_jobs = client.get("/jobs", query_string={"location": "Lahore"}).get_json()
    print(f"Jobs in Lahore: {[j['title'] for j in lahore_jobs]}")

    application = client.post(
        f"/jobs/{backend['id']}/applications",
        json={
            "applicant_name": "Usman Wajid",
            "applicant_email": "usman@example.com",
            "cover_letter": "7th-semester CS student, MERN + AI/ML experience.",
        },
    ).get_json()
    print(
        f"Application #{application['id']} for job #{backend['id']} "
        f"— status {application['status']}"
    )

    for step in ("reviewed", "shortlisted", "hired"):
        moved = client.post(
            f"/applications/{application['id']}/status", json={"status": step}
        ).get_json()
        print(f"Application #{application['id']} -> {moved['status']}")

    closed = client.post(f"/jobs/{backend['id']}/close").get_json()
    print(f"Job #{backend['id']} status: {closed['status']}")

    late = client.post(
        f"/jobs/{backend['id']}/applications",
        json={"applicant_name": "Late", "applicant_email": "late@example.com"},
    )
    print(f"Applying to a closed job: {late.status_code} "
          f"({late.get_json()['error']})")

    reopened = client.post(f"/jobs/{backend['id']}/reopen").get_json()
    print(f"Job #{backend['id']} status after reopen: {reopened['status']}")


if __name__ == "__main__":
    main()
