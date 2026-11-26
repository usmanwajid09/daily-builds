"""Demo: three students, a few marks each, and the attendance report."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attendance_api.app import create_app


def main():
    client = create_app().test_client()

    students = {}
    for name in ("Ayesha", "Bilal", "Chen"):
        created = client.post("/students", json={"name": name}).get_json()
        students[name] = created["id"]
        print(f"Created student: {created['name']} (id {created['id']})")

    marks = {
        "Ayesha": [("2026-11-23", "present"), ("2026-11-24", "present"),
                   ("2026-11-25", "absent")],
        "Bilal": [("2026-11-23", "present"), ("2026-11-24", "late"),
                  ("2026-11-25", "present")],
        "Chen": [("2026-11-23", "absent"), ("2026-11-24", "absent")],
    }
    for name, days in marks.items():
        for day, status in days:
            resp = client.post(
                f"/students/{students[name]}/marks",
                json={"date": day, "status": status},
            )
            assert resp.status_code == 201, resp.get_json()

    for name in ("Ayesha", "Bilal", "Chen"):
        report = client.get(
            f"/students/{students[name]}/attendance"
        ).get_json()
        print(
            f"{name}: {report['present']} present, {report['late']} late, "
            f"{report['absent']} absent over {report['total_days']} days -> "
            f"{report['attendance_percentage']}% attendance"
        )

    summary = client.get("/summary").get_json()
    print(
        f"Class summary: {summary['total_students']} students, "
        f"{summary['total_marks']} marks, "
        f"overall {summary['overall_attendance_percentage']}% attendance"
    )

    dup = client.post(
        f"/students/{students['Ayesha']}/marks",
        json={"date": "2026-11-23", "status": "present"},
    )
    print(f"Duplicate mark rejected: {dup.status_code} {dup.get_json()}")
    bad = client.post(
        f"/students/{students['Ayesha']}/marks",
        json={"date": "2026-11-26", "status": "half-day"},
    )
    print(f"Invalid status rejected: {bad.status_code} {bad.get_json()}")


if __name__ == "__main__":
    main()
