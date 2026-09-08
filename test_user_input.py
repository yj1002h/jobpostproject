"""Smoke test for the FastAPI routes / validation branches (no network/model calls)."""

from fastapi.testclient import TestClient

from user_input import app

client = TestClient(app)


def test_pages_render():
    assert client.get("/").status_code == 200
    assert client.get("/analyze").status_code == 200


def test_resolve_occupation():
    r = client.post("/api/resolve-occupation", data={"occupation": "data scientist"})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["candidates"]) <= 5
    assert body["candidates"][0]["title"] == "Data Scientists"
    # ranked by match score, best first
    scores = [c["match_score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)

    # missing occupation
    r = client.post("/api/resolve-occupation", data={})
    assert r.status_code == 422

    # gibberish with no plausible match
    r = client.post("/api/resolve-occupation", data={"occupation": "xzq!!qzx"})
    assert r.status_code == 400


def test_analyze_validation():
    # missing occupation/region
    r = client.post("/api/analyze", files={"resume": ("resume.pdf", b"x")})
    assert r.status_code in (400, 422)

    # missing resume
    r = client.post("/api/analyze", data={"occupation": "Data Scientist", "region": "Pittsburgh"})
    assert r.status_code == 400

    # bad resume extension
    r = client.post(
        "/api/analyze",
        data={"occupation": "Data Scientist", "region": "Pittsburgh"},
        files={"resume": ("resume.exe", b"x")},
    )
    assert r.status_code == 400

    print("ok: routes render and /api/analyze validation branches pass")


if __name__ == "__main__":
    test_pages_render()
    test_resolve_occupation()
    test_analyze_validation()
