"""End-to-end resume analysis tests -- real O*NET baseline, real sentence-
transformer model, no network/API keys required (skips the Adzuna-backed
posting boost in skill_matcher.get_boosted_skill_profile; see its module
docstring). Complements test_user_input.py, which deliberately stays
model-free and only checks routing/validation."""

from io import BytesIO

from resume_analysis import DEMONSTRATED, NO_EVIDENCE, WEAK, analyze_resume, classify_skill_gaps
from skill_matcher import get_occupation_baseline, resolve_occupation

# A software-example category (see skill_matcher.build_occupation_skill_index)
# -- Python/SQL/R/etc. all embed under this O*NET skill name for Data Scientists.
PROGRAMMING_CATEGORY = "Object or component oriented development software"
# Present in the baseline but never mentioned by either sample resume below.
UNMENTIONED_SKILL = "Negotiation"

STRONG_RESUME = """
Designed and built Python programs for machine learning.
Wrote SQL queries to analyze large production datasets.
Automated data pipelines and deployed them with Docker on AWS.
Developed dashboards in Tableau for stakeholder reporting.
Led a complex problem solving initiative that reduced customer churn by 18 percent.
Authored technical documentation and presented results at quarterly reviews.
"""

WEAK_RESUME = """
Familiar with Python from university coursework.
Some exposure to SQL during a class project.
Currently learning Tableau and excited to grow in data science.
Basic understanding of statistics from an introductory course.
"""


class _FakeUpload:
    """Stands in for FastAPI's UploadFile -- analyze_resume only reads
    .filename and .file.read()."""

    def __init__(self, filename: str, text: str):
        self.filename = filename
        self.file = BytesIO(text.encode())


def _baseline():
    title = resolve_occupation("data scientist")
    onet_code, skills = get_occupation_baseline(title)
    return onet_code, skills


def test_strong_resume_shows_demonstrated_evidence():
    onet_code, skills = _baseline()
    result = analyze_resume(_FakeUpload("resume.txt", STRONG_RESUME), onet_code, skills)

    assert result["status"] == "ok"
    programming = result["skills"][PROGRAMMING_CATEGORY]
    assert programming["indication_level"] == DEMONSTRATED
    assert programming["presence_score"] > 0.35
    assert all(0.0 <= s["presence_score"] <= 1.0 for s in result["skills"].values())

    # never mentioned -> no evidence at all
    assert result["skills"][UNMENTIONED_SKILL]["indication_level"] == NO_EVIDENCE
    assert result["skills"][UNMENTIONED_SKILL]["presence_score"] == 0.0


def test_weak_resume_scores_lower_than_strong_resume():
    onet_code, skills = _baseline()
    strong = analyze_resume(_FakeUpload("strong.txt", STRONG_RESUME), onet_code, skills)
    weak = analyze_resume(_FakeUpload("weak.txt", WEAK_RESUME), onet_code, skills)

    weak_programming = weak["skills"][PROGRAMMING_CATEGORY]
    strong_programming = strong["skills"][PROGRAMMING_CATEGORY]

    # aspirational/coursework language caps the tone at WEAK regardless of
    # how well "familiar with Python" matches the Python embedding
    assert weak_programming["indication_level"] == WEAK
    assert weak_programming["presence_score"] < strong_programming["presence_score"]


def test_gap_classification_reflects_real_presence_scores():
    onet_code, skills = _baseline()
    result = analyze_resume(_FakeUpload("resume.txt", STRONG_RESUME), onet_code, skills)
    presence = result["skills"]

    # stand in for skill_matcher.get_boosted_skill_profile's output without
    # the network-dependent Adzuna posting boost -- raw O*NET importance only
    boosted_skills = [{"skill_name": name, "boosted_importance": importance} for name, importance, *_ in skills]
    classified = classify_skill_gaps(boosted_skills, presence)

    categories = {s["skill_name"]: s["category"] for s in classified}
    assert set(categories.values()) <= {"missing", "need_work", "deemed_enough"}

    # strongly evidenced skill should never come back as an outright gap
    assert categories[PROGRAMMING_CATEGORY] != "missing"

    # display order: missing block, then need_work, then deemed_enough
    order_rank = {"missing": 0, "need_work": 1, "deemed_enough": 2}
    order = [order_rank[c] for c in categories.values()]
    assert order == sorted(order)


if __name__ == "__main__":
    test_strong_resume_shows_demonstrated_evidence()
    test_weak_resume_scores_lower_than_strong_resume()
    test_gap_classification_reflects_real_presence_scores()
    print("ok: end-to-end resume analysis against real O*NET baseline + model")
