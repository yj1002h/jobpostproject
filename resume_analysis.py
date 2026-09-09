"""
Resume parsing: extract text from an uploaded resume (PDF/DOCX/TXT) and score
every skill in the target occupation's O*NET baseline with a "presence
measure" -- how strongly the resume's own language backs up that skill.

Not limited to skills a keyword search would catch: every skill in the
occupation's baseline is scored against the resume via the same
sentence-embedding approach skill_matcher.py uses to compare postings
(skill_matcher.build_occupation_skill_index), so implicit or paraphrased
evidence counts too, not just an exact skill-name match.

presence_score(skill) = clip(sum over resume sentences that semantically
match the skill of indication_score(sentence) * confidence(sentence), 0, 1)

- confidence comes from cosine similarity between the sentence and the
  skill's O*NET-derived embed text, scaled 0..1 from MATCH_THRESHOLD to a
  perfect match -- the same linear scaling skill_matcher._boost uses, so a
  sentence just over the threshold barely counts and a near-exact paraphrase
  counts fully.
- indication_score comes from a light tone/cue heuristic on the sentence
  itself, since "interested in Python" and "built a Python ML pipeline" can
  be equally strong semantic matches to "Python" but are not equally good
  evidence -- aspirational phrasing is capped low regardless of similarity.
"""

import re
from io import BytesIO

import pandas as pd
from sentence_transformers import SentenceTransformer

from skill_matcher import MATCH_THRESHOLD, build_occupation_skill_index, get_model, split_sentences

# --- indication tiers -------------------------------------------------
# ponytail: cue lists are a hand-tuned heuristic, not a trained tone
# classifier -- upgrade to a small fine-tuned classifier if misreads show up
# on real resumes.
NO_EVIDENCE, WEAK, EXPLICIT, DEMONSTRATED = "no_evidence", "weak", "explicit", "demonstrated"
INDICATION_SCORES = {WEAK: 0.35, EXPLICIT: 0.7, DEMONSTRATED: 1.0}

_WEAK_CUES = [
    "interested in", "familiar with", "exposure to",
    r"some (?:knowledge|experience|understanding) (?:of|with|in)",
    r"basic (?:knowledge|understanding|familiarity)",
    r"coursework (?:in|on)", "introduction to", "aspiring",
    "eager to learn", "willing to learn", r"excited (?:about|to)",
    "currently learning", "studying",
]
_WEAK_CUE_RE = re.compile(r"\b(?:" + "|".join(_WEAK_CUES) + r")\b", re.IGNORECASE)

_DEMONSTRATED_VERBS = [
    "built", "build", "building", r"develop(?:ed|ing)?", r"design(?:ed|ing)?",
    r"implement(?:ed|ing)?", "engineered", r"engineer(?:ing)?", "architected",
    r"deploy(?:ed|ing)?", r"ship(?:ped|ping)?", r"launch(?:ed|ing)?",
    r"automat(?:ed|ing|e)", r"optimiz(?:ed|ing|e)", "led", r"lead(?:ing)?",
    r"manag(?:ed|ing|e)", r"creat(?:ed|ing|e)", r"deliver(?:ed|ing)?",
    r"reduc(?:ed|ing|e)", r"increas(?:ed|ing|e)", r"improv(?:ed|ing|e)",
    "founded", r"found(?:ing)?", r"establish(?:ed|ing)?", r"migrat(?:ed|ing|e)",
    r"scal(?:ed|ing|e)", "authored", r"author(?:ing)?", r"publish(?:ed|ing)?",
    r"integrat(?:ed|ing|e)", r"train(?:ed|ing)?", r"analyz(?:ed|ing|e)",
    "wrote", "written", "writing", r"refactor(?:ed|ing)?", r"debugg(?:ed|ing)",
    "debug", "tested", "testing",
]
_DEMONSTRATED_CONTEXT = [
    "project", "capstone", "thesis", "publication", "github",
    "open[- ]source", "hackathon", "internship", "research",
]
_DEMONSTRATED_RE = re.compile(
    r"\b(?:" + "|".join(_DEMONSTRATED_VERBS + _DEMONSTRATED_CONTEXT) + r")\b", re.IGNORECASE
)


def classify_tone(sentence: str) -> str:
    """Aspirational language ("interested in", "familiar with") caps a match
    at WEAK no matter how strong the semantic similarity is. Concrete
    build/ship language or a project/internship mention marks it
    DEMONSTRATED. Anything else that clears the similarity threshold is a
    plain EXPLICIT mention (e.g. a bare skills-list line)."""
    if _WEAK_CUE_RE.search(sentence):
        return WEAK
    if _DEMONSTRATED_RE.search(sentence):
        return DEMONSTRATED
    return EXPLICIT


# --- resume text extraction --------------------------------------------
# ponytail: extracts the text layer only (pypdf / python-docx) -- not true
# image OCR. Scanned/image-only resumes will yield no text and score empty.
# Upgrade path: pytesseract + pdf2image (needs a system poppler/tesseract
# install) if that turns out to matter for real uploads.

def extract_resume_text(filename: str, content: bytes) -> str:
    """Extract plain text from a PDF, DOCX, or TXT resume."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        return content.decode("utf-8", errors="ignore")

    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        from docx import Document
        doc = Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(
        f"Unsupported resume format '.{ext}' -- upload a PDF, DOCX, or TXT file "
        "(legacy .doc isn't supported; please re-save as .docx)."
    )


# --- presence scoring -----------------------------------------------------

def score_resume_presence(
    resume_text: str,
    onet_code: str,
    skills: list[tuple],
    threshold: float = MATCH_THRESHOLD,
    model: SentenceTransformer | None = None,
) -> dict[str, dict]:
    """For every skill in the occupation's O*NET baseline, find every resume
    sentence that semantically supports it and combine indication tone with
    match confidence into one presence score in [0, 1]."""
    baseline_names = list(dict.fromkeys(name for name, *_ in skills))
    results = {
        name: {"presence_score": 0.0, "indication_level": NO_EVIDENCE, "evidence": []}
        for name in baseline_names
    }

    skill_index = build_occupation_skill_index(onet_code, skills)
    sentences = split_sentences(resume_text)
    if skill_index.empty or not sentences:
        return results

    model = model or get_model()
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    skill_embeddings = model.encode(
        skill_index["embed_text"].tolist(), normalize_embeddings=True, show_progress_bar=False
    )
    sims = sentence_embeddings @ skill_embeddings.T  # sentences x skill_rows

    # collapse duplicate embedding rows (e.g. several software examples under
    # one skill category) down to one column per skill_name, keeping the row
    # each sentence matched best
    sims_by_skill = pd.DataFrame(sims, columns=skill_index["skill_name"]).T.groupby(level=0).max().T

    for sentence, row in zip(sentences, sims_by_skill.itertuples(index=False)):
        tone = None  # computed lazily, only once per sentence that clears threshold for any skill
        for skill_name, sim in zip(sims_by_skill.columns, row):
            if sim < threshold:
                continue
            if tone is None:
                tone = classify_tone(sentence)
            confidence = (sim - threshold) / (1 - threshold)
            contribution = INDICATION_SCORES[tone] * confidence

            entry = results[skill_name]
            entry["evidence"].append(
                {
                    "sentence": sentence,
                    "tone": tone,
                    "confidence": round(float(confidence), 3),
                    "match_score": round(float(sim), 3),
                }
            )
            entry["presence_score"] = min(1.0, entry["presence_score"] + contribution)

    for entry in results.values():
        if entry["evidence"]:
            entry["indication_level"] = max(
                (e["tone"] for e in entry["evidence"]), key=lambda t: INDICATION_SCORES[t]
            )
            entry["evidence"].sort(key=lambda e: e["confidence"], reverse=True)
        entry["presence_score"] = round(entry["presence_score"], 3)

    return results


# --- gap classification ----------------------------------------------
# presence_score and onet_importance are built by different processes (an
# NLP heuristic vs. a human-survey scale) and aren't comparable at face
# value even though both are clipped to [0, 1] -- see the project memory on
# this. So each is percentile-ranked within its own distribution first
# (this skill's presence vs. every other skill's presence in THIS resume;
# this skill's importance vs. every other skill's importance in THIS
# occupation) before being compared. Percentile ranks are always on the same
# 0..1 scale by construction, so the two are safe to subtract once ranked.

MISSING, NEED_WORK, DEEMED_ENOUGH = "missing", "need_work", "deemed_enough"
_CATEGORY_ORDER = {MISSING: 0, NEED_WORK: 1, DEEMED_ENOUGH: 2}


def classify_skill_gaps(boosted_skills: list[dict], presence_by_skill: dict[str, dict]) -> list[dict]:
    """Combine skill_matcher's boosted importance with resume presence into a
    three-way gap classification (missing / need_work / deemed_enough),
    returned in display order: missing first, then need_work, then
    deemed_enough, sorted by importance (descending) within each group.

    Every dict from `boosted_skills` is returned unchanged plus one added
    "category" key -- display format is otherwise untouched."""
    importance = pd.Series({s["skill_name"]: s["boosted_importance"] for s in boosted_skills})
    presence = pd.Series(
        {name: presence_by_skill.get(name, {}).get("presence_score", 0.0) for name in importance.index}
    )

    # rank(pct=True): 0 = lowest in this distribution, 1 = highest: ties
    # (e.g. every skill with zero evidence) share the same rank
    priority = importance.rank(pct=True) - presence.rank(pct=True)  # -1..1, higher = bigger gap

    def _category(gap: float) -> str:
        if gap >= 1 / 3:
            return MISSING
        if gap <= -1 / 3:
            return DEEMED_ENOUGH
        return NEED_WORK

    classified = [
        {**skill, "category": _category(priority[skill["skill_name"]])} for skill in boosted_skills
    ]
    classified.sort(key=lambda s: (_CATEGORY_ORDER[s["category"]], -s["boosted_importance"]))
    return classified


def analyze_resume(resume_file, onet_code: str, skills: list[tuple]) -> dict:
    """Takes the uploaded FastAPI UploadFile plus the target occupation's
    O*NET skill baseline (skill_matcher.get_occupation_baseline) and returns
    a presence measure per skill. Never raises -- extraction/format problems
    come back as a status flag so the caller can degrade gracefully."""
    content = resume_file.file.read()

    try:
        text = extract_resume_text(resume_file.filename, content)
    except ValueError as exc:
        return {"filename": resume_file.filename, "status": "error", "error": str(exc), "skills": {}}

    if not text.strip():
        return {
            "filename": resume_file.filename,
            "status": "error",
            "error": "Couldn't extract any text from this file -- it may be a scanned/image-only document.",
            "skills": {},
        }

    presence = score_resume_presence(text, onet_code, skills)
    return {"filename": resume_file.filename, "status": "ok", "skills": presence}


if __name__ == "__main__":
    _sample_skills = [
        ("Python", 0.6, "software", "high"),
        ("Critical Thinking", 0.5, "essential", "high"),
        ("Public Speaking", 0.4, "essential", "high"),
    ]
    _sample_resume = """
    Wrote production Python code for data pipelines and automated ETL jobs.
    Interested in public speaking and hoping to improve.
    """

    profile = score_resume_presence(_sample_resume, "15-2051.00", _sample_skills)
    for name, data in profile.items():
        print(f"{name:20s} {data['indication_level']:12s} presence={data['presence_score']:.3f}")

    # sanity checks on relative ordering, not exact values
    assert profile["Python"]["indication_level"] == DEMONSTRATED
    assert profile["Public Speaking"]["indication_level"] == WEAK
    assert profile["Critical Thinking"]["indication_level"] == NO_EVIDENCE
    assert profile["Python"]["presence_score"] > profile["Public Speaking"]["presence_score"] > profile["Critical Thinking"]["presence_score"]
    for data in profile.values():
        assert 0.0 <= data["presence_score"] <= 1.0
    print("\n(tone/confidence ordering + [0,1] bounds verified)")

    # --- classify_skill_gaps: synthetic, deterministic (no model call) ---
    _boosted_skills = [
        {"skill_name": "critical_no_evidence", "boosted_importance": 0.95},
        {"skill_name": "trivial_well_evidenced", "boosted_importance": 0.15},
        {"skill_name": "balanced_lo", "boosted_importance": 0.5},
        {"skill_name": "balanced_hi", "boosted_importance": 0.55},
    ]
    _presence_by_skill = {
        "critical_no_evidence": {"presence_score": 0.0},
        "trivial_well_evidenced": {"presence_score": 0.95},
        "balanced_lo": {"presence_score": 0.5},
        "balanced_hi": {"presence_score": 0.55},
    }
    classified = classify_skill_gaps(_boosted_skills, _presence_by_skill)
    categories = {s["skill_name"]: s["category"] for s in classified}
    print("\n" + "\n".join(f"{s['skill_name']:24s} {s['category']}" for s in classified))

    assert categories["critical_no_evidence"] == MISSING
    assert categories["trivial_well_evidenced"] == DEEMED_ENOUGH
    # display order: missing block, then need_work block, then deemed_enough block
    order = [_CATEGORY_ORDER[s["category"]] for s in classified]
    assert order == sorted(order)
    # original skill dict fields untouched -- only "category" added
    assert all(k in classified[0] for k in ("skill_name", "boosted_importance", "category"))
    print("(gap classification + display ordering verified)")
