"""
Per-occupation pipeline: student gives a desired occupation -> resolve it to
an O*NET occupation in the archive built by data_intake_onet.py -> pull that
occupation's baseline skill list -> fetch live Adzuna postings for that same
occupation title -> NLP-compare the postings against the skill list -> boost
the importance of skills the postings back up.

Why embeddings, not keyword/TF-IDF matching: most O*NET skills are abstract
dispositions ("Critical Thinking", "Active Listening") that are essentially
never spelled out verbatim in a job posting -- a posting implies them through
phrasing like "must be able to evaluate competing proposals and justify a
recommendation". A bi-encoder sentence embedding model captures that semantic
relationship; plain lexical overlap (TF-IDF, keyword search) does not.

For software skills, we embed the concrete product name (O*NET's "Workplace
Example" column, e.g. "QuickBooks") rather than the broad category ("Accounting
software"), scoped to the specific occupation, since the same category maps to
different concrete tools across occupations. A posting that says "QuickBooks"
then sits almost on top of that label in embedding space.

Archive immutability: data_intake_onet.build_skill_archive() is loaded once
and cached -- it's never mutated. Its "skills" column holds Python lists, and
pandas hands back the *same* list object on every `.loc[title]` access, so
mutating it in place would silently corrupt the shared archive for every future
call. Each call to get_boosted_skill_profile() therefore deep-copies the one
occupation row's skill list before boosting it, and returns that copy -- the
archive's own importance values never change no matter how many times a
student calls this.
"""

import copy
import difflib
import re
from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer

from data_intake_adzuna import get_jobs
from data_intake_onet import DATA_DIR, build_skill_archive

MODEL_NAME = "all-MiniLM-L6-v2"

# Below this cosine similarity a sentence/skill pair is treated as unrelated
# rather than a weak match. MiniLM cosine scores for genuinely related short
# texts typically land >0.35; tune against real postings as needed.
MATCH_THRESHOLD = 0.35

# Largest amount a skill's importance can be boosted by (at posting_score == 1.0).
# Scales linearly from 0 at MATCH_THRESHOLD up to MAX_BOOST at a perfect match.
MAX_BOOST = 0.3


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_archive() -> pd.DataFrame:
    """The O*NET occupation/skill archive, loaded once and never mutated."""
    return build_skill_archive(DATA_DIR)


@lru_cache(maxsize=1)
def _title_lookup() -> dict[str, str]:
    """lowercased title/related-title -> canonical archive title (the index key)."""
    lookup: dict[str, str] = {}
    for title, row in get_archive().iterrows():
        lookup.setdefault(title.lower(), title)
        for related in row["related_titles"]:
            lookup.setdefault(related.lower(), title)
    return lookup


def resolve_occupation(occupation_query: str) -> str | None:
    """Map free-text student input (e.g. "management_consultant") to the
    canonical O*NET occupation title used as the archive's index key."""
    lookup = _title_lookup()
    norm = occupation_query.strip().lower().replace("_", " ")
    if norm in lookup:
        return lookup[norm]
    close = difflib.get_close_matches(norm, lookup.keys(), n=1, cutoff=0.6)
    return lookup[close[0]] if close else None


@lru_cache(maxsize=1)
def _software_examples_table() -> pd.DataFrame:
    return pd.read_csv(
        f"{DATA_DIR}/software_skills.csv",
        usecols=["O*NET-SOC Code", "Element Name", "Workplace Example"],
    ).dropna(subset=["Workplace Example"])


def build_occupation_skill_index(onet_code: str, skills: list[tuple]) -> pd.DataFrame:
    """One embedding row per skill (generic essential/transferable) or per
    concrete product example (software), scoped to this occupation."""
    software_examples = _software_examples_table()
    rows = []
    for skill_name, _importance, source, _confidence in skills:
        if source != "software":
            rows.append({"skill_name": skill_name, "source": source, "embed_text": skill_name})
            continue
        examples = software_examples[
            (software_examples["O*NET-SOC Code"] == onet_code)
            & (software_examples["Element Name"] == skill_name)
        ]["Workplace Example"].unique()
        if len(examples) == 0:
            # No concrete example on file for this occupation -- fall back to
            # matching on the category name itself so the skill still gets scored.
            rows.append({"skill_name": skill_name, "source": source, "embed_text": skill_name})
        else:
            rows.extend(
                {"skill_name": skill_name, "source": source, "embed_text": example}
                for example in examples
            )
    return pd.DataFrame(rows)


_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?:\s*[•·]\s*)")


def clean_description(description: str) -> str:
    text = _TAG_RE.sub(" ", description or "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(description: str) -> list[str]:
    text = clean_description(description)
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) > 3]


def score_skills_against_postings(
    descriptions: list[str],
    skill_index: pd.DataFrame,
    model: SentenceTransformer | None = None,
) -> dict[str, float]:
    """Best cosine-similarity score per skill_name, maxed over every sentence
    in every posting and every embedding row that shares that skill_name
    (e.g. several software product examples under one category)."""
    if skill_index.empty:
        return {}

    sentences = [s for d in descriptions for s in split_sentences(d)]
    if not sentences:
        return {name: 0.0 for name in skill_index["skill_name"].unique()}

    model = model or get_model()
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    skill_embeddings = model.encode(
        skill_index["embed_text"].tolist(), normalize_embeddings=True, show_progress_bar=False
    )

    # cosine similarity, since both sets of embeddings are L2-normalized this
    # is just the dot product: (sentences x skills)
    sims = sentence_embeddings @ skill_embeddings.T
    best_per_row = sims.max(axis=0)

    scores: dict[str, float] = {}
    for row, score in zip(skill_index.itertuples(), best_per_row):
        scores[row.skill_name] = max(scores.get(row.skill_name, 0.0), float(score))
    return scores


def _boost(
    importance: float, score: float, threshold: float, max_boost: float
) -> tuple[float, float]:
    """Linear boost from 0 at `threshold` up to `max_boost` at score == 1.0.
    Below threshold the posting isn't taken as real evidence for the skill."""
    if score < threshold:
        return importance, 0.0
    boost = max_boost * (score - threshold) / (1 - threshold)
    return min(1.0, importance + boost), boost


def get_boosted_skill_profile(
    occupation_query: str,
    num_jobs: int = 25,
    location: str = "Pittsburgh",
    threshold: float = MATCH_THRESHOLD,
    max_boost: float = MAX_BOOST,
) -> dict:
    """Resolve `occupation_query` to an O*NET occupation, pull its skill
    baseline from the archive, fetch live Adzuna postings for that occupation,
    and return a per-call COPY of the skill list with importance boosted
    wherever the postings semantically back up a skill.

    The shared archive is never mutated -- every call gets its own deep copy,
    so importance scores don't drift upward the more often this is called.
    """
    title = resolve_occupation(occupation_query)
    if title is None:
        raise ValueError(f"No O*NET occupation found matching {occupation_query!r}")

    row = get_archive().loc[title]
    onet_code = row["onet_code"]
    skills = copy.deepcopy(row["skills"])  # archive.loc hands back the SAME list object

    jobs = get_jobs(title, location=location, num_jobs=num_jobs)
    descriptions = [job.get("description", "") for job in jobs]

    skill_index = build_occupation_skill_index(onet_code, skills)
    posting_scores = score_skills_against_postings(descriptions, skill_index)

    boosted_skills = []
    for skill_name, importance, source, confidence in skills:
        score = posting_scores.get(skill_name, 0.0)
        boosted_importance, boost = _boost(importance, score, threshold, max_boost)
        boosted_skills.append(
            {
                "skill_name": skill_name,
                "source": source,
                "onet_importance": importance,
                "confidence": confidence,
                "posting_match_score": round(score, 3),
                "boost_applied": round(boost, 3),
                "boosted_importance": round(boosted_importance, 3),
            }
        )
    boosted_skills.sort(key=lambda s: s["boosted_importance"], reverse=True)

    return {
        "occupation_query": occupation_query,
        "resolved_title": title,
        "onet_code": onet_code,
        "jobs_pulled": len(jobs),
        "skills": boosted_skills,
    }


if __name__ == "__main__":
    from data_intake_adzuna import student_input

    profile = get_boosted_skill_profile(student_input)
    print(f"Query: {profile['occupation_query']!r} -> resolved: {profile['resolved_title']} ({profile['onet_code']})")
    print(f"Pulled {profile['jobs_pulled']} live postings\n")

    for s in profile["skills"]:
        marker = "*" if s["boost_applied"] > 0 else " "
        print(
            f"{marker} {s['boosted_importance']:.3f} (base {s['onet_importance']:.3f} "
            f"+{s['boost_applied']:.3f})  [{s['source']:12s}]  {s['skill_name']}  "
            f"(posting match {s['posting_match_score']:.3f})"
        )

    # Sanity check: confirm the shared archive itself was never touched.
    original_row = get_archive().loc[profile["resolved_title"]]
    for skill_name, importance, _source, _confidence in original_row["skills"]:
        boosted = next(s for s in profile["skills"] if s["skill_name"] == skill_name)
        assert importance == boosted["onet_importance"], "archive importance was mutated!"
    print("\n(archive left untouched -- verified original importances are unmodified)")
