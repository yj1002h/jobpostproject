"""
Builds the occupation -> skill-archive table that the skill-gap app will query.

For each O*NET occupation this produces one row containing:
  - title            : the occupation's canonical O*NET title (the key)
  - onet_code        : O*NET-SOC Code (kept alongside title as the real join key,
                        since titles can drift/be renamed across O*NET releases
                        while codes stay stable)
  - skills           : list of (skill_name, importance, source, confidence) tuples,
                        pooled from essential_skills.csv, transferable_skills.csv
                        and software_skills.csv
  - related_titles   : sorted list of alternate/"also called" job titles for the
                        occupation, from job_titles.csv

Data source: O*NET database extracts in preliminary_data_onet/.

Importance normalization
-------------------------
essential_skills.csv and transferable_skills.csv both report an "Importance"
scale (Scale ID "IM") on O*NET's fixed 1-5 rating scale, alongside a "Level"
scale we don't want here. We keep only Scale Name == "Importance" rows and
normalize with the scale's known bounds, (value - 1) / (5 - 1), giving 0-1.
Using the fixed theoretical bounds (rather than min/max of observed values)
keeps a normalized score comparable across occupations and across future runs
if new data is appended.

software_skills.csv has no Importance/Data Value column at all -- it only
flags each software tool as "Hot Technology" (Y/N) and "In Demand" (Y/N).
There is no principled way to convert that into the same 1-5 importance
score O*NET analysts assigned to essential/transferable skills, so rather
than invent a number we keep it on a separate 0-1 scale built only from
those two flags (0.5 base + 0.25 per flag) and tag every software skill's
source as "software" so the app can decide how to weight/display it
differently from analyst-rated importance.

Baseline vs. future job-posting data
-------------------------------------
This O*NET-derived importance is meant as a *baseline*, to later be blended
with importance signals mined from real job postings (e.g. how often a
skill/tool is explicitly requested). To make that future blend possible, we
attach a `confidence` value (0-1) to every skill tuple, alongside importance:

  - essential/transferable skills: confidence is derived from each rating's
    published 95% CI width (Upper CI Bound - Lower CI Bound), which O*NET
    already provides per skill per occupation. A tight CI (analysts agreed)
    yields confidence near 1; a wide CI (analysts disagreed / sparse data)
    pulls it down. Computed as 1 - ci_width / 4, clipped to [0, 1] (4 is the
    widest a CI can be on a 1-5 scale). This is a per-row statistical measure,
    not a flat constant -- checked against the data, CI widths range 0-1.88
    with a median of ~0.49 (~0.88 confidence), so this comfortably spreads
    scores instead of collapsing them to one value. (O*NET's raw sample size,
    "N", was checked too but is constant at 8 respondents for every row here,
    so it carries no discriminating information and wasn't used.)
  - software skills: fixed at a low 0.3. The Hot Technology/In Demand flags
    are a coarse, non-statistical proxy, and job postings are expected to be
    a *much* stronger signal specifically for named tools/software (postings
    tend to explicitly list required software), so this baseline should be
    easy for that future source to outweigh once available.

Downstream, a future job-posting pipeline can combine two importance
estimates for the same skill with a confidence-weighted average, e.g.
`(imp_a * conf_a + imp_b * conf_b) / (conf_a + conf_b)`, rather than either
overwriting the other outright.
"""

import pandas as pd

DATA_DIR = "preliminary_data_onet"


def _load_importance_skills(path: str, source: str) -> pd.DataFrame:
    """Load essential_skills.csv or transferable_skills.csv and return
    (O*NET-SOC Code, Element Name, importance 0-1, source, confidence 0-1) rows."""
    df = pd.read_csv(
        path,
        usecols=[
            "O*NET-SOC Code",
            "Element Name",
            "Scale Name",
            "Data Value",
            "Lower CI Bound",
            "Upper CI Bound",
        ],
    )
    df = df[df["Scale Name"] == "Importance"].copy()
    df["importance"] = (df["Data Value"] - 1) / (5 - 1)
    ci_width = df["Upper CI Bound"] - df["Lower CI Bound"]
    df["confidence"] = (1 - ci_width / 4).clip(0, 1)
    df["source"] = source
    return df[["O*NET-SOC Code", "Element Name", "importance", "source", "confidence"]]


def _load_software_skills(path: str) -> pd.DataFrame:
    """Load software_skills.csv and return a proxy 0-1 importance built from
    the Hot Technology / In Demand flags, with a fixed low confidence since
    it's a coarse proxy meant to be outweighed by future job-posting data
    (see module docstring)."""
    df = pd.read_csv(
        path,
        usecols=["O*NET-SOC Code", "Element Name", "Hot Technology", "In Demand"],
    )
    df["importance"] = 0.5
    df["importance"] += (df["Hot Technology"] == "Y") * 0.25
    df["importance"] += (df["In Demand"] == "Y") * 0.25
    df["confidence"] = 0.3
    df["source"] = "software"
    df = df.drop_duplicates(subset=["O*NET-SOC Code", "Element Name"])
    return df[["O*NET-SOC Code", "Element Name", "importance", "source", "confidence"]]


def build_skill_archive(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame keyed by occupation title with pooled skills and
    related job titles, ready to back the resume skill-gap tool."""

    occupations = pd.read_csv(
        f"{data_dir}/occupation_data.csv",
        usecols=["O*NET-SOC Code", "Title"],
    )

    skills = pd.concat(
        [
            _load_importance_skills(f"{data_dir}/essential_skills.csv", "essential"),
            _load_importance_skills(f"{data_dir}/transferable_skills.csv", "transferable"),
            _load_software_skills(f"{data_dir}/software_skills.csv"),
        ],
        ignore_index=True,
    )
    skills["importance"] = skills["importance"].round(3)
    skills["confidence"] = skills["confidence"].round(3)
    skills["skill_tuple"] = list(
        zip(
            skills["Element Name"],
            skills["importance"],
            skills["source"],
            skills["confidence"],
        )
    )
    skills_by_code = (
        skills.groupby("O*NET-SOC Code")["skill_tuple"].apply(list).rename("skills")
    )

    job_titles = pd.read_csv(
        f"{data_dir}/job_titles.csv",
        usecols=["O*NET-SOC Code", "Job Title"],
    )
    related_by_code = (
        job_titles.groupby("O*NET-SOC Code")["Job Title"]
        .apply(lambda s: sorted(set(s.dropna())))
        .rename("related_titles")
    )

    archive = (
        occupations.rename(columns={"O*NET-SOC Code": "onet_code", "Title": "title"})
        .join(skills_by_code, on="onet_code")
        .join(related_by_code, on="onet_code")
    )
    archive["skills"] = archive["skills"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    archive["related_titles"] = archive["related_titles"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    return archive.set_index("title")


if __name__ == "__main__":
    archive = build_skill_archive()
    print(archive.shape)
    print(archive.head())

    sample = archive.index[0]
    print(f"\nExample row for {sample!r}:")
    print("onet_code:", archive.loc[sample, "onet_code"])
    print("skills (name, importance, source, confidence) - first 5:")
    for s in archive.loc[sample, "skills"][:5]:
        print(" ", s)
    print("related_titles (first 5):", archive.loc[sample, "related_titles"][:5])



