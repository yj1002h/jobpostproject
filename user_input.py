"""
Front-end entry point (FastAPI): homepage + occupation/region/resume form,
wired to analysis.analyze_resume for the resume and
skill_matcher.get_boosted_skill_profile(occupation, location=region) for the
skill profile.
"""

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analysis import analyze_resume
from skill_matcher import get_boosted_skill_profile

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_RESUME_EXTENSIONS = {"pdf", "doc", "docx", "txt"}

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _allowed_resume(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_RESUME_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse(request, "analyze.html")


@app.post("/api/analyze")
async def analyze(
    occupation: str = Form(...),
    region: str = Form(...),
    resume: UploadFile | None = None,
):
    occupation = occupation.strip()
    region = region.strip()

    if not occupation or not region:
        raise HTTPException(400, "Occupation and region are required.")
    if resume is None or not resume.filename:
        raise HTTPException(400, "A resume file is required.")
    if not _allowed_resume(resume.filename):
        raise HTTPException(400, "Resume must be a PDF, Word doc, or text file.")

    try:
        skill_profile = get_boosted_skill_profile(occupation, location=region)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    resume_result = analyze_resume(resume)

    return {"skill_profile": skill_profile, "resume": resume_result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("user_input:app", reload=True)
