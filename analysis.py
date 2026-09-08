import os
from dotenv import load_dotenv
import requests

load_dotenv()  # Reads variables from .env


def analyze_resume(resume_file) -> dict:
    """Placeholder for resume parsing/extraction -- takes the uploaded
    FastAPI UploadFile from user_input.py. Real analysis (extracting
    skills/experience to blend into the boosted skill profile) lands later."""
    return {
        "filename": resume_file.filename,
        "status": "received",
        "note": "Resume parsing not implemented yet.",
    }

