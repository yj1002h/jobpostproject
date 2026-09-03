import os
import requests
from dotenv import load_dotenv

load_dotenv()

student_input="Data Scientist" #used when prompted 

def get_jobs(job_title, location="Pittsburgh", num_jobs=50):
    """
    Return up to num_jobs Adzuna job postings for a job title and location.
    """

    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"

    params = {
        "app_id": os.getenv("ADZUNA_ID"),
        "app_key": os.getenv("ADZUNA_KEY"),
        "what": job_title,
        "where": location,
        "results_per_page": num_jobs,
        "content-type": "application/json",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    return data.get("results", [])

#sentiment vector code the skill and description of the job. If the match very close enhace the skill level

def sentiment_match_skill():
    jobs = get_jobs(student_input)
    
    return

if __name__ == "__main__":
    jobs = get_jobs("data scientist")
    for job in jobs:
        print("Title:", job.get("title"))
        print("Company:", job.get("company", {}).get("display_name"))
        print("Location:", job.get("location", {}).get("display_name"))
        print("Description:", job.get("description"))
        print("Link:", job.get("redirect_url"))
        print("------")
