"""Ask Claude for skill-building activities for one skill gap, using the
student's resume text (if available) as context."""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_SYSTEM = "You are a career coach who designs concrete, college-level skill-building activities."


def _build_prompt(skill: str, resume_text: str) -> str:
    resume_context = resume_text.strip() or "No resume text available."
    return (
        f"A student wants to strengthen the skill: {skill}.\n\n"
        f"Their resume:\n{resume_context}\n\n"
        "Suggest college-level activities (inside or outside of school) that would build this "
        "skill. Split the suggestions into two groups:\n"
        '1. "building_on_existing": 2 or more activities that extend or deepen something already '
        "on their resume.\n"
        '2. "new_activities": 2 or more brand-new activities unrelated to anything on their resume.\n\n'
        "Respond with ONLY valid JSON, no other text, in exactly this shape:\n"
        '{"building_on_existing": [{"title": "...", "description": "..."}], '
        '"new_activities": [{"title": "...", "description": "..."}]}'
    )


def _parse_activity_response(raw: str) -> dict:
    """Parse the model's JSON reply; fall back to raw text if it didn't comply."""
    try:
        data = json.loads(raw)
        return {
            "building_on_existing": data.get("building_on_existing", []),
            "new_activities": data.get("new_activities", []),
        }
    except json.JSONDecodeError:
        return {"building_on_existing": [], "new_activities": [], "raw": raw}


def call_activity_generation(skill: str, resume_text: str = "") -> dict:
    """Returns {"building_on_existing": [{"title","description"}, ...],
    "new_activities": [...]} -- or a "raw" text fallback if Claude's reply
    wasn't valid JSON."""
    client = Anthropic(api_key=os.environ.get("CLAUDE_KEY"))
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1500,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(skill, resume_text)}],
    )
    text_block = next(b for b in message.content if b.type == "text")
    return _parse_activity_response(text_block.text)


if __name__ == "__main__":
    valid = _parse_activity_response(
        '{"building_on_existing": [{"title": "A", "description": "d"}], "new_activities": []}'
    )
    assert valid["building_on_existing"] == [{"title": "A", "description": "d"}]
    assert "raw" not in valid

    broken = _parse_activity_response("not json")
    assert broken["building_on_existing"] == [] and broken["raw"] == "not json"

    print("(activity response parsing verified)")
