"""Step 3 and 4: detailed match score for one role, and suggested CV changes."""
from llm import ask_json

MAX_CV_CHARS = 10000
MAX_JD_CHARS = 6000

SCORE_PROMPT = """You are a strict, fair technical recruiter.
Compare the CANDIDATE CV with the TARGET ROLE. Judge only what the CV actually states; never assume skills it doesn't show.
Scoring guide: 85-100 meets nearly every requirement; 60-84 credible applicant with some gaps; below 60 significant gaps.
Location is NOT part of the score.

Reply with JSON exactly like this example (replace the values):
{
  "score": 72,
  "verdict": "One sentence overall verdict",
  "breakdown": {"skills": 70, "experience": 65, "education": 80, "keywords": 60},
  "matched": ["requirement the CV clearly shows"],
  "missing": ["requirement the CV lacks or doesn't show"],
  "strengths": ["short sentence"],
  "gaps": ["short sentence"]
}
Keep each list to at most 6 items. All numbers are 0-100.

TARGET ROLE:
{target}

CANDIDATE CV:
\"\"\"
{cv}
\"\"\"
"""

CHANGES_PROMPT = """You are an expert CV editor. Suggest edits so this CV fits the target role better.
Hard rules:
- Never invent employers, job titles, dates, degrees, tools or numbers the CV doesn't contain.
- If a stronger bullet needs a figure the CV lacks, write a placeholder in square brackets like [add % improvement].
- Reword, reorder and emphasise what is already true. Mirror the role's wording where it is honest.
{gaps}
Reply with JSON exactly like this example (replace the values):
{
  "headline": "Suggested CV headline",
  "summary": "Rewritten professional summary, 3-4 sentences",
  "changes": [
    {"section": "Experience - Company", "priority": "high", "current": "exact text from the CV", "suggested": "rewritten text", "why": "one sentence"}
  ],
  "add_keywords": ["term to add only if true for the candidate"],
  "remove": ["thing to cut or shorten, with reason"],
  "projected_score": 80
}
Give 5 to 8 changes, most important first. "priority" is high, medium or low.
"current" must be copied from the CV word for word, or "Not present" if it's new.

TARGET ROLE:
{target}

CANDIDATE CV:
\"\"\"
{cv}
\"\"\"
"""


def _clamp(value):
    try:
        return max(0, min(100, round(float(value))))
    except (TypeError, ValueError):
        return 0


def _strings(value, limit=8):
    if not isinstance(value, list):
        value = [value] if value else []
    return [str(v) for v in value if v][:limit]


def score_match(cv_text, target):
    data = ask_json(SCORE_PROMPT.replace("{target}", target[:MAX_JD_CHARS]).replace("{cv}", cv_text[:MAX_CV_CHARS]))
    breakdown = data.get("breakdown") if isinstance(data.get("breakdown"), dict) else {}
    return {
        "score": _clamp(data.get("score")),
        "verdict": str(data.get("verdict") or ""),
        "breakdown": {k: _clamp(breakdown[k]) for k in ("skills", "experience", "education", "keywords")
                      if breakdown.get(k) is not None},
        "matched": _strings(data.get("matched")),
        "missing": _strings(data.get("missing")),
        "strengths": _strings(data.get("strengths")),
        "gaps": _strings(data.get("gaps")),
    }


def suggest_changes(cv_text, target, missing=None):
    gaps = f"Known gaps from scoring: {', '.join(missing)}.\n" if missing else ""
    prompt = (
        CHANGES_PROMPT.replace("{gaps}", gaps)
        .replace("{target}", target[:MAX_JD_CHARS])
        .replace("{cv}", cv_text[:MAX_CV_CHARS])
    )
    data = ask_json(prompt)
    changes = []
    for c in data.get("changes") or []:
        if not isinstance(c, dict):
            continue
        priority = str(c.get("priority", "medium")).lower()
        changes.append({
            "section": str(c.get("section") or "General"),
            "priority": priority if priority in ("high", "medium", "low") else "medium",
            "current": str(c.get("current") or "Not present"),
            "suggested": str(c.get("suggested") or ""),
            "why": str(c.get("why") or ""),
        })
    projected = data.get("projected_score")
    return {
        "headline": str(data.get("headline") or ""),
        "summary": str(data.get("summary") or ""),
        "changes": [c for c in changes if c["suggested"]],
        "add_keywords": _strings(data.get("add_keywords"), 12),
        "remove": _strings(data.get("remove")),
        "projected_score": _clamp(projected) if projected is not None else None,
    }
