"""The seven tools the agent can call.

Every tool here is a plain Python function: arguments in, dictionary out. No
classes, no decorators, no framework. The model never runs this code - it only
asks for a tool by name with JSON arguments, and agent.py calls the matching
function below and hands the return value back.

Read this file top to bottom and you know everything the agent is able to do.
"""

from __future__ import annotations

import json

import azure_client
import learner_db

# What each target role is expected to know. This is the mock "role framework"
# an L&D team would normally maintain in a competency system.
ROLE_SKILL_REQUIREMENTS: dict[str, list[str]] = {
    "ai application developer": [
        "python", "llm fundamentals", "prompt engineering", "api integration",
        "embeddings", "rag", "vector databases", "tool calling", "agent design",
        "evaluation", "deployment", "responsible ai",
    ],
    "ai engineer": [
        "python", "llm fundamentals", "prompt engineering", "embeddings", "rag",
        "vector databases", "agent design", "evaluation", "deployment",
        "azure ai services", "responsible ai",
    ],
    "prompt engineer": [
        "llm fundamentals", "prompt engineering", "structured output",
        "prompt evaluation", "evaluation", "responsible ai",
    ],
    "machine learning engineer": [
        "python", "numpy", "pandas", "data wrangling", "evaluation",
        "deployment", "ci cd", "monitoring",
    ],
    "data engineer": [
        "python", "pandas", "data wrangling", "api integration", "deployment",
        "ci cd", "monitoring",
    ],
}

# Roles the learner already holds map to skills we can assume they have, so the
# agent does not put an experienced backend developer through an intro course.
ROLE_IMPLIED_SKILLS: dict[str, list[str]] = {
    "java developer": ["oop", "backend development", "unit testing", "rest apis", "sql", "ci cd"],
    "java": ["oop", "backend development", "unit testing", "rest apis", "sql", "ci cd"],
    "backend developer": ["backend development", "rest apis", "unit testing", "sql"],
    "python developer": ["python", "backend development", "unit testing"],
    "data analyst": ["sql", "data wrangling", "pandas"],
    "qa engineer": ["unit testing", "evaluation"],
}

DEFAULT_TARGET_ROLE = "ai application developer"

# Asking the model the same (query, skills) twice in one run returns the first
# answer, so the agent's repair passes cannot be handed a different set of course
# ids halfway through building a plan.
_SEARCH_MEMO: dict[str, dict] = {}


def _load_courses() -> list[dict]:
    """Every course the model has proposed so far, read back from SQLite.

    There is no knowledge_base.json. The catalogue is generated on demand by
    search_courses and cached in the courses table, so this returns whatever the
    model has surfaced for this learner (and any earlier learner) up to now.
    """
    learner_db.init_db()
    return learner_db.all_courses()


def _normalise(items: list[str]) -> list[str]:
    """Lowercase and strip a list of skill strings, dropping blanks and duplicates."""
    seen: list[str] = []
    for item in items or []:
        cleaned = str(item).strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _match_role(role: str, table: dict[str, list[str]]) -> str | None:
    """Find the closest key in a role table. Exact match first, then substring."""
    if not role:
        return None

    role = role.strip().lower()
    if role in table:
        return role

    for key in table:
        if key in role or role in key:
            return key
    return None


# ==========================================================================
# TOOL 1
# ==========================================================================

def get_employee_profile(employee_id: str) -> dict:
    """Read a learner's profile from SQLite, creating an empty record if new.

    Returns the profile plus whether it already existed, what is still unknown,
    and any courses already completed - the three things the agent needs to
    decide whether it can plan yet.
    """
    learner_db.init_db()

    learner = learner_db.get_learner(employee_id)
    existed = learner is not None

    if not existed:
        learner = learner_db.create_learner(employee_id)

    missing = [
        field for field in ("current_role", "target_role", "hours_per_week")
        if not learner.get(field)
    ]

    return {
        "employee_id": employee_id,
        "profile_existed": existed,
        "profile": {
            "name": learner.get("name"),
            "current_role": learner.get("current_role"),
            "experience_years": learner.get("experience_years"),
            "current_skills": learner.get("current_skills", []),
            "target_role": learner.get("target_role"),
            "hours_per_week": learner.get("hours_per_week"),
            "timeline_months": learner.get("timeline_months"),
        },
        "missing_fields": missing,
        "completed_courses": learner_db.completed_course_ids(employee_id),
        "has_existing_plan": learner_db.get_plan(employee_id) is not None,
        "note": (
            "Fields listed in missing_fields are not stored yet. Take them from what "
            "the learner told you in the conversation, or ask a clarifying question."
        ),
    }


# ==========================================================================
# TOOL 2
# ==========================================================================

def get_skill_assessment(
    current_skills: list[str],
    target_role: str,
    current_role: str | None = None,
    experience_years: float | None = None,
) -> dict:
    """Compare the skills a learner has against what the target role requires.

    Skills implied by the learner's stated current role (a Java developer already
    knows OOP and REST) are credited automatically, so the gap list is the real
    work to be done and nothing more.

    current_role and experience_years are optional but worth passing: the role is
    credited for implied skills, and both are echoed back so the orchestrator can
    store them on the learner's profile for next time.
    """
    have = _normalise(current_skills)

    # An explicitly stated current role is treated as a skill claim too, so the
    # model does not have to smuggle "java developer" inside current_skills.
    if current_role:
        role_as_skill = str(current_role).strip().lower()
        if role_as_skill and role_as_skill not in have:
            have.append(role_as_skill)

    # Credit anything the learner's own words imply about their existing role.
    for skill in have[:]:
        implied_key = _match_role(skill, ROLE_IMPLIED_SKILLS)
        if implied_key:
            for extra in ROLE_IMPLIED_SKILLS[implied_key]:
                if extra not in have:
                    have.append(extra)

    role_key = _match_role(target_role, ROLE_SKILL_REQUIREMENTS)
    if role_key is None:
        role_key = DEFAULT_TARGET_ROLE
        matched_exactly = False
    else:
        matched_exactly = True

    required = ROLE_SKILL_REQUIREMENTS[role_key]
    already_have = [skill for skill in required if skill in have]
    gaps = [skill for skill in required if skill not in have]

    coverage = round(100 * len(already_have) / len(required)) if required else 0

    return {
        "target_role": target_role,
        "current_role": current_role,
        "experience_years": experience_years,
        "matched_role_framework": role_key,
        "matched_exactly": matched_exactly,
        "required_skills": required,
        "skills_already_held": already_have,
        "skill_gaps": gaps,
        "coverage_percent": coverage,
        "transferable_skills": [skill for skill in have if skill not in required],
    }


# ==========================================================================
# TOOL 3
# ==========================================================================

CATALOGUE_SYSTEM_PROMPT = """You are a course catalogue for a corporate L&D team. Given a \
list of skills an employee needs, return real, publicly available online courses or \
certifications that teach those skills.

Return ONLY a JSON object in this shape, with no commentary and no markdown fences:
{
  "courses": [
    {
      "id": "SHORT-STABLE-ID",
      "title": "exact course title",
      "provider": "Coursera | Microsoft Learn | DeepLearning.AI | Udemy | edX | ...",
      "topic": "one short topic label",
      "level": "Beginner | Intermediate | Advanced",
      "duration_hours": 12,
      "prerequisites": ["ID-OF-ANOTHER-COURSE-IN-THIS-RESPONSE"],
      "skills_taught": ["lowercase skill", "..."],
      "course_link": "https://provider.example/course-page",
      "content": "two sentences on what it covers"
    }
  ]
}

Rules:
- Return between 4 and 8 courses, ordered easiest first.
- "id" must be UPPERCASE, hyphenated, derived from provider and topic, e.g.
  "DLAI-PROMPT-ENG" or "MSLEARN-AZURE-AI-FUNDAMENTALS". Reuse the same id for the same
  course every time so plans stay stable.
- "duration_hours" must be a realistic whole number of study hours, not weeks.
- "prerequisites" may only reference ids you return in THIS response. Use [] if there
  are none. Never invent a dependency just to look thorough.
- "skills_taught" must use the exact lowercase skill strings from the request wherever
  the course genuinely teaches them, so gap matching works.
- Only include courses you are reasonably confident exist. Prefer well-known providers.
- Do not include a course twice.
"""


def search_courses(query: str = "", skills_needed: list[str] | None = None) -> dict:
    """Ask the model for courses that teach the missing skills, and cache them.

    There is no local catalogue file. The model IS the catalogue: it proposes
    courses, they are written to the courses table in SQLite, and every later step
    (plan validation, completion recording) reads them back from there. That
    caching is what keeps course ids stable across the agent's repair passes.

    Because the courses come from a language model, ids and links are unverified.
    Every result is marked model_generated: true for exactly that reason.
    """
    learner_db.init_db()

    wanted = _normalise(skills_needed or [])
    if not wanted and not (query or "").strip():
        return {
            "error": "Pass skills_needed (from get_skill_assessment) and/or a query.",
            "courses": [],
        }

    memo_key = (query or "").strip().lower() + "|" + ",".join(wanted)
    if memo_key in _SEARCH_MEMO:
        cached = _SEARCH_MEMO[memo_key]
        return {**cached, "served_from": "in-process cache"}

    user_prompt = (
        f"Skills the employee needs: {', '.join(wanted) if wanted else '(none given)'}\n"
        f"Topic hint: {query or '(none)'}\n\n"
        "Return courses that together cover as many of those skills as possible."
    )

    payload = azure_client.ask_for_json(CATALOGUE_SYSTEM_PROMPT, user_prompt)
    proposed = (payload or {}).get("courses") or []

    if not proposed:
        return {
            "error": "The model did not return any usable courses. Try again or reword the query.",
            "query": query,
            "skills_needed": wanted,
            "courses": [],
        }

    learner_db.upsert_courses(proposed)

    results = []
    for course in proposed:
        course_id = str(course.get("id") or "").strip().upper()
        stored = learner_db.get_course(course_id)
        if stored is None:
            continue

        taught = _normalise(stored.get("skills_taught", []))
        results.append({
            "id": stored["id"],
            "title": stored["title"],
            "provider": stored.get("provider"),
            "topic": stored.get("topic"),
            "level": stored.get("level"),
            "duration_hours": stored.get("duration_hours"),
            "prerequisites": stored.get("prerequisites", []),
            "skills_taught": stored.get("skills_taught", []),
            "course_link": stored.get("course_link"),
            "summary": (stored.get("content") or "")[:220],
            "matched_skill_gaps": [skill for skill in wanted if skill in taught],
            "model_generated": True,
        })

    results.sort(key=lambda c: c["duration_hours"] or 0)

    uncovered = [
        skill for skill in wanted
        if not any(skill in _normalise(r["skills_taught"]) for r in results)
    ]

    result = {
        "query": query,
        "skills_needed": wanted,
        "result_count": len(results),
        "courses": results,
        "skills_with_no_course": uncovered,
        "source": "model-generated, cached in the courses table",
        "note": (
            "These courses came from the language model, not a verified catalogue - the "
            "titles and links must be checked by a human before being sent to a learner. "
            "prerequisites lists course ids that must be scheduled in an earlier or the "
            "same week, unless the learner already has those skills. Use ONLY these ids "
            "in the plan."
        ),
    }
    _SEARCH_MEMO[memo_key] = result
    return result


# ==========================================================================
# TOOL 4
# ==========================================================================

def update_learning_plan(employee_id: str, plan: dict) -> dict:
    """Persist the finished plan for this learner, overwriting any previous one.

    Creates the learner row first if the plan arrives before the profile does,
    and returns a small summary so the caller can see what was written.
    """
    learner_db.init_db()

    if learner_db.get_learner(employee_id) is None:
        learner_db.create_learner(employee_id)

    if isinstance(plan, str):            # some models hand back a JSON string
        plan = json.loads(plan)

    weeks = plan.get("weekly_plan", []) or []
    total_hours = sum(float(week.get("hours") or 0) for week in weeks)
    course_ids = [cid for week in weeks for cid in week.get("courses", [])]

    saved = learner_db.save_plan(employee_id, plan)

    return {
        "status": "saved",
        "employee_id": employee_id,
        "plan_version": saved["version"],
        "weeks_planned": len(weeks),
        "distinct_courses": len(set(course_ids)),
        "total_hours": round(total_hours, 1),
        "saved_at": saved["updated_at"],
    }


# ==========================================================================
# TOOL 5
# ==========================================================================

def record_completion(employee_id: str, course_id: str) -> dict:
    """Mark one course complete and return the learner's updated progress.

    Handles the awkward cases instead of raising: no plan yet, a course id that
    is not in the plan, a course id that is not in the catalogue at all, and a
    repeat completion.
    """
    learner_db.init_db()

    course_id = (course_id or "").strip().upper()
    catalogue_ids = {course["id"] for course in _load_courses()}
    planned = learner_db.plan_course_ids(employee_id)
    already_done = learner_db.completed_course_ids(employee_id)

    if not catalogue_ids:
        return {
            "status": "empty_catalogue",
            "course_id": course_id,
            "message": (
                "No courses have been generated yet, so there is nothing to validate "
                f"{course_id} against. Nothing was recorded. Generate a learning plan "
                "first - that is what populates the catalogue."
            ),
        }

    if course_id not in catalogue_ids:
        return {
            "status": "unknown_course",
            "course_id": course_id,
            "message": (
                f"{course_id} is not a known course id. Nothing was recorded. Known ids "
                "are the ones the model has proposed so far, listed below."
            ),
            "valid_course_ids": sorted(catalogue_ids),
        }

    if course_id in already_done:
        return {
            "status": "already_recorded",
            "course_id": course_id,
            "message": f"{course_id} was already marked complete for {employee_id}.",
            "progress": _progress(employee_id, planned),
        }

    in_plan = course_id in planned
    learner_db.log_completion(employee_id, course_id, was_in_plan=in_plan)

    if not planned:
        status, message = "recorded_without_plan", (
            f"{course_id} was recorded, but {employee_id} has no learning plan yet, "
            "so there is no progress to measure it against."
        )
    elif not in_plan:
        status, message = "recorded_off_plan", (
            f"{course_id} is not in {employee_id}'s current plan. It was logged as "
            "off-plan learning and does not count toward plan progress. Consider "
            "regenerating the plan if their goals have changed."
        )
    else:
        status, message = "recorded", f"{course_id} marked complete for {employee_id}."

    return {
        "status": status,
        "course_id": course_id,
        "was_in_plan": in_plan,
        "message": message,
        "progress": _progress(employee_id, planned),
    }


def _progress(employee_id: str, planned: list[str]) -> dict:
    """Progress against the current plan only - off-plan courses are counted separately."""
    done = learner_db.completed_course_ids(employee_id)
    done_in_plan = [cid for cid in planned if cid in done]
    remaining = [cid for cid in planned if cid not in done]
    percent = round(100 * len(done_in_plan) / len(planned)) if planned else 0

    return {
        "courses_in_plan": len(planned),
        "completed_in_plan": len(done_in_plan),
        "percent_complete": percent,
        "remaining_courses": remaining,
        "completed_off_plan": [cid for cid in done if cid not in planned],
    }


# ==========================================================================
# TOOL 7
# ==========================================================================

# Fields save_learner_profile accepts. Anything else the model sends is dropped
# rather than stored, so a chatty model cannot invent columns.
PROFILE_FIELDS = (
    "name", "current_role", "experience_years", "current_skills",
    "target_role", "hours_per_week", "timeline_months",
)


def save_learner_profile(employee_id: str, **fields) -> dict:
    """Store what the learner said about themselves, and return a receipt only.

    This is the "collect learner information" step made explicit. The learner types
    their name and details once; they belong in the database so progress can be
    tracked against them later, NOT in the plan that gets printed back. So this
    returns a short confirmation - which fields landed, which are still unknown -
    and never echoes the values. Nothing personal travels back through the model's
    context just to be repeated on screen.
    """
    learner_db.init_db()

    if learner_db.get_learner(employee_id) is None:
        learner_db.create_learner(employee_id)

    updates = {key: fields.get(key) for key in PROFILE_FIELDS if fields.get(key) is not None}
    learner = learner_db.update_learner(employee_id, **updates) or {}

    still_missing = [
        field for field in ("current_role", "target_role", "hours_per_week")
        if not learner.get(field)
    ]

    return {
        "status": "saved",
        "employee_id": employee_id,
        "fields_saved": sorted(updates),
        "still_missing": still_missing,
        "note": (
            "Stored for progress tracking. Do NOT repeat these details, and especially "
            "not the learner's name, back in the plan - they are already on file."
        ),
    }


# ==========================================================================
# TOOL 8
# ==========================================================================

def record_weeks_completed(employee_id: str, weeks_completed: float) -> dict:
    """Record progress by week: everything scheduled up to week N is finished.

    Learners think in weeks, not course ids - "I've done the first two weeks", not
    "I completed MSLEARN-PYTHON-FUNDAMENTALS". This converts one into the other.

    A course only counts as complete when its LAST scheduled week falls inside the
    range. A 12-hour course spanning weeks 3-5 is not finished at the end of week 4,
    and recording it as done would overstate progress and skew every later plan.

    The declared week count is stored on the learner as well, because it is a fact
    the completions log cannot always reproduce: if no course happens to finish
    inside the range, nothing is logged, and "I did two weeks" would vanish by the
    next session. Weeks elapsed and courses completed are related but not the same
    thing, so both are kept.
    """
    learner_db.init_db()

    try:
        weeks = int(float(weeks_completed))
    except (TypeError, ValueError):
        return {
            "status": "bad_input",
            "message": f"weeks_completed must be a number, got {weeks_completed!r}.",
        }

    if weeks < 0:
        return {"status": "bad_input", "message": "weeks_completed cannot be negative."}

    record = learner_db.get_plan_record(employee_id)
    if record is None:
        return {
            "status": "no_plan",
            "message": (
                f"{employee_id} has no stored plan, so there are no weeks to complete. "
                "Build a plan first."
            ),
        }

    plan_weeks = record["plan"].get("weekly_plan", []) or []
    total_weeks = len(plan_weeks)

    capped = weeks > total_weeks
    weeks = min(weeks, total_weeks)

    # Where each course starts and ends in the plan.
    first_week: dict[str, int] = {}
    last_week: dict[str, int] = {}
    for index, week in enumerate(plan_weeks, 1):
        for course_id in week.get("courses") or []:
            cid = str(course_id).strip().upper()
            first_week.setdefault(cid, index)
            last_week[cid] = index

    already_done = set(learner_db.completed_course_ids(employee_id))
    finished, newly, in_progress = [], [], []

    for cid, ends in sorted(last_week.items(), key=lambda item: item[1]):
        if ends <= weeks:
            finished.append(cid)
            if cid not in already_done:
                learner_db.log_completion(employee_id, cid, was_in_plan=True)
                newly.append(cid)
        elif first_week[cid] <= weeks:
            in_progress.append(cid)

    if learner_db.get_learner(employee_id) is None:
        learner_db.create_learner(employee_id)
    learner_db.update_learner(employee_id, weeks_completed=weeks)

    planned = learner_db.plan_course_ids(employee_id)
    progress = _progress(employee_id, planned)

    message = f"Weeks 1-{weeks} of {total_weeks} marked complete for {employee_id}."
    if weeks == 0:
        message = f"Progress reset to zero weeks for {employee_id}."
    if capped:
        message += f" The plan is only {total_weeks} weeks long, so it was capped there."
    if newly:
        message += f" That completed {len(newly)} course(s)."
    if in_progress:
        message += (
            f" {len(in_progress)} course(s) are part-way through and were left open: "
            + ", ".join(in_progress) + "."
        )
    elif not newly and weeks:
        message += (
            " No course finishes inside that range yet, so the week count moved but the "
            "course percentage did not."
        )

    return {
        "status": "recorded",
        "weeks_completed": weeks,
        "weeks_in_plan": total_weeks,
        "weeks_remaining_in_plan": max(0, total_weeks - weeks),
        "newly_completed": newly,
        "already_completed": [cid for cid in finished if cid not in newly],
        "still_in_progress": in_progress,
        "progress": progress,
        "message": message,
    }


# ==========================================================================
# TOOL 6
# ==========================================================================

def get_learning_progress(employee_id: str) -> dict:
    """Read the stored plan and measure the learner's real position in it.

    This is the tool that makes *adjusting* a path possible rather than only
    building one. Replanning needs three things the other tools do not give:
    which weeks are already spent, which courses are still outstanding and how
    many hours they need, and how much of the original timeline is left. With
    those, the model can rebuild the remainder instead of starting over.

    Whether the plan is still achievable is answered here, in Python, not left to
    the model to work out: remaining_hours against hours_available is exactly the
    arithmetic small models get wrong.
    """
    learner_db.init_db()

    learner = learner_db.get_learner(employee_id) or {}
    record = learner_db.get_plan_record(employee_id)
    done = learner_db.completed_course_ids(employee_id)

    if record is None:
        return {
            "employee_id": employee_id,
            "has_plan": False,
            "completed_courses": done,
            "message": (
                f"{employee_id} has no stored plan, so there is nothing to adjust. Build a "
                "plan first with get_skill_assessment and search_courses."
            ),
        }

    plan = record["plan"]
    weeks = plan.get("weekly_plan", []) or []
    planned = learner_db.plan_course_ids(employee_id)
    remaining = [cid for cid in planned if cid not in done]
    catalogue = {course["id"]: course for course in _load_courses()}

    # The first week that still holds an unfinished course. Everything before it
    # is treated as time already spent. Project weeks with no courses fall inside
    # that prefix, so an empty week cannot silently consume the timeline on its own.
    derived_elapsed = len(weeks)
    for index, week in enumerate(weeks):
        ids = [str(cid).strip().upper() for cid in (week.get("courses") or [])]
        if any(cid not in done for cid in ids):
            derived_elapsed = index
            break

    # Two sources, and they can disagree. Completions imply weeks; a learner saying
    # "I did two weeks" states them outright. Take whichever is further along: if
    # they finished a course early the log knows best, and if they worked through a
    # long course without finishing it only their own account knows.
    declared_elapsed = int(float(learner.get("weeks_completed") or 0))
    weeks_elapsed = min(len(weeks), max(derived_elapsed, declared_elapsed))

    hours_per_week = learner.get("hours_per_week")
    timeline_months = learner.get("timeline_months")
    total_weeks = int(round(timeline_months * 4.345)) if timeline_months else len(weeks)
    weeks_remaining = max(0, total_weeks - weeks_elapsed)

    remaining_detail: list[dict] = []
    remaining_hours = 0.0
    for course_id in remaining:
        course = catalogue.get(course_id)
        if course is None:
            remaining_detail.append({
                "id": course_id,
                "title": None,
                "note": "in the plan but no longer in the catalogue - drop it or search again",
            })
            continue

        hours = float(course.get("duration_hours") or 0)
        remaining_hours += hours
        remaining_detail.append({
            "id": course_id,
            "title": course["title"],
            "level": course.get("level"),
            "duration_hours": hours,
            "prerequisites": course.get("prerequisites", []),
        })

    hours_available = (
        round(float(hours_per_week) * weeks_remaining, 1) if hours_per_week else None
    )
    on_track = hours_available is None or remaining_hours <= hours_available

    log = learner_db.completion_log(employee_id)

    return {
        "employee_id": employee_id,
        "has_plan": True,
        "plan_version": record["version"],
        "plan_updated_at": record["updated_at"],
        "hours_per_week": hours_per_week,
        "timeline_months": timeline_months,
        "weeks_in_plan": len(weeks),
        "weeks_elapsed": weeks_elapsed,
        "weeks_elapsed_declared": declared_elapsed,
        "weeks_elapsed_from_completions": derived_elapsed,
        "weeks_remaining": weeks_remaining,
        "progress": _progress(employee_id, planned),
        "remaining_courses": remaining_detail,
        "remaining_hours": round(remaining_hours, 1),
        "hours_available": hours_available,
        "on_track": on_track,
        "completion_history": log[-10:],
        "note": (
            "ARITHMETIC, already computed - do not recalculate it. The outstanding courses "
            f"need {round(remaining_hours, 1)} h and the learner has "
            f"{hours_available if hours_available is not None else 'an unknown number of'} h "
            f"left ({hours_per_week or '?'} h/week x {weeks_remaining} weeks). "
            + (
                "That fits, so an adjusted plan can keep every outstanding course - just "
                "re-sequence what is left and renumber the weeks from 1."
                if on_track else
                "That does NOT fit. Drop whole outstanding courses, keeping the ones closest "
                "to the target role, until the remaining durations fit the hours available. "
                "Do not extend past the learner's timeline."
            )
            + " Courses in progress.completed_in_plan are finished: never schedule them again, "
            "and treat their skills as held when checking prerequisites. Courses in "
            "progress.completed_off_plan were learned outside the plan - credit those skills too."
        ),
    }
