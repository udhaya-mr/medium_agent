"""Agent evaluation harness.

An agent that "seems to work" is not evidence of anything. This file is the
evidence: a fixed set of checks, each one either passing or failing, run against a
throwaway database so it can never touch real learner data.

Two tiers, because they cost different things:

  OFFLINE (default, free)   No model calls at all. Exercises everything Python is
                            responsible for: constraint parsing, the skill-gap
                            framework, plan validation, completion bookkeeping,
                            progress arithmetic, and the final-shape dispatch.
                            This is where most agent bugs actually live.

  ONLINE (--online, costs)  Real agent runs against Azure OpenAI. Checks the things
                            only a live model can demonstrate: that it clarifies
                            instead of inventing, that it calls tools in a sensible
                            order, that its plan survives validation with zero
                            repairs needed, and that a replan drops completed work.

Run:
    python eval.py                 # offline only, no tokens spent
    python eval.py --online        # everything, several model calls
    python eval.py --suite validation --suite progress
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The harness must never write to the real learner.db. LEARNER_DB is read when
# learner_db is imported, so it has to be set BEFORE any project import below.
EVAL_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_learner.db")
os.environ["LEARNER_DB"] = EVAL_DB

import agent                      # noqa: E402  (import order is deliberate)
import azure_client               # noqa: E402
import learner_db                 # noqa: E402
import tools                      # noqa: E402


# ==========================================================================
# Fixtures
# ==========================================================================

# A fake catalogue with known durations and one real prerequisite chain
# (EV-PY -> EV-LLM -> EV-RAG), so sequencing and hour checks have something
# deterministic to bite on. Nothing here comes from the model.
FIXTURE_COURSES = [
    {"id": "EV-PY", "title": "Python for Java Developers", "provider": "Eval",
     "topic": "python", "level": "Beginner", "duration_hours": 10,
     "prerequisites": [], "skills_taught": ["python"], "source": "eval-fixture"},
    {"id": "EV-LLM", "title": "LLM Fundamentals", "provider": "Eval",
     "topic": "llm", "level": "Intermediate", "duration_hours": 12,
     "prerequisites": ["EV-PY"], "skills_taught": ["llm fundamentals", "prompt engineering"],
     "source": "eval-fixture"},
    {"id": "EV-RAG", "title": "RAG in Practice", "provider": "Eval",
     "topic": "rag", "level": "Advanced", "duration_hours": 15,
     "prerequisites": ["EV-LLM"], "skills_taught": ["rag", "embeddings"],
     "source": "eval-fixture"},
    {"id": "EV-EVAL", "title": "Evaluating LLM Applications", "provider": "Eval",
     "topic": "evaluation", "level": "Intermediate", "duration_hours": 8,
     "prerequisites": [], "skills_taught": ["evaluation"], "source": "eval-fixture"},
]

REQUEST_FULL = (
    "I am a 6-year Java developer. I want to become an AI application developer in "
    "six months. I can spend 5 hours per week."
)


def reset_db() -> None:
    """Start every run from an empty database so results are reproducible."""
    if os.path.exists(EVAL_DB):
        os.remove(EVAL_DB)
    learner_db.init_db()


def seed_courses() -> None:
    learner_db.upsert_courses(FIXTURE_COURSES)


def week(number: int, courses: list[str], hours: float,
         assignment: str = "Ship one working example.") -> dict:
    return {"week": number, "focus": "eval fixture", "courses": courses,
            "hours": hours, "assignment": assignment}


def plan_of(weeks: list[dict], **overrides) -> dict:
    base = {
        "learner_summary": "Eval fixture learner.",
        "skill_gaps": ["python", "llm fundamentals"],
        "weekly_plan": weeks,
        "estimated_completion": "5 weeks",
    }
    base.update(overrides)
    return base


# A plan that satisfies every rule: EV-PY needs 10 h so it gets 2 weeks at
# 5 h/week, EV-LLM needs 12 h so it gets 3, and EV-LLM's prerequisite is earlier.
GOOD_PLAN = plan_of([
    week(1, ["EV-PY"], 5),
    week(2, ["EV-PY"], 5),
    week(3, ["EV-LLM"], 5),
    week(4, ["EV-LLM"], 5),
    week(5, ["EV-LLM"], 2),
])

GOOD_CONSTRAINTS = {"hours_per_week": 5, "max_weeks": 8}


# ==========================================================================
# Result collection
# ==========================================================================

class Results:
    """Flat list of (suite, check, passed, detail). Printed as it goes."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []
        self.suite = ""

    def start(self, suite: str, note: str = "") -> None:
        self.suite = suite
        print()
        print(f"--- {suite} " + "-" * max(0, 70 - len(suite)))
        if note:
            print(f"    {note}")

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        passed = bool(passed)
        self.rows.append((self.suite, name, passed, detail))
        mark = "PASS" if passed else "FAIL"
        line = f"  [{mark}] {name}"
        if detail and not passed:
            line += f"\n         {detail}"
        print(line)
        return passed

    def equals(self, name: str, actual, expected) -> bool:
        return self.check(name, actual == expected, f"expected {expected!r}, got {actual!r}")

    def summary(self) -> int:
        total = len(self.rows)
        failed = [row for row in self.rows if not row[2]]
        print()
        print("=" * 78)
        print(f"EVALUATION SUMMARY: {total - len(failed)}/{total} checks passed")
        print("=" * 78)

        by_suite: dict[str, list[bool]] = {}
        for suite, _, passed, _ in self.rows:
            by_suite.setdefault(suite, []).append(passed)
        for suite, flags in by_suite.items():
            print(f"  {suite:<26} {sum(flags)}/{len(flags)}")

        if failed:
            print()
            print("  FAILED CHECKS")
            for suite, name, _, detail in failed:
                print(f"    {suite} / {name}")
                if detail:
                    print(f"      {detail}")
        print("=" * 78)
        return 1 if failed else 0


# ==========================================================================
# OFFLINE SUITE: constraint parsing
# ==========================================================================

def suite_constraints(r: Results) -> None:
    r.start("constraints", "infer_constraints() reads the numbers; resolve_constraints() "
                           "falls back to the profile")

    full = agent.infer_constraints(REQUEST_FULL)
    r.equals("hours per week from 'S hours per week'", full["hours_per_week"], 5.0)
    r.equals("timeline from 'six months' (word number)", full["timeline_months"], 6.0)
    r.equals("weeks derived from months", full["max_weeks"], 26)
    r.equals("experience from '6-year developer'", full["experience_years"], 6.0)

    slash = agent.infer_constraints("I can do 3 hrs/week towards Kubernetes.")
    r.equals("hours from 'hrs/week'", slash["hours_per_week"], 3.0)

    explicit = agent.infer_constraints("10 hours each week for 12 weeks.")
    r.equals("hours from 'each week'", explicit["hours_per_week"], 10.0)
    r.equals("weeks stated directly", explicit["max_weeks"], 12.0)

    vague = agent.infer_constraints("I want to get into AI.")
    r.check("nothing invented when nothing is stated",
            vague["hours_per_week"] is None and vague["max_weeks"] is None,
            f"got {vague}")

    # Profile fallback: the learner says nothing about capacity, but we stored it.
    learner_db.create_learner("EVAL-FALLBACK", hours_per_week=5, timeline_months=6)
    fallback = agent.resolve_constraints("EVAL-FALLBACK", "Please adjust my plan.")
    r.equals("hours filled in from the stored profile", fallback["hours_per_week"], 5)
    r.equals("weeks derived from the stored timeline", fallback["max_weeks"], 26)

    override = agent.resolve_constraints("EVAL-FALLBACK", "I only have 3 hours a week now.")
    r.equals("a newly stated capacity beats the stored one", override["hours_per_week"], 3.0)
    r.equals("timeline still comes from the profile", override["timeline_months"], 6)


# ==========================================================================
# OFFLINE SUITE: skill gap framework
# ==========================================================================

def suite_skills(r: Results) -> None:
    r.start("skills", "get_skill_assessment() credits implied skills instead of "
                      "re-teaching them")

    java = tools.get_skill_assessment(
        ["java", "spring boot", "sql"],
        "AI application developer",
        current_role="java developer",
        experience_years=6,
    )
    r.check("python is reported as a gap", "python" in java["skill_gaps"],
            f"gaps: {java['skill_gaps']}")
    r.check("OOP credited from the current role, not re-taught",
            "oop" in java["transferable_skills"],
            f"transferable: {java['transferable_skills']}")
    r.check("target role matched the framework exactly", java["matched_exactly"] is True)
    r.equals("experience is echoed back for the profile", java["experience_years"], 6)

    unknown = tools.get_skill_assessment(["python"], "Quantum Blockchain Ninja")
    r.check("an unknown target role does not crash", unknown["matched_exactly"] is False)
    r.equals("it falls back to the default framework",
             unknown["matched_role_framework"], tools.DEFAULT_TARGET_ROLE)

    partial = tools.get_skill_assessment(
        ["llm fundamentals", "prompt engineering"], "prompt engineer")
    r.check("already-held skills are excluded from the gaps",
            "llm fundamentals" in partial["skills_already_held"]
            and "llm fundamentals" not in partial["skill_gaps"],
            f"held: {partial['skills_already_held']} gaps: {partial['skill_gaps']}")
    r.check("coverage is above zero when skills overlap",
            partial["coverage_percent"] > 0, f"coverage: {partial['coverage_percent']}")


# ==========================================================================
# OFFLINE SUITE: structured output handling
# ==========================================================================

def suite_json(r: Results) -> None:
    r.start("json", "extract_json() survives the ways models wrap JSON")

    r.equals("bare object", azure_client.extract_json('{"a": 1}'), {"a": 1})
    r.equals("markdown fenced", azure_client.extract_json('```json\n{"a": 1}\n```'), {"a": 1})
    r.equals("wrapped in prose",
             azure_client.extract_json('Sure!\n{"a": 1}\nHope that helps.'), {"a": 1})
    r.equals("no JSON at all returns None",
             azure_client.extract_json("I could not do that."), None)


def suite_dispatch(r: Results) -> None:
    r.start("dispatch", "classify_final() routes the three contracted output shapes")

    r.equals("clarification shape",
             agent.classify_final({"needs_clarification": True, "questions": ["?"]}),
             "clarification")
    r.equals("progress shape via progress_update",
             agent.classify_final({"progress_update": True, "message": "done"}), "progress")
    r.equals("progress shape via a bare progress block",
             agent.classify_final({"progress": {"percent_complete": 10}}), "progress")
    r.equals("plan shape", agent.classify_final(GOOD_PLAN), "plan")
    r.equals("anything else is unknown, not silently a plan",
             agent.classify_final({"hello": "world"}), "unknown")

    # The model once dated a 35-week plan two and a half years out, so the finish date
    # is computed in Python and overwrites whatever it wrote.
    from datetime import date
    long_plan = plan_of([week(n, [], 3) for n in range(1, 36)])
    r.equals("35 weeks from a fixed start lands on the right date",
             agent._stamp_completion(long_plan, date(2026, 8, 7)),
             "35 weeks, finishing on or around 2027-04-09")
    r.equals("a five-week plan too",
             agent._stamp_completion(GOOD_PLAN, date(2026, 8, 7)),
             "5 weeks, finishing on or around 2026-09-11")


# ==========================================================================
# OFFLINE SUITE: plan validation
# ==========================================================================

def suite_validation(r: Results) -> None:
    r.start("validation", "validate_plan() is the guard rail - each broken plan must be "
                          "caught by name")

    violations, notes = agent.validate_plan(GOOD_PLAN, GOOD_CONSTRAINTS)
    r.check("a compliant plan produces zero violations", not violations,
            f"violations: {violations}")
    r.check("notes still report the total hours",
            any("Plan totals" in note for note in notes), f"notes: {notes}")

    bad_cases = [
        ("a week over the hours limit", "exceeds",
         plan_of([week(1, ["EV-PY"], 9), week(2, ["EV-PY"], 5)]), GOOD_CONSTRAINTS),

        ("a course id the model invented", "not a real course id",
         plan_of([week(1, ["EV-NOPE"], 5)]), GOOD_CONSTRAINTS),

        ("a long course crammed into one week", "needs 15",
         plan_of([week(1, ["EV-RAG"], 5)]), GOOD_CONSTRAINTS),

        ("a prerequisite scheduled after its course", "before its prerequisite",
         plan_of([week(1, ["EV-LLM"], 5), week(2, ["EV-LLM"], 5), week(3, ["EV-LLM"], 2),
                  week(4, ["EV-PY"], 5), week(5, ["EV-PY"], 5)]), GOOD_CONSTRAINTS),

        ("a gap in the week numbering", "Week numbers must run",
         plan_of([week(1, ["EV-PY"], 5), week(3, ["EV-PY"], 5)]), GOOD_CONSTRAINTS),

        ("a week with no assignment", "'assignment' is empty",
         plan_of([week(1, ["EV-PY"], 5, assignment=""), week(2, ["EV-PY"], 5)]),
         GOOD_CONSTRAINTS),

        ("a plan longer than the timeline", "Compress it",
         GOOD_PLAN, {"hours_per_week": 5, "max_weeks": 2}),

        ("a missing top-level field", "learner_summary",
         plan_of([week(1, ["EV-PY"], 5), week(2, ["EV-PY"], 5)], learner_summary=""),
         GOOD_CONSTRAINTS),

        ("an empty plan", "weekly_plan is missing or empty",
         plan_of([]), GOOD_CONSTRAINTS),
    ]

    for name, expected, plan, constraints in bad_cases:
        found, _ = agent.validate_plan(plan, constraints)
        r.check(f"caught: {name}",
                any(expected in violation for violation in found),
                f"no violation contained {expected!r}; got {found}")

    # A prerequisite absent from the plan is a note, not a failure: an experienced
    # developer should not be forced through an introductory course.
    waived = plan_of([week(1, ["EV-RAG"], 5), week(2, ["EV-RAG"], 5), week(3, ["EV-RAG"], 5)])
    found, notes = agent.validate_plan(waived, GOOD_CONSTRAINTS)
    r.check("an absent prerequisite is waived, not failed",
            not any("prerequisite" in v for v in found)
            and any("waived" in note for note in notes),
            f"violations: {found} notes: {notes}")


# ==========================================================================
# OFFLINE SUITE: completion bookkeeping
# ==========================================================================

def suite_empty_catalogue(r: Results) -> None:
    """Must run before the fixtures are seeded - that is the whole point."""
    r.start("empty catalogue", "nothing can be recorded before any course exists")

    result = tools.record_completion("EVAL-EMPTY", "EV-PY")
    r.equals("recording against an empty catalogue is refused",
             result["status"], "empty_catalogue")
    r.check("and nothing was written",
            learner_db.completed_course_ids("EVAL-EMPTY") == [],
            f"log: {learner_db.completed_course_ids('EVAL-EMPTY')}")


def suite_completions(r: Results) -> None:
    r.start("completions", "record_completion() returns a status for every awkward case "
                           "instead of raising")

    emp = "EVAL-EMP-1"
    profile = tools.get_employee_profile(emp)
    r.check("a new employee gets a record created",
            profile["profile_existed"] is False and profile["profile"]["current_role"] is None)
    r.check("and the unknown fields are named",
            set(profile["missing_fields"]) == {"current_role", "target_role", "hours_per_week"},
            f"missing: {profile['missing_fields']}")

    r.equals("an unknown course id is rejected",
             tools.record_completion(emp, "EV-NOPE")["status"], "unknown_course")
    r.check("the rejection lists the valid ids",
            "EV-PY" in tools.record_completion(emp, "EV-NOPE")["valid_course_ids"])

    r.equals("a completion with no plan yet is still logged",
             tools.record_completion(emp, "EV-PY")["status"], "recorded_without_plan")

    tools.update_learning_plan(emp, plan_of([
        week(1, ["EV-LLM"], 5), week(2, ["EV-LLM"], 5), week(3, ["EV-LLM"], 2),
        week(4, ["EV-RAG"], 5), week(5, ["EV-RAG"], 5), week(6, ["EV-RAG"], 5),
    ]))

    in_plan = tools.record_completion(emp, "EV-LLM")
    r.equals("a planned course is recorded", in_plan["status"], "recorded")
    r.equals("progress counts it", in_plan["progress"]["completed_in_plan"], 1)
    r.equals("percentage is against the plan only",
             in_plan["progress"]["percent_complete"], 50)
    r.equals("the earlier off-plan course is kept separate",
             in_plan["progress"]["completed_off_plan"], ["EV-PY"])

    r.equals("a repeat completion is not double counted",
             tools.record_completion(emp, "EV-LLM")["status"], "already_recorded")

    off = tools.record_completion(emp, "EV-EVAL")
    r.equals("a real course outside the plan is flagged", off["status"], "recorded_off_plan")
    r.check("and excluded from plan progress",
            off["progress"]["percent_complete"] == 50,
            f"percent: {off['progress']['percent_complete']}")

    # Persistence: a second "process" reading the same employee sees the history.
    reread = tools.get_employee_profile(emp)
    r.check("state survives for the next run",
            reread["profile_existed"] is True and reread["has_existing_plan"] is True
            and "EV-LLM" in reread["completed_courses"],
            f"reread: {reread}")

    plan_before = learner_db.get_plan_record(emp)["version"]
    tools.update_learning_plan(emp, plan_of([week(1, ["EV-EVAL"], 5)]))
    r.equals("overwriting a plan bumps the version",
             learner_db.get_plan_record(emp)["version"], plan_before + 1)


# ==========================================================================
# OFFLINE SUITE: profile collection
# ==========================================================================

def suite_profile(r: Results) -> None:
    r.start("profile", "save_learner_profile() stores the details and hands back a receipt, "
                       "not the values")

    emp = "EVAL-PROFILE-1"
    receipt = tools.save_learner_profile(
        emp,
        name="Priya Raman",
        current_role="Java developer",
        experience_years=6,
        target_role="AI application developer",
        hours_per_week=5,
        timeline_months=6,
    )
    r.equals("it reports success", receipt["status"], "saved")
    r.check("it names the fields it stored",
            {"name", "current_role", "hours_per_week"} <= set(receipt["fields_saved"]),
            f"saved: {receipt['fields_saved']}")
    r.equals("nothing is still missing", receipt["still_missing"], [])

    # The receipt goes back into the model's context. The values must not.
    r.check("the receipt does not echo the values back",
            "Priya Raman" not in json.dumps(receipt),
            f"receipt leaked the name: {receipt}")

    stored = learner_db.get_learner(emp)
    r.equals("the name reached the database", stored["name"], "Priya Raman")
    r.equals("so did the role", stored["current_role"], "Java developer")
    r.equals("and the capacity", stored["hours_per_week"], 5)

    # A partial save must not blank out what is already there.
    tools.save_learner_profile(emp, hours_per_week=3)
    kept = learner_db.get_learner(emp)
    r.equals("a partial update leaves the name alone", kept["name"], "Priya Raman")
    r.equals("and applies the new value", kept["hours_per_week"], 3)

    # A field the schema does not define is dropped, not stored, not fatal.
    odd = tools.save_learner_profile(emp, favourite_colour="blue")
    r.check("an unknown field is ignored rather than stored",
            "favourite_colour" not in odd["fields_saved"], f"saved: {odd['fields_saved']}")

    fresh = tools.save_learner_profile("EVAL-PROFILE-2", name="Sam")
    r.check("an unknown employee is created on the spot",
            learner_db.get_learner("EVAL-PROFILE-2") is not None)
    r.check("and the fields they did not give are reported missing",
            set(fresh["still_missing"]) == {"current_role", "target_role", "hours_per_week"},
            f"missing: {fresh['still_missing']}")


# ==========================================================================
# OFFLINE SUITE: identity and week-based progress
# ==========================================================================

def suite_identity(r: Results) -> None:
    r.start("identity", "the email is the identity - one address, one record")

    r.equals("case and whitespace are normalised",
             learner_db.key_for_email("  Deepak@LevelShift.com "), "deepak@levelshift.com")
    r.equals("mail-client angle brackets are stripped",
             learner_db.key_for_email("<deepak@levelshift.com>"), "deepak@levelshift.com")
    r.equals("two spellings resolve to one key",
             learner_db.key_for_email("DEEPAK@levelshift.com"),
             learner_db.key_for_email("deepak@levelshift.com "))

    r.check("a plausible address is accepted", agent._looks_like_email("a.b@c.co"))
    for bad in ("deepak", "deepak@", "@levelshift.com", "deepak@levelshift",
                "two words@x.com", "a@b@c.com"):
        r.check(f"rejected: {bad!r}", not agent._looks_like_email(bad))


def suite_weeks(r: Results) -> None:
    r.start("weeks", "record_weeks_completed() converts weeks into course completions "
                     "without overstating them")

    # 3 h/week. EV-PY (10 h) spans weeks 1-4, EV-EVAL (8 h) spans weeks 5-7.
    emp = "EVAL-WEEKS-1"
    learner_db.create_learner(emp, hours_per_week=3, timeline_months=3)
    tools.update_learning_plan(emp, plan_of([
        week(1, ["EV-PY"], 3), week(2, ["EV-PY"], 3), week(3, ["EV-PY"], 3),
        week(4, ["EV-PY"], 1), week(5, ["EV-EVAL"], 3), week(6, ["EV-EVAL"], 3),
        week(7, ["EV-EVAL"], 2),
    ]))

    r.equals("no plan means nothing to record",
             tools.record_weeks_completed("EVAL-WEEKS-NOBODY", 2)["status"], "no_plan")
    r.equals("a non-numeric week count is refused",
             tools.record_weeks_completed(emp, "soon")["status"], "bad_input")

    # Two weeks in, the four-week course is NOT finished.
    two = tools.record_weeks_completed(emp, 2)
    r.equals("it records the week count", two["weeks_completed"], 2)
    r.equals("a part-finished course is not marked complete", two["newly_completed"], [])
    r.equals("it is reported as still in progress", two["still_in_progress"], ["EV-PY"])
    r.equals("so the course percentage stays at zero",
             two["progress"]["percent_complete"], 0)
    r.check("and the message names the course left open",
            "part-way through" in two["message"] and "EV-PY" in two["message"],
            two["message"])

    # The declared weeks must survive even though no completion was logged.
    resumed = tools.get_learning_progress(emp)
    r.equals("the declared weeks are remembered", resumed["weeks_elapsed"], 2)
    r.equals("even though completions imply none",
             resumed["weeks_elapsed_from_completions"], 0)
    r.equals("remaining weeks come off the timeline", resumed["weeks_remaining"], 11)

    # Week 4 finishes EV-PY.
    four = tools.record_weeks_completed(emp, 4)
    r.equals("finishing the span completes the course", four["newly_completed"], ["EV-PY"])
    r.equals("progress moves to half the plan", four["progress"]["percent_complete"], 50)
    r.equals("nothing is left part-way", four["still_in_progress"], [])

    # Re-recording the same range must not double count.
    again = tools.record_weeks_completed(emp, 4)
    r.equals("re-recording adds nothing new", again["newly_completed"], [])
    r.equals("and reports what was already done", again["already_completed"], ["EV-PY"])
    r.equals("the percentage is unchanged", again["progress"]["percent_complete"], 50)

    # More weeks than the plan has.
    over = tools.record_weeks_completed(emp, 99)
    r.equals("it caps at the plan length", over["weeks_completed"], 7)
    r.check("and says so", "capped" in over["message"], over["message"])
    r.equals("finishing every week completes every course",
             over["progress"]["percent_complete"], 100)

    # A correction downwards is allowed - the learner is the source of truth.
    back = tools.record_weeks_completed(emp, 1)
    r.equals("a correction downwards is accepted", back["weeks_completed"], 1)
    r.check("but completions already logged are not erased",
            back["progress"]["percent_complete"] == 100,
            f"percent: {back['progress']['percent_complete']}")

    # A plan that opens with a courseless project week: recording it moves the week
    # count with nothing to complete and nothing yet started.
    quiet = "EVAL-WEEKS-2"
    learner_db.create_learner(quiet, hours_per_week=3, timeline_months=1)
    tools.update_learning_plan(quiet, plan_of([
        week(1, [], 3), week(2, ["EV-EVAL"], 3), week(3, ["EV-EVAL"], 3),
        week(4, ["EV-EVAL"], 2),
    ]))
    empty = tools.record_weeks_completed(quiet, 1)
    r.equals("a courseless week still counts as done", empty["weeks_completed"], 1)
    r.equals("with nothing completed", empty["newly_completed"], [])
    r.equals("and nothing started", empty["still_in_progress"], [])
    r.check("the message explains the percentage did not move",
            "No course finishes inside that range" in empty["message"], empty["message"])


# ==========================================================================
# OFFLINE SUITE: progress and replan arithmetic
# ==========================================================================

def suite_progress(r: Results) -> None:
    r.start("progress", "get_learning_progress() computes the replan arithmetic in Python, "
                        "not in the model")

    r.equals("no plan means nothing to adjust",
             tools.get_learning_progress("EVAL-NOBODY")["has_plan"], False)

    # 5 h/week, 2 months => 9 weeks total. EV-PY (10 h) fills weeks 1-2 and is
    # finished; EV-LLM (12 h) fills weeks 3-5 and is not.
    on_track = "EVAL-EMP-2"
    learner_db.create_learner(on_track, hours_per_week=5, timeline_months=2)
    tools.update_learning_plan(on_track, plan_of([
        week(1, ["EV-PY"], 5), week(2, ["EV-PY"], 5),
        week(3, ["EV-LLM"], 5), week(4, ["EV-LLM"], 5), week(5, ["EV-LLM"], 2),
    ]))
    tools.record_completion(on_track, "EV-PY")

    report = tools.get_learning_progress(on_track)
    r.equals("finished weeks are counted as spent", report["weeks_elapsed"], 2)
    r.equals("remaining weeks come off the timeline, not the plan length",
             report["weeks_remaining"], 7)
    r.equals("outstanding hours add up", report["remaining_hours"], 12.0)
    r.equals("available hours are hours x remaining weeks", report["hours_available"], 35.0)
    r.check("a learner with room to spare is on track", report["on_track"] is True)
    r.equals("only outstanding courses are listed",
             [course["id"] for course in report["remaining_courses"]], ["EV-LLM"])
    r.equals("progress percentage is unchanged by the arithmetic",
             report["progress"]["percent_complete"], 50)
    r.check("the completion history is returned",
            len(report["completion_history"]) == 1
            and report["completion_history"][0]["course_id"] == "EV-PY",
            f"history: {report['completion_history']}")

    # 1 h/week, 1 month => 4 weeks, 4 hours. EV-RAG needs 15. Impossible.
    behind = "EVAL-EMP-3"
    learner_db.create_learner(behind, hours_per_week=1, timeline_months=1)
    tools.update_learning_plan(behind, plan_of([week(1, ["EV-RAG"], 1)]))

    stuck = tools.get_learning_progress(behind)
    r.check("an impossible remainder is flagged, not glossed over",
            stuck["on_track"] is False,
            f"remaining {stuck['remaining_hours']} h vs {stuck['hours_available']} h")
    r.check("the note tells the model to drop courses",
            "Drop whole outstanding courses" in stuck["note"], stuck["note"])

    # The replan ceiling is the weeks left, not the original timeline.
    replan = agent.resolve_constraints(on_track, "Adjust my plan please.", mode="replan")
    r.equals("replan is validated against the remaining weeks", replan["max_weeks"], 7)
    r.equals("and it knows how many weeks are already gone", replan["weeks_elapsed"], 2)

    plain = agent.resolve_constraints(on_track, "Adjust my plan please.")
    r.equals("a fresh plan is still validated against the full timeline",
             plain["max_weeks"], 9)


def suite_extension(r: Results) -> None:
    r.start("extension", "asking for MORE time must raise the ceiling, not be capped at "
                         "the old deadline")

    # 5 h/week, 2 months (9 weeks). Two weeks spent, so 7 remain.
    emp = "EVAL-EXTEND-1"
    learner_db.create_learner(emp, hours_per_week=5, timeline_months=2)
    tools.update_learning_plan(emp, plan_of([
        week(1, ["EV-PY"], 5), week(2, ["EV-PY"], 5),
        week(3, ["EV-LLM"], 5), week(4, ["EV-LLM"], 5), week(5, ["EV-LLM"], 2),
    ]))
    tools.record_completion(emp, "EV-PY")

    # Saying nothing about time keeps the old ceiling.
    quiet = agent.resolve_constraints(emp, "Please rework my plan.", mode="chat")
    r.equals("silence keeps the remaining-weeks ceiling", quiet["max_weeks"], 7)
    r.check("and is not flagged as an extension", not quiet.get("timeline_extended"))

    # Asking for six more months must NOT be capped at the 7 weeks left.
    longer = agent.resolve_constraints(emp, "Give me six months instead.", mode="chat")
    r.equals("a longer window becomes the new ceiling", longer["max_weeks"], 26)
    r.check("the extension is flagged for the prompt", longer["timeline_extended"] is True)
    r.check("the stored total is rebased to spent + new window",
            abs(longer["timeline_months"] - round((2 + 26) / 4.345, 1)) < 0.05,
            f"timeline_months: {longer['timeline_months']}")

    # More hours per week, same window.
    faster = agent.resolve_constraints(emp, "I can do 12 hours a week now.", mode="chat")
    r.equals("a new weekly capacity is taken from the message",
             faster["hours_per_week"], 12.0)
    r.equals("with the week ceiling untouched", faster["max_weeks"], 7)

    # A shorter window still narrows, and is not called an extension.
    shorter = agent.resolve_constraints(emp, "I only have 3 weeks left.", mode="chat")
    r.equals("a shorter window narrows the ceiling", shorter["max_weeks"], 3)
    r.check("and is not flagged as an extension",
            shorter["timeline_extended"] is False, f"{shorter}")

    # Both at once - the case the learner actually asked about.
    both = agent.resolve_constraints(
        emp, "I can do 10 hours a week now and give me 4 months.", mode="chat")
    r.equals("hours and weeks both move together", both["hours_per_week"], 10.0)
    r.equals("weeks come from the stated months", both["max_weeks"], 17)
    r.check("and the extra room is flagged", both["timeline_extended"] is True)
    r.equals("so the budget the model is handed grows",
             agent._capacity(both)[2], round(10.0 * 17, 1))

    # A learner with no plan is unaffected by any of this.
    fresh = agent.resolve_constraints(
        "EVAL-EXTEND-NOBODY", "I want to learn RAG, 4 hours a week for 8 weeks.",
        mode="chat")
    r.equals("a learner with no plan just uses what they said", fresh["max_weeks"], 8.0)
    r.check("with no elapsed weeks to subtract", "weeks_elapsed" not in fresh)


# ==========================================================================
# ONLINE SUITE: real agent runs
# ==========================================================================

def _run(r: Results, name: str, request: str, employee_id: str, mode: str = "plan",
         history: list[dict] | None = None):
    """Run the agent, turning an API failure into a failed check rather than a crash."""
    try:
        return agent.process_request(request, employee_id, mode, history)
    except Exception as exc:                       # noqa: BLE001 - report, do not propagate
        r.check(f"{name}: agent run completed", False, f"{type(exc).__name__}: {exc}")
        return None


def suite_online(r: Results) -> None:
    r.start("online", "real Azure OpenAI runs - these cost tokens")

    # --- 1. ambiguous input: must ask, must not guess, must not touch the DB ---
    vague = _run(r, "vague", "I want to get into AI.", "EVAL-ON-VAGUE")
    if vague:
        r.equals("no capacity given -> asks instead of planning", vague["kind"], "clarification")
        r.check("and calls zero tools while asking", vague["call_order"] == [],
                f"called: {vague['call_order']}")
        r.check("nothing was written for that learner",
                learner_db.get_plan("EVAL-ON-VAGUE") is None)

    no_target = _run(r, "no target", "I have 4 hours a week to learn.", "EVAL-ON-NOTARGET")
    if no_target:
        r.equals("no target given -> asks instead of planning",
                 no_target["kind"], "clarification")

    # --- 1b. a brand-new learner must be asked for the whole profile ----------
    def said(result: dict) -> str:
        """What the agent put to the learner, whether as JSON questions or plain speech."""
        questions = (result.get("payload") or {}).get("questions") or []
        return (" ".join(questions) + " " + str(result.get("spoken") or "")).lower()

    opening = _run(r, "opening", agent.OPENING_NUDGE, "EVAL-ON-NEW")
    if opening:
        r.check("a new learner is asked, not planned for",
                opening["kind"] in ("clarification", "spoken"), f"kind: {opening['kind']}")
        asked = said(opening)
        for label, words in (
            ("name", ("name",)),
            ("current role", ("role", "doing", "job")),
            ("experience", ("experience", "long", "years")),
            ("target", ("aiming", "target", "become", "goal", "want")),
            ("capacity", ("hours", "time", "week")),
        ):
            r.check(f"the opening question asks for {label}",
                    any(word in asked for word in words), f"asked: {asked!r}")
        r.check("and it is one question, not a six-field form",
                len((opening["payload"] or {}).get("questions") or []) <= 2,
                f"{len((opening['payload'] or {}).get('questions') or [])} separate questions")

    # A learner whose name and role are already stored must not be asked for them again.
    tools.save_learner_profile("EVAL-ON-PARTIAL", name="Meera", current_role="QA engineer")
    partial = _run(r, "partial", agent.OPENING_NUDGE, "EVAL-ON-PARTIAL",
                   history=agent._profile_context_history("EVAL-ON-PARTIAL"))
    if partial:
        asked = said(partial)
        r.check("it says something rather than discarding the turn",
                asked.strip() != "", f"kind: {partial['kind']} raw: {partial['raw'][:120]!r}")
        r.check("a stored name is not asked for twice", "name" not in asked, f"asked: {asked!r}")
        r.check("but the missing capacity still is",
                any(word in asked for word in ("hours", "time", "week")), f"asked: {asked!r}")

    # --- 2. the full request: must plan, and the plan must be legal -----------
    full = _run(r, "full", REQUEST_FULL, "EVAL-ON-PLAN")
    if full:
        r.equals("a complete request produces a plan", full["kind"], "plan")
        order = full["call_order"]
        r.check("it assessed skills before searching for courses",
                "get_skill_assessment" in order and "search_courses" in order
                and order.index("get_skill_assessment") < order.index("search_courses"),
                f"call order: {order}")
        r.check("the plan passed validation", not full["violations"],
                f"violations: {full['violations']}")

        weeks = (full["payload"] or {}).get("weekly_plan", [])
        r.check("every week has a practical assignment",
                bool(weeks) and all(w.get("assignment") for w in weeks),
                "at least one week has no assignment")
        r.check("the profile was saved for next time",
                (learner_db.get_learner("EVAL-ON-PLAN") or {}).get("target_role") is not None,
                f"learner: {learner_db.get_learner('EVAL-ON-PLAN')}")

        # --- 3. reporting a completion -> progress, not a new plan ------------
        planned = learner_db.plan_course_ids("EVAL-ON-PLAN")
        if planned:
            first = planned[0]
            done = _run(r, "completion",
                        f"I have finished course {first}. Record it and tell me my progress.",
                        "EVAL-ON-PLAN")
            if done:
                r.equals("a completion returns progress, not a plan", done["kind"], "progress")
                r.check("record_completion was the tool chosen",
                        "record_completion" in done["call_order"],
                        f"called: {done['call_order']}")
                r.check("the completion is in the database",
                        first in learner_db.completed_course_ids("EVAL-ON-PLAN"))

            # --- 4. replanning after progress changed ------------------------
            revised = _run(r, "replan",
                           "My situation has changed and I need my plan adjusted. I only have "
                           "3 hours a week now. Give me a revised plan for the remaining weeks.",
                           "EVAL-ON-PLAN", mode="replan")
            if revised:
                r.equals("a replan returns a plan", revised["kind"], "plan")
                r.check("it read progress before rebuilding",
                        "get_learning_progress" in revised["call_order"],
                        f"called: {revised['call_order']}")

                new_weeks = (revised["payload"] or {}).get("weekly_plan", [])
                scheduled = {
                    str(cid).strip().upper()
                    for w in new_weeks for cid in (w.get("courses") or [])
                }
                r.check("the completed course is not scheduled again",
                        first not in scheduled, f"{first} is still in the revised plan")
                r.check("the revision restarts at week 1",
                        bool(new_weeks) and new_weeks[0].get("week") == 1,
                        f"first week: {new_weeks[0] if new_weeks else None}")
                r.check("every week respects the new 3 h/week limit",
                        all((w.get("hours") or 0) <= 3 for w in new_weeks),
                        f"hours: {[w.get('hours') for w in new_weeks]}")
                r.check("the revised plan passed validation", not revised["violations"],
                        f"violations: {revised['violations']}")
                r.check("it was stored as a new version",
                        (learner_db.get_plan_record("EVAL-ON-PLAN") or {}).get("version", 0) > 1)


# ==========================================================================
# Entry point
# ==========================================================================

OFFLINE_SUITES = {
    "empty_catalogue": suite_empty_catalogue,   # must run before seeding
    "constraints": suite_constraints,
    "skills": suite_skills,
    "json": suite_json,
    "dispatch": suite_dispatch,
    "validation": suite_validation,
    "profile": suite_profile,
    "identity": suite_identity,
    "completions": suite_completions,
    "weeks": suite_weeks,
    "progress": suite_progress,
    "extension": suite_extension,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the learning path agent.")
    parser.add_argument("--online", action="store_true",
                        help="Also run real agent runs against Azure OpenAI. Costs tokens.")
    parser.add_argument("--suite", action="append", default=[],
                        choices=sorted(OFFLINE_SUITES) + ["online"],
                        help="Run only these suites. Repeatable.")
    parser.add_argument("--keep-db", action="store_true",
                        help="Leave eval_learner.db behind for inspection.")
    args = parser.parse_args()

    selected = args.suite or list(OFFLINE_SUITES)
    if args.online and "online" not in selected:
        selected.append("online")

    print("=" * 78)
    print("LEARNING PATH AGENT - EVALUATION")
    print("=" * 78)
    print(f"  database : {EVAL_DB} (throwaway - the real learner.db is never touched)")
    print(f"  suites   : {', '.join(selected)}")
    print(f"  model    : {'yes - this run costs tokens' if 'online' in selected else 'no - offline only'}")

    reset_db()
    results = Results()

    # empty_catalogue has to see a catalogue with nothing in it.
    if "empty_catalogue" in selected:
        suite_empty_catalogue(results)
    seed_courses()

    for name in selected:
        if name in ("empty_catalogue", "online"):
            continue
        OFFLINE_SUITES[name](results)

    if "online" in selected:
        suite_online(results)

    exit_code = results.summary()

    if not args.keep_db and os.path.exists(EVAL_DB):
        os.remove(EVAL_DB)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
