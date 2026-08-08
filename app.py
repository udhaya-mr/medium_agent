"""Flask web server for the Learning Path Agent — Jinja2 template edition.

Routes:
    GET  /                  → render main UI via Jinja2 template
    POST /api/session/start → identify learner by email (instant, no LLM)
    POST /api/chat/open     → fire AI opening greeting (async, for new learners)
    POST /api/chat          → send a user message, get AI reply
    GET  /api/progress      → raw progress dict
    GET  /api/plan          → stored plan JSON
"""

from __future__ import annotations

import sys
import threading
import traceback
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import os
sys.path.insert(0, os.path.dirname(__file__))

import agent
import learner_db
import tools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
CORS(app)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _get_session(employee_id: str) -> dict:
    with _sessions_lock:
        if employee_id not in _sessions:
            _sessions[employee_id] = {
                "employee_id": employee_id,
                "history": [],
                "lock": threading.Lock(),
            }
        return _sessions[employee_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_email(value: str) -> bool:
    if value.count("@") != 1 or " " in value:
        return False
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


def _format_plan(plan: dict) -> dict:
    """Enrich plan weeks with course details from the catalogue."""
    catalogue = {c["id"]: c for c in tools._load_courses()}
    weeks = []
    for week in plan.get("weekly_plan", []):
        courses = []
        for cid in (week.get("courses") or []):
            cid_up = str(cid).strip().upper()
            c = catalogue.get(cid_up, {})
            courses.append({
                "id": cid_up,
                "title": c.get("title", cid_up),
                "provider": c.get("provider", ""),
                "level": c.get("level", ""),
                "duration_hours": c.get("duration_hours", 0),
                "course_link": c.get("course_link", ""),
            })
        weeks.append({
            "week": week.get("week"),
            "focus": week.get("focus", ""),
            "hours": week.get("hours", 0),
            "assignment": week.get("assignment", ""),
            "courses": courses,
        })
    return {
        "learner_summary": plan.get("learner_summary", ""),
        "skill_gaps": plan.get("skill_gaps", []),
        "estimated_completion": plan.get("estimated_completion", ""),
        "weekly_plan": weeks,
    }


# ---------------------------------------------------------------------------
# Page Routes  (rendered by Jinja2)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main single-page UI via Jinja2."""
    learner_db.init_db()
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/session/start", methods=["POST"])
def session_start():
    """Identify learner by email. Returns instantly — no LLM call."""
    data = request.get_json(silent=True) or {}
    raw_email = (data.get("email") or "").strip()

    if not _looks_like_email(raw_email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    learner_db.init_db()
    employee_id = learner_db.key_for_email(raw_email)
    sess = _get_session(employee_id)

    with sess["lock"]:
        learner = learner_db.get_learner(employee_id) or {}
        has_plan = learner_db.get_plan(employee_id) is not None
        known_name = learner.get("name")

        # Prime conversation history so agent never re-asks stored facts.
        sess["history"] = agent._profile_context_history(employee_id)

        if has_plan:
            record = learner_db.get_plan_record(employee_id)
            progress = tools.get_learning_progress(employee_id)
            return jsonify({
                "employee_id": employee_id,
                "known_name": known_name,
                "has_plan": True,
                "plan": _format_plan(record["plan"]),
                "plan_version": record["version"],
                "progress": progress,
                "message": (
                    f"Welcome back{', ' + known_name if known_name else ''}! "
                    "I've loaded your learning plan."
                ),
            })

        # New learner — return immediately; frontend calls /api/chat/open next.
        return jsonify({
            "employee_id": employee_id,
            "known_name": known_name,
            "has_plan": False,
            "needs_opening": True,
        })


@app.route("/api/chat/open", methods=["POST"])
def chat_open():
    """Generate the AI's opening greeting for a brand-new learner."""
    data = request.get_json(silent=True) or {}
    employee_id = (data.get("employee_id") or "").strip()
    if not employee_id:
        return jsonify({"error": "No employee_id"}), 400

    sess = _get_session(employee_id)
    with sess["lock"]:
        opening = None
        try:
            opening = agent.process_request(
                agent.OPENING_NUDGE, employee_id, "chat", sess["history"]
            )
        except Exception:
            traceback.print_exc()

        fallback = (
            "Hi! I'm your Learning Path Agent. Tell me your name, "
            "current role, how long you've been doing it, what you're "
            "aiming for, and how many hours a week you can study."
        )
        msg = fallback
        if opening:
            sess["history"] = opening["history"]
            kind = opening.get("kind")
            if kind == "clarification":
                questions = (opening.get("payload") or {}).get("questions") or []
                msg = " ".join(str(q) for q in questions) or fallback
            elif kind == "spoken":
                msg = opening.get("spoken") or fallback
            elif opening.get("raw") and "{" not in (opening.get("raw") or ""):
                msg = opening["raw"]

        return jsonify({"agent_message": msg})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a user message and receive the agent's reply."""
    data = request.get_json(silent=True) or {}
    employee_id = (data.get("employee_id") or "").strip()
    user_message = (data.get("message") or "").strip()

    if not employee_id:
        return jsonify({"error": "No session. Start a session first."}), 400
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    sess = _get_session(employee_id)
    with sess["lock"]:
        try:
            result = agent.process_request(
                user_message, employee_id, "chat", sess["history"]
            )
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500

        sess["history"] = result.get("history", sess["history"])

        kind = result.get("kind", "unknown")
        payload = result.get("payload") or {}
        response: dict[str, Any] = {"kind": kind}

        if kind == "clarification":
            questions = payload.get("questions") or []
            response["agent_message"] = " ".join(str(q) for q in questions)

        elif kind == "progress":
            response["agent_message"] = str(payload.get("message") or "")
            response["progress"] = payload.get("progress") or {}

        elif kind == "plan":
            response["agent_message"] = "Your personalised learning plan is ready! 🎉"
            response["plan"] = _format_plan(payload)
            saved = result.get("saved") or {}
            response["plan_version"] = saved.get("plan_version")

        elif kind == "spoken":
            response["agent_message"] = result.get("spoken", "")

        else:
            raw = result.get("raw", "")
            response["agent_message"] = (
                raw if (raw and "{" not in raw)
                else "I wasn't able to form a clear response. Try rephrasing?"
            )

        # Always refresh progress if a plan exists.
        try:
            prog = tools.get_learning_progress(employee_id)
            if prog.get("has_plan"):
                response["progress"] = prog
        except Exception:
            pass

        return jsonify(response)


@app.route("/api/progress")
def api_progress():
    employee_id = (request.args.get("employee_id") or "").strip()
    if not employee_id:
        return jsonify({"error": "employee_id required"}), 400
    learner_db.init_db()
    try:
        return jsonify(tools.get_learning_progress(employee_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/plan")
def api_plan():
    employee_id = (request.args.get("employee_id") or "").strip()
    if not employee_id:
        return jsonify({"error": "employee_id required"}), 400
    learner_db.init_db()
    record = learner_db.get_plan_record(employee_id)
    if not record:
        return jsonify({"has_plan": False})
    return jsonify({
        "has_plan": True,
        "plan_version": record["version"],
        "plan": _format_plan(record["plan"]),
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    learner_db.init_db()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
