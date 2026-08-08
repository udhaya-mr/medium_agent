"""Medium Agent - Personalized Learning Path Agent (Level 2).

Level 1 was retrieve-then-answer: one prompt in, one answer out. This is an
agent. The difference is the loop in `run_agent()`:

    model decides what it needs
      -> asks for a tool by name with JSON arguments
      -> Python executes that tool and returns real data
      -> model sees the result and decides the next step
      -> repeat until it can produce the final answer

Tool selection is done by the model's own native function calling. There is no
regex router and no keyword matching anywhere in this file - the only thing
Python decides is *how* to execute a tool the model already chose.

The model's final message must be one of three JSON shapes:

  1. clarification    {"needs_clarification": true, "questions": [...]}
  2. learning plan    {"learner_summary": ..., "skill_gaps": [...],
                       "weekly_plan": [...], "estimated_completion": ...}
  3. progress update  {"progress_update": true, "message": ..., "progress": {...}}

Two ways to run it.

INTERACTIVE (no arguments) - the normal way. Asks for whatever details are missing,
builds the plan, then stays open so completions can be recorded and the percentage
checked for as long as the learner wants:

    python agent.py --employee-id EMP-1001

ONE-SHOT (a request as an argument) - for scripts and demos:

    python agent.py "I am a 6-year Java developer. I want to become an AI
                     application developer in six months. I can spend 5 hours per week."
    python agent.py --employee-id EMP-1001 --complete AI-101
    python agent.py --employee-id EMP-1001 --replan "I only have 3 hours a week now"
    python agent.py --employee-id EMP-1001 --show-plan
    python agent.py --employee-id EMP-1001 --progress
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, timedelta

from openai import AzureOpenAI

import learner_db
import tools
from azure_client import build_client, call_with_retry, extract_json

MAX_TURNS = 12          # hard stop so a confused model cannot loop forever
MAX_OUTPUT_TOKENS = 8000  # a 26-week plan is a lot of JSON; reasoning models need headroom

# The tool trace is the teaching material, but it also dumps the learner's profile
# and every course record to the terminal. Off by default so the plan is the output;
# --verbose brings it back for anyone learning how the loop works.
VERBOSE = False


def _trace(message: str = "") -> None:
    """Print only in verbose mode. Everything learner-identifying goes through here."""
    if VERBOSE:
        print(message)


def _step(message: str) -> None:
    """One short line per step, so a 40-second run does not look frozen."""
    print(message)


# ==========================================================================
# System prompt - the agent's operating instructions
# ==========================================================================

SYSTEM_PROMPT = """You are a Learning Path Agent for the LevelShift L&D team. You build \
personalized, sequenced learning journeys for employees using the tools provided.

TOOLS AVAILABLE
  get_employee_profile   - read or create the learner's stored record
  save_learner_profile   - store the details they gave (name, role, experience, capacity)
  get_skill_assessment   - compare current skills against a target role, get the gap list
  search_courses         - generate real courses that close specific skill gaps
  get_learning_progress  - read the stored plan and how far through it the learner really is
  update_learning_plan   - persist a finished plan (the caller does this for you; do not call it)
  record_completion      - mark ONE NAMED COURSE complete and return updated progress
  record_weeks_completed - mark a NUMBER OF WEEKS complete and return updated progress

RULE 0 - THIS IS A CONVERSATION, NOT A FORM
You are talking with one learner across many turns and you can see everything already
said. Work out what they want *right now* from the conversation, and do that:

  they have told you enough and have no plan   -> build one (RULE 2)
  something essential is genuinely missing     -> ask for it, conversationally (RULE 1)
  they are reporting progress                  -> record it (RULE 4)
  they are asking how they are doing           -> get_learning_progress, then report it
  their hours, timeline or goal has changed    -> revise the plan (RULE 5)
  they are asking a question about their plan  -> answer it from the tools, no new plan

Never ask for something they already told you, in this turn or an earlier one. Never ask
for something a tool can read - their profile, plan, completions and progress are all in
the database. Look them up instead of asking.

Talk like a colleague who happens to be good at this. No menus, no numbered forms, no
"please select an option". One or two sentences of plain speech, then the JSON.

RULE 1 - COLLECT THE WHOLE PROFILE BEFORE THE FIRST PLAN, IN AS FEW TURNS AS POSSIBLE
Applies to a learner's FIRST plan only. Skip it entirely when they are reporting progress
or asking for an adjustment - it is all on file by then, so read it with
get_learning_progress instead of asking.

These are what a personalised plan is built on, and all five must be known before you
build one:
  (a) NAME       - what to call them.
  (b) CURRENT ROLE - what they do now, e.g. "Java developer". This is what lets
                     get_skill_assessment credit the skills they already have.
  (c) EXPERIENCE - roughly how many years.
  (d) TARGET     - the role, technology or certification they want to reach.
  (e) CAPACITY   - hours per week they can study.
Timeline in months is optional: if they do not give one, choose something sensible and say
what you chose.

Check the WHOLE conversation and the stored profile first, and never ask for anything
already known. If any of (a) to (e) is still missing, do NOT call search_courses, do NOT
build a plan, and do NOT invent values. Reply with only:
  {"needs_clarification": true, "questions": ["..."]}

Ask for EVERYTHING still missing in ONE natural question, the way a person would say it
out loud:
  - nothing known yet -> "Before I put a plan together - what's your name, what are you
    doing at the moment and for how long, what are you aiming for, and how much time can
    you give it each week?"
  - only capacity missing -> "How much time can you give it each week?"
Never one field per turn. Never a numbered list. Two collecting turns is fine; five is an
interrogation and means you asked badly.

The moment you have them, call save_learner_profile with the whole lot before anything
else, then carry straight on to RULE 2 in the same turn.

Their employee_id is an email address. Do NOT mine it for their name and then greet them
by it while still asking what their name is - that reads as though you were not listening.
Either they have told you their name or it is on file; otherwise just ask.

If they skip something after you have asked once, let it go and work with what you have -
a missing name is not worth a second question. Judge presence, never precision: a named
target is enough, so never ask them to narrow it down, pick a specialisation, or confirm a
timeline they already gave. And never ask what skills they already have - that is what
get_skill_assessment works out from their role.

RULE 2 - GATHER EVIDENCE WITH TOOLS
When you have (a) to (e), work in this order:
  1. get_employee_profile  - see what is already stored and what they have completed
  1b. save_learner_profile - store every detail they stated about themselves, including
                             their name if they gave one. Pass only what they actually
                             said. This is bookkeeping: it goes in the database so their
                             progress has something to hang off, and it is NOT content for
                             the plan.
  2. get_skill_assessment  - pass the learner's stated skills, their current role and
                             years of experience if they gave them (e.g. current_skills
                             ["java", "spring", "sql"], current_role "java developer",
                             experience_years 6) plus the target role, to get a concrete
                             skill_gaps list. current_role and experience_years are also
                             how those details get saved to the learner's profile.
  3. search_courses        - call it with the skill gaps to get course ids.
                             Call it more than once if different gap clusters need it.
You MUST call search_courses before writing a plan. There is no fixed catalogue: the ids,
titles and durations only exist once search_courses has returned them. Never write a course
id into the plan that search_courses did not give you in this conversation - a plan
referencing an unknown id fails validation and is rejected.

RULE 3 - RESPECT THE CONSTRAINTS
  - Every week's "hours" must be less than or equal to the learner's stated hours per week.
  - A course takes as many weeks as its duration_hours needs. Repeat the SAME course id
    in every week it spans. Weeks needed = duration_hours / hours per week, ROUNDED UP.
    At 5 hours/week: a 10-hour course needs 2 weeks, a 12-hour course needs 3 weeks (not
    2), an 18-hour course needs 4 weeks, a 28-hour course needs 6 weeks. A long course can
    never be finished inside one short week.
  - Budget first, then choose courses. Available hours = hours per week x number of weeks
    in the timeline; the message gives you this total. The duration_hours of the courses
    you schedule must add up to no more than that total. If they do not fit, DROP courses
    until they do - never stretch the plan past the learner's timeline instead.
  - When dropping, keep the courses closest to the target role and cut the ones the
    learner's existing experience already covers. For an experienced developer in another
    language, prefer a conversion course such as Python for Java Developers over a
    beginner programming course; do not schedule both.
  - The total number of weeks must fit inside the learner's stated timeline.
  - A NEW number the learner gives you in this conversation replaces the stored one, and it
    can go UP as well as down. "I can do 10 hours a week now" and "give me eight months
    instead" both mean the plan should GROW to use the extra room: add back the courses that
    would not fit before, deepen the ones that were rushed, and stretch a cramped schedule
    out. Do not simply reprint the old plan with a bigger ceiling - the CAPACITY line in the
    message is the new budget, and a plan that uses only half of it is the wrong answer.
    Equally, more hours per week means FEWER weeks for the same course, not the same weeks
    with idle time in them.
  - Respect prerequisites: a course's prerequisite ids must appear in an earlier or the
    same week, UNLESS the learner already holds those skills or already completed them -
    in that case say so in the week's "focus" or in learner_summary.
  - Credit existing experience. Do not send a senior developer through a beginner course
    in a language-agnostic skill they already have.

RULE 4 - FINAL OUTPUT FORMAT
Your final message must be raw JSON only. No markdown fences, no commentary.
For a learning plan, use exactly this shape:
{
  "learner_summary": "string - who they are, what they bring, what the journey does",
  "skill_gaps": ["string", "..."],
  "weekly_plan": [
    {"week": 1, "focus": "string", "courses": ["COURSE-ID"], "hours": 5,
     "assignment": "string - one concrete deliverable for that week"}
  ],
  "estimated_completion": "string - e.g. '26 weeks, on or around 2027-02-05'"
}
Weeks must start at 1 and increase by 1 with no gaps. "courses" may be empty for a
project or revision week, but "assignment" must always be filled in.

Do NOT put the learner's name, employee id or any other personal detail in
"learner_summary" or anywhere else in the plan. Those are already saved. Describe them by
what matters to the plan - "an experienced backend developer moving into AI" - not by who
they are. The plan is the output; their details are records.

If the learner is reporting progress instead of asking for a plan, record it and reply with:
  {"progress_update": true, "message": "string", "progress": { ... the tool's progress ... }}

Pick the right recording tool by what they actually said:
  - a course name or id ("I finished the RAG course")  -> record_completion
  - a number of weeks ("I've completed two weeks", "I'm through week 5", "done the first
    3 weeks")                                          -> record_weeks_completed
Never convert weeks into course ids yourself; record_weeks_completed knows which courses
those weeks contain and which are only part-way through. Put the tool's own message and
progress block into your reply rather than recomputing any of the numbers.

RULE 5 - ADJUSTING AN EXISTING PLAN
When the learner's situation has changed - they finished courses, fell behind, got ahead,
have less time per week now, or changed target - you are REVISING a plan, not writing a
new one. Work in this order:
  1. get_learning_progress  - ALWAYS first, and always before search_courses. It tells you
                              which weeks are already spent, which courses are still
                              outstanding and what they cost in hours, how many hours are
                              left, and whether what remains still fits.
  2. search_courses         - only if the target changed, or an outstanding course has
                              vanished from the catalogue, or a new gap has no course yet.
                              The ids in "remaining_courses" are already valid and already
                              have titles and durations: REUSE THEM DIRECTLY. Do not call
                              search_courses to re-find a course you were already given, or
                              you will schedule a near-duplicate under a different id and
                              lose the learner's history against the original.
  3. Reply with a plan in the RULE 4 shape covering ONLY the remaining work:
     - Renumber weeks from 1. Week 1 of a revision is the learner's NEXT week, not the
       week they originally started.
     - Never schedule a course the tool lists as completed, whether it was in the plan or
       off-plan. Treat the skills of every completed course as held, which means their
       prerequisites are already met - do not re-add them.
     - If "on_track" is false, DROP whole outstanding courses, keeping the ones closest to
       the target role, until the durations fit "hours_available". Never push the plan past
       "weeks_remaining".
     - Every week's "hours" must respect the learner's CURRENT hours per week, which may be
       higher or lower than the figure the original plan used. If they stated a new number,
       that new number wins over the stored one.
     - If they have given you MORE time - more hours a week, or a longer timeline - use it.
       Put back outstanding courses you had to drop, and add the ones the skill gaps still
       call for. A revision that shortens the plan when the learner just asked for more room
       is wrong.
     - Say in "learner_summary" what changed, what you dropped, and what you resequenced,
       so the learner can see why their path moved.
If get_learning_progress returns "has_plan": false there is nothing to revise. Build a new
plan under RULE 2 instead.
"""


# ==========================================================================
# Tool definitions in Azure OpenAI function-calling format
# ==========================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_profile",
            "description": (
                "Read an employee's stored learning profile, creating an empty record if "
                "they are new. Returns the stored profile, which fields are still unknown, "
                "any courses already completed, and whether a plan already exists. Call "
                "this first, before assessing skills."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "The employee id, e.g. 'EMP-1001'.",
                    }
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_learner_profile",
            "description": (
                "Store the details the learner gave about themselves - name, current role, "
                "years of experience, skills, target role, hours per week, timeline - so "
                "progress can be tracked against them later. Call this once, early, with "
                "everything they stated. Pass only fields they actually gave; never guess. "
                "Returns a short receipt, not the values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "name": {
                        "type": "string",
                        "description": "The learner's name, if they gave one.",
                    },
                    "current_role": {"type": "string", "description": "e.g. 'Java developer'."},
                    "experience_years": {"type": "number"},
                    "current_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills they said they already have.",
                    },
                    "target_role": {"type": "string"},
                    "hours_per_week": {"type": "number"},
                    "timeline_months": {"type": "number"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_skill_assessment",
            "description": (
                "Compare a learner's current skills against the skills their target role "
                "requires, and return the concrete skill gaps. Pass current_role as well "
                "(e.g. 'java developer') so implied skills such as OOP and REST are "
                "credited automatically and the profile can be saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Skills the learner already has, e.g. "
                            "['java', 'spring boot', 'sql']."
                        ),
                    },
                    "target_role": {
                        "type": "string",
                        "description": "The role, technology or certification they are aiming for.",
                    },
                    "current_role": {
                        "type": "string",
                        "description": (
                            "Their present job title, e.g. 'java developer'. Credited for "
                            "implied skills and stored on the profile."
                        ),
                    },
                    "experience_years": {
                        "type": "number",
                        "description": "Years of experience, if the learner stated them.",
                    },
                },
                "required": ["current_skills", "target_role"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": (
                "Get real, publicly available courses that teach specific missing skills. "
                "Returns course ids, providers, durations in hours, prerequisites and the "
                "skills each course teaches, and caches them so the ids stay stable. This "
                "is the ONLY source of valid course ids - none exist until you call it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text topic hint, e.g. 'retrieval augmented generation'.",
                    },
                    "skills_needed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skill gaps to close, taken from get_skill_assessment.",
                    },
                },
                "required": ["skills_needed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_learning_progress",
            "description": (
                "Read an employee's stored plan and how far through it they actually are: "
                "which weeks are spent, which courses are still outstanding and how many "
                "hours they need, how many weeks of the timeline remain, and whether what "
                "is left still fits. Call this FIRST whenever you are adjusting or "
                "re-sequencing an existing plan, before search_courses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_learning_plan",
            "description": (
                "Persist a finished learning plan for an employee, overwriting any "
                "previous plan. Normally the orchestrator calls this automatically after "
                "you produce your final plan, so you do not need to call it yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "plan": {
                        "type": "object",
                        "description": "The full plan object in the required final shape.",
                        "additionalProperties": True,
                    },
                },
                "required": ["employee_id", "plan"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_weeks_completed",
            "description": (
                "Record progress by week when the learner talks in weeks rather than "
                "course names - 'I've finished the first two weeks', 'I'm through week 5'. "
                "Marks every course whose last scheduled week falls in that range as "
                "complete, stores the week count, and returns updated progress. Use this "
                "instead of record_completion whenever they give a number of weeks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "weeks_completed": {
                        "type": "number",
                        "description": "How many weeks of the plan are done, counting from week 1.",
                    },
                },
                "required": ["employee_id", "weeks_completed"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_completion",
            "description": (
                "Mark one course complete for an employee and return their updated "
                "progress against the current plan. Handles courses that are not in the "
                "plan, unknown course ids and repeat completions without failing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string"},
                    "course_id": {
                        "type": "string",
                        "description": "Catalogue course id, e.g. 'AI-201'.",
                    },
                },
                "required": ["employee_id", "course_id"],
                "additionalProperties": False,
            },
        },
    },
]

# Name -> Python function. This mapping is the entire "router".
TOOL_FUNCTIONS = {
    "get_employee_profile": tools.get_employee_profile,
    "save_learner_profile": tools.save_learner_profile,
    "get_skill_assessment": tools.get_skill_assessment,
    "search_courses": tools.search_courses,
    "get_learning_progress": tools.get_learning_progress,
    "update_learning_plan": tools.update_learning_plan,
    "record_completion": tools.record_completion,
    "record_weeks_completed": tools.record_weeks_completed,
}

# What each step is called in quiet mode. Deliberately says what is happening
# without echoing any of the values being handled.
TOOL_STEP_LABELS = {
    "get_employee_profile": "Reading the learner record",
    "save_learner_profile": "Saving the profile details",
    "get_skill_assessment": "Working out the skill gaps",
    "search_courses": "Finding courses that close them",
    "get_learning_progress": "Checking progress so far",
    "update_learning_plan": "Storing the plan",
    "record_completion": "Recording the completion",
    "record_weeks_completed": "Recording the weeks you finished",
}


# ==========================================================================
# Azure OpenAI client
# ==========================================================================

def call_model(client: AzureOpenAI, deployment: str, messages: list[dict]):
    """One chat completion with the tool definitions attached."""
    kwargs = dict(
        model=deployment,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    try:
        return call_with_retry(lambda: client.chat.completions.create(**kwargs))
    except Exception as exc:
        # Older, non-reasoning deployments reject max_completion_tokens and want
        # max_tokens instead. Retry once with the legacy parameter name.
        if "max_completion_tokens" in str(exc):
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
            return call_with_retry(lambda: client.chat.completions.create(**kwargs))
        raise


# ==========================================================================
# The agent loop
# ==========================================================================

def run_agent(user_request: str, employee_id: str, constraints: dict | None = None,
              mode: str = "plan", history: list[dict] | None = None) -> dict:
    """Run the tool-calling loop until the model produces a final JSON answer.

    `history` is the plain-text conversation so far - what the learner said and what the
    agent said back, with no tool calls in it. Passing it is what turns a series of
    one-shot requests into an actual conversation: the model can see that the target
    role was mentioned three turns ago and stop asking for it.
    """
    client, deployment = build_client()
    learner_db.init_db()

    prompt = f"employee_id: {employee_id}\n\nLearner request:\n{user_request}"

    # Small models are unreliable at multiplying, and the whole plan hinges on this
    # one number. Compute it in Python and hand it over rather than hoping.
    constraints = constraints or {}
    capacity = _capacity(constraints)
    if capacity is not None:
        hours, weeks, total = capacity
        window = ("left in their timeline" if constraints.get("weeks_elapsed")
                  else "available in total")
        prompt += (
            f"\n\nCAPACITY (computed for you, use it as a hard ceiling): "
            f"{hours} hours/week x {weeks} weeks = {total} hours {window}. "
            f"The duration_hours of every course you schedule must add up to {total} or less, "
            f"and the plan must not exceed {weeks} weeks."
        )

    if constraints.get("timeline_extended"):
        prompt += (
            "\n\nThey have just given themselves MORE time than the old plan assumed. The "
            "capacity above is the new, larger budget - build a plan that actually uses it. "
            "Put back outstanding courses, add what the remaining skill gaps call for, and "
            "give rushed courses the weeks they need. Returning a plan that fits the old, "
            "smaller budget would ignore what they asked for."
        )

    if mode == "replan":
        prompt += (
            "\n\nThis is an ADJUSTMENT to an existing plan, not a new one. Follow RULE 5: "
            "call get_learning_progress first, then return a revised plan covering only the "
            "remaining work, with the weeks renumbered from 1."
        )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": prompt},
    ]

    call_order: list[str] = []          # for the trace summary at the end
    observed_args: dict[str, dict] = {}  # what the model extracted, reused to backfill the DB

    _trace("=" * 78)
    _trace("AGENT RUN")
    _trace("=" * 78)
    _trace(f"employee_id : {employee_id}")
    _trace(f"deployment  : {deployment}")
    _trace(f"mode        : {mode}")
    _trace(f"request     : {user_request}")
    _trace("-" * 78)

    _step("Working on it" + (" (adjusting the existing plan)" if mode == "replan" else "") + "...")

    for turn in range(1, MAX_TURNS + 1):
        response = call_model(client, deployment, messages)
        message = response.choices[0].message

        # --- no tool calls means this is the final answer ------------------
        if not message.tool_calls:
            _trace(f"[turn {turn}] model returned a final message (no tool call).")
            _trace("-" * 78)
            _trace("TOOL CALL ORDER: " + (
                " -> ".join(f"{i}.{name}" for i, name in enumerate(call_order, 1))
                if call_order else "(none - answered without tools)"
            ))
            _trace("-" * 78)
            return {
                "raw": message.content or "",
                "call_order": call_order,
                "observed_args": observed_args,
                "messages": messages,
                "client": client,
                "deployment": deployment,
            }

        # --- the model asked for one or more tools ------------------------
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            call_order.append(name)
            observed_args[name] = args

            # Quiet mode names the step but never the arguments - those carry the
            # learner's role, skills and name.
            _trace(f"[turn {turn}] TOOL CALL #{len(call_order)}: {name}")
            _trace(f"           args   : {_short(json.dumps(args), 400)}")
            if not VERBOSE:
                _step(f"  . {TOOL_STEP_LABELS.get(name, name)}")

            result = execute_tool(name, args)
            _trace(f"           result : {_short(json.dumps(result), 700)}")
            _trace("")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    raise SystemExit(f"Agent did not finish within {MAX_TURNS} turns. Tools called: {call_order}")


def execute_tool(name: str, args: dict) -> dict:
    """Run one tool by name. Errors are returned to the model, not raised."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"No such tool: {name}", "available_tools": sorted(TOOL_FUNCTIONS)}

    try:
        return func(**args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:                      # keep the loop alive
        return {"error": f"{name} failed: {type(exc).__name__}: {exc}"}


# ==========================================================================
# Parsing and validating what the model produced
# ==========================================================================

def parse_final_json(raw: str) -> dict | None:
    """Pull a JSON object out of the model's final message, fences and prose and all."""
    return extract_json(raw)


WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}


def _number(token: str) -> float | None:
    token = token.strip().lower()
    if token in WORD_NUMBERS:
        return float(WORD_NUMBERS[token])
    try:
        return float(token)
    except ValueError:
        return None


def infer_constraints(text: str) -> dict:
    """Best-effort read of hours/week and timeline from the request.

    This is NOT how the agent decides anything - the model does that. These
    numbers are only used afterwards to check the model's plan against what the
    learner actually said, and to store the profile for next time. If a value
    cannot be found, the corresponding check is skipped rather than guessed.
    """
    lowered = (text or "").lower()
    numeric = r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve)"

    hours = None
    match = re.search(numeric + r"\s*(?:h|hr|hrs|hour|hours)\b[^.]{0,20}?\b(?:per|a|each|/)\s*week", lowered)
    if not match:
        match = re.search(numeric + r"\s*(?:h|hr|hrs|hour|hours)\s*/\s*(?:wk|week)", lowered)
    if match:
        hours = _number(match.group(1))

    months = None
    match = re.search(numeric + r"[\s-]*months?", lowered)
    if match:
        months = _number(match.group(1))

    weeks = None
    match = re.search(numeric + r"[\s-]*weeks?\b", lowered)
    if match:
        weeks = _number(match.group(1))
    if weeks is None and months is not None:
        weeks = round(months * 4.345)

    # "6-year Java developer", "6 years of experience". Only ever stored on the
    # profile, never used in a check, so a false positive costs nothing.
    experience = None
    match = re.search(numeric + r"[\s-]*(?:year|yr)s?\b", lowered)
    if match:
        experience = _number(match.group(1))

    return {
        "hours_per_week": hours,
        "timeline_months": months,
        "max_weeks": weeks,
        "experience_years": experience,
    }


def resolve_constraints(employee_id: str, user_request: str, mode: str = "plan") -> dict:
    """The numbers the plan will be checked against, from the request and the profile.

    What the learner just said always wins. Anything they did not restate is filled in
    from their stored profile - which is what makes "3 hours a week now" work: the new
    capacity comes from the message, the timeline from the database.

    The subtle case is a timeline stated mid-plan. "Give me eight months instead" means
    eight months FROM NOW, so it has to be allowed to push the ceiling PAST the weeks
    left on the old plan. Cap it at the old deadline and asking for more time would be
    silently ignored, which is the one thing a learner would never forgive. When that
    happens the stored total is rebased to weeks already spent plus the new window, so
    later sessions still work out the remaining weeks correctly.
    """
    constraints = infer_constraints(user_request)
    stated_weeks = constraints["max_weeks"]        # None unless they said so just now

    learner_db.init_db()          # this may be the first thing to touch a fresh database
    learner = learner_db.get_learner(employee_id) or {}
    if constraints["hours_per_week"] is None:
        constraints["hours_per_week"] = learner.get("hours_per_week")
    if constraints["timeline_months"] is None:
        constraints["timeline_months"] = learner.get("timeline_months")
    if constraints["max_weeks"] is None and constraints["timeline_months"]:
        constraints["max_weeks"] = round(constraints["timeline_months"] * 4.345)

    if mode in ("replan", "chat"):
        progress = tools.get_learning_progress(employee_id)
        if progress.get("has_plan"):
            elapsed = progress["weeks_elapsed"]
            constraints["weeks_elapsed"] = elapsed

            if stated_weeks:
                # A window named just now is time from here. Rebase the stored total so
                # weeks_remaining is still right in the next session.
                constraints["max_weeks"] = int(stated_weeks)
                constraints["timeline_months"] = round((elapsed + stated_weeks) / 4.345, 1)
                constraints["timeline_extended"] = stated_weeks > progress["weeks_remaining"]
            elif progress["weeks_remaining"] > 0:
                # Fall back to the full timeline only if there is genuinely nothing left,
                # so an overrun learner still gets a checkable plan instead of every week
                # failing against a ceiling of zero.
                constraints["max_weeks"] = progress["weeks_remaining"]

    return constraints


def validate_plan(plan: dict, constraints: dict) -> tuple[list[str], list[str]]:
    """Check the plan against the catalogue and the learner's constraints.

    Returns (violations, notes). Violations are hard problems that get sent back
    to the model for one repair pass. Notes are informational.
    """
    violations: list[str] = []
    notes: list[str] = []

    catalogue = {course["id"]: course for course in tools._load_courses()}
    weeks = plan.get("weekly_plan")

    if not isinstance(weeks, list) or not weeks:
        return ["weekly_plan is missing or empty."], notes

    for field in ("learner_summary", "skill_gaps", "estimated_completion"):
        if not plan.get(field):
            violations.append(f"Required field '{field}' is missing or empty.")

    budget = constraints.get("hours_per_week")
    max_weeks = constraints.get("max_weeks")

    # Week numbering must be 1, 2, 3, ... with no gaps or repeats.
    numbers = [week.get("week") for week in weeks]
    if numbers != list(range(1, len(weeks) + 1)):
        violations.append(
            f"Week numbers must run 1..{len(weeks)} with no gaps or repeats; got {numbers}."
        )

    # week number -> position, so prerequisite ordering can be checked.
    scheduled_week: dict[str, int] = {}
    for index, week in enumerate(weeks, 1):
        for course_id in week.get("courses", []) or []:
            scheduled_week.setdefault(str(course_id).strip().upper(), index)

    waived: set[tuple[str, str]] = set()

    for index, week in enumerate(weeks, 1):
        label = f"week {week.get('week', index)}"

        hours = week.get("hours")
        if not isinstance(hours, (int, float)):
            violations.append(f"{label}: 'hours' must be a number, got {hours!r}.")
        elif budget is not None and hours > budget + 0.01:
            violations.append(
                f"{label}: {hours} hours exceeds the learner's limit of {budget} hours/week."
            )

        if not week.get("assignment"):
            violations.append(f"{label}: 'assignment' is empty.")

        for course_id in week.get("courses", []) or []:
            cid = str(course_id).strip().upper()
            course = catalogue.get(cid)
            if course is None:
                violations.append(f"{label}: '{course_id}' is not a real course id.")
                continue

            for prereq in course.get("prerequisites", []):
                prereq_week = scheduled_week.get(prereq)
                if prereq_week is None:
                    # Report once per course/prerequisite pair, not once per week the
                    # course spans, or a five-week course prints the same note five times.
                    if (cid, prereq) not in waived:
                        waived.add((cid, prereq))
                        notes.append(
                            f"{cid} lists prerequisite {prereq}, which is not in the plan "
                            "- treated as waived by existing experience."
                        )
                elif prereq_week > index:
                    violations.append(
                        f"{label}: {cid} is scheduled before its prerequisite {prereq} "
                        f"(week {weeks[prereq_week - 1].get('week', prereq_week)})."
                    )

    # Every scheduled course needs its full duration_hours allocated somewhere in
    # the plan. Without this check a model happily drops a 28-hour course into a
    # single 5-hour week, which reads fine and is impossible to actually do.
    allocated: dict[str, float] = {}
    week_count: dict[str, int] = {}
    for week in weeks:
        course_ids = [str(c).strip().upper() for c in (week.get("courses") or [])]
        hours = week.get("hours") if isinstance(week.get("hours"), (int, float)) else 0
        if not course_ids:
            continue
        share = hours / len(course_ids)
        for cid in course_ids:
            allocated[cid] = allocated.get(cid, 0) + share
            week_count[cid] = week_count.get(cid, 0) + 1

    for cid, given in sorted(allocated.items()):
        course = catalogue.get(cid)
        if course is None:
            continue
        needed = course.get("duration_hours") or 0
        if not needed:
            notes.append(f"{cid} has no duration_hours, so its hour coverage was not checked.")
        elif given < needed - 0.01:
            # Say exactly how many weeks the course needs. "Schedule more weeks"
            # is advice a small model reliably gets wrong; "give it 3 weeks" is not.
            if budget:
                want = math.ceil(needed / budget)
                violations.append(
                    f"{cid} ({course['title']}) needs {needed} h, which is {want} weeks at "
                    f"{budget} h/week, but you gave it {week_count[cid]} week(s) "
                    f"({round(given, 1)} h). Schedule {cid} in exactly {want} weeks, or drop it."
                )
            else:
                violations.append(
                    f"{cid} ({course['title']}) needs {needed} h but only {round(given, 1)} h "
                    f"are scheduled for it. Repeat {cid} across more weeks, or drop it."
                )

    if max_weeks is not None and len(weeks) > max_weeks + 1:
        violations.append(
            f"The plan is {len(weeks)} weeks long but the learner's timeline is about "
            f"{int(max_weeks)} weeks. Compress it."
        )

    total_hours = sum(w.get("hours") or 0 for w in weeks if isinstance(w.get("hours"), (int, float)))
    notes.append(f"Plan totals {round(total_hours, 1)} hours across {len(weeks)} weeks.")

    return violations, notes


def _stamp_completion(plan: dict, today: date | None = None) -> str:
    """Set the finish date from the plan length, in Python rather than in the model.

    Trusting the model here was a mistake worth recording: asked to revise a plan to 35
    weeks in August 2026 it wrote "on or around 2029-05-07", roughly two years out. Weeks
    times seven days is arithmetic, so it is computed here and overwrites whatever the
    model wrote. The model still has to produce the field - a plan with no end date fails
    validation - it just does not get to do the sum.
    """
    weeks = len(plan.get("weekly_plan") or [])
    finish = (today or date.today()) + timedelta(weeks=weeks)
    return f"{weeks} weeks, finishing on or around {finish.isoformat()}"


def _capacity(constraints: dict) -> tuple[float, int, float] | None:
    """(hours per week, weeks, total hours) if both constraints are known."""
    hours = constraints.get("hours_per_week")
    weeks = constraints.get("max_weeks")
    if not hours or not weeks:
        return None
    return hours, int(weeks), round(hours * weeks, 1)


def _budget_note(plan: dict, constraints: dict) -> str:
    """Spell out the hours arithmetic so the model does not have to do it."""
    capacity = _capacity(constraints)
    if capacity is None:
        return ""

    hours, weeks, available = capacity
    catalogue = {course["id"]: course for course in tools._load_courses()}
    scheduled = {
        str(cid).strip().upper()
        for week in plan.get("weekly_plan", []) or []
        for cid in (week.get("courses") or [])
    }
    required = sum((catalogue[cid].get("duration_hours") or 0) for cid in scheduled if cid in catalogue)

    note = (
        f"\n\nARITHMETIC (already computed - do not recalculate): the learner has "
        f"{hours} h/week x {weeks} weeks = {available} h available. The {len(scheduled)} "
        f"courses you scheduled need {required} h of study time in total."
    )
    if required > available:
        note += (
            f" That is {round(required - available, 1)} h more than they have. You MUST DROP "
            f"whole courses until the remaining durations total {available} h or less. Do not "
            f"add weeks - the plan cannot exceed {weeks} weeks. Keep the courses closest to "
            f"the target role."
        )
    return note


def repair_plan(session: dict, plan: dict, violations: list[str],
                constraints: dict, attempt: int) -> dict | None:
    """Send the violations back to the model for one corrective pass."""
    _trace("-" * 78)
    _trace(f"VALIDATION FAILED (repair attempt {attempt}) - sending problems back to the model:")
    for problem in violations:
        _trace(f"  ! {problem}")
    if not VERBOSE:
        _step(f"  . Plan broke {len(violations)} constraint(s) - repairing (attempt {attempt})")

    messages = session["messages"] + [
        {"role": "assistant", "content": json.dumps(plan)},
        {
            "role": "user",
            "content": (
                "Your plan failed validation:\n- " + "\n- ".join(violations) +
                _budget_note(plan, constraints) +
                "\n\nReturn the corrected, complete plan as raw JSON in the same shape. "
                "Do not add commentary. Do not call any tools."
            ),
        },
    ]

    response = call_model(session["client"], session["deployment"], messages)
    return parse_final_json(response.choices[0].message.content or "")


# ==========================================================================
# CLI output
# ==========================================================================

def print_plan(plan: dict, employee_id: str, version: int | None = None,
               revised: bool = False) -> None:
    catalogue = {course["id"]: course for course in tools._load_courses()}

    heading = "REVISED LEARNING PLAN" if revised or (version or 1) > 1 else "PERSONALIZED LEARNING PLAN"
    if version:
        heading += f" (version {version})"

    print()
    print("=" * 78)
    print(f"{heading} - {employee_id}")
    print("=" * 78)
    print()
    print("LEARNER SUMMARY")
    for line in _wrap(plan.get("learner_summary", ""), 74):
        print(f"  {line}")
    print()

    gaps = plan.get("skill_gaps", []) or []
    print(f"SKILL GAPS ({len(gaps)})")
    for gap in gaps:
        print(f"  - {gap}")
    print()

    weeks = plan.get("weekly_plan", []) or []
    total_hours = sum(w.get("hours") or 0 for w in weeks)
    print(f"WEEKLY PLAN ({len(weeks)} weeks, {round(total_hours, 1)} hours total)")
    print("-" * 78)

    for week in weeks:
        print(f"  Week {week.get('week'):>2}  |  {week.get('hours')} h  |  {week.get('focus', '')}")
        for course_id in week.get("courses", []) or []:
            course = catalogue.get(str(course_id).strip().upper())
            if course:
                provider = f"{course.get('provider')}, " if course.get("provider") else ""
                print(f"            {course_id:<22} {course['title']} "
                      f"({provider}{course.get('level')}, {course.get('duration_hours')} h)")
                print(f"            {' ' * 22} {course.get('course_link')}")
            else:
                print(f"            {course_id:<22} (unknown id - not generated by search_courses)")
        if not week.get("courses"):
            print(f"            {'(no course)':<12} project / consolidation week")
        for line in _wrap("Assignment: " + str(week.get("assignment", "")), 62):
            print(f"            {line}")
        print()

    print("-" * 78)
    print(f"ESTIMATED COMPLETION: {plan.get('estimated_completion', 'not stated')}")
    print()
    print("NOTE: courses above were generated by the language model, not read from a")
    print("      verified catalogue. Check every title and link before sending this to")
    print("      a learner - model-generated course URLs are often wrong or dead.")
    print("=" * 78)


def print_clarification(payload: dict) -> None:
    """Print the agent's question as speech.

    It used to be a boxed, numbered form, which read like a web form and invited
    form-shaped answers. The learner is having a conversation, so the questions run
    together as one short paragraph and the reply can be a sentence.
    """
    questions = [str(q).strip() for q in (payload.get("questions") or []) if str(q).strip()]
    print()
    for line in _wrap(" ".join(questions), 74):
        print(f"  {line}")


def print_progress(payload: dict) -> None:
    """The agent's spoken reply, then the percentage as a bar rather than a field dump."""
    print()
    for line in _wrap(str(payload.get("message", "")), 74):
        print(f"  {line}")

    progress = payload.get("progress") or {}
    if progress.get("courses_in_plan"):
        print()
        print(_progress_bar(progress))
        remaining = progress.get("remaining_courses") or []
        if remaining:
            print(f"  Still to go: {', '.join(str(c) for c in remaining[:4])}"
                  + (" ..." if len(remaining) > 4 else ""))


def print_progress_report(report: dict) -> None:
    """Print get_learning_progress straight from SQLite - no model call, no tokens."""
    employee_id = report.get("employee_id")

    # Read the name here rather than through the tool, so it never enters the
    # model's context. This report is a records view, which is the one place the
    # stored details are the point.
    name = (learner_db.get_learner(employee_id) or {}).get("name")
    who = f"{name} ({employee_id})" if name else str(employee_id)

    print()
    print("=" * 78)
    print(f"PROGRESS REPORT - {who}")
    print("=" * 78)

    if not report.get("has_plan"):
        print(f"  {report.get('message')}")
        completed = report.get("completed_courses") or []
        if completed:
            print(f"  Completed anyway (off-plan): {', '.join(completed)}")
        print("=" * 78)
        return

    progress = report.get("progress", {})
    print(f"  plan version        : {report.get('plan_version')} "
          f"(updated {report.get('plan_updated_at')})")
    print(f"  capacity            : {report.get('hours_per_week')} h/week, "
          f"{report.get('timeline_months')} month timeline")
    print(f"  weeks               : {report.get('weeks_elapsed')} spent, "
          f"{report.get('weeks_remaining')} remaining "
          f"(plan is {report.get('weeks_in_plan')} weeks)")
    print(f"  courses             : {progress.get('completed_in_plan')}/"
          f"{progress.get('courses_in_plan')} complete "
          f"({progress.get('percent_complete')}%)")

    off_plan = progress.get("completed_off_plan") or []
    if off_plan:
        print(f"  off-plan learning   : {', '.join(off_plan)}")

    print(f"  hours outstanding   : {report.get('remaining_hours')} h needed vs "
          f"{report.get('hours_available')} h available")
    print(f"  on track            : {'yes' if report.get('on_track') else 'NO - needs replanning'}")

    remaining = report.get("remaining_courses") or []
    if remaining:
        print()
        print("  STILL OUTSTANDING")
        for course in remaining:
            title = course.get("title") or course.get("note") or "(unknown)"
            hours = course.get("duration_hours")
            print(f"    {course['id']:<24} {title}" + (f" ({hours} h)" if hours else ""))

    history = report.get("completion_history") or []
    if history:
        print()
        print("  COMPLETION HISTORY")
        for event in history:
            flag = "in plan" if event["was_in_plan"] else "off plan"
            print(f"    {event['completed_at']}  {event['course_id']:<24} ({flag})")

    if not report.get("on_track"):
        print()
        print("  Run with --replan to have the agent rebuild the remaining weeks.")
    print("=" * 78)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars total]"


# ==========================================================================
# Entry point
# ==========================================================================

MAX_REPAIRS = 3


def classify_final(payload: dict) -> str:
    """Which of the three contracted shapes the model's final JSON is.

    Dispatch lives here rather than inline in process_request so it can be tested
    without spending a single token - see eval.py.
    """
    if payload.get("needs_clarification"):
        return "clarification"
    if payload.get("progress_update") or ("progress" in payload and "weekly_plan" not in payload):
        return "progress"
    if "weekly_plan" in payload:
        return "plan"
    return "unknown"


MAX_HISTORY_TURNS = 8     # user+assistant pairs kept, so the prompt cannot grow forever


def _summarise_for_history(result: dict, saved: dict | None = None) -> str:
    """What the agent said, compressed to what is worth remembering.

    A 26-week plan is thousands of tokens of JSON. Replaying it on every turn would
    crowd out the actual conversation and cost more each time, so history keeps a
    one-line account instead. The plan itself is in SQLite, and the model can read it
    back with a tool on the rare turn it needs the detail.
    """
    kind = result.get("kind")
    payload = result.get("payload") or {}

    if kind == "spoken":
        return str(result.get("spoken") or "")
    if kind == "clarification":
        return "I asked them: " + " ".join(payload.get("questions") or [])
    if kind == "progress":
        return str(payload.get("message") or "I reported their progress.")
    if kind == "plan":
        weeks = len(payload.get("weekly_plan") or [])
        version = (saved or {}).get("plan_version", "?")
        gaps = ", ".join((payload.get("skill_gaps") or [])[:8])
        return (
            f"I built and saved a {weeks}-week plan (version {version}) covering: {gaps}. "
            "It is stored - read it with get_learning_progress rather than asking them."
        )
    return "I failed to produce a usable answer on that turn."


def _extend_history(history: list[dict] | None, said: str, replied: str) -> list[dict]:
    turns = list(history or []) + [
        {"role": "user", "content": said},
        {"role": "assistant", "content": replied},
    ]
    return turns[-MAX_HISTORY_TURNS * 2:]


def process_request(user_request: str, employee_id: str, mode: str = "plan",
                    history: list[dict] | None = None) -> dict:
    """Run the agent end to end and return everything a caller might want to check.

    main() only needs the exit code, but the session needs the updated conversation
    history and the evaluation harness needs the shape the model produced, which tools
    it chose and in what order, and whether the plan survived validation. Returning all
    of it keeps eval.py from re-implementing this pipeline and testing its own copy.
    """
    constraints = resolve_constraints(employee_id, user_request, mode)
    session = run_agent(user_request, employee_id, constraints, mode, history)
    payload = parse_final_json(session["raw"])

    result = {
        "exit_code": 0,
        "kind": "unparsable",
        "payload": payload,
        "raw": session["raw"],
        "call_order": session["call_order"],
        "constraints": constraints,
        "violations": [],
        "notes": [],
        "history": history or [],
    }

    def finish(saved: dict | None = None) -> dict:
        result["history"] = _extend_history(
            history, user_request, _summarise_for_history(result, saved)
        )
        return result

    if payload is None:
        spoken = _clean(session["raw"])
        if spoken and "{" not in spoken:
            # The agent talked instead of wrapping it in JSON. That is a formatting slip
            # on a conversational turn, not a failure - so say what it said. Throwing away
            # a perfectly good sentence because it lacked braces would be the worst of
            # both worlds, and it is exactly what used to happen.
            result["kind"] = "spoken"
            result["spoken"] = spoken
            print()
            for line in _wrap(spoken, 74):
                print(f"  {line}")
            return finish()

        print("  I could not put that into a usable shape. Say it another way?")
        _trace("Raw model output was: " + session["raw"])
        result["exit_code"] = 1
        return finish()

    result["kind"] = classify_final(payload)

    # --- clarification path: no tools, no plan, no database write ---------
    if result["kind"] == "clarification":
        print_clarification(payload)
        return finish()

    # --- progress update path --------------------------------------------
    if result["kind"] == "progress":
        print_progress(payload)
        return finish()

    # --- plan path --------------------------------------------------------
    if result["kind"] == "unknown":
        print("Unrecognised final JSON shape:")
        print(json.dumps(payload, indent=2))
        result["exit_code"] = 1
        return finish()

    _trace(f"Constraints for validation (this request first, stored profile as fallback): "
           f"{constraints}")

    violations, notes = validate_plan(payload, constraints)
    for attempt in range(1, MAX_REPAIRS + 1):
        if not violations:
            break
        repaired = repair_plan(session, payload, violations, constraints, attempt)
        if not repaired or "weekly_plan" not in repaired:
            print(f"  Repair attempt {attempt} did not return a usable plan; keeping the previous one.")
            break
        payload = repaired
        violations, notes = validate_plan(payload, constraints)

    _trace("-" * 78)
    _trace("VALIDATION")
    for note in notes:
        _trace(f"  . {note}")

    # Violations stay visible even in quiet mode. A plan that breaks the learner's
    # stated limits is the one thing they must not have to run --verbose to discover.
    if violations:
        print(f"  ! PLAN STILL VIOLATES CONSTRAINTS AFTER {MAX_REPAIRS} REPAIR ATTEMPTS:")
        for problem in violations:
            print(f"  ! {problem}")
        print("  ! It is saved so you can inspect it, but review it before use.")
    else:
        _trace("  . All constraint checks passed.")

    payload["estimated_completion"] = _stamp_completion(payload)

    # Persist the plan, then backfill the profile from what the model extracted.
    saved = tools.update_learning_plan(employee_id, payload)
    _trace("-" * 78)
    _trace(f"PERSISTED: {json.dumps(saved)}")

    # The model already parsed the learner's sentence into fields when it called
    # get_skill_assessment. Reusing those arguments is how the profile gets filled
    # in without a second extraction step - update_learner ignores anything None.
    profile_args = session["observed_args"].get("save_learner_profile", {})
    assessment_args = session["observed_args"].get("get_skill_assessment", {})

    def stated(field):
        """save_learner_profile is the intended source; the assessment is the fallback."""
        for source in (profile_args, assessment_args):
            if source.get(field) is not None:
                return source[field]
        return None

    learner_db.update_learner(
        employee_id,
        name=stated("name"),
        current_skills=stated("current_skills"),
        current_role=stated("current_role"),
        experience_years=stated("experience_years") or constraints.get("experience_years"),
        target_role=stated("target_role"),
        hours_per_week=constraints.get("hours_per_week"),
        timeline_months=constraints.get("timeline_months"),
    )

    print_plan(payload, employee_id, version=saved.get("plan_version"),
               revised=(mode == "replan"))

    result.update({
        "payload": payload,
        "violations": violations,
        "notes": notes,
        "saved": saved,
        "exit_code": 2 if violations else 0,
    })
    return finish(saved)


def handle_request(user_request: str, employee_id: str, mode: str = "plan") -> int:
    """CLI entry point: run the agent and collapse the result to an exit code."""
    return process_request(user_request, employee_id, mode)["exit_code"]


# ==========================================================================
# Interactive session
# ==========================================================================

def _looks_like_email(value: str) -> bool:
    """Cheap sanity check. Not RFC 5322 - just enough to catch a typo or a bare name."""
    if value.count("@") != 1 or " " in value:
        return False
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


def ask_for_email() -> str:
    """Ask for the email and return the learner key derived from it.

    The email is the identity. Enter the same address next week and everything comes
    back - profile, plan, progress. Enter a new one and you are a new learner.
    """
    while True:
        raw = _clean(input("  Your email: "))
        if _looks_like_email(raw):
            return learner_db.key_for_email(raw)
        print("     -> That does not look like an email. Try name@company.com.")


def _clean(text: str) -> str:
    """Trim input, and drop any BOM or zero-width junk a paste or pipe dragged in.

    A leading U+FEFF is invisible but makes "courses" stop matching "courses", so
    the command silently falls through to the model. Same class of problem as a
    BOM in .env - worth removing at the door.
    """
    return (text or "").replace("﻿", "").replace("​", "").strip()


def _profile_context(employee_id: str) -> str:
    """What the agent already knows, handed over once at the start of a session.

    Without this the model would open by asking for a role and a goal it could have
    read from the database. The fields are stated as facts rather than asked about,
    which is the whole difference between a conversation and an intake form.
    """
    learner = learner_db.get_learner(employee_id)
    if not learner:
        return ""

    facts = []
    for label, field in (
        ("their name is", "name"),
        ("current role", "current_role"),
        ("years of experience", "experience_years"),
        ("target role", "target_role"),
        ("hours available per week", "hours_per_week"),
        ("timeline in months", "timeline_months"),
    ):
        value = learner.get(field)
        if value:
            facts.append(f"{label}: {value}")

    if not facts:
        return ""

    return (
        "ALREADY ON FILE for this learner - treat as given and never ask for any of it:\n  "
        + "\n  ".join(facts)
    )


def _profile_context_history(employee_id: str) -> list[dict]:
    """The stored profile as an opening exchange, ready to prepend to a conversation.

    Shaped as a completed turn rather than a system note so it reads to the model as
    something already settled between them, which is the point: settled facts do not
    get asked about again.
    """
    context = _profile_context(employee_id)
    if not context:
        return []
    return [
        {"role": "user", "content": context},
        {"role": "assistant",
         "content": "Noted - I have their details on file and will not ask for them again."},
    ]


def _progress_bar(progress: dict) -> str:
    """The percentage, spelled out. This is the thing the learner actually asked for."""
    percent = int(progress.get("percent_complete") or 0)
    cells = 24
    filled = round(percent * cells / 100)
    return (
        f"  [{'#' * filled}{'.' * (cells - filled)}]  {percent}% complete"
        f"  ({progress.get('completed_in_plan', 0)} of "
        f"{progress.get('courses_in_plan', 0)} courses)"
    )


def _resume_summary(employee_id: str) -> bool:
    """Where a returning learner left off. Returns whether there was a plan to resume."""
    report = tools.get_learning_progress(employee_id)
    if not report.get("has_plan"):
        return False

    weeks_done = report["weeks_elapsed"]
    total = report["weeks_in_plan"]
    progress = report["progress"]

    print()
    print(f"  Picking up where you left off - plan version {report['plan_version']},"
          f" last updated {str(report['plan_updated_at'])[:10]}.")
    print(f"  You have completed {weeks_done} of {total} weeks"
          f" ({progress['completed_in_plan']} of {progress['courses_in_plan']} courses).")
    print(_progress_bar(progress))

    if not report["on_track"]:
        print(f"  Heads up: {report['remaining_hours']} h of courses remain but only "
              f"{report['hours_available']} h are left. Worth talking about.")
    return True


def _safely(label: str, func, *args, **kwargs):
    """Run something that talks to Azure without letting a failure end the session.

    The learner asked for a terminal that does not stop. An expired key, a rate
    limit or a model that loops past MAX_TURNS should cost them a turn, not their
    session - the plan and progress are already in SQLite either way.
    """
    try:
        return func(*args, **kwargs)
    except SystemExit as exc:                  # missing config, or MAX_TURNS exceeded
        print(f"\n  ! {label} stopped: {exc}")
    except Exception as exc:                   # noqa: BLE001 - keep the loop alive
        print(f"\n  ! {label} failed: {type(exc).__name__}: {exc}")
    print("  ! Nothing was lost - your plan and progress are saved. Try again.")
    return None


LEAVE_WORDS = {"quit", "exit", "bye", "goodbye", "q", "that's all", "thats all", "nothing"}

# Sent on behalf of the session when a brand-new learner appears, so the AGENT opens the
# conversation and asks for the profile in its own words. A hardcoded greeting cannot do
# that job: it either asks for nothing, which is how a new learner ended up with no name on
# file, or it asks for six fields in a row, which is the form this was meant to replace.
OPENING_NUDGE = (
    "[Session start. This learner has no plan yet. Introduce yourself in one short "
    "sentence, then ask - in a single natural question - for whatever you still need under "
    "RULE 1 before you can build their first plan. Reply in the needs_clarification shape, "
    "putting your greeting and your question together in one entry.]"
)


def session_loop(employee_id: str | None = None) -> int:
    """A conversation with the agent. Identify by email, then just talk.

    There is no command menu and no intake questionnaire. Everything the learner types
    goes to the model with the conversation so far, and the model decides what it
    means: build a plan, ask the one thing it still needs, record progress, report
    progress, or rebuild the plan around new hours and weeks.

    The one piece of scaffolding left is the email, because the agent has to know
    whose records to read before it can avoid asking about them.
    """
    print("=" * 78)
    print("  LEARNING PATH AGENT")
    print("=" * 78)

    if employee_id is None:
        print("  Your email tells me whose plan this is - everything else we can just talk"
              " about.")
        print()
        employee_id = ask_for_email()

    learner = learner_db.get_learner(employee_id)
    has_plan = learner_db.get_plan(employee_id) is not None
    known_name = (learner or {}).get("name")

    # The profile goes in as stated fact, so the agent opens knowing who it is talking
    # to instead of asking for things already in the database.
    history: list[dict] = _profile_context_history(employee_id)

    print()
    if known_name:
        print(f"  Welcome back, {known_name}.")

    if has_plan:
        _resume_summary(employee_id)
        record = learner_db.get_plan_record(employee_id)
        print_plan(record["plan"], employee_id, version=record["version"])
        print()
        print("  Tell me how you're getting on, or what you'd like to change.")
        print("  (Say 'quit' when you're done - everything is saved as we go.)")
    else:
        # No plan yet, so the AGENT speaks first and asks for whatever RULE 1 still
        # needs, in its own words. It already has whatever is on file, so a
        # half-finished profile gets asked about once and never again.
        print("  (Say 'quit' any time - everything is saved as we go.)")
        opening = _safely("Opening", process_request, OPENING_NUDGE, employee_id,
                          "chat", history)
        if opening:
            history = opening["history"]

    while True:
        try:
            line = _clean(input("\nYou: "))
        except (EOFError, KeyboardInterrupt):
            print("\n  Saved. Talk soon.")
            return 0

        if not line:
            continue

        if line.lower().strip(" .!") in LEAVE_WORDS:
            print("  Saved. Talk soon.")
            return 0

        # Everything else is the model's call. "chat" mode means the constraints are
        # resolved against whatever they just said, falling back to their record, and a
        # timeline named now is allowed to extend past the old one.
        result = _safely("That", process_request, line, employee_id, "chat", history)
        if result:
            history = result["history"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Personalized Learning Path Agent")
    parser.add_argument("request", nargs="*", help="The learner's request in plain English.")
    parser.add_argument("--email", help="The learner's email - their identity. In the "
                                        "interactive session this is asked for instead.")
    parser.add_argument("--employee-id", help="Identify by raw key instead of email. "
                                              "Defaults to EMP-1001 for one-shot runs.")
    parser.add_argument("--name", help="The learner's name. Saved to their record; never "
                                       "printed in the plan.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show the full tool trace: every call, its arguments and its "
                             "result. Off by default because the arguments carry the "
                             "learner's profile.")
    parser.add_argument("--complete", metavar="COURSE_ID",
                        help="Report a completed course through the agent.")
    parser.add_argument("--replan", action="store_true",
                        help="Adjust the stored plan for the remaining weeks. Any text given "
                             "as the request describes what changed, e.g. --replan "
                             "\"I only have 3 hours a week now\".")
    parser.add_argument("--show-plan", action="store_true",
                        help="Print the stored plan from SQLite without calling the model.")
    parser.add_argument("--progress", action="store_true",
                        help="Print progress against the stored plan. No model call.")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    learner_db.init_db()

    # The email is the identity, so it wins when both are given. None means the
    # interactive session should ask - the one path where it is not needed up front.
    identity = learner_db.key_for_email(args.email) if args.email else args.employee_id
    known_identity = identity or "EMP-1001"     # one-shot runs keep the old default

    # Anything typed in as a field is stored straight away, before the model is
    # involved at all. It is a record, not part of the request.
    if args.name:
        tools.save_learner_profile(known_identity, name=args.name)

    if args.show_plan:
        record = learner_db.get_plan_record(known_identity)
        if record is None:
            print(f"No stored plan for {known_identity}.")
            return 1
        print_plan(record["plan"], known_identity, version=record["version"])
        return 0

    if args.progress:
        print_progress_report(tools.get_learning_progress(known_identity))
        return 0

    if args.complete:
        return handle_request(
            f"I have finished course {args.complete}. Please record it and tell me my progress.",
            known_identity,
        )

    if args.replan:
        changed = " ".join(args.request).strip()
        request = (
            "My situation has changed and I need my learning plan adjusted."
            + (f" What changed: {changed}" if changed else "")
            + " Read my progress first, then give me a revised plan covering only the"
              " remaining weeks, renumbered from week 1."
        )
        return handle_request(request, known_identity, mode="replan")

    request = " ".join(args.request).strip()

    # No request on the command line means the interactive session: identify by
    # email, resume or collect, then stay open for progress and adjustments.
    if not request:
        try:
            return session_loop(identity)
        except (EOFError, KeyboardInterrupt):
            print("\n  Saved. Goodbye.")
            return 0

    return handle_request(request, known_identity)


if __name__ == "__main__":
    sys.exit(main())
