# Medium Agent - Personalized Learning Path Agent (Level 2)

An agent that turns a sentence like

> "I am a 6-year Java developer. I want to become an AI application developer in six
> months. I can spend 5 hours per week."

into a sequenced, week-by-week learning plan with practical assignments, remembers it
against the learner's email so the same address picks up where it left off weeks later,
and rebuilds the remainder when their progress or capacity changes.

## What makes this Level 2 and not Level 1

| | Level 1 (RAG Q&A) | Level 2 (this project) |
|---|---|---|
| Control flow | one prompt, one answer | a loop the model drives, across a multi-turn conversation |
| Data access | retrieve top-k chunks, inject | model picks tools and reads their results |
| Decisions | none - the model just answers | model chooses which tool, with which arguments, in what order |
| Memory | none | SQLite profile, plan and completion log, keyed by email |
| Output | free text | validated JSON contract |
| Revision | not applicable | reads recorded progress and rebuilds the remaining weeks |
| Evidence it works | read the answers | 135 offline + 29 live checks in `eval.py` |

The model is doing the reasoning and the tool selection. Python only executes the tool
the model already chose, and checks its arithmetic afterwards. There is no keyword router
anywhere in the codebase.

## Files

| File | What it holds |
|---|---|
| [learner_db.py](learner_db.py) | SQLite setup and helpers for the `learners`, `learning_plans`, `completions` and `courses` tables |
| [tools.py](tools.py) | The eight tools, as eight plain functions. No classes, no framework |
| [agent.py](agent.py) | Tool schemas, the agent loop, plan validation, CLI |
| [azure_client.py](azure_client.py) | The shared `AzureOpenAI` client, so `agent.py` and `tools.py` don't import each other |
| [eval.py](eval.py) | The evaluation harness - 135 offline checks plus 29 live agent checks |
| [.env.example](.env.example) | Template for the four Azure OpenAI variables |
| [requirements.txt](requirements.txt) | `openai` and `python-dotenv`. Everything else is stdlib |

There is **no `knowledge_base.json`**. See the next section.

## What the terminal shows

The plan is the output. The learner's details are records.

`save_learner_profile` stores the name, role, experience and capacity and returns a
**receipt** - which fields landed, which are still missing - never the values. So nothing
personal travels back through the model's context just to be printed again. The system
prompt also forbids putting the name or employee id in `learner_summary`.

The tool trace is off by default for the same reason: tool *arguments* carry the profile.
Quiet mode names each step instead:

```
Working on it...
  . Reading the learner record
  . Saving the profile details
  . Working out the skill gaps
  . Finding courses that close them
```

`--verbose` restores the full trace - every call, its arguments, its result, and the
`TOOL CALL ORDER` summary - which is the version worth showing when teaching the loop.
Constraint violations print in **both** modes: a plan that breaks the learner's stated
limits is the one thing nobody should need a flag to discover.

The name appears in exactly one place: the `--progress` report, which is a records view and
reads it straight from SQLite rather than through the tool.

## The eight tools

| Tool | What it does | Who calls it |
|---|---|---|
| `get_employee_profile(employee_id)` | Reads or creates the learner's record; reports which fields are still unknown | the model, first |
| `save_learner_profile(employee_id, ...)` | Stores name, role, experience, target and capacity. Returns a receipt, not the values | the model, early; also the `--name` flag |
| `get_skill_assessment(current_skills, target_role, current_role, experience_years)` | Gap list against the role framework, crediting skills the current role implies | the model |
| `search_courses(query, skills_needed)` | Generates courses that close specific gaps and caches them | the model |
| `get_learning_progress(employee_id)` | Weeks spent, courses outstanding, hours left, and whether the remainder still fits | the model, when adjusting |
| `update_learning_plan(employee_id, plan)` | Persists a plan, bumping its version | the orchestrator, after validation |
| `record_completion(employee_id, course_id)` | Marks one named course complete and returns updated progress | the model |
| `record_weeks_completed(employee_id, weeks_completed)` | Marks weeks 1-N complete, only finishing courses that end inside the range | the model, when told "I've done two weeks" |

`save_learner_profile` is the profile-collection step, called once and early. The
orchestrator also backfills from the arguments the model passed to it and to
`get_skill_assessment`, so a field the model mentioned but forgot to save still lands -
there is never a second extraction pass over the learner's sentence.

## Where courses come from

The model is the catalogue. `search_courses(query, skills_needed)` makes its own Azure
OpenAI call asking for real, publicly available courses that teach the requested skills,
and gets back id, title, provider, level, `duration_hours`, `prerequisites`,
`skills_taught` and `course_link` for each.

Generated courses are written to the `courses` table in SQLite. That cache is what makes
the rest of the system work:

- **Ids stay stable** across the agent's repair passes. If `search_courses` returned a
  different set of ids on every call, a plan could never be validated or corrected.
- **Plans stay checkable.** `validate_plan()` looks up durations and prerequisites in the
  table, so the hours and sequencing rules still apply to invented courses.
- **Completions work in a later process.** `record_completion` can validate a course id
  weeks later, in a run that never called `search_courses`.

Within a single run, identical `(query, skills_needed)` arguments are memoised, so a
repair pass cannot be handed a different catalogue halfway through building a plan.

> **These courses are unverified.** They come from a language model, not from a course
> catalogue you control. Expect wrong titles, mismatched ids and dead links - in testing
> the model returned an id of `DLAI-LLM-FUNDAMENTALS` for a course titled *"ChatGPT Prompt
> Engineering for Developers"*, and invented a prerequisite requiring *Python for
> Everybody* before an introduction to LLMs. Every result is flagged
> `model_generated: true`, `search_courses` returns `source: model-generated`, and the CLI
> prints a warning under each plan. **A human must check every title and link before a
> plan reaches an employee.** If you need trustworthy course data, point
> `search_courses` at your real LMS API instead - that is the only change required.

## Setup

### 1. Install

```powershell
cd c:\Users\udhaya_m\Desktop\Medium_Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure your Azure credentials

Copy the template, then edit `.env` **in VS Code** - never paste keys into a chat window,
a terminal command, or source code.

```powershell
Copy-Item .env.example .env
code .env
```

Fill in the four values:

| Variable | Where to get it |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure Portal > your Azure OpenAI resource > **Keys and Endpoint** > *Endpoint*. Looks like `https://your-resource.openai.azure.com`. No trailing slash, and do not append `/openai`. |
| `AZURE_OPENAI_API_KEY` | Same blade > **KEY 1** (or KEY 2). Click the copy icon and paste straight into `.env`. Treat it like a password; rotate it if it ever leaks. |
| `AZURE_OPENAI_API_VERSION` | Use `2025-04-01-preview`. It must be a version that supports tool calling. |
| `AZURE_OPENAI_DEPLOYMENT` | Azure AI Foundry > **Deployments** > the *deployment name* you chose, e.g. `gpt-5.4-nano`. This is **not** the public model name - on Azure the value passed as `model=` is your deployment name. |

Two things that catch people out:

- **Save `.env` as UTF-8 without BOM.** A byte-order mark makes the first variable name
  unreadable, and you get a confusing "Missing environment variables:
  AZURE_OPENAI_ENDPOINT" even though the line is clearly there. In VS Code the encoding
  is in the status bar; pick *UTF-8*, not *UTF-8 with BOM*.
- `.env` is in `.gitignore`. Keep it that way.

Verify without spending tokens on a full run:

```powershell
python learner_db.py     # creates learner.db
```

### 3. Run it

**Interactive session - the normal way to run it:**

```powershell
python agent.py
```

It asks for one thing - your **email** - and after that you just talk to it. There is no
command menu and no intake form. Every line you type goes to the model with the
conversation so far, and the model works out what you meant.

**The agent speaks first.** A new email has no plan, so the session sends it a nudge and it
opens the conversation itself, asking for everything it needs in one question. A real
first session, verbatim except for the plan body:

```
  Your email: rahul@gmail.com

  Before I put a plan together - what's your name, what are you doing at the
  moment and for how long, what are you aiming for, and how much time can
  you give it each week?

You: I'm Rahul, Java developer for 6 years, want to become an AI application developer, 5 hours a week

  . Reading the learner record
  . Saving the profile details
  . Working out the skill gaps
  . Finding courses that close them
  . Plan broke 4 constraint(s) - repairing (attempt 1)

  [ ... 26-week plan, 130 hours ... ]
```

One question, one answer, done - and the whole profile is on file:

```
  name             : 'Rahul'          target_role      : 'AI application developer'
  current_role     : 'Java developer' hours_per_week   : 5.0
  experience_years : 6.0              timeline_months  : None   (agent chose 26 weeks)
```

The greeting is **not** hardcoded. A fixed one cannot win: it either asks for nothing -
which is how a learner ended up with a plan and no name on file - or it reads out six
fields in a row, which is the form this was meant to replace. So RULE 1 lists the five
things a personalised plan needs (name, current role, experience, target, capacity) and
requires them to be asked for in **one** natural question, and the agent writes that
question itself.

Fields already stored are never asked about again. A learner with a name and role on file
but no target gets asked only for the target and the hours, because `_profile_context()`
hands the known fields over as settled fact - *"ALREADY ON FILE, treat as given and never
ask for any of it"* - rather than leaving the agent to discover them.

Then say what you like, in your own words:

```
You: I've finished the first four weeks

  . Checking progress so far
  . Recording the weeks you finished
  [#####...................]  20% complete  (2 of 10 courses)
  Still to go: COUR-LLM-API-INTEGRATION, DLAI-RAG-EMBEDDINGS-VDB ...

You: how am I doing?

  . Checking progress so far
  You're on track: 4 of 35 weeks completed (~20%). You've got 8 more courses...
```

Nothing there is a command. `record_weeks_completed` and `get_learning_progress` were the
model's choices. The only literal words the session still recognises are `quit` / `exit` /
`bye`, because leaving should not need a model call.

If the agent replies in plain prose instead of the JSON contract - which it occasionally
does on a conversational turn, having just been asked to introduce itself - the sentence is
printed as speech rather than discarded. That used to produce *"I could not put that into a
usable shape"* in place of a perfectly good question. A reply with no `{` in it is treated
as something said; only malformed JSON is reported as a failure.

Come back with the **same email** and it resumes - no questions, no model call:

```
  Welcome back, Deepak.
  Picking up where you left off - plan version 2, last updated 2026-08-07.
  You have completed 4 of 35 weeks (2 of 10 courses).
  [#####...................]  20% complete  (2 of 10 courses)

  [ ... your plan ... ]

  Tell me how you're getting on, or what you'd like to change.
```

## Saying "give me more time" has to actually work

Tell it you have more room and the plan grows to use it. Measured on a live run:

```
You: actually I can do 10 hours a week now, and give me 8 months instead of 6

  REVISED LEARNING PLAN (version 2)
  WEEKLY PLAN (35 weeks, 338 hours total)     <- was 26 weeks, 130 hours
```

That took a specific fix, because the obvious implementation quietly breaks it. On a
revision the week ceiling is normally `weeks_remaining` - the weeks left on the *old*
timeline - which is right when nothing has changed and wrong the moment someone asks for
more. Capped that way, "give me 8 months" would have been silently clamped to the 26 weeks
already agreed, and the plan would have come back the same size.

So `resolve_constraints()` treats a timeline stated **in this turn** as time available from
now, and lets it exceed what was left. It then rebases the stored total to weeks already
spent plus the new window, so the next session still computes the remaining weeks
correctly. The prompt is told explicitly that the budget grew, because a model handed a
bigger ceiling will otherwise happily return the old plan unchanged.

More hours per week works the same way, and means *fewer* weeks per course rather than the
same weeks with idle time in them. Saying you have **less** time still narrows the plan -
the `extension` eval suite pins both directions, plus the case where you say nothing about
time at all and the old ceiling must stay put.

## The email is the identity

`learners.employee_id` holds the normalised email, so one address is one record with one
plan and one progress history. `key_for_email()` lowercases, trims and strips the angle
brackets mail clients paste, which means `Deepak@LevelShift.com` and
`deepak@levelshift.com ` are the same learner rather than two who each think they are the
only one.

The session branches on whether that key exists:

| | New address | Known address |
|---|---|---|
| Opening | "Tell me where you are now and where you want to get to" | "Welcome back, Deepak" |
| Questions | whatever the agent still needs, conversationally | none - the profile goes in as stated fact |
| Plan | built once it has enough | read from SQLite and reprinted |
| Progress | starts at zero | resumes where it stopped |
| Model calls to get there | one plan generation | **none** |

A returning learner's stored fields are injected into the conversation as a fact, not a
question: *"ALREADY ON FILE - treat as given and never ask for any of it: current role:
Java developer, target role: AI application developer, ..."*. Without that the agent opens
by asking for things it could have read, which is the exact behaviour that made the old
version feel like a form.

`_looks_like_email()` is a cheap sanity check - one `@`, a dot in the domain, no spaces -
not RFC 5322. It exists to catch a typo or a bare name before it becomes a second learner
record, not to validate deliverability.

## How the conversation is kept

`run_agent()` takes a `history` argument: the plain-text turns so far, what the learner
said and what the agent said back. That is what lets the model see the target role was
mentioned three turns ago and stop asking for it.

History holds **no tool calls and no tool results**, only text. Two reasons:

- **Size.** A 35-week plan is thousands of tokens of JSON. Replaying it every turn would
  crowd out the conversation and cost more each time. `_summarise_for_history()` records
  *"I built and saved a 35-week plan (version 2) covering: python, rag, ..."* instead, and
  points the model at `get_learning_progress` for the detail. The plan is in SQLite; it
  does not need to be in the prompt.
- **Correctness.** An assistant message with `tool_calls` is only valid when followed by
  matching `tool` results. Keeping partial tool exchanges across turns is a reliable way to
  get 400s from the API. Dropping them wholesale cannot go wrong.

Trimmed to `MAX_HISTORY_TURNS` (8 exchanges), so a long session cannot grow the prompt
without bound.

## Weeks and courses are not the same thing

`week 2` does **not** mean "two courses done". A 10-hour course at 5 hours a week spans
weeks 1-3, so at the end of week 2 it is genuinely unfinished. `record_weeks_completed`
only completes a course when its **last** scheduled week falls inside the range, and
names the ones it left open. Rounding that up would overstate progress and skew every
plan built from it afterwards.

Which creates a problem worth pointing out: if no course happens to finish inside the
range, nothing lands in the completions log, and "I did two weeks" would vanish by the
next session. So the declared week count is stored on the learner too, in
`learners.weeks_completed`. `get_learning_progress` then takes whichever source is
further along:

- **completions imply weeks** - finished the week-1-to-3 course, so 3 weeks are gone
- **the learner states weeks** - ground through half a long course, which only they know

Two facts, two sources, `max()` of the two. The eval suite pins both directions.

`learners.weeks_completed` was added after the first version of the schema, so `init_db()`
adds it with `ALTER TABLE` when it is missing - the cheap version of a migration, which is
all a single-file SQLite database needs.

An Azure failure mid-session - expired key, rate limit, a model that loops past
`MAX_TURNS` - costs you a turn, not the session. `progress`, `courses`, `plan` and
`profile` keep working regardless, because they read SQLite and never call the model.

**One-shot - for scripts and demos:**

```powershell
# Generate a plan. --email identifies the learner, exactly as the session does.
python agent.py --email deepak@levelshift.com "I am a 6-year Java developer. I want to become an AI application developer in six months. I can spend 5 hours per week."

# Report a completed course (goes through the agent, which picks record_completion)
python agent.py --email deepak@levelshift.com --complete DLAI-PROMPT-ENG

# Adjust the path now that progress has changed. Any text describes what changed.
python agent.py --email deepak@levelshift.com --replan "I only have 3 hours a week now"

# Read state back out of SQLite - no model call, no tokens
python agent.py --email deepak@levelshift.com --show-plan
python agent.py --email deepak@levelshift.com --progress

# Save a name without starting a session, and show the full tool trace
python agent.py --email deepak@levelshift.com --name "Deepak" --verbose "..."
```

`--employee-id` still works for a raw key instead of an email, and one-shot runs given
neither fall back to `EMP-1001`. `--email` wins if both are given.

Exit codes: `0` success, `1` unusable model output, `2` plan saved but still violates a
constraint after all repair attempts.

### 4. Check it still works

```powershell
python eval.py            # 135 checks, no model calls, no tokens
python eval.py --online   # adds 29 live agent checks
```

## The tool-calling flow

```
                 +-------------------------------------------+
                 |  messages = [system prompt, user request] |
                 +-------------------------------------------+
                                    |
                                    v
        +---------------------------------------------------------+
        |  client.chat.completions.create(                        |
        |      model=<deployment>, messages=..., tools=TOOL_SCHEMAS)|
        +---------------------------------------------------------+
                    |                              |
       message.tool_calls is set          no tool_calls -> final answer
                    |                              |
                    v                              v
   +-----------------------------+     +---------------------------+
   | for each tool call:         |     | parse JSON, validate,     |
   |   look up TOOL_FUNCTIONS    |     | repair if needed, persist,|
   |   run the Python function   |     | pretty-print              |
   |   append role:"tool" result |     +---------------------------+
   +-----------------------------+
                    |
                    +---> loop back to the model (max MAX_TURNS = 12)
```

Concretely, one turn of the loop:

1. **Python sends** the conversation plus eight JSON tool schemas.
2. **The model replies** with `tool_calls`, e.g.
   `get_skill_assessment({"current_skills": ["Java developer"], "target_role": "AI application developer"})`.
   It chose that tool and those arguments; nothing in Python suggested them.
3. **Python executes** `tools.get_skill_assessment(...)` and appends the return value as a
   `{"role": "tool", "tool_call_id": ..., "content": <json>}` message. The `tool_call_id`
   is what ties a result back to the request.
4. **Repeat.** The model now sees the real skill gaps and calls `search_courses`, often
   several times, once per cluster of gaps. Note that `search_courses` makes its own
   nested model call to generate the courses - so one turn of the outer agent loop can
   contain an inner completion. That is worth pointing out to students: a tool is just a
   Python function, and nothing stops it calling an LLM itself.
5. When it has enough, it replies with **no** `tool_calls` and its content is the final JSON.

Every call is printed as it happens, and the run ends with the order:

```
TOOL CALL ORDER: 1.get_employee_profile -> 2.get_skill_assessment -> 3.search_courses -> ...
```

### The three final shapes

The model's last message is always raw JSON in one of three shapes, and `handle_request()`
dispatches on which key is present:

| Key present | Meaning | What Python does |
|---|---|---|
| `needs_clarification` | not enough information to plan | print the questions, call no tools, write nothing |
| `weekly_plan` | a finished plan | validate, repair, persist, print |
| `progress_update` | a completion was recorded | print the progress figures |

## Guard rails

The model proposes; Python checks. `validate_plan()` enforces:

- **Weekly hours** never exceed the learner's stated capacity.
- **Course hours are real.** Each course must be allocated its full `duration_hours`
  across the weeks it appears in. This is the check that matters most - without it a model
  will happily drop a 28-hour course into a single 5-hour week, which reads perfectly and
  is impossible to do.
- **Timeline** - the plan cannot run past the stated number of months.
- **Course ids exist** in the `courses` table, meaning `search_courses` actually returned
  them. This stops the model writing plausible-looking ids straight into the plan without
  ever looking them up.
- **Prerequisites** are not scheduled after the course that needs them. A prerequisite
  absent from the plan entirely is reported as *waived by existing experience* rather than
  failed, so a senior Java developer is not forced through introductory Python.
- **Week numbering** runs 1..N with no gaps, and every week has an assignment.

If anything fails, the violations go back to the model for up to `MAX_REPAIRS = 3`
corrective passes. Two details make the repair actually work on a small model:

- Python computes the capacity arithmetic (`5 h/week x 26 weeks = 130 h`) and the sum of
  the scheduled course durations, and states them. Small models are unreliable at this
  multiplication, and the whole plan depends on it.
- Violations name the exact fix - "schedule AI-311 in exactly 3 weeks" - rather than
  "schedule more weeks", which is advice a small model reliably gets wrong.

Constraint numbers are read from the request by a small regex in `infer_constraints()`.
That regex **never** routes tools or picks values for the plan - it only checks the model's
output afterwards and stores the profile. If it cannot find a value, the matching check is
skipped rather than guessed.

`resolve_constraints()` then layers the stored profile underneath: whatever the learner
just said wins, and anything they did not restate is filled in from the database. That is
what makes `--replan "3 hours a week now"` work - the new capacity comes from the sentence,
the six-month timeline comes from the profile. On a replan the week ceiling is not the full
timeline any more but `weeks_remaining`, so a revision cannot quietly buy itself the weeks
the learner has already spent.

## Adjusting a plan when progress changes

A plan that cannot be revised is a document, not a learning path. `--replan` is the second
agent behaviour, and it works from real recorded state rather than from the learner's
description of it.

`get_learning_progress()` derives, in Python:

| Value | How |
|---|---|
| `weeks_elapsed` | The first week still holding an unfinished course. Everything before it is time already spent |
| `weeks_remaining` | The timeline in weeks minus `weeks_elapsed` - not the length of the old plan |
| `remaining_hours` | The `duration_hours` of every outstanding course, added up |
| `hours_available` | `hours_per_week x weeks_remaining` |
| `on_track` | Whether the second fits inside the third |

`on_track` is decided here rather than by the model, for the same reason the capacity
arithmetic is: it is a multiplication and a comparison, and small models get those wrong.
The tool also returns a `note` that states the numbers and says what to do about them -
keep everything and resequence, or drop courses until it fits.

RULE 5 in the system prompt then constrains the revision: renumber from week 1, never
reschedule a completed course, treat completed courses' skills as held so their
prerequisites count as met, respect the learner's *current* hours per week, reuse the
course ids already in `remaining_courses` instead of searching for near-duplicates, and
say in `learner_summary` what changed.

A verified live run: a learner 6 weeks in with 1 of 7 courses done, dropping from 5 to 3
hours a week, produced a 20-week 45-hour version 2 plan starting at week 1, with the
completed course gone and two under-allocated courses caught and repaired by
`validate_plan` before it was saved.

## Behaviour worth demoing

**Clarifies before planning.** Given "I want to get into AI" the agent calls **zero** tools
and asks for the missing capacity and target. Nothing is written to the database. The rule
tests *presence*, not precision: a stated goal is never queried for refinement, so
"become an AI application developer, 5 hours per week" plans immediately.

**Handles off-plan completions.** `record_completion()` returns a status instead of
raising, for four awkward cases:

| Situation | `status` |
|---|---|
| course is in the current plan | `recorded` |
| course exists but is not in the plan | `recorded_off_plan` - logged separately, excluded from plan progress |
| learner has no plan yet | `recorded_without_plan` |
| course id not in the catalogue | `unknown_course` - nothing is written |
| already completed | `already_recorded` |

**Remembers.** Re-run for the same employee id and `get_employee_profile` returns the
stored profile, the completion history, and `has_existing_plan: true`; `learning_plans`
keeps a `version` counter that increments on every overwrite.

**Adjusts.** `--replan` rebuilds only what is left, from the completion log rather than the
learner's account of it. See the section above.

## Evaluation

`python eval.py` is the answer to "how do you know it works". Every check either passes or
fails against a throwaway database, so the real `learner.db` is never touched.

Two tiers, because they cost different things:

| | Checks | Cost | Covers |
|---|---|---|---|
| offline (default) | 135 | none | everything Python owns: constraint parsing, the gap framework, `extract_json`, output-shape dispatch, all nine `validate_plan` rules, all five completion statuses, email identity, week-based progress, timeline extension, and the replan arithmetic |
| `--online` | 29 | real model calls | what only a live model shows: does it clarify instead of guessing, does it call tools in a sensible order, does its plan pass validation, does a replan drop completed work |

The split matters. Most agent bugs are not in the model's reasoning, they are in the Python
around it - so most of the checks cost nothing to run and can go in a pre-commit hook.

```powershell
python eval.py                                  # offline, free
python eval.py --online                         # everything
python eval.py --suite validation --suite progress
python eval.py --keep-db                        # leave eval_learner.db for inspection
```

The online tier asserts on behaviour, not on wording:

- "I want to get into AI" produces `needs_clarification` **and zero tool calls**, and writes
  nothing to the database.
- The full Java-developer request produces a plan, with `get_skill_assessment` called before
  `search_courses`, every week carrying an assignment, and **zero** validation violations.
- Reporting a completion returns `progress_update`, not a fresh plan, and the completion is
  actually in SQLite afterwards.
- A replan calls `get_learning_progress` first, never reschedules the completed course,
  restarts at week 1, respects the *new* hours-per-week figure, and is stored as version 2.

Last full run: **135/135 offline, 28/29 online** - the one failure was a plan the repair loop could not fix in three passes, reported loudly rather than hidden. See [Guard rails](#guard-rails).

The offline tier found a real bug on its first run: `with sqlite3.connect(...)` commits but
does not close, so every helper leaked an open handle and Windows would not let the database
file be deleted. `learner_db.connect()` is now a context manager that closes. That is the
argument for the harness in one line - the leak was invisible in normal use.

## Course schema

`search_courses` asks the model for these fields per course, and they are what the
`courses` table stores:

| Field | Why it exists |
|---|---|
| `id` | Stable uppercase slug. The only handle a plan may reference |
| `title`, `provider`, `topic`, `level`, `content` | Human-readable detail for the printed plan |
| `course_link` | Provider's course page. **Unverified** |
| `skills_taught` | Array, so skill-gap matching has real values to match on |
| `prerequisites` | Array of course ids, so sequencing can be enforced rather than hoped for |
| `duration_hours` | Number, so the hours-per-week constraint can be checked |

The last three are what make the behaviour requirements enforceable rather than
aspirational. A course with no `duration_hours` is reported as unchecked rather than
silently passing.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Missing environment variables: AZURE_OPENAI_ENDPOINT` when the line exists | `.env` saved as UTF-8 **with** BOM. Re-save without it. |
| `404` from Azure | The deployment name does not exist on this resource, or the api-version is too old. Check Azure AI Foundry > Deployments. |
| `401` / `403` | Key belongs to a different resource, or was rotated. |
| `429` | Out of quota for the deployment - raise TPM or wait. |
| Empty final message | Reasoning models spend tokens thinking. Raise `MAX_OUTPUT_TOKENS`. |
| `openai.APIConnectionError` / `getaddrinfo failed` | DNS or egress blocked for the Python process (corporate proxy or sandbox). |

## Exercises

1. Add a ninth tool, `get_team_skill_gaps(team_id)`, and let the agent plan for a whole team.
2. Point `search_courses` at a real LMS API instead of the model. Nothing else has to change,
   and the fabrication problem goes away.
3. Delete the capacity arithmetic injected in `run_agent()` and watch how much worse the
   plans get. That difference is the lesson about what to compute in code versus what to
   delegate to the model.
4. Break something on purpose - drop the `on_track` calculation, or let `validate_plan` skip
   the hour-coverage check - and see which `eval.py` checks go red. A harness you have never
   seen fail is not evidence of anything.
5. Add an online eval case for a target role that is not in `ROLE_SKILL_REQUIREMENTS` (say
   "SRE"). It falls back to the default framework today; decide whether that is right, and
   write the check that pins your answer.
6. Track calendar time instead of inferring `weeks_elapsed` from completions. Store a start
   date, and see how many new awkward cases appear once a learner can simply be late.
