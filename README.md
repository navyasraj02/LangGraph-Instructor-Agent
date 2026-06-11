# Moodle Instructor Agent

LangGraph-based agent that logs into Moodle as an instructor, finds an assignment submission, grades it, saves feedback, and verifies the grade.

## Stack

- LangGraph for workflow orchestration
- OpenAI LLM for navigation decisions and grading
- Playwright for browser automation
- Moodle running locally with Docker
- LangSmith for tracing

## Agent Workflow

1. Logs into Moodle as an instructor.
2. Navigates to an assignment submission that needs grading.
3. Grades the assignment submission .
4. Verifies that the submission is graded.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

## Run

python instructor_agent.py