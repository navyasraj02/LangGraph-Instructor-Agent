"""Entry point: CLI arguments, browser/planner wiring, and run loop."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from opentelemetry import trace as otel_trace

from browser import BrowserController
from logging_config import logger
from planner import SelectorPlanner
from tracing import initialize_tracing
from workflow import InstructorAgent, InstructorState

load_dotenv()

DEFAULT_MAX_STEPS = 20


def session_file_for_url(base_url: str, role: str = "instructor") -> str:
    host = urlparse(base_url).netloc or "default"
    safe_host = host.replace(":", "_")
    return os.path.join(".auth", f"{safe_host}_{role}.json")


async def run_instructor_agent(headless: bool, login_url: str | None, max_steps: int) -> InstructorState:
    moodle_url = os.getenv("MOODLE_URL", "http://127.0.0.1:8080").rstrip("/")
    resolved_login_url = login_url or f"{moodle_url}/login/index.php"
    session_path = None if os.getenv("DISABLE_SESSION_PERSISTENCE") else session_file_for_url(resolved_login_url)

    browser = BrowserController(headless=headless, storage_state_path=session_path)
    planner = SelectorPlanner()
    await browser.start()
    try:
        agent = InstructorAgent(browser=browser, planner=planner, max_steps=max_steps)
        app = agent.build_graph()
        initial_state: InstructorState = {
            "login_url": resolved_login_url,
            "moodle_url": moodle_url,
            "phase_status": {
                "login": "SKIP",
                "find_assignment": "SKIP",
                "grade_assignment": "SKIP",
                "verify_graded": "SKIP",
            },
        }
        tracer = otel_trace.get_tracer("moodle.instructor.agent")
        with tracer.start_as_current_span("Moodle Instructor Agent"):
            final_state = await app.ainvoke(initial_state)
        logger.info(f"Instructor agent finished: {final_state['phase_status']}")
        return final_state
    finally:
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Moodle Instructor LangGraph Agent")
    parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode.")
    parser.add_argument("--login-url", default=None, help="Optional full Moodle login URL.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max LLM/action steps per node.")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    initialize_tracing()
    asyncio.run(run_instructor_agent(args.headless, args.login_url, args.max_steps))


if __name__ == "__main__":
    main()
