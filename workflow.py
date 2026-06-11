"""LangGraph workflow: phases and InstructorAgent orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
from urllib.parse import parse_qs, urlparse

from langgraph.graph import END, StateGraph
from typing_extensions import NotRequired

from browser import BrowserController
from logging_config import logger
from planner import (
    SelectorPlanner,
    execute_action,
    read_full_online_text_submission,
    submit_grade_and_feedback,
)


class InstructorState(TypedDict):
    login_url: str
    moodle_url: str
    phase_status: dict[str, str]
    assignment_title: NotRequired[str]
    student_name: NotRequired[str]
    grade_value: NotRequired[str]
    feedback: NotRequired[str]
    verification_summary: NotRequired[str]


@dataclass
class PhaseResult:
    done: bool
    summary: str = ""
    assignment_title: str = ""
    student_name: str = ""
    grade_value: str = ""
    feedback: str = ""


def is_login_url(url: str) -> bool:
    """Return True when Moodle has redirected the browser to its login page."""
    parsed = urlparse(url)
    return "/login/" in parsed.path.lower()


def is_grader_url(url: str) -> bool:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    return query.get("action") == ["grader"] and bool((query.get("userid") or [""])[0])


class InstructorAgent:
    def __init__(self, browser: BrowserController, planner: SelectorPlanner, max_steps: int) -> None:
        self.browser = browser
        self.planner = planner
        self.max_steps = max_steps

    async def _observe(self) -> tuple[str, str, str]:
        url = await self.browser.get_current_url()
        tree = await self.browser.get_accessibility_tree()
        text = await self.browser.get_page_text()
        return url, tree or "No accessible elements found.", text or "No visible text content."

    async def _run_phase(
        self,
        *,
        phase: Literal["login", "find_assignment", "grade_assignment", "verify_graded"],
        goal: str,
        memory: dict[str, Any],
    ) -> PhaseResult:
        last_summary = ""
        for step in range(1, self.max_steps + 1):
            url, tree, text = await self._observe()
            if phase == "find_assignment" and is_grader_url(url):
                return PhaseResult(done=True, summary="Specific student submission grading page is open.")

            plan = await self.planner.plan(
                phase=phase,
                goal=goal,
                url=url,
                accessibility_tree=tree,
                visible_text=text,
                memory=memory,
            )
            last_summary = str(plan.get("summary", "")).strip()
            logger.info(f"{phase}: step {step} - {last_summary or 'planned'}")

            if bool(plan.get("done")):
                if phase == "login" and memory.get("auth_probe_url"):
                    await self.browser.navigate(str(memory["auth_probe_url"]))
                    await self.browser.wait_for_load()
                    current = await self.browser.get_current_url()
                    if is_login_url(current):
                        logger.info("login: auth probe reached login page; continuing login")
                        login_url = memory.get("login_url")
                        if login_url:
                            await self.browser.navigate(str(login_url))
                            await self.browser.wait_for_load()
                        continue

                return PhaseResult(
                    done=True,
                    summary=last_summary,
                    assignment_title=str(plan.get("assignment_title", "") or ""),
                    student_name=str(plan.get("student_name", "") or ""),
                    grade_value=str(plan.get("grade_value", "") or ""),
                    feedback=str(plan.get("feedback", "") or ""),
                )

            actions = plan.get("actions") or []
            if not isinstance(actions, list) or not actions:
                raise RuntimeError(f"{phase}: LLM returned no actions before done=true")

            for action in actions[:2]:
                if not isinstance(action, dict):
                    raise RuntimeError(f"{phase}: invalid action object: {action}")
                result = await execute_action(self.browser, action)
                logger.info(f"{phase}: {result}")

        raise RuntimeError(f"{phase}: did not complete within {self.max_steps} steps. Last: {last_summary}")

    async def login_node(self, state: InstructorState) -> InstructorState:
        username = os.getenv("TEACHER_USER") or os.getenv("INSTRUCTOR_USER")
        password = os.getenv("TEACHER_PASS") or os.getenv("INSTRUCTOR_PASS")
        if not username or not password:
            raise ValueError("Set TEACHER_USER/TEACHER_PASS or INSTRUCTOR_USER/INSTRUCTOR_PASS in .env")

        auth_probe_url = f"{state['moodle_url']}/my/"
        await self.browser.navigate(auth_probe_url)
        await self.browser.wait_for_load()
        current = await self.browser.get_current_url()
        if not is_login_url(current):
            logger.info("login: auth probe passed; session is valid")
            state["phase_status"]["login"] = "OK (cached)"
            await self.browser.save_storage_state()
            return state

        await self.browser.navigate(state["login_url"])
        await self.browser.wait_for_load()
        result = await self._run_phase(
            phase="login",
            goal=(
                "Log in to Moodle as an instructor. "
                "Use username and password from memory. The phase is done only after the page is no longer "
                "the login form and the /my/ dashboard auth probe works."
            ),
            memory={
                "username": username,
                "password": password,
                "login_url": state["login_url"],
                "auth_probe_url": auth_probe_url,
            },
        )

        await self.browser.navigate(auth_probe_url)
        await self.browser.wait_for_load()
        current = await self.browser.get_current_url()
        if is_login_url(current):
            self.browser.clear_storage_state()
            raise RuntimeError("Instructor login failed; /my/ still redirects to the login page.")

        state["phase_status"]["login"] = "OK" if result.done else "FAIL"
        await self.browser.save_storage_state()
        return state

    async def find_assignment_node(self, state: InstructorState) -> InstructorState:
        auth_probe_url = f"{state['moodle_url']}/my/"
        await self.browser.navigate(auth_probe_url)
        await self.browser.wait_for_load()
        current = await self.browser.get_current_url()
        if is_login_url(current):
            raise RuntimeError("Cannot find assignments because instructor is not logged in.")

        result = await self._run_phase(
            phase="find_assignment",
            goal=(
                "Find an assignment submission that needs grading and open its grading page. "
                "Useful Moodle paths include Dashboard, My courses, a course page, Assignments, "
                "'View all submissions', and 'Grade'. The phase is done only when a specific "
                "student submission grading page is open. Do not attempt to log in during this phase."
            ),
            memory={"moodle_url": state["moodle_url"]},
        )
        state["phase_status"]["find_assignment"] = "OK"
        if result.assignment_title:
            state["assignment_title"] = result.assignment_title
        if result.student_name:
            state["student_name"] = result.student_name
        return state

    async def grade_assignment_node(self, state: InstructorState) -> InstructorState:
        grader_url = await self.browser.get_current_url()
        assignment_question, student_answer = await read_full_online_text_submission(self.browser, grader_url)
        grading = await self.planner.grade_submission(
            assignment_question=assignment_question,
            student_answer=student_answer,
            assignment_title=state.get("assignment_title", ""),
            student_name=state.get("student_name", ""),
        )
        grade_value = str(grading.grade)
        feedback = grading.feedback
        logger.info(f"grade_assignment: LLM grade {grade_value}/100 - {feedback}")

        logger.info("grade_assignment: returning to grader page")
        await self.browser.navigate(grader_url)
        await self.browser.wait_for_load()
        await submit_grade_and_feedback(self.browser, grade_value, feedback)
        state["phase_status"]["grade_assignment"] = "OK"
        state["grade_value"] = grade_value
        state["feedback"] = feedback
        return state

    async def verify_graded_node(self, state: InstructorState) -> InstructorState:
        result = await self._run_phase(
            phase="verify_graded",
            goal=(
                "Verify the current assignment submission has been graded. "
                "Look for saved grade values, Graded status, feedback, or Moodle confirmation text. "
                "If needed, navigate back to the submissions table and inspect the student's row."
            ),
            memory={
                "assignment_title": state.get("assignment_title", ""),
                "student_name": state.get("student_name", ""),
                "expected_grade": state.get("grade_value", ""),
                "expected_feedback": state.get("feedback", ""),
            },
        )
        state["phase_status"]["verify_graded"] = "OK"
        state["verification_summary"] = result.summary
        return state

    def build_graph(self):
        graph = StateGraph(InstructorState)
        graph.add_node("login", self.login_node)
        graph.add_node("find_assignment", self.find_assignment_node)
        graph.add_node("grade_assignment", self.grade_assignment_node)
        graph.add_node("verify_graded", self.verify_graded_node)
        graph.set_entry_point("login")
        graph.add_edge("login", "find_assignment")
        graph.add_edge("find_assignment", "grade_assignment")
        graph.add_edge("grade_assignment", "verify_graded")
        graph.add_edge("verify_graded", END)
        return graph.compile()
