"""LLM planning, browser action execution, and Moodle grading."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from openai import AsyncOpenAI

from browser import BrowserController
from logging_config import logger

MAX_TEXT_CHARS = 6000
MAX_TREE_CHARS = 12000

SelectorAction = dict[str, Any]


@dataclass
class GradingResult:
    grade: int
    feedback: str


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return cleaned


def _parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(cleaned.lstrip())
    if not isinstance(parsed, dict):
        raise RuntimeError(f"LLM JSON response must be an object, got: {type(parsed).__name__}")
    return parsed


def _compact(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n...\n{tail}"


# ---------------------------------------------------------------------------
# Moodle URL helper
# ---------------------------------------------------------------------------

def full_online_text_url_from_grader_url(grader_url: str) -> str | None:
    """Build Moodle's full online-text submission URL from a grader URL."""
    parsed = urlparse(grader_url)
    query = parse_qs(parsed.query)
    assignment_id = (query.get("id") or [""])[0]
    user_id = (query.get("userid") or [""])[0]
    if not assignment_id or not user_id:
        return None
    full_query = urlencode(
        {
            "id": assignment_id,
            "sid": user_id,
            "gid": user_id,
            "plugin": "onlinetext",
            "action": "viewpluginassignsubmission",
            "returnaction": "grading",
            "returnparams": "",
        }
    )
    return urlunparse(parsed._replace(query=full_query))


# ---------------------------------------------------------------------------
# Browser action execution
# ---------------------------------------------------------------------------

_FORM_INPUT_ROLES = frozenset({"textbox", "searchbox", "spinbutton"})
_FORM_CONTROL_TAGS = frozenset({"input", "textarea", "select"})
# Moodle login pages use these stable ids; avoids matching <label>Username</label>.
_MOODLE_LOGIN_SELECTORS: dict[str, str] = {
    "username": "#username",
    "password": "#password",
}


async def visible_locator(locator, occurrence: int = 0, *, prefer_form_control: bool = False):
    visible_seen = 0
    form_controls: list = []
    count = await locator.count()
    for index in range(count):
        candidate = locator.nth(index)
        try:
            if not await candidate.is_visible(timeout=1000):
                continue
            if prefer_form_control:
                tag = await candidate.evaluate("el => el.tagName.toLowerCase()")
                if tag in _FORM_CONTROL_TAGS:
                    form_controls.append(candidate)
                    continue
                if tag == "label":
                    continue
            if visible_seen == occurrence:
                return candidate
            visible_seen += 1
        except Exception:
            pass
    if prefer_form_control and form_controls:
        return form_controls[min(occurrence, len(form_controls) - 1)]
    return locator.nth(occurrence)


def locator_from_action(page, action: SelectorAction):
    selector = str(action.get("selector", "")).strip()
    if selector:
        return page.locator(selector)
    role = str(action.get("role", "")).strip()
    name = str(action.get("name", "")).strip()
    if not role or not name:
        raise RuntimeError(f"Action requires role/name or selector: {action}")

    if role in _FORM_INPUT_ROLES:
        moodle_selector = _MOODLE_LOGIN_SELECTORS.get(name.lower())
        if moodle_selector:
            return page.locator(moodle_selector)
        # Label-linked control first; avoid get_by_text which matches <label> on Moodle login.
        return page.get_by_label(name, exact=False).or_(page.get_by_role(role, name=name))

    by_role = page.get_by_role(role, name=name)
    by_text = page.get_by_text(name, exact=True)
    return by_role.or_(by_text)


def _prefers_form_control(action: SelectorAction) -> bool:
    role = str(action.get("role", "")).strip().lower()
    return role in _FORM_INPUT_ROLES


async def type_editor(page, text: str) -> None:
    editable = page.locator('[contenteditable="true"]')
    if await editable.count() > 0:
        await editable.first.click(timeout=10000)
        await editable.first.fill(text, timeout=10000)
        return
    frame_body = page.frame_locator("iframe").first.locator("body")
    await frame_body.click(timeout=10000)
    await page.keyboard.press("Control+A")
    await page.keyboard.type(text)


async def execute_action(browser: BrowserController, action: SelectorAction) -> str:
    kind = str(action.get("action", "")).strip().lower()
    occurrence = int(action.get("occurrence", 0) or 0)
    page = browser.page

    prefer_input = _prefers_form_control(action)

    if kind == "click":
        locator = await visible_locator(
            locator_from_action(page, action), occurrence, prefer_form_control=prefer_input
        )
        await locator.click(timeout=10000)
    elif kind == "type":
        text = str(action.get("text", ""))
        locator = await visible_locator(
            locator_from_action(page, action), occurrence, prefer_form_control=prefer_input
        )
        await locator.fill(text, timeout=10000)
    elif kind == "type_editor":
        await type_editor(page, str(action.get("text", "")))
    elif kind == "select":
        text = str(action.get("text", ""))
        locator = await visible_locator(locator_from_action(page, action), occurrence)
        await locator.select_option(label=text, timeout=10000)
    elif kind == "check":
        locator = await visible_locator(locator_from_action(page, action), occurrence)
        await locator.check(timeout=10000)
    elif kind == "goto":
        url = str(action.get("url", "")).strip()
        if not url:
            raise RuntimeError("goto action requires url")
        await browser.navigate(url)
    elif kind == "wait":
        seconds = float(action.get("seconds", 1) or 1)
        await asyncio.sleep(max(0.1, min(seconds, 5.0)))
    else:
        raise RuntimeError(f"Unsupported selector action: {kind}")

    await browser.wait_for_load()
    current = await browser.get_current_url()
    return f"{kind} ok -> {current}"


# ---------------------------------------------------------------------------
# Moodle grading DOM helpers
# ---------------------------------------------------------------------------

async def extract_online_text_submission(browser: BrowserController) -> tuple[str, str]:
    boxes = browser.page.locator(".box.py-3")
    count = await boxes.count()
    texts: list[str] = []
    for index in range(count):
        text = (await boxes.nth(index).inner_text(timeout=5000)).strip()
        if text:
            texts.append(text)

    if len(texts) >= 2:
        question = texts[0]
        answer = max(texts[1:], key=len)
        if len(answer) >= 50:
            return question, answer

    page_text = (await browser.get_page_text()).strip()
    return "", page_text


async def read_full_online_text_submission(browser: BrowserController, grader_url: str) -> tuple[str, str]:
    full_submission_url = full_online_text_url_from_grader_url(grader_url)
    if full_submission_url:
        logger.info("grade_assignment: opening full online text submission")
        await browser.navigate(full_submission_url)
        await browser.wait_for_load()
        question, answer = await extract_online_text_submission(browser)
        logger.info(
            "grade_assignment: direct online text extraction "
            f"(question {len(question)} chars, answer {len(answer)} chars)"
        )
        if len(answer) >= 50:
            return question, answer

    logger.info("grade_assignment: using View full fallback")
    await browser.navigate(grader_url)
    await browser.wait_for_load()
    view_full = browser.page.get_by_role("button", name="View full", exact=True)
    if await view_full.count() > 0:
        await view_full.first.click(timeout=10000)
        await asyncio.sleep(1)

    question, answer = await extract_online_text_submission(browser)
    logger.info(
        "grade_assignment: View full extraction "
        f"(question {len(question)} chars, answer {len(answer)} chars)"
    )
    if len(answer) < 50:
        raise RuntimeError("Could not extract the full online-text submission from Moodle.")
    return question, answer


async def submit_grade_and_feedback(browser: BrowserController, grade_value: str, feedback: str) -> None:
    page = browser.page
    grade_input = page.locator('input[name="grade"]').first
    if await grade_input.count() > 0:
        await grade_input.fill(grade_value, timeout=10000)
    else:
        await page.get_by_role("textbox").first.fill(grade_value, timeout=10000)
    await type_editor(page, feedback)
    await page.get_by_role("button", name=re.compile(r"^Save changes$")).click(timeout=10000)
    await browser.wait_for_load()


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------

class SelectorPlanner:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env for the instructor agent.")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
        self.client = AsyncOpenAI(api_key=api_key)

    async def _complete_json(self, messages: list[dict[str, str]], error_label: str) -> dict[str, Any]:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        content = completion.choices[0].message.content or "{}"
        try:
            return _parse_llm_json(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON {error_label}: {content}") from exc

    async def plan(
        self,
        *,
        phase: str,
        goal: str,
        url: str,
        accessibility_tree: str,
        visible_text: str,
        memory: dict[str, Any],
    ) -> dict[str, Any]:
        system = (
            "You control a Moodle page through Playwright. "
            "Use the accessibility tree to choose selectors. Return only JSON. "
            "Prefer role/name selectors from the tree over CSS. "
            "Take the smallest useful next step, then wait for a fresh page observation. "
            "Never invent credentials; use only values provided in the user task. "
            "If the phase goal is satisfied, return done=true and no actions."
        )
        user = {
            "phase": phase,
            "goal": goal,
            "current_url": url,
            "memory": memory,
            "allowed_actions": [
                {
                    "action": "click",
                    "role": "button|link|textbox|checkbox|radio|tab|menuitem|option",
                    "name": "accessible name",
                    "occurrence": 0,
                },
                {
                    "action": "type",
                    "role": "textbox|searchbox|spinbutton",
                    "name": "accessible name",
                    "text": "text to fill",
                    "occurrence": 0,
                },
                {"action": "type_editor", "text": "feedback text"},
                {"action": "select", "role": "combobox", "name": "accessible name", "text": "option"},
                {"action": "check", "role": "checkbox|radio", "name": "accessible name"},
                {"action": "goto", "url": "absolute Moodle URL"},
                {"action": "wait", "seconds": 1},
            ],
            "response_schema": {
                "done": "boolean",
                "summary": "short status",
                "assignment_title": "optional title being graded",
                "student_name": "optional student name",
                "grade_value": "optional grade entered or observed",
                "feedback": "optional feedback entered or observed",
                "actions": "array of one or two allowed action objects",
            },
            "accessibility_tree": _compact(accessibility_tree, MAX_TREE_CHARS),
            "visible_text": _compact(visible_text, MAX_TEXT_CHARS),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ]
        parsed = await self._complete_json(messages, "selector plan")
        parsed.setdefault("actions", [])
        parsed.setdefault("done", False)
        return parsed

    async def grade_submission(
        self,
        *,
        assignment_question: str,
        student_answer: str,
        assignment_title: str = "",
        student_name: str = "",
    ) -> GradingResult:
        system = (
            "You are grading a Moodle assignment submission. "
            "Use only the assignment question and student's answer provided. "
            "Rubric: Correctness, meaning whether the answer properly answers the question, is worth 100 points. "
            "Return only JSON with an integer grade from 0 to 100 and short feedback for the student. "
            "If student_answer is non-empty, grade that answer; do not say the answer is missing."
        )
        user = {
            "assignment_title": assignment_title,
            "student_name": student_name,
            "rubric": "Correctness / whether it answers the question properly: 100 points",
            "response_schema": {
                "grade": "integer from 0 to 100",
                "feedback": "one or two short sentences",
            },
            "assignment_question": _compact(assignment_question, MAX_TEXT_CHARS // 2),
            "student_answer": _compact(student_answer, MAX_TEXT_CHARS),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ]
        parsed = await self._complete_json(messages, "grading result")

        try:
            grade = int(parsed.get("grade"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"LLM grading result missing integer grade: {parsed}") from exc
        grade = max(0, min(100, grade))

        feedback = str(parsed.get("feedback", "")).strip()
        if not feedback:
            feedback = f"Score: {grade}/100."
        return GradingResult(grade=grade, feedback=feedback)
