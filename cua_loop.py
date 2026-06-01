from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


DEFAULT_CONTAINER = "cua-image"
DEFAULT_DISPLAY = ":99"
DEFAULT_MAX_TURNS = 50
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_DIR = "cua-runs"


class SafetyCheckRequired(RuntimeError):
    pass


class StaleWindowError(RuntimeError):
    def __init__(
        self,
        window_id: str,
        *,
        logical_id: str | None = None,
        reason: str,
    ) -> None:
        self.window_id = window_id
        self.logical_id = logical_id
        self.reason = reason
        label = f"{logical_id} ({window_id})" if logical_id else window_id
        super().__init__(f"Stale window {label}: {reason}")


DockerExec = Callable[[str, str, bool], str | bytes]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def docker_exec(cmd: str, container_name: str, decode: bool = True) -> str | bytes:
    output = subprocess.check_output(
        ["docker", "exec", container_name, "sh", "-c", cmd],
        stderr=subprocess.PIPE,
    )
    if decode:
        return output.decode("utf-8", errors="replace")
    return output


@dataclass
class VM:
    display: str = DEFAULT_DISPLAY
    container_name: str = DEFAULT_CONTAINER
    executor: DockerExec = docker_exec

    def exec(self, cmd: str, decode: bool = True) -> str | bytes:
        return self.executor(cmd, self.container_name, decode)


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def display_prefix(vm: VM) -> str:
    return f"DISPLAY={shell_quote(vm.display)}"


@dataclass
class WindowInfo:
    logical_id: str
    window_id: str
    title: str
    pid: int | None
    geometry: dict[str, int]


@dataclass
class WindowAssignment:
    logical_id: str
    window_id: str
    pid: int | None = None


WindowTarget = WindowAssignment | None


@dataclass
class StageResult:
    what_they_did: str
    steps_taken: str
    failure_reason: str


@dataclass
class TaskProgress:
    original_prompt: str
    results_by_window: dict[str, list[StageResult]] = field(default_factory=dict)


class PlannerOutputError(ValueError):
    pass


class PlannerWindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str = Field(min_length=1)
    kind: Literal["browser", "terminal"]
    reason: str

    @field_validator("logical_id")
    @classmethod
    def _strip_logical_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("logical_id is required")
        return stripped


class PlannerAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: str = Field(min_length=1)
    subtasks: list[str] = Field(min_length=1)

    @field_validator("window")
    @classmethod
    def _strip_window(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("window is required")
        return stripped

    @field_validator("subtasks")
    @classmethod
    def _strip_subtasks(cls, value: list[str]) -> list[str]:
        subtasks = [item.strip() for item in value if item.strip()]
        if not subtasks:
            raise ValueError("at least one subtask is required")
        return subtasks


class PlannerPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    final_response: str
    windows_to_open: list[PlannerWindowRequest]
    assignments: list[PlannerAssignment]

    @model_validator(mode="after")
    def _validate_done_shape(self) -> "PlannerPlan":
        if self.done:
            if self.windows_to_open or self.assignments:
                raise ValueError("done plans cannot open windows or assign subtasks")
            if not self.final_response.strip():
                raise ValueError("done plans must include final_response")
        return self


ALLOWED_WINDOW_OPEN_COMMANDS: dict[str, str] = {
    "browser": "firefox-esr",
    "terminal": "xfce4-terminal",
}
DEFAULT_SUBAGENTS = 8
DEFAULT_MAX_STAGES = 20
DEFAULT_PLANNER_RETRIES = 2


PLANNER_GUIDELINES = """\
You are the central planner for a pool of computer-use subagents in one Linux virtual display.

Return only strict JSON matching the requested schema. Plan one stage at a time.
- Each assignment must target exactly one visible window logical_id from the latest registry.
- windows_to_open launches windows for future planner stages; do not assign subtasks to those requested logical_ids until they appear in a later registry.
- Each assignment's subtasks must be independent of every other assignment in the same stage.
- If a dependency exists between windows, serialize it by assigning only the prerequisite work in this stage.
- Do cross-window reasoning yourself after subagents return; do not ask a subagent to reason across windows.
- Use windows_to_open only for allowlisted semantic kinds: browser or terminal.
- Set done true only when the original user prompt is complete, and include final_response.
"""


def planner_text_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "cua_stage_plan",
            "strict": True,
            "schema": PlannerPlan.model_json_schema(),
        },
    }


def task_window_key(window_id: WindowTarget) -> str | None:
    if window_id is None:
        return None
    return window_id.logical_id or window_id.window_id


def append_stage_result(
    progress: TaskProgress,
    window_id: WindowTarget,
    result: StageResult,
) -> bool:
    key = task_window_key(window_id)
    if key is None:
        return False
    progress.results_by_window.setdefault(key, []).append(result)
    return True


def format_task_progress_ledger(progress: TaskProgress) -> str:
    lines: list[str] = []
    for window, results in progress.results_by_window.items():
        lines.append(f"## {window}")
        for index, result in enumerate(results, start=1):
            lines.extend(
                [
                    f"### Iteration {index}",
                    f"- What they did: {result.what_they_did}",
                    f"- Steps taken: {result.steps_taken}",
                    f"- Failure reason: {result.failure_reason}",
                ],
            )
    return "\n".join(lines) if lines else "No stage results yet."


def build_planner_prompt(progress: TaskProgress) -> str:
    return "\n\n".join(
        [
            PLANNER_GUIDELINES,
            f"Original prompt:\n{progress.original_prompt}",
            f"Progress ledger:\n{format_task_progress_ledger(progress)}",
        ],
    )


def latest_screenshots_by_window(
    screenshots: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for screenshot in screenshots:
        window = screenshot.get("window")
        if window:
            latest[str(window)] = screenshot
    return latest


def latest_screenshots_for_progress(
    progress: TaskProgress,
    screenshots: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = latest_screenshots_by_window(screenshots)
    return [
        latest[window]
        for window in progress.results_by_window
        if window in latest
    ]


def parse_planner_plan(response: Any) -> PlannerPlan:
    text = extract_final_response(response)
    if not text:
        raise PlannerOutputError("Planner response did not contain JSON text.")
    try:
        return PlannerPlan.model_validate_json(text)
    except ValidationError as error:
        raise PlannerOutputError(f"Planner response did not match schema: {error}") from error


def validate_planner_plan(
    plan: PlannerPlan,
    windows: Iterable[WindowInfo],
    *,
    max_assignments: int,
) -> None:
    if max_assignments < 1:
        raise PlannerOutputError("At least one subagent is required.")
    if plan.done:
        return

    if len(plan.assignments) > max_assignments:
        raise PlannerOutputError(
            f"Planner assigned {len(plan.assignments)} windows, but only {max_assignments} subagents are available.",
        )

    seen_open_requests: set[str] = set()
    for request in plan.windows_to_open:
        if request.logical_id in seen_open_requests:
            raise PlannerOutputError(f"Duplicate windows_to_open logical_id: {request.logical_id}")
        seen_open_requests.add(request.logical_id)
        if request.kind not in ALLOWED_WINDOW_OPEN_COMMANDS:
            raise PlannerOutputError(f"Unsupported window kind: {request.kind}")

    known_windows = {window.logical_id for window in windows}
    seen_assignments: set[str] = set()
    for assignment in plan.assignments:
        if assignment.window in seen_assignments:
            raise PlannerOutputError(f"Duplicate assignment for window: {assignment.window}")
        seen_assignments.add(assignment.window)
        if assignment.window not in known_windows:
            raise PlannerOutputError(f"Assignment references unknown window: {assignment.window}")

    if not plan.windows_to_open and not plan.assignments:
        raise PlannerOutputError("Planner must either finish, open a window, or assign subtasks.")


def window_assignments_by_logical(windows: Iterable[WindowInfo]) -> dict[str, WindowAssignment]:
    return {
        window.logical_id: window_assignment_from_info(window)
        for window in windows
    }


SubagentInvoker = Callable[
    [WindowTarget, str],
    StageResult | Awaitable[StageResult],
]


@dataclass
class PooledSubagent:
    name: str
    invoke: SubagentInvoker
    busy: bool = False


class ComputerUseSubagentPool:
    def __init__(self, subagents: Iterable[PooledSubagent]) -> None:
        self._subagents = list(subagents)

    @property
    def capacity(self) -> int:
        return len(self._subagents)

    async def invoke(self, window_id: WindowTarget, prompt: str) -> StageResult:
        subagent = self._available_subagent()
        if subagent is None:
            raise RuntimeError("No available subagents.")

        subagent.busy = True
        try:
            result = subagent.invoke(window_id, prompt)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as error:
            return StageResult(
                what_they_did="",
                steps_taken="",
                failure_reason=str(error),
            )
        finally:
            subagent.busy = False

    def _available_subagent(self) -> PooledSubagent | None:
        return next((subagent for subagent in self._subagents if not subagent.busy), None)


def normalize_window_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("Window id is empty")
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError as error:
        raise ValueError(f"Invalid window id: {value!r}") from error


def target_window_id(window_id: WindowTarget) -> str | None:
    if window_id is None:
        return None
    return window_id.window_id


def target_logical_id(window_id: WindowTarget) -> str | None:
    if window_id is None:
        return None
    return window_id.logical_id


def _parse_wmctrl_lpxg(output: str) -> list[WindowInfo]:
    windows: list[WindowInfo] = []
    counts: dict[str, int] = {}
    for line in output.splitlines():
        parts = line.split(None, 9)
        if len(parts) < 9:
            continue
        try:
            window_id = parts[0]
            pid = int(parts[2])
            x = int(parts[3])
            y = int(parts[4])
            width = int(parts[5])
            height = int(parts[6])
            normalize_window_id(window_id)
        except ValueError:
            continue

        title = parts[9] if len(parts) > 9 else ""
        wm_class = parts[7] if parts[7] != "N/A" else None
        base = _logical_id_base(wm_class, title)
        counts[base] = counts.get(base, 0) + 1

        windows.append(
            WindowInfo(
                logical_id=f"{base}_{counts[base]}",
                window_id=window_id,
                title=title,
                pid=pid if pid > 0 else None,
                geometry={
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
            ),
        )
    return windows


def _logical_id_base(wm_class: str | None, title: str) -> str:
    candidates = []
    if wm_class:
        candidates.append(wm_class.split(".")[-1])
    if title:
        candidates.append(title.split()[0])

    for candidate in candidates:
        safe = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
        if safe:
            return safe
    return "window"


def _logical_id_base_from_snapshot(logical_id: str) -> str:
    base, separator, suffix = logical_id.rpartition("_")
    if separator and base and suffix.isdecimal():
        return base
    return logical_id or "window"


class WindowLogicalIdRegistry:
    """Assign stable logical ids to X11 windows for one orchestrated prompt."""

    def __init__(self) -> None:
        self._logical_ids_by_window: dict[int, str] = {}
        self._next_suffix_by_base: dict[str, int] = {}

    def apply(self, windows: Iterable[WindowInfo]) -> list[WindowInfo]:
        return [self._with_stable_logical_id(window) for window in windows]

    def _with_stable_logical_id(self, window: WindowInfo) -> WindowInfo:
        normalized_window_id = normalize_window_id(window.window_id)
        logical_id = self._logical_ids_by_window.get(normalized_window_id)
        if logical_id is None:
            base = _logical_id_base_from_snapshot(window.logical_id)
            suffix = self._next_suffix_by_base.get(base, 1)
            logical_id = f"{base}_{suffix}"
            self._next_suffix_by_base[base] = suffix + 1
            self._logical_ids_by_window[normalized_window_id] = logical_id

        return WindowInfo(
            logical_id=logical_id,
            window_id=window.window_id,
            title=window.title,
            pid=window.pid,
            geometry=dict(window.geometry),
        )


def list_windows(vm: VM) -> list[WindowInfo]:
    raw = vm.exec(f"{display_prefix(vm)} wmctrl -lpxG")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return _parse_wmctrl_lpxg(raw)


def window_assignment_from_info(window: WindowInfo) -> WindowAssignment:
    return WindowAssignment(
        logical_id=window.logical_id,
        window_id=window.window_id,
        pid=window.pid,
    )


def find_window_info(windows: Iterable[WindowInfo], window_id: Any) -> WindowInfo | None:
    expected_id = normalize_window_id(window_id)
    return next(
        (
            window
            for window in windows
            if normalize_window_id(window.window_id) == expected_id
        ),
        None,
    )


def find_window_assignment(
    windows: Iterable[WindowInfo],
    window_id: Any,
) -> WindowAssignment | None:
    window = find_window_info(windows, window_id)
    if window is None:
        return None
    return window_assignment_from_info(window)


def validate_window_assignment(
    vm: VM,
    assignment: WindowAssignment,
    *,
    windows: list[WindowInfo] | None = None,
) -> WindowInfo:
    registry = windows if windows is not None else list_windows(vm)
    match = find_window_info(registry, assignment.window_id)
    if match is None:
        raise StaleWindowError(
            assignment.window_id,
            logical_id=assignment.logical_id,
            reason="window id is not present in registry",
        )
    if assignment.pid is not None and match.pid != assignment.pid:
        raise StaleWindowError(
            assignment.window_id,
            logical_id=assignment.logical_id,
            reason=f"pid changed from {assignment.pid} to {match.pid}",
        )
    return match


@dataclass
class ArbiterRequest:
    kind: str
    future: asyncio.Future[Any]
    action: Any | None = None
    window_id: WindowTarget = None


class InputArbiter:
    def __init__(self, vm: VM) -> None:
        self.vm = vm
        self._queue: asyncio.Queue[ArbiterRequest] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> "InputArbiter":
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())
        return self

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.join()
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def __aenter__(self) -> "InputArbiter":
        return self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    async def submit_action(self, action: Any, window_id: WindowTarget = None) -> None:
        await self._submit("action", action=action, window_id=window_id)

    async def screenshot(self, window_id: WindowTarget = None) -> bytes:
        return await self._submit("screenshot", window_id=window_id)

    async def _submit(
        self,
        kind: str,
        *,
        action: Any | None = None,
        window_id: WindowTarget = None,
    ) -> Any:
        self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        await self._queue.put(
            ArbiterRequest(kind=kind, future=future, action=action, window_id=window_id),
        )
        return await future

    async def _run(self) -> None:
        while True:
            request = await self._queue.get()
            try:
                if request.kind == "action":
                    result = await asyncio.to_thread(
                        handle_computer_actions,
                        self.vm,
                        [request.action],
                        request.window_id,
                    )
                elif request.kind == "screenshot":
                    result = await asyncio.to_thread(
                        capture_screenshot,
                        self.vm,
                        request.window_id,
                    )
                else:
                    raise ValueError(f"Unsupported arbiter request: {request.kind}")
            except Exception as error:
                if not request.future.done():
                    request.future.set_exception(error)
            else:
                if not request.future.done():
                    request.future.set_result(result)
            finally:
                self._queue.task_done()


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return to_plain(vars(value))
    return str(value)


def normalize_xdotool_key(key: str) -> str:
    lookup = key.strip().lower()
    aliases = {
        "alt": "alt",
        "apostrophe": "apostrophe",
        "backspace": "BackSpace",
        "backslash": "backslash",
        "comma": "comma",
        "cmd": "super",
        "command": "super",
        "control": "ctrl",
        "ctrl": "ctrl",
        "delete": "Delete",
        "down": "Down",
        "enter": "Return",
        "equal": "equal",
        "equals": "equal",
        "esc": "Escape",
        "escape": "Escape",
        "grave": "grave",
        "hyphen": "minus",
        "left": "Left",
        "leftbracket": "bracketleft",
        "meta": "super",
        "minus": "minus",
        "option": "alt",
        "period": "period",
        "plus": "plus",
        "return": "Return",
        "right": "Right",
        "rightbracket": "bracketright",
        "semicolon": "semicolon",
        "shift": "shift",
        "slash": "slash",
        "space": "space",
        "super": "super",
        "tab": "Tab",
        "up": "Up",
        "-": "minus",
    }
    return aliases.get(lookup, key)


def normalize_drag_path(path: Iterable[Any]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in path:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        else:
            x = getattr(point, "x", None)
            y = getattr(point, "y", None)

        if x is None or y is None:
            continue
        points.append((int(round(float(x))), int(round(float(y)))))
    return points


def with_modifiers(vm: VM, keys: Iterable[str], callback: Callable[[], None]) -> None:
    modifiers = [normalize_xdotool_key(key) for key in keys if str(key).strip()]
    for key in modifiers:
        vm.exec(f"{display_prefix(vm)} xdotool keydown {shell_quote(key)}")
    try:
        callback()
    finally:
        for key in reversed(modifiers):
            vm.exec(f"{display_prefix(vm)} xdotool keyup {shell_quote(key)}")


def _button_number(button: Any) -> int:
    if isinstance(button, int):
        return button
    lookup = str(button or "left").lower()
    if lookup == "right":
        return 3
    if lookup in {"middle", "wheel"}:
        return 2
    return 1


def _action_keys(action: Any) -> list[str]:
    keys = get_value(action, "keys")
    if keys is None:
        key = get_value(action, "key")
        return [str(key)] if key else []
    if isinstance(keys, str):
        return [keys]
    return [str(key) for key in keys]


def _number(action: Any, *keys: str, default: float = 0) -> float:
    for key in keys:
        value = get_value(action, key)
        if value is not None:
            return float(value)
    return default


def activate_window(vm: VM, window_id: WindowTarget) -> None:
    resolved = target_window_id(window_id)
    if not resolved:
        return
    quoted = shell_quote(resolved)
    try:
        vm.exec(f"{display_prefix(vm)} xdotool windowactivate --sync {quoted} windowraise {quoted}")
    except Exception as error:
        raise StaleWindowError(
            resolved,
            logical_id=target_logical_id(window_id),
            reason=f"activation failed: {error}",
        ) from error


def ensure_target_window_is_active(vm: VM, window_id: WindowTarget) -> None:
    resolved = target_window_id(window_id)
    if not resolved:
        return
    try:
        active = vm.exec(f"{display_prefix(vm)} xdotool getactivewindow")
        if isinstance(active, bytes):
            active = active.decode("utf-8", errors="replace")
        active_id = normalize_window_id(active)
        expected_id = normalize_window_id(resolved)
    except Exception as error:
        raise StaleWindowError(
            resolved,
            logical_id=target_logical_id(window_id),
            reason=f"active-window check failed: {error}",
        ) from error
    if active_id != expected_id:
        raise StaleWindowError(
            resolved,
            logical_id=target_logical_id(window_id),
            reason=f"active window is {active.strip()}, expected {resolved}",
        )


def mousemove_command(vm: VM, x: int, y: int, window_id: WindowTarget) -> str:
    resolved = target_window_id(window_id)
    if resolved:
        return (
            f"{display_prefix(vm)} xdotool mousemove --window "
            f"{shell_quote(resolved)} {x} {y}"
        )
    return f"{display_prefix(vm)} xdotool mousemove {x} {y}"


def handle_computer_actions(
    vm: VM,
    actions: Iterable[Any],
    window_id: WindowTarget = None,
) -> None:
    for action in actions:
        action_type = str(get_value(action, "type", ""))
        x = int(round(_number(action, "x")))
        y = int(round(_number(action, "y")))
        button = _button_number(get_value(action, "button", "left"))
        keys = _action_keys(action)

        activate_window(vm, window_id)
        if action_type in {"type", "keypress"}:
            ensure_target_window_is_active(vm, window_id)

        def run_action() -> None:
            if action_type == "click":
                vm.exec(
                    f"{mousemove_command(vm, x, y, window_id)} click {button}",
                )
            elif action_type == "double_click":
                vm.exec(
                    f"{mousemove_command(vm, x, y, window_id)} click --repeat 2 {button}",
                )
            elif action_type == "move":
                vm.exec(mousemove_command(vm, x, y, window_id))
            elif action_type == "scroll":
                dx = _number(action, "scroll_x", "scrollX", "delta_x", "deltaX")
                dy = _number(action, "scroll_y", "scrollY", "delta_y", "deltaY")
                vm.exec(mousemove_command(vm, x, y, window_id))
                if dy:
                    scroll_button = 5 if dy > 0 else 4
                    for _ in range(max(1, int(abs(dy) // 100) or 1)):
                        vm.exec(f"{display_prefix(vm)} xdotool click {scroll_button}")
                if dx:
                    scroll_button = 7 if dx > 0 else 6
                    for _ in range(max(1, int(abs(dx) // 100) or 1)):
                        vm.exec(f"{display_prefix(vm)} xdotool click {scroll_button}")
            elif action_type == "type":
                text = str(get_value(action, "text", ""))
                vm.exec(
                    f"{display_prefix(vm)} xdotool type --delay 0 {shell_quote(text)}",
                )
            elif action_type == "keypress":
                normalized = [normalize_xdotool_key(key) for key in keys]
                if not normalized:
                    raise ValueError("keypress action did not include keys")
                vm.exec(
                    f"{display_prefix(vm)} xdotool key "
                    + shell_quote("+".join(normalized)),
                )
            elif action_type == "drag":
                path = normalize_drag_path(get_value(action, "path", []))
                if len(path) < 2:
                    raise ValueError("drag action did not include a valid path")
                start_x, start_y = path[0]
                vm.exec(f"{mousemove_command(vm, start_x, start_y, window_id)} mousedown 1")
                for point_x, point_y in path[1:]:
                    vm.exec(mousemove_command(vm, point_x, point_y, window_id))
                vm.exec(f"{display_prefix(vm)} xdotool mouseup 1")
            elif action_type == "wait":
                duration_ms = _number(action, "ms", "duration_ms", default=2000)
                time.sleep(max(0, duration_ms) / 1000)
            elif action_type == "screenshot":
                return
            else:
                raise ValueError(f"Unsupported computer action: {action_type}")

        with_modifiers(vm, keys if action_type != "keypress" else [], run_action)


def capture_screenshot(vm: VM, window_id: WindowTarget = None) -> bytes:
    resolved = target_window_id(window_id)
    target = shell_quote(resolved or "root")
    try:
        screenshot = vm.exec(
            f"export DISPLAY={shell_quote(vm.display)} && import -window {target} png:-",
            decode=False,
        )
    except Exception as error:
        if resolved:
            raise StaleWindowError(
                resolved,
                logical_id=target_logical_id(window_id),
                reason=f"screenshot failed: {error}",
            ) from error
        raise
    if not isinstance(screenshot, bytes):
        return screenshot.encode("utf-8")
    return screenshot


def sanitize_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-")[:80] or "item"


class TrajectoryRecorder:
    def __init__(
        self,
        output_root: Path,
        *,
        prompt: str,
        model: str,
        vm: VM,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}-{secrets.token_hex(4)}"
        self.run_dir = output_root / self.run_id
        self.responses_dir = self.run_dir / "responses"
        self.screenshots_dir = self.run_dir / "screenshots"
        self.prompts_dir = self.run_dir / "prompts"
        self.actions_path = self.run_dir / "actions.jsonl"
        self.calls_path = self.run_dir / "computer_calls.jsonl"
        self.trajectory_path = self.run_dir / "trajectory.json"
        self.window_snapshots_path = self.run_dir / "window_snapshots.jsonl"
        self.actions: list[dict[str, Any]] = []
        self.responses: list[str] = []
        self.screenshots: list[dict[str, Any]] = []
        self.started_at = utc_now()
        self.trajectory: dict[str, Any] = {
            "version": 1,
            "run_id": self.run_id,
            "status": "running",
            "prompt": prompt,
            "model": model,
            "container": vm.container_name,
            "display": vm.display,
            "started_at": self.started_at,
            "completed_at": None,
            "final_response": None,
            "error": None,
            "files": {
                "actions": "actions.jsonl",
                "computer_calls": "computer_calls.jsonl",
                "responses": "responses/",
                "screenshots": "screenshots/",
                "window_snapshots": "window_snapshots.jsonl",
            },
            "responses": self.responses,
            "screenshots": self.screenshots,
        }

        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path.write_text("", encoding="utf-8")
        self.calls_path.write_text("", encoding="utf-8")
        self.window_snapshots_path.write_text("", encoding="utf-8")
        self.flush_trajectory()

    def relative(self, path: Path) -> str:
        return path.relative_to(self.run_dir).as_posix()

    def flush_trajectory(self) -> None:
        self.trajectory_path.write_text(
            json.dumps(self.trajectory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def flush_actions(self) -> None:
        lines: list[str] = []
        previous_turn: int | None = None

        for action in self.actions:
            turn = action.get("turn")
            if previous_turn is not None and turn != previous_turn:
                lines.append("")
            lines.append(json.dumps(action, sort_keys=True))
            previous_turn = turn

        if lines:
            lines.append("")

        payload = "\n".join(lines)
        self.actions_path.write_text(payload, encoding="utf-8")

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def record_response(self, turn: int, response: Any) -> Path:
        path = self.responses_dir / f"turn-{turn:03d}.json"
        path.write_text(
            json.dumps(to_plain(response), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relative_path = self.relative(path)
        if relative_path not in self.responses:
            self.responses.append(relative_path)
            self.flush_trajectory()
        return path

    def record_computer_call(self, turn: int, call: Any) -> None:
        self.append_jsonl(
            self.calls_path,
            {
                "call": to_plain(call),
                "call_id": get_value(call, "call_id"),
                "recorded_at": utc_now(),
                "turn": turn,
            },
        )

    def record_action_decision(
        self,
        *,
        action: Any,
        action_index: int,
        call_id: Any,
        turn: int,
    ) -> int:
        record_id = len(self.actions)
        self.actions.append(
            {
                "action": to_plain(action),
                "action_index": action_index,
                "call_id": call_id,
                "completed_at": None,
                "decided_at": utc_now(),
                "error": None,
                "record_id": record_id,
                "status": "pending",
                "turn": turn,
            },
        )
        self.flush_actions()
        return record_id

    def mark_action_completed(self, record_id: int) -> None:
        self.actions[record_id]["completed_at"] = utc_now()
        self.actions[record_id]["status"] = "completed"
        self.flush_actions()

    def mark_action_failed(self, record_id: int, error: BaseException) -> None:
        self.actions[record_id]["completed_at"] = utc_now()
        self.actions[record_id]["error"] = str(error)
        self.actions[record_id]["status"] = "failed"
        self.flush_actions()

    def record_screenshot(
        self,
        label: str,
        data: bytes,
        *,
        call_id: Any,
        window_id: WindowTarget,
    ) -> Path | None:
        window = task_window_key(window_id)
        if window is None:
            return None

        stem = sanitize_filename(label)
        call_key = sanitize_filename(str(call_id or "unknown"))
        window_key = sanitize_filename(window)
        path = self.screenshots_dir / call_key / window_key / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        artifact = {
            "call_id": call_id,
            "captured_at": utc_now(),
            "label": label,
            "path": self.relative(path),
            "window": window,
        }
        self.screenshots.append(artifact)
        self.flush_trajectory()
        return path

    def record_prompt_artifacts(
        self,
        *,
        call_id: str,
        stage: int,
        actor: Literal["planner", "subagents"],
        prompt_text: str,
        screenshots: Iterable[dict[str, Any]] | None = None,
    ) -> Path:
        call_key = sanitize_filename(call_id)
        artifact_dir = self.prompts_dir / call_key / f"stage_{stage:03d}" / actor
        artifact_dir.mkdir(parents=True, exist_ok=True)

        (artifact_dir / "prompt.txt").write_text(
            prompt_text if prompt_text.endswith("\n") else prompt_text + "\n",
            encoding="utf-8",
        )

        if screenshots is not None:
            for index, artifact in enumerate(screenshots, start=1):
                source_path = artifact.get("path")
                if not source_path:
                    continue
                source = self.run_dir / str(source_path)
                try:
                    data = source.read_bytes()
                except OSError:
                    continue

                source_stem = sanitize_filename(Path(str(source_path)).stem)
                filename = f"{index:03d}-{source_stem}.png"
                destination = artifact_dir / filename
                destination.write_bytes(data)

        return artifact_dir

    def record_window_snapshot(
        self,
        label: str,
        windows: list[WindowInfo],
        *,
        assignment: WindowAssignment | None = None,
    ) -> None:
        self.append_jsonl(
            self.window_snapshots_path,
            {
                "assignment": to_plain(assignment),
                "label": label,
                "recorded_at": utc_now(),
                "windows": to_plain(windows),
            },
        )

    def finish(
        self,
        status: str,
        *,
        error: str | None = None,
        final_response: str | None = None,
    ) -> None:
        self.trajectory["completed_at"] = utc_now()
        self.trajectory["error"] = error
        self.trajectory["final_response"] = final_response
        self.trajectory["status"] = status
        self.flush_trajectory()


def output_items(response: Any) -> list[Any]:
    output = get_value(response, "output", [])
    return list(output or [])


def response_id(response: Any) -> str:
    value = get_value(response, "id")
    if not value:
        raise ValueError("Response is missing an id")
    return str(value)


def computer_calls(response: Any) -> list[Any]:
    return [item for item in output_items(response) if get_value(item, "type") == "computer_call"]


def extract_final_response(response: Any) -> str:
    direct = get_value(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    texts: list[str] = []
    for item in output_items(response):
        if get_value(item, "type") != "message":
            continue
        for part in get_value(item, "content", []) or []:
            text = get_value(part, "text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n\n".join(texts)


def screenshot_input(data: bytes) -> dict[str, Any]:
    return {
        "type": "computer_screenshot",
        "image_url": "data:image/png;base64,"
        + base64.b64encode(data).decode("ascii"),
        "detail": "original",
    }


def input_image(data: bytes) -> dict[str, Any]:
    return {
        "type": "input_image",
        "image_url": "data:image/png;base64,"
        + base64.b64encode(data).decode("ascii"),
        "detail": "original",
    }


def format_window_registry(windows: Iterable[WindowInfo]) -> str:
    return json.dumps(to_plain(list(windows)), indent=2, sort_keys=True)


def screenshot_artifact_bytes(
    recorder: TrajectoryRecorder,
    artifact: dict[str, Any],
) -> bytes | None:
    path = artifact.get("path")
    if not path:
        return None
    screenshot_path = recorder.run_dir / str(path)
    try:
        return screenshot_path.read_bytes()
    except OSError:
        return None


def latest_screenshots_for_windows(
    windows: Iterable[WindowInfo],
    screenshots: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = latest_screenshots_by_window(screenshots)
    return [
        latest[window.logical_id]
        for window in windows
        if window.logical_id in latest
    ]


def build_planner_input(
    *,
    progress: TaskProgress,
    windows: list[WindowInfo],
    recorder: TrajectoryRecorder,
    subagent_capacity: int,
    feedback: str = "",
    latest_screenshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if latest_screenshots is None:
        latest_screenshots = latest_screenshots_for_windows(windows, recorder.screenshots)
    screenshot_lines = [
        f"- {artifact.get('window')}: {artifact.get('path')}"
        for artifact in latest_screenshots
    ]
    text_parts = [
        build_planner_prompt(progress),
        f"Available subagents for this stage: {subagent_capacity}",
        "Latest window registry:\n" + format_window_registry(windows),
        "Latest screenshots:\n"
        + ("\n".join(screenshot_lines) if screenshot_lines else "No screenshots available."),
    ]
    if feedback:
        text_parts.append(f"Previous planner output was invalid:\n{feedback}")

    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": "\n\n".join(text_parts)},
    ]
    for artifact in latest_screenshots:
        image = screenshot_artifact_bytes(recorder, artifact)
        if image is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Screenshot for {artifact.get('window')}: {artifact.get('path')}",
                },
            )
            content.append(input_image(image))

    return [{"role": "user", "content": content}]


async def request_planner_plan(
    *,
    client: Any,
    model: str,
    progress: TaskProgress,
    windows: list[WindowInfo],
    recorder: TrajectoryRecorder,
    subagent_capacity: int,
    stage: int,
    max_retries: int = DEFAULT_PLANNER_RETRIES,
    dump_prompts: bool = False,
) -> PlannerPlan:
    feedback = ""
    last_error: PlannerOutputError | None = None
    for attempt in range(1, max_retries + 2):
        latest_screenshots = latest_screenshots_for_windows(windows, recorder.screenshots)
        planner_input = build_planner_input(
            progress=progress,
            windows=windows,
            recorder=recorder,
            subagent_capacity=subagent_capacity,
            feedback=feedback,
            latest_screenshots=latest_screenshots,
        )
        if dump_prompts:
            prompt_text = str(planner_input[0]["content"][0]["text"])
            recorder.record_prompt_artifacts(
                call_id=f"planner-stage-{stage:03d}-attempt-{attempt:03d}",
                stage=stage,
                actor="planner",
                prompt_text=prompt_text,
                screenshots=latest_screenshots,
            )
        response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            instructions=PLANNER_GUIDELINES,
            input=planner_input,
            text=planner_text_format(),
            truncation="auto",
        )
        recorder.record_response(stage * 100 + attempt, response)
        try:
            plan = parse_planner_plan(response)
            validate_planner_plan(
                plan,
                windows,
                max_assignments=subagent_capacity,
            )
            return plan
        except PlannerOutputError as error:
            last_error = error
            feedback = str(error)
    assert last_error is not None
    raise last_error


async def computer_use_loop_async(
    *,
    client: Any,
    model: str,
    recorder: TrajectoryRecorder,
    response: Any,
    arbiter: InputArbiter,
    window_id: WindowTarget = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Any:
    current_response = response

    for turn in range(1, max_turns + 1):
        recorder.record_response(turn, current_response)
        calls = computer_calls(current_response)

        if not calls:
            final_response = extract_final_response(current_response)
            recorder.finish("completed", final_response=final_response)
            return current_response

        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            pending_safety_checks = get_value(call, "pending_safety_checks", []) or []
            call_id = get_value(call, "call_id")
            recorder.record_computer_call(turn, call)

            if pending_safety_checks:
                message = f"Pending safety checks require operator acknowledgement: {to_plain(pending_safety_checks)}"
                recorder.finish("blocked", error=message)
                raise SafetyCheckRequired(message)

            actions = list(get_value(call, "actions", []) or [])
            pending_screenshot_records: list[int] = []
            for action_index, action in enumerate(actions):
                record_id = recorder.record_action_decision(
                    action=action,
                    action_index=action_index,
                    call_id=call_id,
                    turn=turn,
                )
                try:
                    await arbiter.submit_action(action, window_id=window_id)
                except Exception as error:
                    recorder.mark_action_failed(record_id, error)
                    for pending_record_id in pending_screenshot_records:
                        recorder.mark_action_failed(pending_record_id, error)
                    if target_window_id(window_id) is not None:
                        try:
                            recorder.record_screenshot(
                                f"{turn:03d}-failure-after-call-{call_id or 'unknown'}",
                                await arbiter.screenshot(window_id=window_id),
                                call_id=call_id,
                                window_id=window_id,
                            )
                        except Exception:
                            pass
                    recorder.finish("failed", error=str(error))
                    raise
                if str(get_value(action, "type", "")) == "screenshot":
                    pending_screenshot_records.append(record_id)
                else:
                    recorder.mark_action_completed(record_id)

            if target_window_id(window_id) is None:
                for record_id in pending_screenshot_records:
                    recorder.mark_action_completed(record_id)
                continue

            try:
                screenshot = await arbiter.screenshot(window_id=window_id)
            except Exception as error:
                for record_id in pending_screenshot_records:
                    recorder.mark_action_failed(record_id, error)
                recorder.finish("failed", error=str(error))
                raise
            for record_id in pending_screenshot_records:
                recorder.mark_action_completed(record_id)
            recorder.record_screenshot(
                f"{turn:03d}-after-call-{call_id or 'unknown'}",
                screenshot,
                call_id=call_id,
                window_id=window_id,
            )
            tool_outputs.append(
                {
                    "type": "computer_call_output",
                    "call_id": call_id,
                    "output": screenshot_input(screenshot),
                },
            )

        current_response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            previous_response_id=response_id(current_response),
            tools=[{"type": "computer"}],
            input=tool_outputs,
        )

    message = f"Computer-use loop exhausted the configured {max_turns}-turn budget."
    recorder.finish("failed", error=message)
    raise RuntimeError(message)


def computer_use_loop(
    *,
    client: Any,
    model: str,
    recorder: TrajectoryRecorder,
    response: Any,
    vm: VM,
    window_id: WindowTarget = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Any:
    async def run() -> Any:
        async with InputArbiter(vm) as arbiter:
            return await computer_use_loop_async(
                client=client,
                model=model,
                recorder=recorder,
                response=response,
                arbiter=arbiter,
                window_id=window_id,
                max_turns=max_turns,
            )

    return asyncio.run(run())


def make_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


async def run_computer_use_task_async(
    *,
    client: Any,
    prompt: str,
    vm: VM,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    output_root: Path = Path(DEFAULT_OUTPUT_DIR),
    window_id: WindowTarget = None,
    arbiter: InputArbiter | None = None,
) -> tuple[Any, TrajectoryRecorder]:
    recorder = TrajectoryRecorder(output_root, prompt=prompt, model=model, vm=vm)
    owns_arbiter = arbiter is None
    if arbiter is None:
        arbiter = InputArbiter(vm)
    arbiter.start()

    try:
        if isinstance(window_id, WindowAssignment):
            windows = await asyncio.to_thread(list_windows, vm)
            recorder.record_window_snapshot(
                "stage-start",
                windows,
                assignment=window_id,
            )
            validate_window_assignment(vm, window_id, windows=windows)
        if target_window_id(window_id) is not None:
            recorder.record_screenshot(
                "000-initial",
                await arbiter.screenshot(window_id=window_id),
                call_id="initial",
                window_id=window_id,
            )
        first_response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            tools=[{"type": "computer"}],
            input=prompt,
        )
        final_response = await computer_use_loop_async(
            client=client,
            model=model,
            recorder=recorder,
            response=first_response,
            arbiter=arbiter,
            window_id=window_id,
            max_turns=max_turns,
        )
        return final_response, recorder
    except Exception as error:
        if recorder.trajectory["status"] == "running":
            recorder.finish("failed", error=str(error))
        raise
    finally:
        if owns_arbiter:
            await arbiter.stop()


def run_computer_use_task(
    *,
    client: Any,
    prompt: str,
    vm: VM,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    output_root: Path = Path(DEFAULT_OUTPUT_DIR),
    window_id: WindowTarget = None,
) -> tuple[Any, TrajectoryRecorder]:
    return asyncio.run(
        run_computer_use_task_async(
            client=client,
            prompt=prompt,
            vm=vm,
            model=model,
            max_turns=max_turns,
            output_root=output_root,
            window_id=window_id,
        ),
    )


async def refresh_window_registry(
    *,
    vm: VM,
    recorder: TrajectoryRecorder,
    label: str,
    logical_ids: WindowLogicalIdRegistry | None = None,
) -> list[WindowInfo]:
    windows = await asyncio.to_thread(list_windows, vm)
    if logical_ids is not None:
        windows = logical_ids.apply(windows)
    recorder.record_window_snapshot(label, windows)
    return windows


async def capture_registry_screenshots(
    *,
    arbiter: InputArbiter,
    recorder: TrajectoryRecorder,
    windows: Iterable[WindowInfo],
    label: str,
    call_id: str,
) -> None:
    for window in windows:
        assignment = window_assignment_from_info(window)
        try:
            screenshot = await arbiter.screenshot(window_id=assignment)
        except Exception:
            continue
        recorder.record_screenshot(
            f"{label}-{window.logical_id}",
            screenshot,
            call_id=call_id,
            window_id=assignment,
        )


def open_planned_windows(
    *,
    vm: VM,
    requests: list[PlannerWindowRequest],
) -> None:
    for request in requests:
        command = ALLOWED_WINDOW_OPEN_COMMANDS.get(request.kind)
        if command is None:
            raise PlannerOutputError(f"Unsupported window kind: {request.kind}")
        vm.exec(f"{display_prefix(vm)} {command} >/dev/null 2>&1 &")


def resolve_planner_assignments(
    plan: PlannerPlan,
    windows: Iterable[WindowInfo],
) -> list[tuple[WindowAssignment, list[str]]]:
    available = window_assignments_by_logical(windows)
    resolved: list[tuple[WindowAssignment, list[str]]] = []
    seen_window_ids: set[int] = set()

    for assignment in plan.assignments:
        window = available.get(assignment.window)
        if window is None:
            raise PlannerOutputError(f"Assignment references unknown window: {assignment.window}")
        normalized = normalize_window_id(window.window_id)
        if normalized in seen_window_ids:
            raise PlannerOutputError(f"Multiple assignments resolved to window id: {window.window_id}")
        seen_window_ids.add(normalized)
        resolved.append((window, assignment.subtasks))

    return resolved


def format_subagent_prompt(
    *,
    original_prompt: str,
    stage: int,
    window: WindowAssignment,
    subtasks: list[str],
) -> str:
    lines = [
        "You are a computer-use subagent in a staged multi-window task.",
        f"Original user task: {original_prompt}",
        f"Stage: {stage}",
        f"Assigned window: {window.logical_id} ({window.window_id})",
        "Work only in this assigned window. Do not reason across windows.",
        "Complete these subtasks in order:",
    ]
    lines.extend(f"{index}. {subtask}" for index, subtask in enumerate(subtasks, start=1))
    lines.append(
        "Finish with a concise report of what you did, steps taken, and any blocker.",
    )
    return "\n".join(lines)


def make_computer_use_subagent_pool(
    *,
    client: Any,
    vm: VM,
    model: str,
    max_turns: int,
    output_root: Path,
    arbiter: InputArbiter,
    count: int,
) -> ComputerUseSubagentPool:
    if count < 1:
        raise ValueError("subagent count must be at least 1")

    async def invoke(window_id: WindowTarget, prompt: str) -> StageResult:
        response, sub_recorder = await run_computer_use_task_async(
            client=client,
            prompt=prompt,
            vm=vm,
            model=model,
            max_turns=max_turns,
            output_root=output_root,
            window_id=window_id,
            arbiter=arbiter,
        )
        final_text = extract_final_response(response)
        return StageResult(
            what_they_did=final_text,
            steps_taken=f"Completed CUA run. Artifacts: {sub_recorder.run_dir}",
            failure_reason="",
        )

    return ComputerUseSubagentPool(
        PooledSubagent(name=f"worker-{index}", invoke=invoke)
        for index in range(1, count + 1)
    )


async def run_orchestrated_computer_use_task_async(
    *,
    client: Any,
    prompt: str,
    vm: VM,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_stages: int = DEFAULT_MAX_STAGES,
    output_root: Path = Path(DEFAULT_OUTPUT_DIR),
    subagent_count: int = DEFAULT_SUBAGENTS,
    pool: ComputerUseSubagentPool | None = None,
    arbiter: InputArbiter | None = None,
    planner_max_retries: int = DEFAULT_PLANNER_RETRIES,
    dump_prompts: bool = False,
) -> tuple[str, TrajectoryRecorder]:
    recorder = TrajectoryRecorder(output_root, prompt=prompt, model=model, vm=vm)
    progress = TaskProgress(original_prompt=prompt)
    owns_arbiter = arbiter is None
    if arbiter is None:
        arbiter = InputArbiter(vm)
    arbiter.start()

    if pool is None:
        pool = make_computer_use_subagent_pool(
            client=client,
            vm=vm,
            model=model,
            max_turns=max_turns,
            output_root=output_root,
            arbiter=arbiter,
            count=subagent_count,
        )

    logical_ids = WindowLogicalIdRegistry()

    try:
        windows = await refresh_window_registry(
            vm=vm,
            recorder=recorder,
            label="orchestration-start",
            logical_ids=logical_ids,
        )
        await capture_registry_screenshots(
            arbiter=arbiter,
            recorder=recorder,
            windows=windows,
            label="000-initial",
            call_id="orchestrator-stage-000",
        )

        for stage in range(1, max_stages + 1):
            plan = await request_planner_plan(
                client=client,
                model=model,
                progress=progress,
                windows=windows,
                recorder=recorder,
                subagent_capacity=pool.capacity,
                stage=stage,
                max_retries=planner_max_retries,
                dump_prompts=dump_prompts,
            )

            if plan.done:
                recorder.finish("completed", final_response=plan.final_response)
                return plan.final_response, recorder

            assignments = resolve_planner_assignments(plan, windows)

            if plan.windows_to_open:
                open_planned_windows(
                    vm=vm,
                    requests=plan.windows_to_open,
                )
                windows = await refresh_window_registry(
                    vm=vm,
                    recorder=recorder,
                    label=f"stage-{stage:03d}-after-window-open-requests",
                    logical_ids=logical_ids,
                )

            if not assignments:
                await capture_registry_screenshots(
                    arbiter=arbiter,
                    recorder=recorder,
                    windows=windows,
                    label=f"{stage:03d}-after-open-only",
                    call_id=f"orchestrator-stage-{stage:03d}",
                )
                continue

            subagent_invocations = []
            for window, subtasks in assignments:
                subagent_prompt = format_subagent_prompt(
                    original_prompt=prompt,
                    stage=stage,
                    window=window,
                    subtasks=subtasks,
                )
                if dump_prompts:
                    recorder.record_prompt_artifacts(
                        call_id=f"subagent-stage-{stage:03d}-{window.logical_id}",
                        stage=stage,
                        actor="subagents",
                        prompt_text=subagent_prompt,
                    )
                subagent_invocations.append(pool.invoke(window, subagent_prompt))

            results = await asyncio.gather(*subagent_invocations)

            for (window, _subtasks), result in zip(assignments, results):
                append_stage_result(progress, window, result)

            windows = await refresh_window_registry(
                vm=vm,
                recorder=recorder,
                label=f"stage-{stage:03d}-after-subagents",
                logical_ids=logical_ids,
            )
            await capture_registry_screenshots(
                arbiter=arbiter,
                recorder=recorder,
                windows=windows,
                label=f"{stage:03d}-after-subagents",
                call_id=f"orchestrator-stage-{stage:03d}",
            )

        message = f"Orchestration exhausted the configured {max_stages}-stage budget."
        recorder.finish("failed", error=message)
        raise RuntimeError(message)
    except Exception as error:
        if recorder.trajectory["status"] == "running":
            recorder.finish("failed", error=str(error))
        raise
    finally:
        if owns_arbiter:
            await arbiter.stop()


def run_orchestrated_computer_use_task(
    *,
    client: Any,
    prompt: str,
    vm: VM,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_stages: int = DEFAULT_MAX_STAGES,
    output_root: Path = Path(DEFAULT_OUTPUT_DIR),
    subagent_count: int = DEFAULT_SUBAGENTS,
    dump_prompts: bool = False,
) -> tuple[str, TrajectoryRecorder]:
    return asyncio.run(
        run_orchestrated_computer_use_task_async(
            client=client,
            prompt=prompt,
            vm=vm,
            model=model,
            max_turns=max_turns,
            max_stages=max_stages,
            output_root=output_root,
            subagent_count=subagent_count,
            dump_prompts=dump_prompts,
        ),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-backed OpenAI computer-use loop.")
    parser.add_argument("--prompt", help="Task prompt to send to the computer-use model.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name.")
    parser.add_argument("--display", default=DEFAULT_DISPLAY, help="X11 display inside the container.")
    parser.add_argument("--list-windows", action="store_true", help="Print the current X11 window registry as JSON.")
    parser.add_argument("--model", default=os.environ.get("CUA_MODEL", DEFAULT_MODEL), help="OpenAI model.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum Responses turns.")
    parser.add_argument("--max-stages", type=int, default=DEFAULT_MAX_STAGES, help="Maximum orchestrated planner stages.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for trajectory artifacts.")
    parser.add_argument("--orchestrate", action="store_true", help="Use the staged multi-subagent orchestration loop.")
    parser.add_argument("--subagents", type=int, default=DEFAULT_SUBAGENTS, help="Number of subagents for --orchestrate.")
    parser.add_argument(
        "--dump-prompts",
        action="store_true",
        help="Dump orchestrated planner and subagent prompt artifacts under the run directory.",
    )
    parser.add_argument("--window-id", default=None, help="Optional X11 window id to target for actions and screenshots.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    vm = VM(display=args.display, container_name=args.container)

    if args.list_windows:
        print(
            json.dumps(
                {
                    "captured_at": utc_now(),
                    "container": vm.container_name,
                    "display": vm.display,
                    "windows": to_plain(list_windows(vm)),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return 0

    if not args.prompt:
        print("--prompt is required unless --list-windows is set", file=sys.stderr)
        return 2

    window_assignment: WindowAssignment | None = None
    if args.window_id:
        try:
            windows = list_windows(vm)
            window_assignment = find_window_assignment(windows, args.window_id)
        except ValueError:
            print(
                f"Invalid --window-id {args.window_id!r}; use an X11 id from --list-windows, not a logical_id.",
                file=sys.stderr,
            )
            return 2

        if window_assignment is None:
            print(f"Window id not found: {args.window_id}", file=sys.stderr)
            return 2

    client = make_openai_client()

    try:
        if args.orchestrate:
            final_response, recorder = run_orchestrated_computer_use_task(
                client=client,
                prompt=args.prompt,
                vm=vm,
                model=args.model,
                max_turns=args.max_turns,
                max_stages=args.max_stages,
                output_root=Path(args.output_dir),
                subagent_count=args.subagents,
                dump_prompts=args.dump_prompts,
            )
            final_text = final_response
        else:
            final_response, recorder = run_computer_use_task(
                client=client,
                prompt=args.prompt,
                vm=vm,
                model=args.model,
                max_turns=args.max_turns,
                output_root=Path(args.output_dir),
                window_id=window_assignment,
            )
            final_text = extract_final_response(final_response)
    except SafetyCheckRequired as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Computer-use run failed: {error}", file=sys.stderr)
        return 1

    print(f"Run artifacts: {recorder.run_dir}")
    if final_text:
        print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
