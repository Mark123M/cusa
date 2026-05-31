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
from typing import Any, Awaitable, Callable, Iterable


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


PLANNER_GUIDELINES = "TODO: central planner guidelines placeholder."


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-backed OpenAI computer-use loop.")
    parser.add_argument("--prompt", help="Task prompt to send to the computer-use model.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name.")
    parser.add_argument("--display", default=DEFAULT_DISPLAY, help="X11 display inside the container.")
    parser.add_argument("--list-windows", action="store_true", help="Print the current X11 window registry as JSON.")
    parser.add_argument("--model", default=os.environ.get("CUA_MODEL", DEFAULT_MODEL), help="OpenAI model.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum Responses turns.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for trajectory artifacts.")
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
        final_response, recorder = run_computer_use_task(
            client=client,
            prompt=args.prompt,
            vm=vm,
            model=args.model,
            max_turns=args.max_turns,
            output_root=Path(args.output_dir),
            window_id=window_assignment,
        )
    except SafetyCheckRequired as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Computer-use run failed: {error}", file=sys.stderr)
        return 1

    final_text = extract_final_response(final_response)
    print(f"Run artifacts: {recorder.run_dir}")
    if final_text:
        print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
