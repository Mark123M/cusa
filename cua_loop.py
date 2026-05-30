from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_CONTAINER = "cua-image"
DEFAULT_DISPLAY = ":99"
DEFAULT_MAX_TURNS = 50
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_DIR = "cua-runs"


class SafetyCheckRequired(RuntimeError):
    pass


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


def handle_computer_actions(vm: VM, actions: Iterable[Any]) -> None:
    for action in actions:
        action_type = str(get_value(action, "type", ""))
        x = int(round(_number(action, "x")))
        y = int(round(_number(action, "y")))
        button = _button_number(get_value(action, "button", "left"))
        keys = _action_keys(action)

        def run_action() -> None:
            if action_type == "click":
                vm.exec(
                    f"{display_prefix(vm)} xdotool mousemove {x} {y} click {button}",
                )
            elif action_type == "double_click":
                vm.exec(
                    f"{display_prefix(vm)} xdotool mousemove {x} {y} click --repeat 2 {button}",
                )
            elif action_type == "move":
                vm.exec(f"{display_prefix(vm)} xdotool mousemove {x} {y}")
            elif action_type == "scroll":
                dx = _number(action, "scroll_x", "scrollX", "delta_x", "deltaX")
                dy = _number(action, "scroll_y", "scrollY", "delta_y", "deltaY")
                vm.exec(f"{display_prefix(vm)} xdotool mousemove {x} {y}")
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
                vm.exec(f"{display_prefix(vm)} xdotool mousemove {start_x} {start_y} mousedown 1")
                for point_x, point_y in path[1:]:
                    vm.exec(f"{display_prefix(vm)} xdotool mousemove {point_x} {point_y}")
                vm.exec(f"{display_prefix(vm)} xdotool mouseup 1")
            elif action_type == "wait":
                duration_ms = _number(action, "ms", "duration_ms", default=2000)
                time.sleep(max(0, duration_ms) / 1000)
            elif action_type == "screenshot":
                return
            else:
                raise ValueError(f"Unsupported computer action: {action_type}")

        with_modifiers(vm, keys if action_type != "keypress" else [], run_action)


def capture_screenshot(vm: VM) -> bytes:
    screenshot = vm.exec(
        f"export DISPLAY={shell_quote(vm.display)} && import -window root png:-",
        decode=False,
    )
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
            },
            "responses": self.responses,
            "screenshots": self.screenshots,
        }

        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path.write_text("", encoding="utf-8")
        self.calls_path.write_text("", encoding="utf-8")
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

    def record_screenshot(self, label: str, data: bytes) -> Path:
        stem = sanitize_filename(label)
        path = self.screenshots_dir / f"{stem}.png"
        path.write_bytes(data)
        artifact = {
            "captured_at": utc_now(),
            "label": label,
            "path": self.relative(path),
        }
        self.screenshots.append(artifact)
        self.flush_trajectory()
        return path

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


def computer_use_loop(
    *,
    client: Any,
    model: str,
    recorder: TrajectoryRecorder,
    response: Any,
    vm: VM,
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
            for action_index, action in enumerate(actions):
                record_id = recorder.record_action_decision(
                    action=action,
                    action_index=action_index,
                    call_id=call_id,
                    turn=turn,
                )
                try:
                    handle_computer_actions(vm, [action])
                except Exception as error:
                    recorder.mark_action_failed(record_id, error)
                    try:
                        recorder.record_screenshot(
                            f"{turn:03d}-failure-after-call-{call_id or 'unknown'}",
                            capture_screenshot(vm),
                        )
                    except Exception:
                        pass
                    recorder.finish("failed", error=str(error))
                    raise
                recorder.mark_action_completed(record_id)

            screenshot = capture_screenshot(vm)
            recorder.record_screenshot(
                f"{turn:03d}-after-call-{call_id or 'unknown'}",
                screenshot,
            )
            tool_outputs.append(
                {
                    "type": "computer_call_output",
                    "call_id": call_id,
                    "output": screenshot_input(screenshot),
                },
            )

        current_response = client.responses.create(
            model=model,
            previous_response_id=response_id(current_response),
            tools=[{"type": "computer"}],
            input=tool_outputs,
        )

    message = f"Computer-use loop exhausted the configured {max_turns}-turn budget."
    recorder.finish("failed", error=message)
    raise RuntimeError(message)


def make_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def run_computer_use_task(
    *,
    client: Any,
    prompt: str,
    vm: VM,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    output_root: Path = Path(DEFAULT_OUTPUT_DIR),
) -> tuple[Any, TrajectoryRecorder]:
    recorder = TrajectoryRecorder(output_root, prompt=prompt, model=model, vm=vm)
    recorder.record_screenshot("000-initial", capture_screenshot(vm))

    try:
        first_response = client.responses.create(
            model=model,
            tools=[{"type": "computer"}],
            input=prompt,
        )
        final_response = computer_use_loop(
            client=client,
            model=model,
            recorder=recorder,
            response=first_response,
            vm=vm,
            max_turns=max_turns,
        )
        return final_response, recorder
    except Exception as error:
        if recorder.trajectory["status"] == "running":
            recorder.finish("failed", error=str(error))
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-backed OpenAI computer-use loop.")
    parser.add_argument("--prompt", required=True, help="Task prompt to send to the computer-use model.")
    parser.add_argument("--container", default=DEFAULT_CONTAINER, help="Docker container name.")
    parser.add_argument("--display", default=DEFAULT_DISPLAY, help="X11 display inside the container.")
    parser.add_argument("--model", default=os.environ.get("CUA_MODEL", DEFAULT_MODEL), help="OpenAI model.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Maximum Responses turns.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for trajectory artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    client = make_openai_client()
    vm = VM(display=args.display, container_name=args.container)

    try:
        final_response, recorder = run_computer_use_task(
            client=client,
            prompt=args.prompt,
            vm=vm,
            model=args.model,
            max_turns=args.max_turns,
            output_root=Path(args.output_dir),
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
