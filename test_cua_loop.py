from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cua_loop import (
    InputArbiter,
    SafetyCheckRequired,
    VM,
    run_computer_use_task,
)


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake response queued.")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line]


def final_response(text="Done."):
    return {
        "id": "resp_final",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }


def make_vm(commands, *, fail_on: str | None = None):
    def executor(cmd: str, container_name: str, decode: bool = True):
        commands.append(cmd)
        if fail_on and fail_on in cmd:
            raise RuntimeError("boom")
        if "import -window" in cmd:
            return b"png-bytes" if not decode else "png-bytes"
        return "" if decode else b""

    return VM(display=":99", container_name="cua-image", executor=executor)


def test_first_request_uses_computer_tool(tmp_path):
    commands = []
    client = FakeClient([final_response()])
    _, recorder = run_computer_use_task(
        client=client,
        prompt="Inspect the desktop.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    assert client.responses.requests[0]["tools"] == [{"type": "computer"}]
    assert client.responses.requests[0]["input"] == "Inspect the desktop."
    trajectory = read_json(recorder.trajectory_path)
    assert trajectory["status"] == "completed"
    assert trajectory["screenshots"][0]["path"] == "screenshots/000-initial.png"


def test_screenshot_first_turn_returns_screenshot_output(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_1",
                        "actions": [{"type": "screenshot"}],
                    },
                ],
            },
            final_response("Complete."),
        ],
    )

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Look around.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    assert len(client.responses.requests) == 2
    second_input = client.responses.requests[1]["input"]
    assert second_input == [
        {
            "type": "computer_call_output",
            "call_id": "call_1",
            "output": {
                "type": "computer_screenshot",
                "image_url": "data:image/png;base64,cG5nLWJ5dGVz",
                "detail": "original",
            },
        },
    ]
    assert Path(recorder.run_dir, "screenshots/001-after-call-call_1.png").exists()
    actions = read_jsonl(recorder.actions_path)
    assert actions[0]["action"]["type"] == "screenshot"
    assert actions[0]["status"] == "completed"


def test_batched_actions_execute_in_order_and_are_logged(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_batch",
                        "actions": [
                            {"type": "click", "x": 10, "y": 20, "button": "left"},
                            {"type": "type", "text": "hello"},
                        ],
                    },
                ],
            },
            final_response(),
        ],
    )

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Click and type.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool mousemove 10 20 click 1",
        "DISPLAY=:99 xdotool type --delay 0 hello",
    ]
    actions = read_jsonl(recorder.actions_path)
    assert [item["action"]["type"] for item in actions] == ["click", "type"]
    assert [item["status"] for item in actions] == ["completed", "completed"]


def test_window_targeted_click_focuses_and_uses_window_relative_coordinates(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_window",
                        "actions": [{"type": "click", "x": 10, "y": 20}],
                    },
                ],
            },
            final_response(),
        ],
    )

    run_computer_use_task(
        client=client,
        prompt="Click in the assigned window.",
        vm=make_vm(commands),
        output_root=tmp_path,
        window_id="0x123",
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool windowactivate --sync 0x123 windowraise 0x123",
        "DISPLAY=:99 xdotool mousemove --window 0x123 10 20 click 1",
    ]


def test_window_targeted_screenshot_uses_window_id(tmp_path):
    commands = []
    client = FakeClient([final_response()])

    run_computer_use_task(
        client=client,
        prompt="Inspect the assigned window.",
        vm=make_vm(commands),
        output_root=tmp_path,
        window_id="0x123",
    )

    import_commands = [cmd for cmd in commands if "import -window" in cmd]
    assert import_commands == [
        "export DISPLAY=:99 && import -window 0x123 png:-",
    ]


def test_input_arbiter_serializes_concurrent_actions_fifo():
    commands = []
    vm = make_vm(commands)

    async def run() -> None:
        async with InputArbiter(vm) as arbiter:
            first = asyncio.create_task(
                arbiter.submit_action({"type": "click", "x": 1, "y": 2}),
            )
            second = asyncio.create_task(
                arbiter.submit_action({"type": "type", "text": "hi"}),
            )
            await asyncio.gather(first, second)

    asyncio.run(run())

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool mousemove 1 2 click 1",
        "DISPLAY=:99 xdotool type --delay 0 hi",
    ]


def test_actions_jsonl_inserts_blank_lines_between_turns(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_1",
                        "actions": [{"type": "click", "x": 1, "y": 2}],
                    },
                ],
            },
            {
                "id": "resp_2",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_2",
                        "actions": [{"type": "click", "x": 3, "y": 4}],
                    },
                ],
            },
            final_response(),
        ],
    )

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Click twice across turns.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    raw_lines = recorder.actions_path.read_text(encoding="utf-8").splitlines()
    assert raw_lines[1] == ""
    assert json.loads(raw_lines[0])["turn"] == 1
    assert json.loads(raw_lines[2])["turn"] == 2


def test_modifier_assisted_action_presses_and_releases_keys(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_shift",
                        "actions": [
                            {
                                "type": "click",
                                "x": 30,
                                "y": 40,
                                "button": "left",
                                "keys": ["SHIFT"],
                            },
                        ],
                    },
                ],
            },
            final_response(),
        ],
    )

    run_computer_use_task(
        client=client,
        prompt="Shift-click.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool keydown shift",
        "DISPLAY=:99 xdotool mousemove 30 40 click 1",
        "DISPLAY=:99 xdotool keyup shift",
    ]


def test_keypress_chords_normalize_symbol_key_names(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_minus",
                        "actions": [
                            {"type": "keypress", "keys": ["CTRL", "MINUS"]},
                            {"type": "keypress", "keys": ["CTRL", "SHIFT", "MINUS"]},
                        ],
                    },
                ],
            },
            final_response(),
        ],
    )

    run_computer_use_task(
        client=client,
        prompt="Zoom out.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool key ctrl+minus",
        "DISPLAY=:99 xdotool key ctrl+shift+minus",
    ]


def test_loop_stops_without_followup_when_no_computer_call(tmp_path):
    commands = []
    client = FakeClient([final_response("All set.")])

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Finish without acting.",
        vm=make_vm(commands),
        output_root=tmp_path,
    )

    assert len(client.responses.requests) == 1
    trajectory = read_json(recorder.trajectory_path)
    assert trajectory["final_response"] == "All set."


def test_safety_checks_stop_without_acknowledgement(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_safety",
                        "pending_safety_checks": [
                            {"code": "requires_ack", "message": "Confirm first."},
                        ],
                        "actions": [{"type": "screenshot"}],
                    },
                ],
            },
        ],
    )

    with pytest.raises(SafetyCheckRequired):
        run_computer_use_task(
            client=client,
            prompt="Do something sensitive.",
            vm=make_vm(commands),
            output_root=tmp_path,
        )

    run_dirs = list(tmp_path.iterdir())
    trajectory = read_json(run_dirs[0] / "trajectory.json")
    assert trajectory["status"] == "blocked"
    assert "Pending safety checks" in trajectory["error"]
    assert len(client.responses.requests) == 1
    assert [cmd for cmd in commands if "xdotool" in cmd] == []


def test_action_failure_is_persisted(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_fail",
                        "actions": [{"type": "click", "x": 5, "y": 6}],
                    },
                ],
            },
        ],
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_computer_use_task(
            client=client,
            prompt="Click.",
            vm=make_vm(commands, fail_on="xdotool mousemove 5 6 click 1"),
            output_root=tmp_path,
        )

    run_dirs = list(tmp_path.iterdir())
    actions = read_jsonl(run_dirs[0] / "actions.jsonl")
    trajectory = read_json(run_dirs[0] / "trajectory.json")
    assert actions[0]["status"] == "failed"
    assert actions[0]["error"] == "boom"
    assert trajectory["status"] == "failed"
    assert (run_dirs[0] / "screenshots/001-failure-after-call-call_fail.png").exists()
