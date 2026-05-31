from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import cua_loop
from cua_loop import (
    ComputerUseSubagentPool,
    InputArbiter,
    PlannerAssignment,
    PlannerOutputError,
    PlannerPlan,
    PlannerWindowRequest,
    PooledSubagent,
    SafetyCheckRequired,
    StageResult,
    StaleWindowError,
    TaskProgress,
    VM,
    WindowAssignment,
    _parse_wmctrl_lpxg,
    append_stage_result,
    build_planner_prompt,
    format_task_progress_ledger,
    latest_screenshots_by_window,
    latest_screenshots_for_progress,
    list_windows,
    normalize_window_id,
    open_planned_windows,
    run_computer_use_task,
    run_orchestrated_computer_use_task_async,
    validate_planner_plan,
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


def planner_response(payload):
    return final_response(json.dumps(payload))


def make_vm(commands, *, fail_on: str | None = None, responses=None):
    responses = responses or {}

    def executor(cmd: str, container_name: str, decode: bool = True):
        commands.append(cmd)
        if fail_on and fail_on in cmd:
            raise RuntimeError("boom")
        for pattern, value in responses.items():
            if pattern in cmd:
                result = value(cmd) if callable(value) else value
                if isinstance(result, BaseException):
                    raise result
                return result
        if "import -window" in cmd:
            return b"png-bytes" if not decode else "png-bytes"
        return "" if decode else b""

    return VM(display=":99", container_name="cua-image", executor=executor)


WMCTRL_SAMPLE = "\n".join(
    [
        "0x03a00007  0 1234 10 40 900 700 Navigator.firefox host-a Example - Mozilla Firefox",
        "not-a-window",
        "0x03c00001  0 2345 30 50 640 480 xfce4-terminal.Xfce4-terminal host-a Terminal",
    ],
)


def registry_responses():
    return {
        "wmctrl -lpxG": WMCTRL_SAMPLE,
    }


def direct_assignment(
    window_id: str = "0x03a00007",
    logical_id: str = "firefox_1",
    pid: int | None = 1234,
) -> WindowAssignment:
    return WindowAssignment(logical_id=logical_id, window_id=window_id, pid=pid)


def test_task_progress_ledger_groups_stage_results_by_window():
    progress = TaskProgress(original_prompt="Prepare the report.")
    assert append_stage_result(
        progress,
        WindowAssignment(logical_id="firefox_1", window_id="0x03a00007"),
        StageResult(
            what_they_did="Opened the source page.",
            steps_taken="Focused Firefox and inspected the visible tabs.",
            failure_reason="",
        ),
    )
    assert append_stage_result(
        progress,
        WindowAssignment(logical_id="xfce4_terminal_1", window_id="0x03c00001"),
        StageResult(
            what_they_did="Tried to run the export command.",
            steps_taken="Focused the terminal and entered the command.",
            failure_reason="Command failed because credentials were missing.",
        ),
    )
    assert not append_stage_result(
        progress,
        None,
        StageResult(
            what_they_did="Ignored desktop-wide result.",
            steps_taken="No window was available.",
            failure_reason="",
        ),
    )

    assert progress.results_by_window == {
        "firefox_1": [
            StageResult(
                what_they_did="Opened the source page.",
                steps_taken="Focused Firefox and inspected the visible tabs.",
                failure_reason="",
            ),
        ],
        "xfce4_terminal_1": [
            StageResult(
                what_they_did="Tried to run the export command.",
                steps_taken="Focused the terminal and entered the command.",
                failure_reason="Command failed because credentials were missing.",
            ),
        ],
    }
    assert format_task_progress_ledger(progress) == "\n".join(
        [
            "## firefox_1",
            "### Iteration 1",
            "- What they did: Opened the source page.",
            "- Steps taken: Focused Firefox and inspected the visible tabs.",
            "- Failure reason: ",
            "## xfce4_terminal_1",
            "### Iteration 1",
            "- What they did: Tried to run the export command.",
            "- Steps taken: Focused the terminal and entered the command.",
            "- Failure reason: Command failed because credentials were missing.",
        ],
    )
    prompt = build_planner_prompt(progress)
    assert "You are the central planner" in prompt
    assert "Original prompt:\nPrepare the report." in prompt
    assert "Progress ledger:\n## firefox_1" in prompt


def test_latest_screenshot_helpers_ignore_windowless_artifacts():
    progress = TaskProgress(
        original_prompt="Work across windows.",
        results_by_window={
            "firefox_1": [StageResult("Read page", "Focused Firefox", "")],
            "terminal_1": [StageResult("Ran command", "Focused Terminal", "")],
        },
    )
    screenshots = [
        {"path": "screenshots/call_1/firefox_1/old.png", "window": "firefox_1"},
        {"path": "screenshots/call_2/firefox_1/new.png", "window": "firefox_1"},
        {"path": "screenshots/call_3/terminal_1/latest.png", "window": "terminal_1"},
        {"path": "screenshots/call_4/desktop/ignored.png"},
    ]

    latest = latest_screenshots_by_window(screenshots)
    assert latest == {
        "firefox_1": {"path": "screenshots/call_2/firefox_1/new.png", "window": "firefox_1"},
        "terminal_1": {"path": "screenshots/call_3/terminal_1/latest.png", "window": "terminal_1"},
    }
    assert latest_screenshots_for_progress(progress, screenshots) == [
        {"path": "screenshots/call_2/firefox_1/new.png", "window": "firefox_1"},
        {"path": "screenshots/call_3/terminal_1/latest.png", "window": "terminal_1"},
    ]


def test_subagent_pool_invokes_same_agent_with_different_windows():
    invocations = []
    first_assignment = direct_assignment("0x123", "paint_1")
    second_assignment = direct_assignment("0x456", "terminal_1")

    async def invoke(window_id, prompt):
        invocations.append((window_id, prompt))
        return StageResult(
            what_they_did=f"Handled {prompt}",
            steps_taken=f"Used {window_id.window_id}",
            failure_reason="",
        )

    pool = ComputerUseSubagentPool([PooledSubagent(name="worker-1", invoke=invoke)])

    async def run():
        first = await pool.invoke(first_assignment, "paint")
        second = await pool.invoke(second_assignment, "terminal")
        return first, second

    first, second = asyncio.run(run())

    assert invocations == [(first_assignment, "paint"), (second_assignment, "terminal")]
    assert first.what_they_did == "Handled paint"
    assert second.steps_taken == "Used 0x456"


def test_planner_plan_validation_accepts_known_windows_and_open_requests():
    windows = _parse_wmctrl_lpxg(WMCTRL_SAMPLE)
    plan = PlannerPlan(
        done=False,
        final_response="",
        windows_to_open=[
            PlannerWindowRequest(logical_id="browser_alias", kind="browser", reason="Need a page."),
        ],
        assignments=[
            PlannerAssignment(window="firefox_1", subtasks=["Inspect the current tab."]),
        ],
    )

    validate_planner_plan(plan, windows, max_assignments=1)


def test_planner_plan_model_rejects_invalid_kind_and_done_shape():
    with pytest.raises(ValueError, match="browser|terminal"):
        PlannerPlan(
            done=False,
            final_response="",
            windows_to_open=[
                {"logical_id": "tool_1", "kind": "spreadsheet", "reason": "Need sheets."},
            ],
            assignments=[],
        )

    with pytest.raises(ValueError, match="final_response"):
        PlannerPlan(
            done=True,
            final_response="",
            windows_to_open=[],
            assignments=[],
        )


@pytest.mark.parametrize(
    ("plan", "capacity", "message"),
    [
        (
            PlannerPlan(
                done=False,
                final_response="",
                windows_to_open=[],
                assignments=[PlannerAssignment(window="missing_1", subtasks=["Look."])],
            ),
            1,
            "unknown window",
        ),
        (
            PlannerPlan(
                done=False,
                final_response="",
                windows_to_open=[],
                assignments=[
                    PlannerAssignment(window="firefox_1", subtasks=["Look."]),
                    PlannerAssignment(window="firefox_1", subtasks=["Look again."]),
                ],
            ),
            2,
            "Duplicate assignment",
        ),
        (
            PlannerPlan(
                done=False,
                final_response="",
                windows_to_open=[],
                assignments=[
                    PlannerAssignment(window="firefox_1", subtasks=["Look."]),
                    PlannerAssignment(window="xfce4_terminal_1", subtasks=["List files."]),
                ],
            ),
            1,
            "only 1 subagents",
        ),
    ],
)
def test_planner_plan_validation_rejects_invalid_stage_shapes(plan, capacity, message):
    windows = _parse_wmctrl_lpxg(WMCTRL_SAMPLE)

    with pytest.raises(PlannerOutputError, match=message):
        validate_planner_plan(plan, windows, max_assignments=capacity)


def test_open_planned_windows_only_launches_allowlisted_command():
    commands = []
    vm = make_vm(commands, responses=registry_responses())

    open_planned_windows(
        vm=vm,
        requests=[
            PlannerWindowRequest(
                logical_id="terminal_alias",
                kind="terminal",
                reason="Need shell.",
            ),
        ],
    )

    assert any("DISPLAY=:99 xfce4-terminal" in command for command in commands)
    assert not any("wmctrl -lpxG" in command for command in commands)


def test_orchestration_runs_stage_in_parallel_and_refreshes_registry(tmp_path):
    commands = []
    subagents_done = False
    refreshed_registry = "\n".join(
        [
            WMCTRL_SAMPLE,
            "0x04100001  0 3456 80 90 500 400 editor.Editor host-a Notes",
        ],
    )

    def registry(_cmd):
        return refreshed_registry if subagents_done else WMCTRL_SAMPLE

    vm = make_vm(commands, responses={"wmctrl -lpxG": registry})
    client = FakeClient(
        [
            planner_response(
                {
                    "done": False,
                    "final_response": "",
                    "windows_to_open": [],
                    "assignments": [
                        {"window": "firefox_1", "subtasks": ["Read the page."]},
                        {"window": "xfce4_terminal_1", "subtasks": ["Check files."]},
                    ],
                },
            ),
            planner_response(
                {
                    "done": True,
                    "final_response": "All windows checked.",
                    "windows_to_open": [],
                    "assignments": [],
                },
            ),
        ],
    )
    started = []

    async def invoke(window_id, prompt):
        nonlocal subagents_done
        started.append((window_id.logical_id, prompt))
        if len(started) == 2:
            subagents_done = True
        while len(started) < 2:
            await asyncio.sleep(0)
        return StageResult(
            what_they_did=f"Handled {window_id.logical_id}",
            steps_taken=f"Prompt included {window_id.logical_id}: {window_id.logical_id in prompt}",
            failure_reason="",
        )

    pool = ComputerUseSubagentPool(
        [
            PooledSubagent("worker-1", invoke),
            PooledSubagent("worker-2", invoke),
        ],
    )

    async def run():
        return await run_orchestrated_computer_use_task_async(
            client=client,
            prompt="Inspect the browser and terminal.",
            vm=vm,
            model="test-model",
            output_root=tmp_path,
            pool=pool,
            planner_max_retries=0,
        )

    final_text, recorder = asyncio.run(run())

    assert final_text == "All windows checked."
    assert [item[0] for item in started] == ["firefox_1", "xfce4_terminal_1"]
    assert len(client.responses.requests) == 2
    assert client.responses.requests[0]["text"]["format"]["type"] == "json_schema"
    second_planner_text = client.responses.requests[1]["input"][0]["content"][0]["text"]
    assert "Handled firefox_1" in second_planner_text
    assert "Handled xfce4_terminal_1" in second_planner_text
    assert "editor_1" in second_planner_text
    assert any(
        part["type"] == "input_image"
        for part in client.responses.requests[1]["input"][0]["content"]
    )

    snapshots = read_jsonl(recorder.window_snapshots_path)
    assert [snapshot["label"] for snapshot in snapshots] == [
        "orchestration-start",
        "stage-001-after-subagents",
    ]
    trajectory = read_json(recorder.trajectory_path)
    assert trajectory["status"] == "completed"
    assert any(item["window"] == "editor_1" for item in trajectory["screenshots"])


def test_orchestration_refreshes_registry_after_window_open_request(tmp_path):
    commands = []
    opened_registry = "\n".join(
        [
            WMCTRL_SAMPLE,
            "0x03d00002  0 3456 60 70 800 600 xfce4-terminal.Xfce4-terminal host-a Terminal",
        ],
    )

    def registry(_cmd):
        if any("xfce4-terminal" in command for command in commands):
            return opened_registry
        return WMCTRL_SAMPLE

    vm = make_vm(commands, responses={"wmctrl -lpxG": registry})
    client = FakeClient(
        [
            planner_response(
                {
                    "done": False,
                    "final_response": "",
                    "windows_to_open": [
                        {"logical_id": "terminal_alias", "kind": "terminal", "reason": "Need shell."},
                    ],
                    "assignments": [],
                },
            ),
            planner_response(
                {
                    "done": True,
                    "final_response": "Terminal is available.",
                    "windows_to_open": [],
                    "assignments": [],
                },
            ),
        ],
    )
    pool = ComputerUseSubagentPool([PooledSubagent("worker-1", lambda _window, _prompt: pytest.fail())])

    async def run():
        return await run_orchestrated_computer_use_task_async(
            client=client,
            prompt="Open a terminal.",
            vm=vm,
            model="test-model",
            output_root=tmp_path,
            pool=pool,
            planner_max_retries=0,
        )

    final_text, recorder = asyncio.run(run())

    assert final_text == "Terminal is available."
    assert any("DISPLAY=:99 xfce4-terminal" in command for command in commands)
    second_planner_text = client.responses.requests[1]["input"][0]["content"][0]["text"]
    assert "xfce4_terminal_2" in second_planner_text
    snapshots = read_jsonl(recorder.window_snapshots_path)
    assert [snapshot["label"] for snapshot in snapshots] == [
        "orchestration-start",
        "stage-001-after-window-open-requests",
    ]


def test_orchestration_cli_uses_opt_in_path(monkeypatch, capsys):
    captured = {}
    vm = make_vm([], responses=registry_responses())
    monkeypatch.setattr(cua_loop, "VM", lambda display, container_name: vm)
    monkeypatch.setattr(cua_loop, "make_openai_client", lambda: object())

    class Recorder:
        run_dir = Path("run")

    def fake_run_orchestrated_computer_use_task(**kwargs):
        captured.update(kwargs)
        return "orchestrated result", Recorder()

    monkeypatch.setattr(
        cua_loop,
        "run_orchestrated_computer_use_task",
        fake_run_orchestrated_computer_use_task,
    )

    result = cua_loop.main(
        [
            "--prompt",
            "Inspect everything.",
            "--orchestrate",
            "--max-stages",
            "7",
        ],
    )

    assert result == 0
    assert captured["subagent_count"] == 8
    assert captured["max_stages"] == 7
    assert "orchestrated result" in capsys.readouterr().out


def test_normalize_window_id_accepts_hex_and_decimal():
    assert normalize_window_id("0x123") == 291
    assert normalize_window_id("291") == 291
    assert normalize_window_id(291) == 291


def test_wmctrl_parser_skips_malformed_lines_and_assigns_logical_ids():
    windows = _parse_wmctrl_lpxg(WMCTRL_SAMPLE)

    assert [window.window_id for window in windows] == ["0x03a00007", "0x03c00001"]
    assert [window.logical_id for window in windows] == ["firefox_1", "xfce4_terminal_1"]
    assert windows[0].geometry == {"x": 10, "y": 40, "width": 900, "height": 700}


def test_list_windows_returns_minimal_registry():
    commands = []
    windows = list_windows(make_vm(commands, responses=registry_responses()))

    assert [window.logical_id for window in windows] == ["firefox_1", "xfce4_terminal_1"]
    assert windows[0].title == "Example - Mozilla Firefox"
    assert windows[0].pid == 1234
    assert "DISPLAY=:99 wmctrl -lpxG" in commands


def test_list_windows_cli_prints_json_without_prompt(monkeypatch, capsys):
    commands = []
    vm = make_vm(commands, responses=registry_responses())
    monkeypatch.setattr(cua_loop, "VM", lambda display, container_name: vm)
    monkeypatch.setattr(
        cua_loop,
        "make_openai_client",
        lambda: pytest.fail("list-windows should not create an OpenAI client"),
    )

    assert cua_loop.main(["--list-windows"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["windows"][0]["logical_id"] == "firefox_1"
    assert payload["windows"][0]["window_id"] == "0x03a00007"


def test_cli_rejects_logical_id_as_window_id(monkeypatch, capsys):
    vm = make_vm([], responses=registry_responses())
    monkeypatch.setattr(cua_loop, "VM", lambda display, container_name: vm)
    monkeypatch.setattr(
        cua_loop,
        "make_openai_client",
        lambda: pytest.fail("invalid window id should not create an OpenAI client"),
    )

    result = cua_loop.main(
        [
            "--prompt",
            "Can you print hello world?",
            "--window-id",
            "xfce4_terminal_1",
        ],
    )

    assert result == 2
    assert "not a logical_id" in capsys.readouterr().err


def test_cli_rejects_missing_window_id(monkeypatch, capsys):
    vm = make_vm([], responses=registry_responses())
    monkeypatch.setattr(cua_loop, "VM", lambda display, container_name: vm)
    monkeypatch.setattr(
        cua_loop,
        "make_openai_client",
        lambda: pytest.fail("missing window id should not create an OpenAI client"),
    )

    result = cua_loop.main(["--prompt", "Click.", "--window-id", "0x999"])

    assert result == 2
    assert "Window id not found: 0x999" in capsys.readouterr().err


def test_cli_marshals_window_id_to_assignment(monkeypatch, capsys):
    captured = {}
    vm = make_vm([], responses=registry_responses())
    monkeypatch.setattr(cua_loop, "VM", lambda display, container_name: vm)
    monkeypatch.setattr(cua_loop, "make_openai_client", lambda: object())

    class Recorder:
        run_dir = Path("run")

    def fake_run_computer_use_task(**kwargs):
        captured.update(kwargs)
        return final_response(), Recorder()

    monkeypatch.setattr(cua_loop, "run_computer_use_task", fake_run_computer_use_task)

    result = cua_loop.main(
        [
            "--prompt",
            "Inspect Firefox.",
            "--window-id",
            "0x03a00007",
        ],
    )

    assert result == 0
    assert isinstance(captured["window_id"], WindowAssignment)
    assert captured["window_id"].logical_id == "firefox_1"
    assert captured["window_id"].window_id == "0x03a00007"
    assert captured["window_id"].pid == 1234
    assert "Done." in capsys.readouterr().out


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
    assert trajectory["screenshots"] == []
    assert [cmd for cmd in commands if "import -window" in cmd] == []


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
        vm=make_vm(commands, responses=registry_responses()),
        output_root=tmp_path,
        window_id=direct_assignment(),
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
    trajectory = read_json(recorder.trajectory_path)
    assert trajectory["screenshots"][0]["path"] == "screenshots/initial/firefox_1/000-initial.png"
    assert trajectory["screenshots"][0]["call_id"] == "initial"
    assert trajectory["screenshots"][0]["window"] == "firefox_1"
    assert Path(recorder.run_dir, "screenshots/call_1/firefox_1/001-after-call-call_1.png").exists()
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
    assert client.responses.requests[1]["input"] == []
    assert read_json(recorder.trajectory_path)["screenshots"] == []
    assert [cmd for cmd in commands if "import -window" in cmd] == []


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
        vm=make_vm(commands, responses=registry_responses()),
        output_root=tmp_path,
        window_id=direct_assignment(),
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool windowactivate --sync 0x03a00007 windowraise 0x03a00007",
        "DISPLAY=:99 xdotool mousemove --window 0x03a00007 10 20 click 1",
    ]


def test_window_targeted_screenshot_uses_window_id(tmp_path):
    commands = []
    client = FakeClient([final_response()])

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Inspect the assigned window.",
        vm=make_vm(commands, responses=registry_responses()),
        output_root=tmp_path,
        window_id=direct_assignment(),
    )

    import_commands = [cmd for cmd in commands if "import -window" in cmd]
    assert import_commands == [
        "export DISPLAY=:99 && import -window 0x03a00007 png:-",
    ]
    trajectory = read_json(recorder.trajectory_path)
    assert trajectory["screenshots"][0]["path"] == "screenshots/initial/firefox_1/000-initial.png"


def test_window_assignment_validates_stage_start_and_records_snapshot(tmp_path):
    commands = []
    client = FakeClient([final_response()])
    assignment = WindowAssignment(
        logical_id="firefox_1",
        window_id="0x03a00007",
        pid=1234,
    )

    _, recorder = run_computer_use_task(
        client=client,
        prompt="Inspect the assigned window.",
        vm=make_vm(commands, responses=registry_responses()),
        output_root=tmp_path,
        window_id=assignment,
    )

    snapshots = read_jsonl(recorder.window_snapshots_path)
    trajectory = read_json(recorder.trajectory_path)
    assert snapshots[0]["label"] == "stage-start"
    assert snapshots[0]["assignment"]["logical_id"] == "firefox_1"
    assert snapshots[0]["windows"][0]["logical_id"] == "firefox_1"
    assert trajectory["files"]["window_snapshots"] == "window_snapshots.jsonl"
    assert trajectory["screenshots"][0]["path"] == "screenshots/initial/firefox_1/000-initial.png"
    assert trajectory["screenshots"][0]["window"] == "firefox_1"
    assert any("wmctrl -lpxG" in cmd for cmd in commands)


def test_window_targeted_type_checks_active_window_before_typing(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_type",
                        "actions": [{"type": "type", "text": "hi"}],
                    },
                ],
            },
            final_response(),
        ],
    )
    assignment = WindowAssignment(logical_id="firefox_1", window_id="0x03a00007")
    responses = registry_responses() | {"xdotool getactivewindow": "0x03a00007\n"}

    run_computer_use_task(
        client=client,
        prompt="Type in the assigned window.",
        vm=make_vm(commands, responses=responses),
        output_root=tmp_path,
        window_id=assignment,
    )

    xdotool_commands = [cmd for cmd in commands if "xdotool" in cmd]
    assert xdotool_commands == [
        "DISPLAY=:99 xdotool windowactivate --sync 0x03a00007 windowraise 0x03a00007",
        "DISPLAY=:99 xdotool getactivewindow",
        "DISPLAY=:99 xdotool type --delay 0 hi",
    ]


def test_window_targeted_type_fails_before_typing_when_focus_check_mismatches(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_type",
                        "actions": [{"type": "type", "text": "hi"}],
                    },
                ],
            },
        ],
    )
    assignment = WindowAssignment(logical_id="firefox_1", window_id="0x03a00007")
    responses = registry_responses() | {"xdotool getactivewindow": "0x999\n"}

    with pytest.raises(StaleWindowError):
        run_computer_use_task(
            client=client,
            prompt="Type in the assigned window.",
            vm=make_vm(commands, responses=responses),
            output_root=tmp_path,
            window_id=assignment,
        )

    run_dirs = list(tmp_path.iterdir())
    actions = read_jsonl(run_dirs[0] / "actions.jsonl")
    assert actions[0]["status"] == "failed"
    assert not any("xdotool type" in cmd for cmd in commands)


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
    assert [cmd for cmd in commands if "import -window" in cmd] == []


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
    assert trajectory["screenshots"] == []
    assert [cmd for cmd in commands if "import -window" in cmd] == []


def test_stale_activation_failure_is_structured_and_not_retried(tmp_path):
    commands = []
    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_window",
                        "actions": [{"type": "click", "x": 5, "y": 6}],
                    },
                ],
            },
        ],
    )

    with pytest.raises(StaleWindowError):
        run_computer_use_task(
            client=client,
            prompt="Click.",
            vm=make_vm(
                commands,
                fail_on="xdotool windowactivate --sync 0x03a00007",
                responses=registry_responses(),
            ),
            output_root=tmp_path,
            window_id=direct_assignment(),
        )

    run_dirs = list(tmp_path.iterdir())
    actions = read_jsonl(run_dirs[0] / "actions.jsonl")
    assert actions[0]["status"] == "failed"
    assert "Stale window firefox_1 (0x03a00007)" in actions[0]["error"]
    assert not any("xdotool mousemove --window 0x03a00007 5 6" in cmd for cmd in commands)


def test_stale_screenshot_failure_marks_screenshot_action_failed(tmp_path):
    commands = []
    import_calls = 0

    def flaky_screenshot(_cmd):
        nonlocal import_calls
        import_calls += 1
        if import_calls == 1:
            return b"png-bytes"
        return RuntimeError("window vanished")

    client = FakeClient(
        [
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_screenshot",
                        "actions": [{"type": "screenshot"}],
                    },
                ],
            },
        ],
    )

    with pytest.raises(StaleWindowError):
        run_computer_use_task(
            client=client,
            prompt="Screenshot.",
            vm=make_vm(
                commands,
                responses=registry_responses() | {"import -window 0x03a00007": flaky_screenshot},
            ),
            output_root=tmp_path,
            window_id=direct_assignment(),
        )

    run_dirs = list(tmp_path.iterdir())
    actions = read_jsonl(run_dirs[0] / "actions.jsonl")
    assert actions[0]["status"] == "failed"
    assert "Stale window firefox_1 (0x03a00007)" in actions[0]["error"]
    assert import_calls == 2
