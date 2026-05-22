"""Tests for the Claude Agent SDK Openlayer integration."""

# Many of the fake_query implementations below take **kwargs / unused prompt /
# options arguments to match the real SDK's call signature; ruff's ARG001
# fires for each of those, so we silence the rule file-wide.
# ruff: noqa: ARG001

import asyncio
from typing import Any, List
from unittest.mock import patch

import pytest

from openlayer.lib.tracing import tracer as ol_tracer

from .claude_agent_sdk_mocks import (
    FakeTextBlock,
    FakeUserMessage,
    FakeToolUseBlock,
    FakeResultMessage,
    FakeThinkingBlock,
    FakeToolResultBlock,
    FakeAssistantMessage,
    make_stream,
    init_system_message,
)

# ---------- shared infra ----------

@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable real network publishing for every test in this module."""
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")
    monkeypatch.setattr(ol_tracer, "_publish", False, raising=False)


def _capture_trace_publish():
    """Return ``(captured, capture_fn)`` pair for patching the publish path.

    NOTE on plan deviation: the plan references a function named
    ``_publish_trace_async`` on the tracer, but the actual public-private
    function is ``_upload_and_publish_trace`` (see
    ``src/openlayer/lib/tracing/tracer.py``). We patch that one and capture
    the in-memory ``Trace`` object passed to it.
    """
    captured: List[Any] = []

    def capture(trace, *args, **kwargs):
        captured.append(trace)

    return captured, capture


# ---------- tests ----------

def test_imports_work():
    """Module imports cleanly even if claude_agent_sdk is not installed."""
    from openlayer.lib.integrations import claude_agent_sdk  # noqa: F401


def test_traced_query_emits_root_agent_step_with_cost_and_tokens():
    """Basic flow: init -> assistant text -> result.

    The root AGENT step gets cost/tokens/session_id populated from the final
    ``ResultMessage``.
    """
    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    messages = [
        init_system_message(session_id="s1"),
        FakeAssistantMessage(content=[FakeTextBlock("Hello back")]),
        FakeResultMessage(
            subtype="success",
            duration_ms=1500,
            num_turns=1,
            session_id="s1",
            total_cost_usd=0.0042,
            result="Hello back",
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
    ]

    async def fake_query(*, prompt, options=None, **kwargs):
        async for m in make_stream(messages):
            yield m

    captured, capture_fn = _capture_trace_publish()

    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi"):
                pass

        asyncio.run(run())

    assert len(captured) == 1, "expected exactly one trace to be published"
    trace_obj = captured[0]
    root_step = trace_obj.steps[0]
    assert root_step.step_type.value == "agent"
    assert root_step.output == "Hello back"
    # Cost / tokens / latency are recorded on the step's metadata
    # (and also surface at trace-level via `post_process_trace`).
    assert root_step.metadata["cost"] == 0.0042
    assert root_step.metadata["tokens"] == 15
    assert root_step.metadata["prompt_tokens"] == 10
    assert root_step.metadata["completion_tokens"] == 5
    assert root_step.metadata["session_id"] == "s1"
    assert root_step.metadata["num_turns"] == 1
    assert root_step.metadata["stop_reason"] == "end_turn"
    assert root_step.latency == 1500


def test_options_metadata_captured_on_root_step():
    """``options.system_prompt`` and ``options.agents`` land on root metadata."""
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    user_options = ClaudeAgentOptions(
        system_prompt="You are a banana expert.",
        model="claude-haiku-4-5",
        max_turns=3,
        allowed_tools=["Read", "Bash"],
        agents={
            "code-reviewer": AgentDefinition(
                description="Reviews code for bugs",
                prompt="You are a strict reviewer. Flag anti-patterns.",
                tools=["Read", "Grep"],
            ),
        },
    )

    messages = [init_system_message(), FakeResultMessage(subtype="success")]

    async def fake_query(*, prompt, options=None, **kwargs):
        async for m in make_stream(messages):
            yield m

    captured, capture_fn = _capture_trace_publish()

    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi", options=user_options):
                pass

        asyncio.run(run())

    root = captured[0].steps[0]
    assert root.metadata["system_prompt"] == "You are a banana expert."
    agents_meta = root.metadata["agents_defined"]
    assert "code-reviewer" in agents_meta
    assert agents_meta["code-reviewer"]["description"] == "Reviews code for bugs"
    assert (
        agents_meta["code-reviewer"]["prompt"]
        == "You are a strict reviewer. Flag anti-patterns."
    )
    assert agents_meta["code-reviewer"]["tools"] == ["Read", "Grep"]
    opts = root.metadata["options"]
    assert opts["model"] == "claude-haiku-4-5"
    assert opts["max_turns"] == 3
    assert opts["allowed_tools"] == ["Read", "Bash"]


def test_assistant_message_emits_chat_completion_step():
    """Each AssistantMessage becomes a CHAT_COMPLETION child of the root step."""
    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    messages = [
        init_system_message(),
        FakeAssistantMessage(
            content=[
                FakeThinkingBlock("planning..."),
                FakeTextBlock("answer turn 1"),
            ],
            model="claude-opus-4-7",
            usage={
                "input_tokens": 12,
                "output_tokens": 4,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            stop_reason="end_turn",
        ),
        FakeResultMessage(subtype="success"),
    ]

    async def fake_query(*, prompt, options=None, **kwargs):
        async for m in make_stream(messages):
            yield m

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi"):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    nested = root_step.steps
    assert len(nested) == 1
    chat = nested[0]
    assert chat.step_type.value == "chat_completion"
    assert chat.model == "claude-opus-4-7"
    assert chat.provider == "anthropic"
    assert "answer turn 1" in chat.output
    assert chat.metadata["thinking"] == "planning..."
    assert chat.prompt_tokens == 12
    assert chat.completion_tokens == 4
    assert chat.tokens == 16
    assert chat.metadata["stop_reason"] == "end_turn"


def _extract_hook_callbacks(options, event: str):
    """Pull our callback functions out of options.hooks[event]."""
    matchers = (getattr(options, "hooks", None) or {}).get(event, []) or []
    callbacks = []
    for m in matchers:
        for cb in getattr(m, "hooks", []) or []:
            callbacks.append(cb)
    return callbacks


def test_tool_call_creates_tool_step_with_input_and_output():
    """A tool call yields a TOOL step with input/output/latency/tool_use_id."""
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    async def fake_query(*, prompt, options=None, **kwargs):
        # The wrapper injects our hooks into options.hooks. Pull them out and
        # invoke at the right moments to simulate the real SDK calling them.
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        post_hooks = _extract_hook_callbacks(options, "PostToolUse")

        yield init_system_message()
        yield FakeAssistantMessage(
            content=[
                FakeTextBlock(""),
                FakeToolUseBlock(id="t1", name="Bash", input={"command": "echo hi"}),
            ]
        )
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo hi"},
                },
                "t1",
                {"signal": None},
            )
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo hi"},
                    "tool_response": "hi",
                },
                "t1",
                {"signal": None},
            )
        yield FakeUserMessage(content=[FakeToolResultBlock(tool_use_id="t1", content="hi")])
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(
                prompt="run echo hi", options=cas.ClaudeAgentOptions()
            ):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    tool_steps = [s for s in root_step.steps if s.step_type.value == "tool"]
    assert len(tool_steps) == 1
    t = tool_steps[0]
    assert t.name == "Bash"
    assert t.inputs == {"command": "echo hi"}
    assert t.output == "hi"
    assert t.metadata["tool_use_id"] == "t1"
    assert t.latency is not None and t.latency >= 0
    assert t.metadata.get("is_error") is False


def test_mcp_tool_name_is_parsed_into_metadata():
    """A tool named ``mcp__playwright__browser_click`` records the parsed metadata."""
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    async def fake_query(*, prompt, options=None, **kwargs):
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        post_hooks = _extract_hook_callbacks(options, "PostToolUse")

        yield init_system_message()
        yield FakeAssistantMessage(
            content=[
                FakeToolUseBlock(
                    id="t1",
                    name="mcp__playwright__browser_click",
                    input={"selector": "#submit"},
                )
            ]
        )
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "mcp__playwright__browser_click",
                    "tool_input": {"selector": "#submit"},
                },
                "t1",
                {"signal": None},
            )
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "mcp__playwright__browser_click",
                    "tool_input": {"selector": "#submit"},
                    "tool_response": "clicked",
                },
                "t1",
                {"signal": None},
            )
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="click", options=cas.ClaudeAgentOptions()):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    tool_steps = [s for s in root_step.steps if s.step_type.value == "tool"]
    assert len(tool_steps) == 1
    t = tool_steps[0]
    assert t.name == "mcp__playwright__browser_click"
    assert t.metadata["mcp_server"] == "playwright"
    assert t.metadata["mcp_tool_name"] == "browser_click"


def test_subagent_messages_nest_under_agent_tool_step():
    """A message with ``parent_tool_use_id`` nests under the spawning Agent ToolStep."""
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    async def fake_query(*, prompt, options=None, **kwargs):
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        post_hooks = _extract_hook_callbacks(options, "PostToolUse")

        yield init_system_message()
        # Parent assistant turn dispatches the Agent tool
        yield FakeAssistantMessage(
            content=[
                FakeToolUseBlock(
                    id="t1",
                    name="Agent",
                    input={
                        "subagent_type": "code-reviewer",
                        "description": "Review the src/ directory",
                        "prompt": "review src/",
                    },
                )
            ]
        )
        # PreToolUse opens the Agent step (AGENT type for subagent dispatches)
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Agent",
                    "tool_input": {
                        "subagent_type": "code-reviewer",
                        "description": "Review the src/ directory",
                        "prompt": "review src/",
                    },
                },
                "t1",
                {"signal": None},
            )
        # Subagent streams its own assistant message *while the Agent step is open*
        yield FakeAssistantMessage(
            content=[FakeTextBlock("subagent turn 1")],
            parent_tool_use_id="t1",
        )
        # Subagent ends; PostToolUse closes the Agent step
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Agent",
                    "tool_input": {"description": "code-reviewer"},
                    "tool_response": "review complete",
                },
                "t1",
                {"signal": None},
            )
        yield FakeUserMessage(
            content=[FakeToolResultBlock(tool_use_id="t1", content="review complete")]
        )
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(
                prompt="dispatch subagent", options=cas.ClaudeAgentOptions()
            ):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    # The Agent tool dispatch is represented as a nested AGENT step (not TOOL)
    # so the subagent's chat/tool steps live inside an agent boundary.
    agent_steps = [s for s in root_step.steps if s.step_type.value == "agent"]
    assert len(agent_steps) == 1, "expected one nested AGENT step for subagent dispatch"
    agent_step = agent_steps[0]
    assert agent_step.name == "Agent: code-reviewer"
    assert agent_step.metadata.get("subagent_type") == "code-reviewer"
    # No sibling TOOL step for the same tool_use_id — Agent dispatches must
    # not double-emit as both AGENT and TOOL.
    tool_steps_for_t1 = [
        s for s in root_step.steps
        if s.step_type.value == "tool" and s.metadata.get("tool_use_id") == "t1"
    ]
    assert tool_steps_for_t1 == [], "Agent dispatch must not also emit a TOOL step"
    # Subagent's assistant turn nests beneath the AGENT step
    subagent_chats = [
        s for s in agent_step.steps if s.step_type.value == "chat_completion"
    ]
    assert len(subagent_chats) == 1, "expected subagent chat completion nested under AGENT step"
    assert "subagent turn 1" in subagent_chats[0].output
    assert subagent_chats[0].metadata.get("parent_tool_use_id") == "t1"


def test_subagent_internal_tool_calls_nest_under_agent_step():
    """A TOOL call made by a subagent must nest inside the spawning AGENT step,
    not escape to the root.

    This is the bug a real audit-pipeline run exposed: the subagent's assistant
    message says ``[tool call: Grep]`` but the actual Grep TOOL step was being
    created as a child of root instead of the ``Agent: ...`` AGENT step.
    """
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    async def fake_query(*, prompt, options=None, **kwargs):
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        post_hooks = _extract_hook_callbacks(options, "PostToolUse")

        yield init_system_message()
        # Dispatcher's turn — emits one Agent tool call (t1) for security-auditor.
        yield FakeAssistantMessage(
            content=[
                FakeToolUseBlock(
                    id="t1",
                    name="Agent",
                    input={
                        "subagent_type": "security-auditor",
                        "description": "audit file",
                        "prompt": "audit src/x.py",
                    },
                )
            ]
        )
        # PreToolUse(Agent, t1) opens the AGENT step.
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Agent",
                    "tool_input": {"subagent_type": "security-auditor"},
                },
                "t1",
                {"signal": None},
            )
        # Subagent's assistant message: declares it will call Grep with tool
        # use id ``g1``. parent_tool_use_id=t1 marks this as a subagent message.
        yield FakeAssistantMessage(
            content=[
                FakeTextBlock("checking for unsafe eval"),
                FakeToolUseBlock(id="g1", name="Grep", input={"pattern": "eval\\("}),
            ],
            parent_tool_use_id="t1",
        )
        # The SDK fires PreToolUse for the subagent's Grep tool. Critically,
        # the subagent's tool_use_id (g1) is unrelated to the dispatcher's t1 —
        # the wrapper must figure out that g1 belongs under the t1 AGENT step
        # by remembering that g1 was declared inside a subagent message.
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Grep",
                    "tool_input": {"pattern": "eval\\("},
                },
                "g1",
                {"signal": None},
            )
        # Subagent receives the tool result.
        yield FakeUserMessage(
            content=[FakeToolResultBlock(tool_use_id="g1", content="no matches")],
            parent_tool_use_id="t1",
        )
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Grep",
                    "tool_input": {"pattern": "eval\\("},
                    "tool_response": "no matches",
                },
                "g1",
                {"signal": None},
            )
        # Subagent wraps up.
        yield FakeAssistantMessage(
            content=[FakeTextBlock("done")], parent_tool_use_id="t1"
        )
        # PostToolUse(Agent, t1) closes the AGENT step.
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Agent",
                    "tool_input": {"subagent_type": "security-auditor"},
                    "tool_response": "done",
                },
                "t1",
                {"signal": None},
            )
        yield FakeUserMessage(
            content=[FakeToolResultBlock(tool_use_id="t1", content="done")]
        )
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="audit", options=cas.ClaudeAgentOptions()):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    # Find the AGENT step for the security-auditor subagent.
    agent_steps = [s for s in root_step.steps if s.step_type.value == "agent"]
    assert len(agent_steps) == 1
    agent_step = agent_steps[0]
    assert agent_step.metadata.get("subagent_type") == "security-auditor"

    # The Grep TOOL step must be a CHILD of the AGENT step, not a child of root.
    nested_tools = [s for s in agent_step.steps if s.step_type.value == "tool"]
    assert (
        len(nested_tools) == 1
    ), "subagent's Grep tool must nest under the AGENT step"
    assert nested_tools[0].name == "Grep"
    assert nested_tools[0].metadata.get("tool_use_id") == "g1"

    # And it must NOT also appear at the root level.
    root_tools = [s for s in root_step.steps if s.step_type.value == "tool"]
    assert root_tools == [], (
        "subagent tools must not escape to root; "
        f"found {[(s.name, s.metadata.get('tool_use_id')) for s in root_tools]}"
    )


def test_result_message_error_subtype_marks_root_step():
    """An error ResultMessage subtype is reflected on the root step's metadata."""
    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    messages = [
        init_system_message(),
        FakeResultMessage(
            subtype="error_max_turns",
            is_error=True,
            result=None,
            stop_reason=None,
            num_turns=10,
        ),
    ]

    async def fake_query(*, prompt, options=None, **kwargs):
        async for m in make_stream(messages):
            yield m

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi"):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    assert root_step.metadata["subtype"] == "error_max_turns"
    assert root_step.metadata["is_error"] is True
    assert root_step.metadata["num_turns"] == 10


def test_post_tool_use_failure_marks_tool_step_as_error():
    """PostToolUseFailure fires instead of PostToolUse — the tool step is marked errored."""
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    async def fake_query(*, prompt, options=None, **kwargs):
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        fail_hooks = _extract_hook_callbacks(options, "PostToolUseFailure")

        yield init_system_message()
        yield FakeAssistantMessage(
            content=[FakeToolUseBlock(id="t1", name="Bash", input={"command": "false"})]
        )
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "false"},
                },
                "t1",
                {"signal": None},
            )
        for h in fail_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUseFailure",
                    "tool_name": "Bash",
                    "tool_input": {"command": "false"},
                    "error": "exit code 1",
                },
                "t1",
                {"signal": None},
            )
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(
                prompt="fail bash", options=cas.ClaudeAgentOptions()
            ):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    tool_steps = [s for s in root_step.steps if s.step_type.value == "tool"]
    assert len(tool_steps) == 1
    t = tool_steps[0]
    assert t.metadata["is_error"] is True
    assert "exit code 1" in (t.output or "")


def test_user_hooks_compose_with_openlayer_hooks():
    """User-provided hooks run alongside ours; neither replaces the other."""
    import claude_agent_sdk as cas

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    user_called: list = []

    async def user_hook(input_data, tool_use_id, context):
        user_called.append(tool_use_id)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "test deny",
            }
        }

    user_options = cas.ClaudeAgentOptions(
        hooks={"PreToolUse": [cas.HookMatcher(hooks=[user_hook])]}
    )

    async def fake_query(*, prompt, options=None, **kwargs):
        # The wrapper should have appended our hook AFTER the user's.
        pre_hooks = _extract_hook_callbacks(options, "PreToolUse")
        post_hooks = _extract_hook_callbacks(options, "PostToolUse")
        assert pre_hooks[0] is user_hook, "user hook must run first"
        assert len(pre_hooks) >= 2, "openlayer hook must be appended after user hook"

        yield init_system_message()
        yield FakeAssistantMessage(
            content=[FakeToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]
        )
        # Simulate the SDK invoking *all* PreToolUse hooks (user first, then ours)
        for h in pre_hooks:
            await h(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "ls"},
                },
                "t1",
                {"signal": None},
            )
        for h in post_hooks:
            await h(
                {
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Bash",
                    "tool_input": {"command": "ls"},
                    "tool_response": "denied",
                },
                "t1",
                {"signal": None},
            )
        yield FakeResultMessage(subtype="success")

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi", options=user_options):
                pass

        asyncio.run(run())

    # User hook was invoked
    assert user_called == ["t1"]
    # Our hook ran too — the tool step exists in the trace
    root_step = captured[0].steps[0]
    tool_steps = [s for s in root_step.steps if s.step_type.value == "tool"]
    assert len(tool_steps) == 1


def test_mcp_env_is_stripped_from_agent_config_metadata():
    """``env`` and ``headers`` of MCP server configs must be redacted."""
    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    messages = [
        init_system_message(
            mcp_servers=[
                {
                    "name": "secret-server",
                    "command": "x",
                    "env": {"API_KEY": "supersecret"},
                    "headers": {"Authorization": "Bearer xyz"},
                }
            ]
        ),
        FakeResultMessage(subtype="success"),
    ]

    async def fake_query(*, prompt, options=None, **kwargs):
        async for m in make_stream(messages):
            yield m

    captured, capture_fn = _capture_trace_publish()
    with patch("claude_agent_sdk.query", fake_query), patch.object(
        ol_tracer, "_publish", True
    ), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture_fn
    ):
        async def run():
            async for _ in traced_query(prompt="hi"):
                pass

        asyncio.run(run())

    root_step = captured[0].steps[0]
    mcp = root_step.metadata["agent_config"]["mcp_servers"]
    assert isinstance(mcp, list)
    server = mcp[0]
    assert "env" not in server, "env must be stripped"
    assert "headers" not in server, "headers must be stripped"
    # Non-secret keys are preserved
    assert server.get("name") == "secret-server"
    # The literal secret must not appear anywhere in the serialized metadata
    serialized = repr(root_step.metadata)
    assert "supersecret" not in serialized
    assert "Bearer xyz" not in serialized


def test_trace_claude_agent_sdk_patches_module_query():
    """``trace_claude_agent_sdk()`` monkey-patches ``claude_agent_sdk.query``."""
    import claude_agent_sdk

    from openlayer.lib.integrations.claude_agent_sdk import trace_claude_agent_sdk

    original = claude_agent_sdk.query
    try:
        trace_claude_agent_sdk()
        assert claude_agent_sdk.query is not original
        assert getattr(claude_agent_sdk.query, "_openlayer_patched", False) is True

        # Idempotent: a second call doesn't double-wrap.
        after_first = claude_agent_sdk.query
        trace_claude_agent_sdk()
        assert claude_agent_sdk.query is after_first
        assert getattr(claude_agent_sdk.query, "_openlayer_patched", False) is True
    finally:
        claude_agent_sdk.query = original


def test_trace_claude_agent_sdk_config_persists():
    """Init kwargs are persisted into the module-level config."""
    import claude_agent_sdk

    from openlayer.lib.integrations import claude_agent_sdk as integration
    from openlayer.lib.integrations.claude_agent_sdk import trace_claude_agent_sdk

    original = claude_agent_sdk.query
    try:
        trace_claude_agent_sdk(
            inference_pipeline_id="pipe-123",
            truncate_tool_output_chars=512,
            capture_thinking=False,
        )
        assert integration._config.inference_pipeline_id == "pipe-123"
        assert integration._config.truncate_tool_output_chars == 512
        assert integration._config.capture_thinking is False
    finally:
        # Reset config for downstream tests
        trace_claude_agent_sdk(
            inference_pipeline_id=None,
            truncate_tool_output_chars=8192,
            capture_thinking=True,
            redact_mcp_env=True,
        )
        claude_agent_sdk.query = original


def test_trace_claude_agent_sdk_patches_claude_sdk_client():
    """``trace_claude_agent_sdk()`` also patches ``ClaudeSDKClient.query`` / ``.receive_response``."""
    import claude_agent_sdk

    from openlayer.lib.integrations.claude_agent_sdk import trace_claude_agent_sdk

    Client = claude_agent_sdk.ClaudeSDKClient
    original_module_query = claude_agent_sdk.query
    original_query = Client.query
    original_receive = Client.receive_response
    try:
        trace_claude_agent_sdk()
        assert Client.query is not original_query
        assert Client.receive_response is not original_receive
        assert getattr(Client, "_openlayer_patched", False) is True

        # Idempotent
        after_first_query = Client.query
        trace_claude_agent_sdk()
        assert Client.query is after_first_query
    finally:
        Client.query = original_query
        Client.receive_response = original_receive
        try:
            del Client._openlayer_patched
        except AttributeError:
            pass
        claude_agent_sdk.query = original_module_query


def test_wrapped_stream_yields_identical_messages_in_identical_order():
    """The wrapper is a pure observer — output must equal the underlying stream."""
    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    original_messages = [
        init_system_message(),
        FakeAssistantMessage(content=[FakeTextBlock("x")]),
        FakeResultMessage(subtype="success"),
    ]

    async def fake_query(*, prompt, options=None, **kwargs):
        for m in original_messages:
            yield m

    with patch("claude_agent_sdk.query", fake_query):
        async def run():
            out = []
            async for m in traced_query(prompt="x"):
                out.append(m)
            return out

        result = asyncio.run(run())

    # Same length, same identities (the wrapper must not substitute), same order
    assert len(result) == len(original_messages)
    for got, expected in zip(result, original_messages):
        assert got is expected

