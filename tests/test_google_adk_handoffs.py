"""Trace shape of Google ADK multi-agent turns (OPEN-12864, OPEN-12518).

Drives a real ``Runner`` with a scripted ``BaseLlm``, so ADK's own flow runs
end to end with no network. The same assertions hold on google-adk 1.x and
2.x: 2.x runs each agent in its own asyncio task, which used to split a turn
into one trace per agent and drop tool/handoff steps.
"""

# google-adk and google-genai aren't installed in the lint env, and pytest's
# fixture decorator hides the autouse functions from static analysis.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedFunction=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportImplicitOverride=false, reportOptionalMemberAccess=false, reportUntypedBaseClass=false

import asyncio
import logging
import importlib
from typing import Any, Dict, List, Optional, AsyncGenerator
from unittest.mock import patch

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("wrapt")

from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse

from openlayer.lib.tracing import enums, tracer as _tracer
from openlayer.lib.integrations import google_adk_tracer

APP_NAME = "bank_app"
USER_ID = "u"
SESSION_ID = "s"

# Each fake model pops its next reply from the list under its ``script`` key: a
# Content, a full LlmResponse, an exception to raise, or a number of seconds to
# hang for (to be cancelled).
SCRIPTS: Dict[str, List[Any]] = {}


class ScriptedLlm(BaseLlm):
    model: str = "gemini-2.5-flash"
    script: str = ""

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:  # noqa: ARG002
        item = SCRIPTS[self.script].pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, (int, float)):
            await asyncio.sleep(item)
            item = _text("too late")
        if isinstance(item, LlmResponse):
            yield item
            return
        yield LlmResponse(
            content=item,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                thoughts_token_count=3,
                total_token_count=18,
            ),
        )


def _call(name: str, **args: Any) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def _text(value: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=value)])


def executar_transferencia_pix(valor: float) -> Dict[str, Any]:
    """Executa uma transferência Pix para outra conta."""
    return {"ok": True, "valor": valor}


@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")
    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


@pytest.fixture(autouse=True)
def _trace_adk():
    SCRIPTS.clear()
    from openlayer.lib.integrations import trace_google_adk

    trace_google_adk()
    yield
    google_adk_tracer._unpatch_google_adk()


@pytest.fixture
def traces():
    """Collect the trace of every completed root step."""
    captured: List[Any] = []
    original = _tracer._handle_trace_completion

    def _capture(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("is_root_step"):
            captured.append(_tracer.get_current_trace())
        return original(*args, **kwargs)

    with patch.object(_tracer, "_handle_trace_completion", _capture):
        yield captured


def _runner(agent: Any) -> Any:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    return Runner(agent=agent, app_name=APP_NAME, session_service=InMemorySessionService())


async def _new_session(runner: Any) -> None:
    await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)


def _turn(runner: Any, message: str) -> Any:
    """The Runner's event generator for one user message."""
    content = types.Content(role="user", parts=[types.Part(text=message)])
    return runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)


async def _until(condition: Any, timeout: float) -> None:
    """Yield to the event loop until ``condition()`` holds or ``timeout`` passes."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)


def _run_turns(agent: Any, *messages: str) -> None:
    runner = _runner(agent)

    async def _drive() -> None:
        await _new_session(runner)
        for message in messages:
            async for _event in _turn(runner, message):
                pass

    asyncio.run(_drive())


def _bank_agents() -> Any:
    from google.adk.agents import LlmAgent

    specialist = LlmAgent(
        name="pix_specialist",
        model=ScriptedLlm(script="pix_specialist"),
        description="Handles Pix transfers.",
        tools=[executar_transferencia_pix],
    )
    return LlmAgent(name="assistant", model=ScriptedLlm(script="assistant"), sub_agents=[specialist])


def _walk(step: Any) -> List[Any]:
    found = [step]
    for child in step.steps:
        found.extend(_walk(child))
    return found


def _all_steps(trace: Any) -> List[Any]:
    return [step for root in trace.steps for step in _walk(root)]


def _find(trace: Any, name: str) -> Optional[Any]:
    return next((step for step in _all_steps(trace) if step.name == name), None)


def _parent_of(trace: Any, target: Any) -> Optional[Any]:
    return next((step for step in _all_steps(trace) if target in step.steps), None)


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
class TestAdkHandoffs:
    def test_turn_is_one_trace_under_a_turn_root(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_call("executar_transferencia_pix", valor=10.0), _text("feito")]

        _run_turns(_bank_agents(), "quero fazer um pix")

        assert len(traces) == 1, "every agent of the turn must land in a single trace"
        (root,) = traces[0].steps
        assert root.name == f"Agent turn: {APP_NAME}"
        assert root.inputs["user_query"] == "quero fazer um pix"
        assert root.output == "feito"
        assert [step.name for step in root.steps] == ["Agent: assistant", "Agent: pix_specialist"]
        errors = [step.name for step in _all_steps(traces[0]) if "error" in (step.metadata or {})]
        assert not errors, f"a clean turn recorded errors on {errors}"

    def test_transfer_becomes_a_handoff_step(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("oi")]

        _run_turns(_bank_agents(), "quero fazer um pix")

        handoff = _find(traces[0], "Handoff: assistant → pix_specialist")
        assert handoff is not None
        assert handoff.step_type == enums.StepType.HANDOFF
        assert handoff.from_component == "assistant"
        assert handoff.to_component == "pix_specialist"
        assert _find(traces[0], "Tool: transfer_to_agent") is None
        assert _parent_of(traces[0], handoff).name == "Agent: assistant"

    def test_tool_with_transfer_in_its_description_is_a_tool(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_call("executar_transferencia_pix", valor=10.0), _text("feito")]

        _run_turns(_bank_agents(), "quero fazer um pix")

        tool = _find(traces[0], "Tool: executar_transferencia_pix")
        assert tool is not None, "a tool whose description mentions 'transfer' must still be traced"
        assert tool.output == {"ok": True, "valor": 10.0}
        # Tool steps are siblings of the LLM calls, not their children.
        assert _parent_of(traces[0], tool).name == "Agent: pix_specialist"

    def test_turn_summary_metadata(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito"), _text("de nada")]

        _run_turns(_bank_agents(), "quero fazer um pix", "obrigado")

        assert len(traces) == 2
        first, second = (trace.metadata for trace in traces)
        assert first == {
            "starting_agent": "assistant",
            "final_agent": "pix_specialist",
            "handoff_count": 1,
            "handoff_path": "assistant > pix_specialist",
        }
        # ADK keeps the specialist active, so the next turn starts there.
        assert second == {
            "starting_agent": "pix_specialist",
            "final_agent": "pix_specialist",
            "handoff_count": 0,
            "handoff_path": "pix_specialist",
        }
        (root,) = traces[0].steps
        assert root.metadata["handoff_path"] == "assistant > pix_specialist"

    def test_agent_tool_is_nested_not_a_handoff(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.tools.agent_tool import AgentTool

        calculator = LlmAgent(name="calculator", model=ScriptedLlm(script="calculator"), description="Does math.")
        boss = LlmAgent(name="boss", model=ScriptedLlm(script="boss"), tools=[AgentTool(agent=calculator)])
        SCRIPTS["boss"] = [_call("calculator", request="2+2"), _text("4!")]
        SCRIPTS["calculator"] = [_text("4")]

        _run_turns(boss, "quanto é 2+2?")

        assert len(traces) == 1, "AgentTool's nested Runner must not start its own turn"
        nested = _find(traces[0], "Agent: calculator")
        assert nested is not None
        assert _parent_of(traces[0], nested).name == "Tool: calculator"
        assert not [step for step in _all_steps(traces[0]) if step.step_type == enums.StepType.HANDOFF]
        assert traces[0].metadata["handoff_count"] == 0
        assert traces[0].metadata["final_agent"] == "boss"

    def test_workflow_agents_are_not_handoffs(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent, SequentialAgent

        first = LlmAgent(name="first", model=ScriptedLlm(script="first"))
        second = LlmAgent(name="second", model=ScriptedLlm(script="second"))
        SCRIPTS["first"] = [_text("one")]
        SCRIPTS["second"] = [_text("two")]

        _run_turns(SequentialAgent(name="pipeline", sub_agents=[first, second]), "go")

        assert len(traces) == 1
        assert traces[0].metadata["handoff_count"] == 0
        assert traces[0].metadata["final_agent"] == "second"

    def test_empty_model_reply_does_not_break_the_turn(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        agent = LlmAgent(name="quiet", model=ScriptedLlm(script="quiet"))
        SCRIPTS["quiet"] = [types.Content(role="model", parts=None)]

        _run_turns(agent, "hello?")

        assert len(traces) == 1
        assert traces[0].metadata["final_agent"] == "quiet"
        assert _find(traces[0], "Agent: quiet").output == "Agent execution completed"


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
class TestAdkHandoffSources:
    """Handoffs come from the transfers ADK actually made."""

    def test_transfer_set_by_a_custom_tool_is_a_handoff(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.tools.tool_context import ToolContext

        def escalate(query: str, tool_context: ToolContext) -> str:  # noqa: ARG001
            """Escalates the request to the Pix desk."""
            tool_context.actions.transfer_to_agent = "pix_specialist"
            return "escalated"

        specialist = LlmAgent(name="pix_specialist", model=ScriptedLlm(script="pix_specialist"), description="Pix.")
        assistant = LlmAgent(
            name="assistant", model=ScriptedLlm(script="assistant"), sub_agents=[specialist], tools=[escalate]
        )
        SCRIPTS["assistant"] = [_call("escalate", query="pix")]
        SCRIPTS["pix_specialist"] = [_text("feito")]

        _run_turns(assistant, "quero fazer um pix")

        assert _find(traces[0], "Tool: escalate") is not None
        assert _find(traces[0], "Handoff: assistant → pix_specialist") is not None
        assert traces[0].metadata["handoff_count"] == 1
        assert traces[0].metadata["handoff_path"] == "assistant > pix_specialist"

    def test_transfer_set_by_a_callback_is_a_handoff(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        def lookup_account(query: str) -> Dict[str, str]:  # noqa: ARG001
            """Looks up the customer's account."""
            return {"status": "blocked"}

        def escalate_blocked(tool: Any, args: Any, tool_context: Any, tool_response: Any) -> None:  # noqa: ARG001
            tool_context.actions.transfer_to_agent = "pix_specialist"

        specialist = LlmAgent(name="pix_specialist", model=ScriptedLlm(script="pix_specialist"), description="Pix.")
        assistant = LlmAgent(
            name="assistant",
            model=ScriptedLlm(script="assistant"),
            sub_agents=[specialist],
            tools=[lookup_account],
            after_tool_callback=escalate_blocked,
        )
        SCRIPTS["assistant"] = [_call("lookup_account", query="pix")]
        SCRIPTS["pix_specialist"] = [_text("desbloqueado")]

        _run_turns(assistant, "minha conta?")

        assert _find(traces[0], "Handoff: assistant → pix_specialist") is not None
        assert traces[0].metadata["handoff_path"] == "assistant > pix_specialist"

    def test_two_transfer_calls_in_one_reply_are_one_handoff(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        billing = LlmAgent(name="billing", model=ScriptedLlm(script="billing"), description="Billing.")
        support = LlmAgent(name="support", model=ScriptedLlm(script="support"), description="Support.")
        triage = LlmAgent(name="triage", model=ScriptedLlm(script="triage"), sub_agents=[billing, support])
        both = [
            types.Part(function_call=types.FunctionCall(name="transfer_to_agent", args={"agent_name": name}))
            for name in ("billing", "support")
        ]
        SCRIPTS["triage"] = [types.Content(role="model", parts=both)]
        SCRIPTS["support"] = [_text("suporte aqui")]

        _run_turns(triage, "help")

        # ADK keeps the last transfer of the reply.
        assert traces[0].metadata["handoff_count"] == 1
        assert traces[0].metadata["handoff_path"] == "triage > support"
        assert _find(traces[0], "Tool: transfer_to_agent") is None

    def test_failed_transfer_call_is_not_a_handoff(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent"), _text("posso ajudar?")]

        _run_turns(_bank_agents(), "oi")

        assert not [step for step in _all_steps(traces[0]) if step.step_type == enums.StepType.HANDOFF]
        # The failed attempt is still visible as a tool call.
        assert _find(traces[0], "Tool: transfer_to_agent") is not None
        assert traces[0].metadata["handoff_count"] == 0
        assert traces[0].metadata["final_agent"] == "assistant"

    def test_transfer_inside_an_agent_tool_stays_inside_it(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.tools.agent_tool import AgentTool

        writer = LlmAgent(name="writer", model=ScriptedLlm(script="writer"), description="Writes drafts.")
        researcher = LlmAgent(
            name="researcher", model=ScriptedLlm(script="researcher"), description="Researches.", sub_agents=[writer]
        )
        boss = LlmAgent(name="boss", model=ScriptedLlm(script="boss"), tools=[AgentTool(agent=researcher)])
        SCRIPTS["boss"] = [_call("researcher", request="write about pix"), _text("aqui está")]
        SCRIPTS["researcher"] = [_call("transfer_to_agent", agent_name="writer")]
        SCRIPTS["writer"] = [_text("rascunho")]

        _run_turns(boss, "go")

        assert len(traces) == 1
        assert traces[0].metadata["handoff_count"] == 0, "a transfer inside an AgentTool isn't the turn's handoff"
        assert traces[0].metadata["handoff_path"] == "boss"
        writer_step = _find(traces[0], "Agent: writer")
        assert writer_step is not None
        assert _parent_of(traces[0], writer_step).name == "Tool: researcher"
        handoff = _find(traces[0], "Handoff: researcher → writer")
        assert handoff is not None
        assert _parent_of(traces[0], handoff).name == "Agent: researcher"

    def test_transfer_inside_a_workflow_agent_stays_inside_it(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent, SequentialAgent

        helper = LlmAgent(name="helper", model=ScriptedLlm(script="helper"), description="Helps.")
        collector = LlmAgent(name="collector", model=ScriptedLlm(script="collector"), sub_agents=[helper])
        finisher = LlmAgent(name="finisher", model=ScriptedLlm(script="finisher"))
        SCRIPTS["collector"] = [_call("transfer_to_agent", agent_name="helper")]
        SCRIPTS["helper"] = [_text("ajudei")]
        SCRIPTS["finisher"] = [_text("fim")]

        _run_turns(SequentialAgent(name="pipeline", sub_agents=[collector, finisher]), "go")

        pipeline = _find(traces[0], "Agent: pipeline")
        assert [step.name for step in pipeline.steps] == ["Agent: collector", "Agent: helper", "Agent: finisher"]
        assert traces[0].metadata["handoff_path"] == "collector > helper"

    def test_final_agent_is_the_target_even_when_it_says_nothing(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [types.Content(role="model", parts=None)]

        _run_turns(_bank_agents(), "quero fazer um pix")

        assert traces[0].metadata["final_agent"] == "pix_specialist"
        assert traces[0].metadata["handoff_path"] == "assistant > pix_specialist"


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
class TestAdkTurnLifecycle:
    """However the caller consumes the Runner, each turn is its own trace."""

    def test_breaking_out_on_the_final_response_keeps_turns_apart(self, traces: List[Any]) -> None:
        import gc

        from google.adk.agents import LlmAgent

        runner = _runner(LlmAgent(name="solo", model=ScriptedLlm(script="solo")))
        SCRIPTS["solo"] = [_text("um"), _text("dois"), _text("três")]

        @_tracer.trace()
        def unrelated() -> int:
            return 1

        leftovers: Dict[str, Any] = {}

        async def _drive() -> None:
            await _new_session(runner)
            expected = 1
            for message in ("a", "b", "c"):
                # ADK's tutorial pattern. The abandoned generator is closed later,
                # from another task.
                async for event in _turn(runner, message):
                    if event.is_final_response():
                        break
                gc.collect()
                await _until(lambda count=expected: len(traces) >= count, timeout=2.0)
                expected += 1
            leftovers["step"] = _tracer.get_current_step()
            leftovers["turn"] = google_adk_tracer._current_turn.get()
            unrelated()

        asyncio.run(_drive())

        turns = [trace for trace in traces if trace.steps[0].name.startswith("Agent turn")]
        assert [trace.steps[0].output for trace in turns] == ["um", "dois", "três"]
        assert all(trace.metadata["final_agent"] == "solo" for trace in turns)
        assert leftovers == {"step": None, "turn": None}, "the caller's context kept the turn"
        assert [trace.steps[0].name for trace in traces if trace not in turns] == ["unrelated"]

    def test_turn_resumed_from_a_new_task_per_event(self, traces: List[Any]) -> None:
        runner = _runner(_bank_agents())
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito")]

        async def _drive() -> None:
            await _new_session(runner)
            events = _turn(runner, "quero fazer um pix")

            async def _next() -> Any:
                return await events.__anext__()

            while True:
                try:
                    # A task per event, as an inactivity timeout or an SSE
                    # heartbeat loop does.
                    await asyncio.ensure_future(_next())
                except StopAsyncIteration:
                    break

        asyncio.run(_drive())

        assert len(traces) == 1
        assert traces[0].metadata["handoff_path"] == "assistant > pix_specialist"
        assert traces[0].steps[0].output == "feito"

    def test_turns_inside_one_traced_function_share_the_columns(self, traces: List[Any]) -> None:
        runner = _runner(_bank_agents())
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito"), _text("de nada")]

        @_tracer.trace_async()
        async def conversation() -> None:
            for message in ("quero fazer um pix", "obrigado"):
                async for _event in _turn(runner, message):
                    pass

        async def _drive() -> None:
            await _new_session(runner)
            await conversation()

        asyncio.run(_drive())

        assert len(traces) == 1
        assert traces[0].metadata == {
            "starting_agent": "assistant",
            "final_agent": "pix_specialist",
            "handoff_count": 1,
            "handoff_path": "assistant > pix_specialist",
        }

    def test_handler_that_returns_on_the_final_response(self, traces: List[Any]) -> None:
        import gc

        runner = _runner(_bank_agents())
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito"), _text("de nada")]

        @_tracer.trace_async()
        async def handler(message: str) -> Optional[str]:
            async for event in _turn(runner, message):
                if event.is_final_response():
                    return event.content.parts[0].text
            return None

        async def _drive() -> None:
            await _new_session(runner)
            assert await handler("quero fazer um pix") == "feito"
            gc.collect()
            assert await handler("obrigado") == "de nada"
            gc.collect()
            await _until(lambda: False, timeout=0.1)

        asyncio.run(_drive())

        handlers = [trace for trace in traces if trace.steps[0].name == "handler"]
        assert len(handlers) == 2
        # Each handler's row has its own turn's columns, written before the
        # turn's generator was closed.
        assert handlers[0].metadata["handoff_path"] == "assistant > pix_specialist"
        assert handlers[1].metadata["handoff_path"] == "pix_specialist"
        assert [step.name for step in handlers[1].steps[0].steps] == ["Agent turn: bank_app"]
        assert [step.name for step in handlers[1].steps[0].steps[0].steps] == ["Agent: pix_specialist"]

    def test_turn_that_reaches_no_agent_keeps_the_columns(self, traces: List[Any]) -> None:
        runner = _runner(_bank_agents())
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito")]

        @_tracer.trace_async()
        async def conversation() -> None:
            async for _event in _turn(runner, "quero fazer um pix"):
                pass
            content = types.Content(role="user", parts=[types.Part(text="oi")])
            with pytest.raises(Exception):  # noqa: B017 - the session doesn't exist
                async for _event in runner.run_async(user_id=USER_ID, session_id="missing", new_message=content):
                    pass

        async def _drive() -> None:
            await _new_session(runner)
            await conversation()

        asyncio.run(_drive())

        assert traces[0].metadata == {
            "starting_agent": "assistant",
            "final_agent": "pix_specialist",
            "handoff_count": 1,
            "handoff_path": "assistant > pix_specialist",
        }

    def test_failed_turn_has_no_completion_output(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        def lookup(account: str) -> str:  # noqa: ARG001
            """Looks up an account."""
            raise ValueError("backend down")

        agent = LlmAgent(name="solo", model=ScriptedLlm(script="solo"), tools=[lookup])
        SCRIPTS["solo"] = [_call("lookup", account="1")]

        with pytest.raises(ValueError):
            _run_turns(agent, "saldo?")

        (root,) = traces[0].steps
        assert root.output is None
        assert root.metadata["error"]["type"] == "ValueError"

    def test_cancelled_turn_records_the_cancellation(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        runner = _runner(LlmAgent(name="solo", model=ScriptedLlm(script="solo")))
        SCRIPTS["solo"] = [5]

        async def _consume() -> None:
            async for _event in _turn(runner, "oi"):
                pass

        async def _drive() -> None:
            await _new_session(runner)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(_consume(), timeout=0.2)

        asyncio.run(_drive())

        (root,) = traces[0].steps
        assert root.output is None
        assert root.metadata["error"]["type"] == "CancelledError"

    def test_resumed_turn_starts_at_the_paused_agent(self, traces: List[Any]) -> None:
        from google.adk.apps import App, ResumabilityConfig
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools.long_running_tool import LongRunningFunctionTool

        def aprovar_pix(valor: float) -> None:  # noqa: ARG001
            """Asks a human to approve a Pix."""
            return None

        specialist = LlmAgent(
            name="pix_specialist",
            model=ScriptedLlm(script="pix_specialist"),
            description="Pix.",
            tools=[LongRunningFunctionTool(aprovar_pix)],
        )
        assistant = LlmAgent(name="assistant", model=ScriptedLlm(script="assistant"), sub_agents=[specialist])
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_call("aprovar_pix", valor=10.0), _text("pix aprovado"), _text("extra")]
        app = App(name=APP_NAME, root_agent=assistant, resumability_config=ResumabilityConfig(is_resumable=True))
        runner = Runner(app=app, session_service=InMemorySessionService())

        async def _drive() -> None:
            await _new_session(runner)
            call_id = invocation_id = None
            async for event in _turn(runner, "quero fazer um pix"):
                invocation_id = event.invocation_id
                for function_call in event.get_function_calls() or []:
                    if function_call.name == "aprovar_pix":
                        call_id = function_call.id
            approval = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id, name="aprovar_pix", response={"status": "approved"}
                        )
                    )
                ],
            )
            async for _event in runner.run_async(
                user_id=USER_ID, session_id=SESSION_ID, invocation_id=invocation_id, new_message=approval
            ):
                pass

        asyncio.run(_drive())

        assert len(traces) == 2
        assert traces[0].metadata["handoff_path"] == "assistant > pix_specialist"
        assert traces[1].metadata["handoff_count"] == 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
class TestAdkOutputs:
    def test_workflow_agent_output_is_its_last_childs_answer(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent, SequentialAgent

        first = LlmAgent(name="first", model=ScriptedLlm(script="first"))
        second = LlmAgent(name="second", model=ScriptedLlm(script="second"))
        SCRIPTS["first"] = [_text("one")]
        SCRIPTS["second"] = [_text("two")]

        _run_turns(SequentialAgent(name="pipeline", sub_agents=[first, second]), "go")

        assert _find(traces[0], "Agent: pipeline").output == "two"

    def test_agent_tool_result_without_summary_is_the_answer(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.tools.agent_tool import AgentTool

        calculator = LlmAgent(name="calculator", model=ScriptedLlm(script="calculator"), description="Does math.")
        boss = LlmAgent(
            name="boss",
            model=ScriptedLlm(script="boss"),
            tools=[AgentTool(agent=calculator, skip_summarization=True)],
        )
        SCRIPTS["boss"] = [_call("calculator", request="2+2")]
        SCRIPTS["calculator"] = [_text("The answer is 4")]

        _run_turns(boss, "quanto é 2+2?")

        assert traces[0].steps[0].output == "The answer is 4"

    def test_thoughts_are_not_part_of_the_answer(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        SCRIPTS["solo"] = [
            types.Content(
                role="model",
                parts=[types.Part(text="**Considering the request**", thought=True), types.Part(text="Pix enviado.")],
            )
        ]

        _run_turns(LlmAgent(name="solo", model=ScriptedLlm(script="solo")), "manda o pix")

        assert traces[0].steps[0].output == "Pix enviado."
        assert _find(traces[0], "Agent: solo").output == "Pix enviado."

    def test_turn_root_carries_the_first_agents_columns(self, traces: List[Any]) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_text("feito")]

        _run_turns(_bank_agents(), "quero fazer um pix")

        row, input_names = _tracer.post_process_trace(traces[0])
        assert row["agent_name"] == "assistant"
        assert {"agent_name", "model", "instruction", "sub_agents", "user_query"} <= set(input_names)
        assert input_names[-1] == "user_query"
        # Every agent of the turn answers the user's message.
        assert _find(traces[0], "Agent: pix_specialist").inputs["user_query"] == "quero fazer um pix"


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
class TestAdkLlmStep:
    def test_llm_step_uses_gemini_provider_and_counts_thinking_tokens(self, traces: List[Any]) -> None:
        from google.adk.agents import LlmAgent

        agent = LlmAgent(name="solo", model=ScriptedLlm(script="solo"))
        SCRIPTS["solo"] = [_text("hi")]

        _run_turns(agent, "hi")

        (llm,) = [step for step in _all_steps(traces[0]) if step.step_type == enums.StepType.CHAT_COMPLETION]
        assert llm.provider == "gemini"
        assert llm.model == "gemini-2.5-flash"
        assert llm.prompt_tokens == 10
        # Thinking tokens are billed as output tokens.
        assert llm.completion_tokens == 8
        assert llm.tokens == 18

    def test_no_context_detach_errors(self, traces: List[Any], caplog: pytest.LogCaptureFixture) -> None:
        SCRIPTS["assistant"] = [_call("transfer_to_agent", agent_name="pix_specialist")]
        SCRIPTS["pix_specialist"] = [_call("executar_transferencia_pix", valor=10.0), _text("feito")]

        with caplog.at_level(logging.ERROR):
            _run_turns(_bank_agents(), "quero fazer um pix")

        assert traces
        assert "Failed to detach context" not in caplog.text


class TestAdkPatchTargets:
    def test_tool_call_function_is_patched(self) -> None:
        """The first target that exists in the installed ADK must be wrapped."""
        for module_name, attr in google_adk_tracer._TOOL_CALL_TARGETS:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, attr):
                assert hasattr(getattr(module, attr), "__wrapped__"), f"{module_name}.{attr} is not patched"
                return
        pytest.fail("none of _TOOL_CALL_TARGETS exists in the installed google-adk")


class TestAdkUsageAndRawOutput:
    @staticmethod
    def _usage(**counts: int) -> Any:
        return LlmResponse(usage_metadata=types.GenerateContentResponseUsageMetadata(**counts))

    def test_native_gemini_thoughts_are_added_to_completion(self) -> None:
        usage = google_adk_tracer._extract_usage_from_response(
            self._usage(prompt_token_count=8, candidates_token_count=6, thoughts_token_count=743, total_token_count=757)
        )
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (8, 749, 757)
        assert usage["breakdown"]["thoughtsTokens"] == 743

    def test_reasoning_already_in_candidates_is_not_added_twice(self) -> None:
        # ADK's LiteLlm maps LiteLLM's completion_tokens (reasoning included) to
        # candidates and the reasoning count to thoughts.
        usage = google_adk_tracer._extract_usage_from_response(
            self._usage(
                prompt_token_count=100, candidates_token_count=500, thoughts_token_count=400, total_token_count=600
            )
        )
        assert (usage["completion_tokens"], usage["total_tokens"]) == (500, 600)

    def test_raw_output_leaves_out_inline_data(self) -> None:
        blob = b"\x89PNG" + b"0" * (1 << 20)
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="here"),
                    types.Part(inline_data=types.Blob(mime_type="image/png", data=blob)),
                ],
            )
        )

        raw = google_adk_tracer._response_json(response)

        assert len(raw) < 1000
        assert "image/png" in raw and "here" in raw
        assert response.content.parts[1].inline_data.data == blob, "the caller's response must not change"
