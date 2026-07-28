"""Real-integration test for ADK / google-genai double-instrumentation.

``google-adk`` depends on ``google-genai``, and ``Gemini.api_client`` IS a
``google.genai.Client`` whose ``aio.models.generate_content`` the ADK flow calls
directly. So registering a ``google.genai`` auto-instrument entry activates it for
every ADK user, and without contextual suppression each ADK LLM call would emit
two chat_completion spans and be billed twice.

Unit tests for ``_adk_span_active`` inject a fake module into ``sys.modules``,
which proves the *check* works but not that the real ``_current_llm_step``
contextvar is visible at the moment our patched method runs. This file drives a
real ``Runner`` + ``LlmAgent`` with both tracers active and asserts the span
count, which is the only thing that discriminates.

No network: ``AsyncModels.generate_content`` is stubbed on the CLASS before the
class-init patch, so ADK's real call path is exercised end to end.
"""

# Neither google-adk nor google-genai is installed in the lint env, so imports
# from the `google` namespace package don't resolve there.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedFunction=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

import asyncio
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("google.genai")
pytest.importorskip("wrapt")

from openlayer.lib.integrations import google_genai_tracer as gg

APP_NAME = "openlayer_open_11891"
USER_ID = "u"
SESSION_ID = "s"


@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    from openlayer.lib.tracing import tracer as _tracer

    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


@pytest.fixture(autouse=True)
def _reset_patches():
    """Both tracers keep module/class-level state; clear it between tests."""
    yield
    from openlayer.lib.integrations.google_adk_tracer import _unpatch_google_adk

    _unpatch_google_adk()
    gg._unpatch_google_genai()


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch):
    """Stub ``AsyncModels.generate_content`` at the class level.

    Installed BEFORE the class-init patch so the per-instance wrapper wraps the
    stub — i.e. ADK -> genai.Client -> our wrapper -> stub, the real chain.
    """
    from google.genai import types, models as genai_models

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="hello from stub")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=5,
            candidates_token_count=3,
            total_token_count=8,
        ),
        model_version="gemini-2.5-flash",
    )

    calls: List[Dict[str, Any]] = []

    async def _impl(_self: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return response

    monkeypatch.setattr(genai_models.AsyncModels, "generate_content", _impl)
    return calls


def _run_one_agent_turn() -> None:
    from google.genai import types
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    agent = LlmAgent(model="gemini-2.5-flash", name="OpenlayerDedupAgent", instruction="test")
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    async def _drive() -> None:
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        message = types.Content(role="user", parts=[types.Part(text="hi")])
        async for _event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message):
            pass

    asyncio.run(_drive())


def _count_chat_completion_steps(step: Any) -> int:
    from openlayer.lib.tracing import enums

    count = 1 if getattr(step, "step_type", None) == enums.StepType.CHAT_COMPLETION else 0
    for nested in getattr(step, "steps", None) or []:
        count += _count_chat_completion_steps(nested)
    return count


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAdkDoubleInstrumentation:
    def test_genai_span_suppressed_when_adk_tracer_is_active(self, stub_model: Any) -> None:
        """Both tracers on: the ADK tracer owns the span, ours must stand down."""
        from openlayer.lib.integrations import trace_google_adk

        trace_google_adk()
        gg._patch_google_genai()

        with patch.object(gg, "add_to_trace") as mock_add:
            _run_one_agent_turn()

        assert stub_model, "the ADK flow must actually have reached genai.Client"
        assert mock_add.call_count == 0, (
            "google_genai_tracer emitted a span inside an ADK LLM step -- "
            "every ADK call would be double-counted and double-billed"
        )

    def test_exactly_one_chat_completion_step_in_the_trace(self, stub_model: Any) -> None:
        """The user-visible property: one LLM call produces one span."""
        from openlayer.lib.tracing import tracer as _tracer
        from openlayer.lib.integrations import trace_google_adk

        trace_google_adk()
        gg._patch_google_genai()

        captured: Dict[str, Any] = {}
        original_handle = _tracer._handle_trace_completion

        def _capture(*args: Any, **kwargs: Any) -> Any:
            current = _tracer.get_current_trace()
            if current is not None and "trace" not in captured:
                captured["trace"] = current
            return original_handle(*args, **kwargs)

        with patch.object(_tracer, "_handle_trace_completion", _capture):
            _run_one_agent_turn()

        assert len(stub_model) == 1, "the agent turn should make exactly one LLM call"

        trace = captured.get("trace")
        assert trace is not None, "no trace was captured"
        total = sum(_count_chat_completion_steps(step) for step in trace.steps)
        assert total == 1, f"expected exactly 1 chat_completion step, found {total}"

    def test_span_emitted_when_only_the_genai_tracer_is_active(self, stub_model: Any) -> None:
        """Control: suppression must be CONTEXTUAL, not 'is ADK importable'.

        ``google_adk_tracer`` stays in ``sys.modules`` once any earlier test has
        imported it, so a module-presence check would wrongly suppress here.
        """
        import sys

        gg._patch_google_genai()

        with patch.object(gg, "add_to_trace") as mock_add:
            _run_one_agent_turn()

        assert stub_model, "the ADK flow must actually have reached genai.Client"
        assert mock_add.call_count >= 1, (
            "with no ADK tracer active, the genai tracer is the only thing that "
            "can trace this call and must not stay silent"
        )
        # Guard the premise of this test: if the module were absent, the control
        # would pass trivially rather than proving contextual behaviour.
        assert gg._ADK_TRACER_MODULE in sys.modules
