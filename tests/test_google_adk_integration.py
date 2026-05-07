"""Tests for the Google ADK tracer integration.

Regression coverage for OPEN-10343: when a user-defined ``before_model_callback``
raises, the Openlayer tracer must let the original exception propagate
unchanged (no chained ``ValueError`` from contextvar cleanup) and must record
the error on the span's metadata without overwriting ``step.output``.
"""

# google-adk and google-genai aren't installed in the lint env, and pytest's
# fixture decorator hides this autouse function from static analysis.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedFunction=false

import asyncio
from typing import Any, Dict, List, Tuple, Optional
from unittest.mock import patch

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("wrapt")

from openlayer.lib.integrations.google_adk_tracer import (
    _record_step_error,
    _safe_reset_contextvar,
)


class ErrorHandling(Exception):
    """User-defined exception raised by a guardrail-style callback."""


@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all tracer publish paths off during the test."""
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    from openlayer.lib.tracing import tracer as _tracer

    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


def _collect_exception_chain(exc: BaseException) -> List[BaseException]:
    """Walk ``__cause__`` / ``__context__`` and return the unique chain."""
    seen: List[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in seen:
        seen.append(current)
        nxt = current.__cause__ if current.__cause__ is not None else current.__context__
        current = nxt
    return seen


class TestGoogleADKCallbackExceptions:
    """OPEN-10343: chained / noisy exceptions from before_model_callback."""

    APP_NAME = "openlayer_open_10343"
    USER_ID = "u"
    SESSION_ID = "s"

    def _build_runner_and_agent(self) -> Tuple[Any, Any]:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        from openlayer.lib.integrations import trace_google_adk

        trace_google_adk()

        async def before_model(callback_context: Any, llm_request: Any, **_: Any) -> None:  # noqa: ARG001
            raise ErrorHandling("input blocked by guardrail (pi_and_jailbreak)")

        agent = LlmAgent(
            model="gemini-2.5-flash",
            name="OpenLayerTestAgent",
            instruction="test",
            before_model_callback=before_model,
        )
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name=self.APP_NAME, session_service=session_service)
        return runner, session_service

    async def _drive_runner(self, runner: Any, session_service: Any) -> None:
        from google.genai import types

        await session_service.create_session(
            app_name=self.APP_NAME, user_id=self.USER_ID, session_id=self.SESSION_ID
        )
        content = types.Content(role="user", parts=[types.Part(text="hi")])
        async for _event in runner.run_async(
            user_id=self.USER_ID, session_id=self.SESSION_ID, new_message=content
        ):
            pass

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_user_exception_propagates_without_chaining(self) -> None:
        """The user's ``ErrorHandling`` must be the only exception surfaced."""
        runner, session_service = self._build_runner_and_agent()

        with pytest.raises(ErrorHandling) as exc_info:
            asyncio.run(self._drive_runner(runner, session_service))

        chain = _collect_exception_chain(exc_info.value)

        # No chained ValueError from contextvars / token-reset cleanup.
        assert not any(
            isinstance(item, ValueError)
            and "was created in a different Context" in str(item)
            for item in chain
        ), f"chained ValueError appeared in exception chain: {chain}"

        # Anything in the chain that isn't the user's exception would itself
        # be a tracer-introduced failure — fail loudly.
        for item in chain:
            assert isinstance(item, ErrorHandling), (
                "unexpected non-ErrorHandling exception in chain: "
                f"{type(item).__name__}: {item}"
            )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_callback_error_recorded_on_step_metadata(self) -> None:
        """The callback step's metadata['error'] should describe the failure."""
        from openlayer.lib.tracing import tracer as _tracer
        from openlayer.lib.tracing.traces import Trace

        captured: Dict[str, Any] = {}
        original_handle = _tracer._handle_trace_completion

        def _capture_handle(*args: Any, **kwargs: Any) -> None:
            current_trace = _tracer.get_current_trace()
            if current_trace is not None and "trace" not in captured:
                captured["trace"] = current_trace
            return original_handle(*args, **kwargs)

        runner, session_service = self._build_runner_and_agent()

        with patch.object(_tracer, "_handle_trace_completion", _capture_handle):
            with pytest.raises(ErrorHandling):
                asyncio.run(self._drive_runner(runner, session_service))

        trace: Trace = captured["trace"]

        callback_steps = self._find_callback_steps(trace)
        assert callback_steps, "no callback steps were recorded on the trace"

        before_model_step = next(
            (s for s in callback_steps if s.metadata.get("callback_type") == "before_model"),
            None,
        )
        assert before_model_step is not None, "before_model callback step missing"

        recorded = before_model_step.metadata.get("error")
        assert recorded is not None, "error metadata not recorded on callback step"
        assert recorded["type"] == "ErrorHandling"
        assert "pi_and_jailbreak" in recorded["message"]

        # And critically: step.output was *not* overwritten with "Error: ...".
        if before_model_step.output is not None:
            assert not (
                isinstance(before_model_step.output, str)
                and before_model_step.output.startswith("Error:")
            ), (
                "step.output should not be overwritten with 'Error: ...'; "
                f"got {before_model_step.output!r}"
            )

    @staticmethod
    def _find_callback_steps(trace: Any) -> List[Any]:
        """Walk the trace tree and return all callback steps (USER_CALL)."""
        from openlayer.lib.tracing import enums

        out: List[Any] = []

        def _walk(step: Any) -> None:
            if getattr(step, "step_type", None) == enums.StepType.USER_CALL:
                if step.metadata.get("is_callback"):
                    out.append(step)
            for child in getattr(step, "steps", []) or []:
                _walk(child)

        for step in getattr(trace, "steps", []) or []:
            _walk(step)
        return out


class TestGoogleADKHelpers:
    """Unit tests for the small helper functions added in OPEN-10343."""

    def test_safe_reset_swallows_cross_context_value_error(self) -> None:
        class FakeVar:
            def reset(self, _token: Any) -> None:
                raise ValueError("<Token at 0x1> was created in a different Context")

        # Should not raise.
        _safe_reset_contextvar(FakeVar(), object())  # type: ignore[arg-type]

    def test_safe_reset_reraises_other_value_errors(self) -> None:
        class FakeVar:
            def reset(self, _token: Any) -> None:
                raise ValueError("some other reason")

        with pytest.raises(ValueError, match="some other reason"):
            _safe_reset_contextvar(FakeVar(), object())  # type: ignore[arg-type]

    def test_safe_reset_no_op_for_none_token(self) -> None:
        class FakeVar:
            def reset(self, _token: Any) -> None:  # pragma: no cover
                raise AssertionError("should not be called when token is None")

        _safe_reset_contextvar(FakeVar(), None)  # type: ignore[arg-type]

    def test_record_step_error_writes_metadata(self) -> None:
        class FakeStep:
            metadata: Optional[Dict[str, Any]] = None
            output: Optional[Any] = "preserved"

        step = FakeStep()
        _record_step_error(step, ErrorHandling("boom"))

        assert step.metadata is not None
        assert step.metadata["error"] == {"type": "ErrorHandling", "message": "boom"}
        # Output left untouched.
        assert step.output == "preserved"

    def test_record_step_error_handles_none_step(self) -> None:
        # Should not raise.
        _record_step_error(None, ErrorHandling("boom"))
