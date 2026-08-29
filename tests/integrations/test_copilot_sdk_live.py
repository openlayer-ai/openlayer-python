"""Live end-to-end test for the GitHub Copilot SDK integration.

Skipped unless OPENLAYER_COPILOT_LIVE_TEST=1 AND the interpreter is Python 3.11+ -- ``github-copilot-sdk`` declares
``requires_python >= 3.11``, while this SDK still supports 3.9, so the unit
tests deliberately run off recorded fixtures instead of the real package.

Run with:
    OPENLAYER_COPILOT_LIVE_TEST=1 GITHUB_TOKEN=$(gh auth token) \\
        pytest tests/integrations/test_copilot_sdk_live.py -v -s
"""

import os
import sys
from typing import Any, List
from unittest.mock import patch

import pytest

from openlayer.lib.tracing import tracer as ol_tracer

# Opt-in is explicit rather than keyed on GITHUB_TOKEN alone: that variable is
# commonly exported by developers and by CI steps that have nothing to do with
# Copilot, and an installation token has no Copilot access -- so keying on it
# would turn a normal `pytest` run into a five-minute live session that then
# fails on auth.
pytestmark = [
    pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="github-copilot-sdk requires Python 3.11+",
    ),
    pytest.mark.skipif(
        os.environ.get("OPENLAYER_COPILOT_LIVE_TEST") != "1",
        reason="set OPENLAYER_COPILOT_LIVE_TEST=1 (and a GitHub token with Copilot access) to run",
    ),
]

_REAL_RESOLVE = ol_tracer._resolve


@pytest.mark.asyncio
async def test_live_copilot_session_publishes_trace(monkeypatch, tmp_path):
    pytest.importorskip("copilot")
    from copilot import CopilotClient, PermissionHandler

    from openlayer.lib.integrations import copilot_sdk

    captured: List[Any] = []

    def capture(trace, *_args, **_kwargs):
        captured.append(trace.to_dict())

    # Force synchronous publishing so capture is deterministic.
    monkeypatch.setattr(
        ol_tracer,
        "_resolve",
        lambda key, *a, **k: (
            False if key == "background_publish_enabled" else _REAL_RESOLVE(key, *a, **k)
        ),
    )

    (tmp_path / "hello.py").write_text("print('hello')\n")
    (tmp_path / "notes.txt").write_text("some notes\n")

    with patch.object(ol_tracer, "_publish", True), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture
    ):
        client = CopilotClient(working_directory=str(tmp_path), log_level="error")
        await client.start()
        try:
            session = await client.create_session(
                working_directory=str(tmp_path),
                on_permission_request=PermissionHandler.approve_all,
                on_event=copilot_sdk.openlayer_event_handler(),
            )
            await session.send_and_wait(
                "Run `ls` with bash and tell me how many files there are, in one sentence.",
                timeout=300,
            )
            await session.disconnect()
        finally:
            await client.stop()

    assert captured, "a trace must be published"
    root = captured[-1][0]

    assert root["type"] == "agent"
    assert root["name"] == copilot_sdk.ROOT_STEP_NAME
    # The root's output is what becomes the row's output column.
    assert isinstance(root["output"], str) and root["output"].strip()
    assert root["inputs"]["prompt"].startswith("Run `ls`")
    assert root["metadata"]["copilot_session_id"]

    chats = [s for s in root["steps"] if s["type"] == "chat_completion"]
    assert chats, "at least one chat completion step"
    for chat in chats:
        assert chat["model"]
        # A real provider slug is what lets Openlayer price the call; "github"
        # would silently yield $0.
        assert chat["provider"] in ("anthropic", "openai", "google", "xai")
        assert chat["cost"] is None, "premium-request units must not be published as cost"
    assert any(c["usageDetails"].get("output_tokens", 0) > 0 for c in chats)

    tools = [s for s in root["steps"] if s["type"] == "tool"]
    assert tools, "the prompt forces a bash tool call"
    assert any(t["name"] == "bash" for t in tools)
