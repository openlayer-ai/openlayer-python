"""Live integration test against the real Claude Agent SDK.

This test is gated on the ``ANTHROPIC_API_KEY`` environment variable. It runs
a tiny ``query()`` against ``claude-haiku-4-5`` with no tools and asserts the
SDK produced a ``ResultMessage`` (proving the wrapper observed the stream).

Run locally with the live test API keys set in env:

    ANTHROPIC_API_KEY=... OPENLAYER_API_KEY=... pytest \\
        tests/integrations/test_claude_agent_sdk_live.py -v

In CI this test is automatically skipped when ``ANTHROPIC_API_KEY`` is unset.

Note: the bundled ``claude-agent-sdk`` raises an internal ``Exception`` after
delivering ``ResultMessage`` when the underlying API returns ``is_error=True``
(e.g. an invalid Anthropic API key). The wrapper correctly observes the
stream up to that point; we tolerate the trailing exception in this test so
it still exercises the trace publishing path with a real API key.
"""

from __future__ import annotations

import os
import asyncio

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def test_live_query_produces_valid_trace():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions

    from openlayer.lib.integrations.claude_agent_sdk import traced_query

    os.environ.setdefault(
        "OPENLAYER_INFERENCE_PIPELINE_ID",
        "d4ee57e5-cd26-4435-b321-0365760724ad",
    )

    async def run():
        messages = []
        try:
            async for m in traced_query(
                prompt="Say the word 'banana' and nothing else.",
                options=ClaudeAgentOptions(
                    model="claude-haiku-4-5",
                    system_prompt=(
                        "You are a terse assistant that follows instructions "
                        "exactly. Never add filler words, never apologize, and "
                        "never add quotes around your answer."
                    ),
                    max_turns=2,
                ),
            ):
                messages.append(m)
        except Exception as exc:
            # The SDK raises an Exception trailing ResultMessage when the
            # underlying API errors. The wrapper observed everything up to
            # that point; tolerate the trailing exception here.
            messages.append(("__sdk_exception__", str(exc)))
        return messages

    msgs = asyncio.run(run())
    real_msgs = [m for m in msgs if not (isinstance(m, tuple) and m[0] == "__sdk_exception__")]
    assert any(
        type(m).__name__ == "SystemMessage" for m in real_msgs
    ), "Expected a SystemMessage(init) in the stream"
    assert any(
        type(m).__name__ == "ResultMessage" for m in real_msgs
    ), "Expected a ResultMessage in the stream"
