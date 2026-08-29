"""Unit tests for the GitHub Copilot SDK integration.

The fixtures are real captured sessions (trimmed of base64 blobs and the 24KB
CLI system prompt), so these tests exercise the actual wire format rather than
an assumption about it.
"""

import json
import pathlib
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from openlayer.lib.tracing import tracer as ol_tracer
from openlayer.lib.integrations import copilot_sdk

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "copilot_sdk"

# Captured before any patching so the fixture can delegate to the real resolver.
_REAL_RESOLVE = ol_tracer._resolve


def load_fixture(name: str) -> List[Dict[str, Any]]:
    with open(FIXTURES / name) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")
    monkeypatch.setattr(ol_tracer, "_publish", False, raising=False)


@pytest.fixture
def published_traces(monkeypatch: pytest.MonkeyPatch):
    """Yield a list that receives each published trace as a dict.

    ``background_publish_enabled`` defaults to True, which hands the trace to a
    ThreadPoolExecutor -- so asserting on the captured list right after the
    collector runs is a race the test only sometimes wins. Force synchronous
    publishing so capture is deterministic regardless of global tracer config
    left behind by other suites.
    """
    captured: List[Any] = []

    def capture(trace, *_args, **_kwargs):
        captured.append(trace.to_dict())

    monkeypatch.setattr(
        ol_tracer,
        "_resolve",
        lambda key, *a, **k: False if key == "background_publish_enabled" else _REAL_RESOLVE(key, *a, **k),
    )

    with patch.object(ol_tracer, "_publish", True), patch.object(
        ol_tracer, "_upload_and_publish_trace", side_effect=capture
    ):
        yield captured



def run_fixture(name: str) -> None:
    collector = copilot_sdk.CopilotTraceCollector()
    for event in load_fixture(name):
        collector.handle(event)


def _find(step_list, predicate):
    """Depth-first search over a published step tree."""
    for step in step_list:
        if predicate(step):
            return step
        found = _find(step.get("steps") or [], predicate)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- #
# Buffering / routing
# --------------------------------------------------------------------------- #


def test_collector_buffers_events_and_keys_by_interaction_id():
    events = load_fixture("session_basic.jsonl")
    collector = copilot_sdk.CopilotTraceCollector()
    for event in events:
        collector.handle(event)

    # The fixture ends with session.idle, which builds and clears interactions.
    assert collector.session_id
    assert collector.built_count == 1


def test_delta_events_are_never_buffered():
    collector = copilot_sdk.CopilotTraceCollector()
    seen_types = set()
    for event in load_fixture("session_basic.jsonl"):
        collector.handle(event)
        for interaction in collector.interactions.values():
            seen_types.update(copilot_sdk._event_type(e) for e in interaction.events)

    assert not any(t.endswith("_delta") for t in seen_types)
    assert "tool.execution_partial_result" not in seen_types


def test_handle_never_raises_on_malformed_event():
    collector = copilot_sdk.CopilotTraceCollector()
    for junk in (None, {}, {"type": "user.message"}, object(), {"type": "assistant.usage"}):
        collector.handle(junk)  # must not raise


# --------------------------------------------------------------------------- #
# Provider + token mapping
# --------------------------------------------------------------------------- #


def test_provider_is_mapped_from_model_prefix_not_labelled_github():
    # Verified against llm-costs.openlayer.com: anthropic/claude-haiku-4.5 and
    # openai/gpt-5.4 resolve; github/* returns "No cost data found".
    assert copilot_sdk._provider_for_model("claude-haiku-4.5") == "anthropic"
    assert copilot_sdk._provider_for_model("gpt-5.4") == "openai"
    assert copilot_sdk._provider_for_model("o3-mini") == "openai"
    assert copilot_sdk._provider_for_model("gemini-2.5-pro") == "google"
    # Unknown prefixes omit provider rather than guess -- unpriced beats wrong.
    assert copilot_sdk._provider_for_model("some-future-model") is None
    assert copilot_sdk._provider_for_model("") is None
    assert copilot_sdk._provider_for_model(None) is None


def test_usage_details_is_a_non_overlapping_partition():
    """Copilot's input_tokens already contains cache reads and writes.

    Observed live: input_tokens=12509, cache_write=12499, cache_read=0, and the
    SDK's own copilot_usage._token_details reported input=10. The partition must
    reproduce that, because the backend sums a price per recognized key.
    """
    usage = {
        "input_tokens": 12509,
        "output_tokens": 224,
        "cache_write_tokens": 12499,
        "cache_read_tokens": 0,
        "reasoning_tokens": 143,
    }
    details = copilot_sdk._usage_details(usage)
    assert details["input_tokens"] == 10
    assert details["output_tokens"] == 224
    assert details["cache_creation_tokens"] == 12499
    assert "cached_tokens" not in details  # zero-valued keys are omitted
    # reasoning stays folded into output_tokens, matching langchain_callback
    assert "reasoning_tokens" not in details


def test_usage_details_handles_cache_reads():
    usage = {
        "input_tokens": 12754,
        "output_tokens": 122,
        "cache_write_tokens": 249,
        "cache_read_tokens": 12499,
    }
    details = copilot_sdk._usage_details(usage)
    assert details["input_tokens"] == 6
    assert details["cached_tokens"] == 12499
    assert details["cache_creation_tokens"] == 249
    assert sum(details.values()) == 12754 + 122  # partition is exact


def test_usage_details_accepts_camel_case():
    """TypeScript-shaped payloads must map identically."""
    details = copilot_sdk._usage_details(
        {"inputTokens": 100, "outputTokens": 10, "cacheReadTokens": 40, "cacheWriteTokens": 0}
    )
    assert details == {"input_tokens": 60, "output_tokens": 10, "cached_tokens": 40}


# --------------------------------------------------------------------------- #
# Trace shape
# --------------------------------------------------------------------------- #


def test_chat_steps_carry_tokens_and_priced_provider(published_traces):
    run_fixture("session_basic.jsonl")

    assert len(published_traces) == 1
    root = published_traces[-1][0]
    chats = [s for s in root["steps"] if s["type"] == "chat_completion"]
    assert len(chats) == 3  # three turns in this fixture
    for chat in chats:
        assert chat["model"] == "claude-haiku-4.5"
        assert chat["provider"] == "anthropic"
        assert chat["usageDetails"]["output_tokens"] > 0
        # Copilot's premium-request figure must NOT be published as cost;
        # leaving cost unset lets Openlayer price it from provider+model.
        assert chat["cost"] is None
        assert chat["metadata"]["copilot_premium_requests"] == 0.33


def test_root_output_is_final_assistant_message_not_empty(published_traces):
    """Openlayer builds a row's output from the root step; an empty root
    silently produces an unusable row."""
    run_fixture("session_tools_subagent.jsonl")

    root = published_traces[-1][0]
    assert root["type"] == "agent"
    assert isinstance(root["output"], str) and len(root["output"]) > 20
    assert "Lisbon" in root["output"]  # the final answer, not a reasoning preamble
    assert root["inputs"]["prompt"].startswith("Do these three things")


def test_concurrent_tool_calls_are_siblings_not_nested(published_traces):
    """Three tool.execution_start fire before any completion, and the
    completions arrive out of order. A step-stack design would nest them."""
    run_fixture("session_tools_subagent.jsonl")

    root = published_traces[-1][0]
    top_level = root["steps"]
    names = [s["name"] for s in top_level]
    assert "bash" in names
    assert "get_weather" in names
    # each concurrent call must be a direct child of the root, not of a sibling
    for name in ("bash", "get_weather"):
        assert any(s["name"] == name for s in top_level)


def test_subagent_dispatch_becomes_nested_agent_step(published_traces):
    run_fixture("session_tools_subagent.jsonl")

    root = published_traces[-1][0]
    subagent = _find(
        root["steps"], lambda s: s["type"] == "agent" and "explore" in s["name"].lower()
    )
    assert subagent is not None, "the `task` dispatch must become an AGENT step"
    # the subagent nests under the `task` tool call, not at the top level
    assert subagent not in root["steps"] or subagent["name"].startswith("subagent:")

    inner_chats = [s for s in subagent["steps"] if s["type"] == "chat_completion"]
    inner_tools = [s for s in subagent["steps"] if s["type"] == "tool"]
    assert len(inner_chats) >= 1
    assert any(t["name"] == "view" for t in inner_tools)


def test_tool_step_captures_arguments_and_failure(published_traces):
    run_fixture("session_basic.jsonl")

    root = published_traces[-1][0]
    bash = _find(root["steps"], lambda s: s["name"] == "bash")
    assert bash is not None
    assert bash["inputs"]["command"] == "ls -lhS"
    # this fixture's bash call was permission-denied
    assert bash["metadata"]["success"] is False
    assert "denied" in json.dumps(bash["metadata"]["error"]).lower()


def test_session_id_is_recorded_for_session_grouping(published_traces):
    run_fixture("session_basic.jsonl")
    root = published_traces[-1][0]
    assert root["metadata"]["copilot_session_id"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def test_user_handler_exception_does_not_break_tracing():
    handler = copilot_sdk.openlayer_event_handler()

    def exploding(_event):
        raise RuntimeError("customer bug")

    composed = copilot_sdk._compose_handlers(exploding, handler)
    composed({"type": "session.start", "data": {"session_id": "s1"}})  # must not raise


def test_composed_handler_still_delivers_to_user_handler():
    seen = []
    handler = copilot_sdk.openlayer_event_handler()
    composed = copilot_sdk._compose_handlers(seen.append, handler)
    composed({"type": "session.start", "data": {"session_id": "s1"}})
    assert len(seen) == 1


# --------------------------------------------------------------------------- #
# Live-shape regressions
#
# The recorded fixtures are JSON, so every value in them is already a plain
# string/dict. The live Python binding is not: it hands back dataclasses and
# Enums. These tests pin the coercions that the fixtures cannot exercise.
# --------------------------------------------------------------------------- #


class _FakeEventType:
    """Stands in for ``copilot.SessionEventType`` (an Enum with a .value)."""

    def __init__(self, value):
        self.value = value


class _FakeData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeEvent:
    def __init__(self, type_value, **data):
        self.type = _FakeEventType(type_value)
        self.data = _FakeData(**data)
        self.agent_id = None
        self.timestamp = "2026-08-27T12:18:55.041000+00:00"


def test_event_type_coerces_enum_to_string():
    """The live binding yields SessionEventType.SESSION_START, not a string.

    Without coercion every comparison silently fails and the integration
    produces no traces at all -- which fixture-only tests cannot catch.
    """
    assert copilot_sdk._event_type(_FakeEvent("session.start")) == "session.start"
    assert copilot_sdk._event_type({"type": "session.idle"}) == "session.idle"
    assert copilot_sdk._event_type(object()) == ""


def test_collector_handles_enum_typed_dataclass_events():
    collector = copilot_sdk.CopilotTraceCollector()
    collector.handle(_FakeEvent("session.start", session_id="sess-1"))
    assert collector.session_id == "sess-1", "enum-typed events must be understood"


def test_jsonable_flattens_dataclasses_and_enums():
    """Tool results arrive as SDK dataclasses; they must survive JSON encoding."""
    result = _FakeData(content="ok", kind=_FakeEventType("text"), nested=_FakeData(n=1))
    flat = copilot_sdk._jsonable(result)
    assert flat["content"] == "ok"
    assert flat["nested"] == {"n": 1}
    json.dumps(flat)  # must not raise


# --------------------------------------------------------------------------- #
# trace_copilot() -- the one-line entry point the docs lead with.
#
# The Copilot SDK requires Python 3.11+, so these drive a stand-in module
# registered in sys.modules rather than the real package.
# --------------------------------------------------------------------------- #


def _run(coro):
    """Run a coroutine on a fresh loop and close it (no ResourceWarning)."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeCopilotClient:
    """Minimal stand-in for ``copilot.CopilotClient``."""

    last_on_event = None

    async def create_session(self, **kwargs):
        type(self).last_on_event = kwargs.get("on_event")
        return "session"


@pytest.fixture
def fake_copilot_module(monkeypatch):
    """Register a fake ``copilot`` module and undo any patch afterwards."""
    import sys
    import types
    import importlib.machinery

    module = types.ModuleType("copilot")
    module.CopilotClient = _FakeCopilotClient
    # _auto probes with importlib.util.find_spec, which raises ValueError for a
    # sys.modules entry whose __spec__ is None -- so the stub needs one to be
    # seen as "installed".
    module.__spec__ = importlib.machinery.ModuleSpec("copilot", None)
    monkeypatch.setitem(sys.modules, "copilot", module)
    _FakeCopilotClient.last_on_event = None
    original = _FakeCopilotClient.create_session
    try:
        yield module
    finally:
        copilot_sdk.untrace_copilot()
        _FakeCopilotClient.create_session = original


def test_trace_copilot_patches_create_session(fake_copilot_module):
    original = fake_copilot_module.CopilotClient.create_session
    copilot_sdk.trace_copilot()
    assert fake_copilot_module.CopilotClient.create_session is not original


def test_trace_copilot_is_idempotent(fake_copilot_module):
    copilot_sdk.trace_copilot()
    patched_once = fake_copilot_module.CopilotClient.create_session
    copilot_sdk.trace_copilot()
    assert fake_copilot_module.CopilotClient.create_session is patched_once


def test_trace_copilot_preserves_a_user_supplied_on_event(fake_copilot_module):
    """Patching must not silently drop the caller's own handler."""
    seen = []
    copilot_sdk.trace_copilot()

    client = fake_copilot_module.CopilotClient()
    _run(client.create_session(on_event=seen.append))

    composed = _FakeCopilotClient.last_on_event
    assert composed is not None
    assert composed is not seen.append, "our handler must wrap, not replace"
    composed({"type": "session.start", "data": {"session_id": "s1"}})
    assert len(seen) == 1, "the user's handler still receives every event"


def test_trace_copilot_traces_sessions_created_after_patching(fake_copilot_module, published_traces):
    """End-to-end through the patch: events fed to the injected handler
    must produce a published trace."""
    copilot_sdk.trace_copilot()
    client = fake_copilot_module.CopilotClient()
    _run(client.create_session())

    handler = _FakeCopilotClient.last_on_event
    assert handler is not None
    for event in load_fixture("session_basic.jsonl"):
        handler(event)

    assert len(published_traces) == 1
    assert published_traces[0][0]["name"] == copilot_sdk.ROOT_STEP_NAME


def test_untrace_copilot_restores_the_original(fake_copilot_module):
    original = fake_copilot_module.CopilotClient.create_session
    copilot_sdk.trace_copilot()
    copilot_sdk.untrace_copilot()
    assert fake_copilot_module.CopilotClient.create_session is original


def test_jsonable_flattens_enum_values():
    """The enum branch of _jsonable, which the dataclass test did not assert."""
    flat = copilot_sdk._jsonable({"kind": _FakeEventType("text"), "n": 1})
    assert flat["kind"] == "text"
    assert flat["n"] == 1


def test_flush_publishes_an_interaction_that_never_went_idle(published_traces):
    """A session torn down mid-flight must not silently drop its buffer.

    ``session.idle`` is the normal trigger; a disconnect before it arrives
    would otherwise lose everything buffered so far.
    """
    collector = copilot_sdk.CopilotTraceCollector()
    for event in load_fixture("session_basic.jsonl"):
        if copilot_sdk._event_type(event) in ("session.idle", "session.shutdown"):
            continue  # simulate a session that never goes idle
        collector.handle(event)

    assert not published_traces, "nothing should publish before the flush"
    collector.flush()
    assert len(published_traces) == 1
    assert published_traces[0][0]["name"] == copilot_sdk.ROOT_STEP_NAME


def test_flush_is_idempotent_after_a_normal_idle(published_traces):
    collector = copilot_sdk.CopilotTraceCollector()
    for event in load_fixture("session_basic.jsonl"):
        collector.handle(event)
    assert len(published_traces) == 1
    collector.flush()  # disconnect after a normal idle
    assert len(published_traces) == 1, "flush must not double-publish"


def test_event_handler_exposes_flush():
    handler = copilot_sdk.openlayer_event_handler()
    assert callable(handler)
    assert callable(handler.flush)
    assert isinstance(handler.collector, copilot_sdk.CopilotTraceCollector)


# --------------------------------------------------------------------------- #
# Metered cost from GitHub's own AIU accounting.
#
# copilot_usage.total_nano_aiu is GitHub's authoritative figure for a call.
# Decoding copilot_usage._token_details shows GitHub's per-token rates for
# claude-haiku-4.5 are Anthropic's list prices scaled by exactly 100 across all
# four token categories, i.e. 1 AIU = $0.01 -- so total_nano_aiu / 1e11 equals
# the priced cost to 12 decimal places. See the constant's docstring for the
# caveat about how far that generalizes.
# --------------------------------------------------------------------------- #


def test_metered_cost_matches_list_price_for_the_observed_session():
    """The exact numbers captured live from claude-haiku-4.5."""
    usage = {
        "copilot_usage": {"total_nano_aiu": 1675375000.0},
        "input_tokens": 12509,
        "output_tokens": 224,
        "cache_write_tokens": 12499,
        "cache_read_tokens": 0,
    }
    metered = copilot_sdk._metered_cost_usd(usage)
    # Anthropic list price for the same partition:
    #   10*1e-6 + 0*1e-7 + 12499*1.25e-6 + 224*5e-6
    assert metered == pytest.approx(0.01675375, abs=1e-9)


def test_metered_cost_is_none_without_aiu():
    assert copilot_sdk._metered_cost_usd({}) is None
    assert copilot_sdk._metered_cost_usd({"copilot_usage": {}}) is None
    assert copilot_sdk._metered_cost_usd({"copilot_usage": {"total_nano_aiu": 0}}) is None


def test_mapped_provider_leaves_cost_for_openlayer_to_price(published_traces):
    """When we know the provider, Openlayer prices it and we get costDetails."""
    run_fixture("session_basic.jsonl")
    chat = published_traces[-1][0]["steps"][0]
    assert chat["provider"] == "anthropic"
    assert chat["cost"] is None, "leave it unset so the backend prices per category"
    # ...but the metered figure is still recorded as a cross-check.
    assert chat["metadata"]["copilot_metered_cost_usd"] == pytest.approx(0.01675375, abs=1e-9)


def test_unmapped_model_falls_back_to_the_metered_cost(published_traces):
    """A model we have no prefix for must land priced, not at $0."""
    events = load_fixture("session_basic.jsonl")
    for event in events:
        data = event.get("data") or {}
        if "model" in data:
            data["model"] = "some-future-model-9"

    collector = copilot_sdk.CopilotTraceCollector()
    for event in events:
        collector.handle(event)

    chat = published_traces[-1][0]["steps"][0]
    assert chat["provider"] is None, "we must not guess a provider"
    assert chat["cost"] == pytest.approx(0.01675375, abs=1e-9), (
        "an unmapped model should still carry GitHub's own metered cost"
    )


def test_metered_cost_reads_a_dataclass_shaped_copilot_usage():
    """Live events nest ``copilot_usage`` as a dataclass, not a dict.

    The fixtures nest plain dicts, so an isinstance(..., dict) guard passes
    every fixture test while silently dropping the real thing.
    """
    usage = {"copilot_usage": _FakeData(total_nano_aiu=1675375000.0)}
    assert copilot_sdk._metered_cost_usd(usage) == pytest.approx(0.01675375, abs=1e-9)


def test_metered_cost_holds_for_a_second_vendor():
    """Verified live on gpt-5-mini: GitHub bills 25/200/2.5 AIU per 1M
    input/output/cache-read, i.e. OpenAI's $0.25/$2.00/$0.025 list prices.
    So 1 AIU = $0.01 is not an Anthropic-only coincidence."""
    metered = copilot_sdk._metered_cost_usd({"copilot_usage": {"total_nano_aiu": 55665000.0}})
    expected = 449 * 0.25e-6 + 9216 * 0.025e-6 + 107 * 2.0e-6
    assert metered == pytest.approx(expected, abs=1e-9)


def test_patch_does_not_double_trace_when_user_passes_our_own_handler(
    fake_copilot_module, published_traces
):
    """Mixing trace_copilot() with an explicit openlayer_event_handler() must
    not publish the interaction twice.

    Both are documented entry points, so a user following the quickstart and
    then copying the "trace specific sessions" snippet ends up with both active
    on the same session. Without a guard that yields two collectors and two
    identical rows per send.
    """
    copilot_sdk.trace_copilot()
    client = fake_copilot_module.CopilotClient()
    _run(client.create_session(on_event=copilot_sdk.openlayer_event_handler()))

    handler = _FakeCopilotClient.last_on_event
    for event in load_fixture("session_basic.jsonl"):
        handler(event)

    assert len(published_traces) == 1, "the interaction must publish exactly once"


# --------------------------------------------------------------------------- #
# Auto-instrumentation: openlayer.lib.init() should pick this up like any other
# supported SDK, so users need not know a Copilot-specific function name.
# --------------------------------------------------------------------------- #


def test_copilot_is_registered_for_auto_instrumentation():
    from openlayer.lib.integrations._auto import REGISTRY

    spec = next((s for s in REGISTRY if s.name == "copilot"), None)
    assert spec is not None, "init(auto_instrument=True) must cover the Copilot SDK"
    assert spec.probe == "copilot", "github-copilot-sdk imports as `copilot`"
    assert spec.unpatch is not None, "must be reversible via unpatch_all()"


def test_auto_instrument_patches_and_unpatches_copilot(fake_copilot_module):
    from openlayer.lib.integrations import _auto

    original = fake_copilot_module.CopilotClient.create_session

    results = _auto.auto_instrument(["copilot"])
    assert results == {"copilot": True}
    assert fake_copilot_module.CopilotClient.create_session is not original

    _auto.unpatch_all()
    assert fake_copilot_module.CopilotClient.create_session is original


def test_auto_instrument_skips_copilot_when_not_installed(monkeypatch):
    import sys

    from openlayer.lib.integrations import _auto

    monkeypatch.setitem(sys.modules, "copilot", None)
    monkeypatch.setattr(_auto, "_is_installed", lambda probe: probe != "copilot")
    assert _auto.auto_instrument(["copilot"]) == {"copilot": False}
