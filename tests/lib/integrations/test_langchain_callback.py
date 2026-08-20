"""Tests for the Openlayer LangChain callback handler.

Regression coverage for OPEN-11315:

1. The synchronous ``OpenlayerHandler`` must wire up the retriever callbacks
   (``on_retriever_start`` / ``on_retriever_end`` / ``on_retriever_error``) so
   synchronous RAG pipelines produce a RETRIEVER step and the auto-populated
   ``context`` column on the trace.

2. A chat-completion turn whose response contains ONLY tool calls (every agent
   iteration in LangGraph / ``create_agent``) must still produce a non-empty
   step output: ``_extract_output`` falls back to serializing the message's
   tool calls, and ``_message_to_dict`` preserves ``tool_calls`` on the
   assistant message.
"""

# The handler under test lives in ``src/openlayer/lib`` (excluded from pyright
# via ``[tool.pyright] ignore``) and depends on LangChain, an OPTIONAL
# integration that is not installed in the lint/type-check environment. So its
# imports don't resolve there and its dynamic types surface as ``Unknown``. Use
# basic mode and disable the missing-import diagnostic for this test module;
# the runtime is guarded by ``pytest.importorskip`` below.
# pyright: basic, reportMissingImports=false

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.documents import Document

from openlayer.lib.tracing import enums, steps
from openlayer.lib.integrations.langchain_callback import (
    LS_PROVIDER_TO_OPENLAYER_MAP,
    LITELLM_PREFIX_TO_PROVIDER_MAP,
    LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP,
    OpenlayerHandler,
)


@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep all tracer publish paths off during the test (fully offline)."""
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")

    from openlayer.lib.tracing import tracer as _tracer

    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    """Build a langchain_core tool-call dict."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --------------------------------------------------------------------------- #
# Fix 2 (b): _message_to_dict preserves tool_calls
# --------------------------------------------------------------------------- #
class TestMessageToDict:
    def test_preserves_tool_calls_on_ai_message(self) -> None:
        handler = OpenlayerHandler()
        tool_calls = [_tool_call("search", {"query": "openlayer"}, "call_1")]
        message = AIMessage(content="", tool_calls=tool_calls)

        result = handler._message_to_dict(message)

        assert result["role"] == "assistant"
        assert "tool_calls" in result, "tool_calls should be preserved"
        assert result["tool_calls"] == tool_calls

    def test_human_message_without_tool_calls_is_backwards_compatible(self) -> None:
        handler = OpenlayerHandler()
        message = HumanMessage(content="hello there")

        result = handler._message_to_dict(message)

        assert result == {"role": "user", "content": "hello there"}
        assert "tool_calls" not in result

    def test_ai_message_without_tool_calls_omits_key(self) -> None:
        handler = OpenlayerHandler()
        message = AIMessage(content="plain answer")

        result = handler._message_to_dict(message)

        assert result == {"role": "assistant", "content": "plain answer"}
        assert "tool_calls" not in result

    def test_tool_message_does_not_raise(self) -> None:
        handler = OpenlayerHandler()
        message = ToolMessage(content="42", tool_call_id="call_1")

        result = handler._message_to_dict(message)

        assert result["content"] == "42"
        assert "tool_calls" not in result


# --------------------------------------------------------------------------- #
# Fix 2 (a): _extract_output falls back to tool calls
# --------------------------------------------------------------------------- #
class TestExtractOutput:
    def test_text_generation_returns_text(self) -> None:
        handler = OpenlayerHandler()
        gen = ChatGeneration(message=AIMessage(content="Hello world"))
        response = LLMResult(generations=[[gen]])

        assert handler._extract_output(response) == "Hello world"

    def test_tool_only_generation_returns_tool_calls(self) -> None:
        handler = OpenlayerHandler()
        tool_calls = [_tool_call("get_weather", {"city": "Rio"}, "call_42")]
        message = AIMessage(content="", tool_calls=tool_calls)
        gen = ChatGeneration(message=message)
        response = LLMResult(generations=[[gen]])

        output = handler._extract_output(response)

        assert output, "tool-only output must not be empty"
        assert "get_weather" in output

    def test_tool_only_generation_via_callbacks(self) -> None:
        """Drive on_chat_model_start / on_llm_end and assert step output is set."""
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()

        handler.on_chat_model_start(
            serialized={"name": "gpt-4o"},
            messages=[[HumanMessage(content="What is the weather in Rio?")]],
            run_id=run_id,
            invocation_params={"model_name": "gpt-4o", "_type": "openai-chat"},
        )

        # Capture the standalone trace before the root step is ended/popped.
        trace = handler._traces_by_root[run_id]
        step = handler.steps[run_id]
        assert isinstance(step, steps.ChatCompletionStep)

        tool_calls = [_tool_call("get_weather", {"city": "Rio"}, "call_42")]
        message = AIMessage(content="", tool_calls=tool_calls)
        response = LLMResult(generations=[[ChatGeneration(message=message)]])

        handler.on_llm_end(response, run_id=run_id)

        assert step.step_type == enums.StepType.CHAT_COMPLETION
        assert step.output, "tool-only chat completion output must not be empty"
        assert "get_weather" in step.output
        assert trace.steps[0] is step


# --------------------------------------------------------------------------- #
# Fix 1: sync handler wires retriever callbacks
# --------------------------------------------------------------------------- #
class TestSyncRetrieverCallbacks:
    def test_handler_exposes_retriever_callbacks(self) -> None:
        handler = OpenlayerHandler()
        assert hasattr(handler, "on_retriever_start")
        assert hasattr(handler, "on_retriever_end")
        assert hasattr(handler, "on_retriever_error")

    def test_sync_retriever_run_produces_step_and_context(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()

        handler.on_retriever_start(
            serialized={"id": ["langchain", "retrievers", "VectorStoreRetriever"]},
            query="what is openlayer?",
            run_id=run_id,
        )

        trace = handler._traces_by_root[run_id]
        step = handler.steps[run_id]
        assert isinstance(step, steps.RetrieverStep)
        assert step.step_type == enums.StepType.RETRIEVER
        assert step.inputs == {"query": "what is openlayer?"}

        documents = [
            Document(page_content="Openlayer is an evaluation platform."),
            Document(page_content="It supports LangChain tracing."),
        ]
        handler.on_retriever_end(documents, run_id=run_id)

        # The retriever step captured the documents...
        assert step.documents == [
            "Openlayer is an evaluation platform.",
            "It supports LangChain tracing.",
        ]
        # ...and the trace gained the auto-populated `context` metadata.
        assert trace.metadata is not None
        assert trace.metadata.get("context") == [
            "Openlayer is an evaluation platform.",
            "It supports LangChain tracing.",
        ]

    def test_sync_retriever_respects_ignore_flag(self) -> None:
        handler = OpenlayerHandler(ignore_retriever=True)
        run_id = uuid.uuid4()

        handler.on_retriever_start(
            serialized={"id": ["VectorStoreRetriever"]},
            query="ignored",
            run_id=run_id,
        )

        assert run_id not in handler.steps

    def test_sync_retriever_error(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()

        handler.on_retriever_start(
            serialized={"id": ["VectorStoreRetriever"]},
            query="boom",
            run_id=run_id,
        )
        step = handler.steps[run_id]

        handler.on_retriever_error(ValueError("retrieval failed"), run_id=run_id)

        assert step.metadata.get("error") == "retrieval failed"
        assert run_id not in handler.steps


# --------------------------------------------------------------------------- #
# OPEN-11315 (medium/low items)
# --------------------------------------------------------------------------- #


def _ai_message_with_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    input_token_details: dict | None = None,
    output_token_details: dict | None = None,
    content: str = "done",
) -> AIMessage:
    """Build an AIMessage carrying standardized usage_metadata (v1 shape)."""
    usage: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if input_token_details is not None:
        usage["input_token_details"] = input_token_details
    if output_token_details is not None:
        usage["output_token_details"] = output_token_details
    return AIMessage(content=content, usage_metadata=usage)


# --------------------------------------------------------------------------- #
# Item 1: usage_metadata-first token extraction + token details
# --------------------------------------------------------------------------- #
class TestTokenExtraction:
    def test_usage_metadata_is_read_first(self) -> None:
        """usage_metadata on the message wins over llm_output token_usage."""
        handler = OpenlayerHandler()
        message = _ai_message_with_usage(input_tokens=11, output_tokens=7, total_tokens=18)
        response = LLMResult(
            generations=[[ChatGeneration(message=message)]],
            # Divergent llm_output that must NOT be used when usage_metadata exists.
            llm_output={
                "token_usage": {
                    "prompt_tokens": 999,
                    "completion_tokens": 999,
                    "total_tokens": 999,
                }
            },
        )

        info = handler._extract_token_info(response)

        assert info["prompt_tokens"] == 11
        assert info["completion_tokens"] == 7
        assert info["tokens"] == 18

    def test_captures_input_and_output_token_details(self) -> None:
        handler = OpenlayerHandler()
        message = _ai_message_with_usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            input_token_details={"cache_read": 80, "cache_creation": 10},
            output_token_details={"reasoning": 25, "audio": 0},
        )
        response = LLMResult(generations=[[ChatGeneration(message=message)]])

        info = handler._extract_token_info(response)

        assert info["tokens"] == 140
        details = info["token_details"]
        assert details["input_token_details"] == {
            "cache_read": 80,
            "cache_creation": 10,
        }
        assert details["output_token_details"] == {"reasoning": 25, "audio": 0}

    def test_no_token_details_when_absent(self) -> None:
        handler = OpenlayerHandler()
        message = _ai_message_with_usage(input_tokens=1, output_tokens=1, total_tokens=2)
        response = LLMResult(generations=[[ChatGeneration(message=message)]])

        info = handler._extract_token_info(response)

        assert "token_details" not in info

    def test_falls_back_to_llm_output_token_usage(self) -> None:
        """When no usage_metadata, the legacy llm_output path still works."""
        handler = OpenlayerHandler()
        gen = ChatGeneration(message=AIMessage(content="hi"))
        response = LLMResult(
            generations=[[gen]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                }
            },
        )

        info = handler._extract_token_info(response)

        assert info == {"prompt_tokens": 5, "completion_tokens": 3, "tokens": 8}

    def test_falls_back_to_ollama_generation_info(self) -> None:
        handler = OpenlayerHandler()
        gen = ChatGeneration(
            message=AIMessage(content="hi"),
            generation_info={"prompt_eval_count": 12, "eval_count": 4},
        )
        response = LLMResult(generations=[[gen]])

        info = handler._extract_token_info(response)

        assert info == {"prompt_tokens": 12, "completion_tokens": 4, "tokens": 16}

    def test_falls_back_to_google_generation_info(self) -> None:
        handler = OpenlayerHandler()
        gen = ChatGeneration(
            message=AIMessage(content="hi"),
            generation_info={
                "usage_metadata": {
                    "prompt_token_count": 9,
                    "candidates_token_count": 6,
                    "total_token_count": 15,
                }
            },
        )
        response = LLMResult(generations=[[gen]])

        info = handler._extract_token_info(response)

        assert info == {"prompt_tokens": 9, "completion_tokens": 6, "tokens": 15}

    def test_token_details_surface_on_step_metadata_via_callbacks(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()

        handler.on_chat_model_start(
            serialized={"name": "claude"},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            invocation_params={"model_name": "claude", "_type": "openai-chat"},
            metadata={"ls_provider": "anthropic"},
        )
        step = handler.steps[run_id]
        assert isinstance(step, steps.ChatCompletionStep)

        message = _ai_message_with_usage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            input_token_details={"cache_read": 80},
        )
        response = LLMResult(generations=[[ChatGeneration(message=message)]])

        handler.on_llm_end(response, run_id=run_id)

        assert step.prompt_tokens == 100
        assert step.completion_tokens == 40
        assert step.tokens == 140
        assert step.metadata["token_details"]["input_token_details"] == {"cache_read": 80}


# --------------------------------------------------------------------------- #
# Item 2: provider detection via ls_provider
# --------------------------------------------------------------------------- #
class TestLsProviderDetection:
    def test_ls_provider_is_primary_source(self) -> None:
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            # _type would map to OpenAI; ls_provider must win.
            invocation_params={"_type": "openai-chat", "model_name": "claude-x"},
            metadata={"ls_provider": "anthropic"},
        )
        assert info["provider"] == "Anthropic"
        assert info["model"] == "claude-x"

    def test_ls_provider_title_cases_unknown_values(self) -> None:
        """Unknown ls_provider values are title-cased but keep their separators.

        Underscores must survive: the cost table is keyed by slugs like
        ``vercel_ai_gateway``, and replacing "_" with " " makes the provider
        un-priceable. See TestProviderCostSlugs.
        """
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "some-model"},
            metadata={"ls_provider": "some_new_provider"},
        )
        assert info["provider"] == "Some_New_Provider"

    def test_falls_back_to_type_map_without_ls_provider(self) -> None:
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"_type": "chat-ollama", "model": "llama3"},
            metadata={},
        )
        assert info["provider"] == "Ollama"

    def test_litellm_prefix_overrides_ls_provider(self) -> None:
        """A LiteLLM proxy reports ls_provider=openai but routes elsewhere."""
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "gemini/gemini-2.5-flash"},
            metadata={"ls_provider": "openai"},
        )
        assert info["provider"] == "Google"
        assert info["model"] == "gemini-2.5-flash"

    def test_chat_model_step_named_from_ls_provider(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            serialized={"name": "model"},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            invocation_params={"_type": "openai-chat"},
            metadata={"ls_provider": "groq"},
        )
        step = handler.steps[run_id]
        assert isinstance(step, steps.ChatCompletionStep)
        assert step.name == "Groq Chat Completion"
        assert step.provider == "Groq"


# --------------------------------------------------------------------------- #
# Item 3: LangGraph metadata (langgraph_node + thread_id -> session_id)
# --------------------------------------------------------------------------- #
class TestLangGraphMetadata:
    def test_langgraph_node_names_chain_step_when_no_explicit_name(self) -> None:
        # When the runnable carries no name, fall back to langgraph_node so graph
        # nodes stay identifiable.
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["langgraph", "utils", "RunnableCallable"]},
            inputs={"messages": []},
            run_id=run_id,
            metadata={"langgraph_node": "agent"},
        )
        step = handler.steps[run_id]
        assert step.name == "agent"

    def test_explicit_name_wins_over_langgraph_node(self) -> None:
        # An explicit runnable name takes precedence over langgraph_node (matches
        # the TS handler's `name ?? langgraph_node ?? id`). langgraph_node is
        # inherited by every run nested inside a node, so preferring it would
        # relabel a node's internal LCEL runs with the node name.
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["langchain_core", "runnables", "RunnableSequence"]},
            inputs={"messages": []},
            run_id=run_id,
            name="RunnableSequence",
            metadata={"langgraph_node": "agent"},
        )
        step = handler.steps[run_id]
        assert step.name == "RunnableSequence"

    def test_chain_name_unchanged_without_langgraph_node(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["langchain", "chains", "LLMChain"]},
            inputs={},
            run_id=run_id,
        )
        step = handler.steps[run_id]
        assert step.name == "LLMChain"

    def test_thread_id_maps_to_session_id(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["graph"]},
            inputs={},
            run_id=run_id,
            metadata={"langgraph_node": "agent", "thread_id": "thread-123"},
        )
        trace = handler._traces_by_root[run_id]
        assert trace.metadata is not None
        assert trace.metadata.get("session_id") == "thread-123"

    def test_thread_id_does_not_clobber_explicit_session(self) -> None:
        from openlayer.lib.tracing.context import UserSessionContext

        UserSessionContext.set_session_id("explicit-session")
        try:
            handler = OpenlayerHandler()
            run_id = uuid.uuid4()
            handler.on_chain_start(
                serialized={"id": ["graph"]},
                inputs={},
                run_id=run_id,
                metadata={"thread_id": "thread-999"},
            )
            trace = handler._traces_by_root[run_id]
            session_in_metadata = (trace.metadata or {}).get("session_id")
            assert session_in_metadata != "thread-999"
        finally:
            UserSessionContext.clear_context()

    def test_thread_id_mapping_opt_out(self) -> None:
        handler = OpenlayerHandler(map_thread_id_to_session=False)
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"id": ["graph"]},
            inputs={},
            run_id=run_id,
            metadata={"thread_id": "thread-123"},
        )
        trace = handler._traces_by_root[run_id]
        assert (trace.metadata or {}).get("session_id") is None


# --------------------------------------------------------------------------- #
# Item 4: v1 content blocks in _message_to_dict
# --------------------------------------------------------------------------- #
class TestV1ContentBlocks:
    def test_plain_string_content_unchanged(self) -> None:
        handler = OpenlayerHandler()
        result = handler._message_to_dict(HumanMessage(content="hello"))
        assert result == {"role": "user", "content": "hello"}
        assert "content_blocks" not in result

    def test_list_of_text_blocks_joined(self) -> None:
        handler = OpenlayerHandler()
        message = AIMessage(
            content=[
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
            ]
        )
        result = handler._message_to_dict(message)
        assert result["content"] == "hello world"
        assert "content_blocks" not in result

    def test_non_text_blocks_preserved_structurally(self) -> None:
        handler = OpenlayerHandler()
        reasoning_block = {"type": "reasoning", "reasoning": "thinking..."}
        message = AIMessage(content=[{"type": "text", "text": "answer"}, reasoning_block])
        result = handler._message_to_dict(message)
        assert result["content"] == "answer"
        assert result["content_blocks"] == [reasoning_block]

    def test_list_of_plain_strings(self) -> None:
        handler = OpenlayerHandler()
        message = AIMessage(content=["foo", "bar"])
        result = handler._message_to_dict(message)
        assert result["content"] == "foobar"
        assert "content_blocks" not in result


# --------------------------------------------------------------------------- #
# Item 5: import simplification keeps HAVE_LANGCHAIN working
# --------------------------------------------------------------------------- #
class TestImportSimplification:
    def test_have_langchain_true_and_schema_from_core(self) -> None:
        from openlayer.lib.integrations import langchain_callback as lc

        assert lc.HAVE_LANGCHAIN is True
        # The schema alias must resolve to langchain_core, not the legacy path.
        assert lc.langchain_schema.__name__.startswith("langchain_core")


# --------------------------------------------------------------------------- #
# OPEN-11695: model name normalization + priced usageDetails/costDetails
# --------------------------------------------------------------------------- #
class TestModelNameNormalization:
    def test_strips_gemini_models_prefix(self) -> None:
        # Customer case: ChatGoogleGenerativeAI reports ls_provider=google_genai
        # and a model with the Gemini Developer API "models/" prefix. Without the
        # strip the cost table lookup misses -> $0.
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={
                "_type": "chat-google-generative-ai",
                "model": "models/gemini-3.5-flash",
            },
            metadata={
                "ls_provider": "google_genai",
                "ls_model_name": "models/gemini-3.5-flash",
            },
        )
        assert info["provider"] == "Google"
        assert info["model"] == "gemini-3.5-flash"

    def test_models_prefix_strip_independent_of_provider(self) -> None:
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "models/gemini-2.5-flash"},
            metadata={},
        )
        assert info["model"] == "gemini-2.5-flash"

    def test_type_anchor_maps_when_ls_provider_absent(self) -> None:
        # OPEN-11695: the exact reported _type string must resolve to Google even
        # without metadata["ls_provider"] (older langchain-google-genai etc.).
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={
                "_type": "chat-google-generative-ai",
                "model": "models/gemini-3.5-flash",
            },
            metadata={},
        )
        assert info["provider"] == "Google"
        assert info["model"] == "gemini-3.5-flash"


class TestUsageDetailsPricing:
    def test_scalar_partition_when_no_details(self) -> None:
        assert OpenlayerHandler._build_usage_details(100, 50) == {
            "input_tokens": 100,
            "output_tokens": 50,
        }

    def test_none_when_no_tokens(self) -> None:
        assert OpenlayerHandler._build_usage_details(0, 0) is None

    def test_cached_tokens_partitioned_non_overlapping(self) -> None:
        # LangChain reports input_tokens INCLUSIVE of cache_read/cache_creation;
        # the backend prices a non-overlapping partition under its own keys.
        details = OpenlayerHandler._build_usage_details(350, 100, {"cache_read": 100, "cache_creation": 200}, None)
        assert details is not None
        assert details == {
            "input_tokens": 50,  # 350 - 100 - 200
            "output_tokens": 100,
            "cached_tokens": 100,
            "cache_creation_tokens": 200,
        }
        assert sum(details.values()) == 350 + 100  # partition conserves total

    def test_audio_broken_out_both_directions(self) -> None:
        details = OpenlayerHandler._build_usage_details(300, 120, {"audio": 30}, {"audio": 20})
        assert details == {
            "input_tokens": 270,
            "output_tokens": 100,
            "audio_input_tokens": 30,
            "audio_output_tokens": 20,
        }

    def test_reasoning_stays_folded_into_output(self) -> None:
        # Reasoning is billed at the output rate; must not be split out or
        # subtracted from output_tokens.
        details = OpenlayerHandler._build_usage_details(100, 240, None, {"reasoning": 200})
        assert details == {"input_tokens": 100, "output_tokens": 240}

    def test_usage_details_set_on_step_via_callbacks(self) -> None:
        handler = OpenlayerHandler()
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            serialized={"name": "gemini"},
            messages=[[HumanMessage(content="hi")]],
            run_id=run_id,
            invocation_params={"model": "models/gemini-3.5-flash"},
            metadata={"ls_provider": "google_genai"},
        )
        step = handler.steps[run_id]
        assert isinstance(step, steps.ChatCompletionStep)
        message = _ai_message_with_usage(
            input_tokens=27131,
            output_tokens=17739,
            total_tokens=44870,
            input_token_details={"cache_read": 10000},
        )
        handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
        # Priced, non-overlapping partition lands on the step (and serializes as
        # the "usageDetails" column the backend prices into costDetails).
        assert step.usage_details == {
            "input_tokens": 17131,
            "output_tokens": 17739,
            "cached_tokens": 10000,
        }
        assert step.to_dict()["usageDetails"] == step.usage_details

    def test_step_omits_usage_details_when_unset(self) -> None:
        assert "usageDetails" not in steps.ChatCompletionStep(name="x").to_dict()


# --------------------------------------------------------------------------- #
# Provider values must be matchable against the llm-costs table
# --------------------------------------------------------------------------- #
class TestProviderCostSlugs:
    """Cost is resolved by lowercasing ``provider`` and matching a llm-costs slug.

    Matching normalizes case but NOT separators (verified against the live API:
    provider "Together AI" priced at $0.0, while "Together_AI" and "together_ai"
    both priced at $0.0065 for the same model and token counts). So any provider
    value containing a space silently prices every row at zero.
    """

    ALL_MAPS = (
        ("LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP", LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP),
        ("LS_PROVIDER_TO_OPENLAYER_MAP", LS_PROVIDER_TO_OPENLAYER_MAP),
        ("LITELLM_PREFIX_TO_PROVIDER_MAP", LITELLM_PREFIX_TO_PROVIDER_MAP),
    )

    # Provider slugs published by https://llm-costs.openlayer.com/v1/costs that these
    # maps target. Vendored rather than fetched so the suite stays offline; refresh with
    #   curl -s https://llm-costs.openlayer.com/v1/costs | jq -r '.costs[].provider' | sort -u
    # Last verified 2026-08-20 against 139 published providers.
    COST_SLUGS = frozenset(
        {
            "anthropic",
            "azure",
            "bedrock",
            "cohere",
            "deepseek",
            "fireworks_ai",
            "google",
            "groq",
            "mistral",
            "ollama",
            "openai",
            "perplexity",
            "replicate",
            "together_ai",
        }
    )

    # Vendors with no provider slug upstream at all: no value can resolve a cost, so the
    # name is display-only. Mirrors the ``null`` entries in openlayer-ts's
    # PROVIDER_COST_SLUG.
    UNPRICED_VENDORS = frozenset({"huggingface"})

    def test_every_provider_value_resolves_a_cost_slug(self) -> None:
        """Stronger than the space check: the value must actually price something.

        Ports the invariant openlayer-ts asserts via PROVIDER_COST_SLUG -- a provider
        is only worth mapping if its canonical name resolves a price, otherwise the
        step gets a nicer label and still costs $0.
        """
        for name, mapping in self.ALL_MAPS:
            for key, value in sorted(mapping.items()):
                slug = value.lower()
                assert slug in self.COST_SLUGS or slug in self.UNPRICED_VENDORS, (
                    f"{name}[{key!r}] = {value!r} lowercases to {slug!r}, which is neither a "
                    f"published cost slug nor a known-unpriced vendor: every row would cost $0"
                )

    def test_no_provider_value_contains_a_space(self) -> None:
        for name, mapping in self.ALL_MAPS:
            offenders = sorted({v for v in mapping.values() if " " in v})
            assert not offenders, f"{name} maps to space-separated values (silent $0): {offenders}"

    @pytest.mark.parametrize(
        ("prefix", "expected_slug"),
        [
            # regressions this class was added for
            ("together_ai", "together_ai"),
            ("fireworks_ai", "fireworks_ai"),
            ("huggingface", "huggingface"),
            # canaries: these already resolved correctly and must keep doing so
            ("gemini", "google"),
            ("vertex_ai", "google"),
            ("anthropic", "anthropic"),
            ("bedrock", "bedrock"),
            ("deepseek", "deepseek"),
        ],
    )
    def test_litellm_prefix_resolves_to_cost_slug(self, prefix: str, expected_slug: str) -> None:
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": f"{prefix}/some-model"},
            metadata={},
        )
        assert info["provider"] is not None
        assert info["provider"].lower() == expected_slug
        assert info["model"] == "some-model"

    @pytest.mark.parametrize(
        ("ls_provider", "expected_slug"),
        [
            ("together", "together_ai"),
            ("fireworks", "fireworks_ai"),
            ("huggingface", "huggingface"),
            ("anthropic", "anthropic"),
            ("google_genai", "google"),
        ],
    )
    def test_ls_provider_resolves_to_cost_slug(self, ls_provider: str, expected_slug: str) -> None:
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "some-model"},
            metadata={"ls_provider": ls_provider},
        )
        assert info["provider"] is not None
        assert info["provider"].lower() == expected_slug

    def test_unmapped_ls_provider_preserves_slug_separators(self) -> None:
        """An unmapped ls_provider that IS already a valid slug must stay matchable."""
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "some-model"},
            metadata={"ls_provider": "vercel_ai_gateway"},
        )
        assert info["provider"] is not None
        assert info["provider"].lower() == "vercel_ai_gateway"
