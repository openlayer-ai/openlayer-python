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
from openlayer.lib.integrations.langchain_callback import OpenlayerHandler


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
        handler = OpenlayerHandler()
        info = handler._extract_model_info(
            serialized={},
            invocation_params={"model": "some-model"},
            metadata={"ls_provider": "some_new_provider"},
        )
        assert info["provider"] == "Some New Provider"

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
