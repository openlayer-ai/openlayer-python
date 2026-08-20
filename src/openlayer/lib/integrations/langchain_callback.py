"""Module with the Openlayer callback handler for LangChain."""

# pylint: disable=unused-argument
import contextvars
import time
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

try:
    # langchain-core is sufficient for this handler. The legacy
    # ``langchain.schema`` / ``langchain.callbacks.base`` import paths were
    # removed in LangChain v1, so we import from ``langchain_core`` directly.
    from langchain_core import messages as langchain_schema
    from langchain_core.callbacks.base import (
        AsyncCallbackHandler,
        BaseCallbackHandler,
    )

    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False


from .. import utils
from ..tracing import enums, steps, tracer, traces

# Provider values below must stay matchable against the llm-costs table: cost is
# resolved by lowercasing the provider and matching a slug exactly. Matching
# normalizes case but NOT separators, so "Together AI" prices every row at $0
# while "Together_AI" prices correctly. Never use spaces in these values.
LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP = {
    "azure-openai-chat": "Azure",
    "openai-chat": "OpenAI",
    "chat-ollama": "Ollama",
    "vertexai": "Google",
    "amazon_bedrock_converse_chat": "Bedrock",
    # ChatGoogleGenerativeAI's _llm_type. Anchored here as well as via
    # ls_provider ("google_genai") so the exact reported string (OPEN-11695)
    # still resolves when metadata["ls_provider"] is absent (older
    # langchain-google-genai, or callers that don't forward it).
    "chat-google-generative-ai": "Google",
}

# LangChain v1 injects a standardized ``metadata["ls_provider"]`` (e.g.
# "openai", "anthropic", "google_genai", "groq"...). This maps those values to
# Openlayer's canonical provider names. Values without an explicit entry fall
# back to a title-cased version of the ls_provider string.
LS_PROVIDER_TO_OPENLAYER_MAP = {
    "openai": "OpenAI",
    "azure": "Azure",
    "azure_openai": "Azure",
    "anthropic": "Anthropic",
    "google": "Google",
    "google_genai": "Google",
    "google_vertexai": "Google",
    "vertexai": "Google",
    "cohere": "Cohere",
    "mistralai": "Mistral",
    "mistral": "Mistral",
    "bedrock": "Bedrock",
    "amazon_bedrock": "Bedrock",
    "ollama": "Ollama",
    "huggingface": "HuggingFace",
    "replicate": "Replicate",
    "together": "Together_AI",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "fireworks": "Fireworks_AI",
    "perplexity": "Perplexity",
}

# LiteLLM model prefixes to provider names.
# When models are accessed via a LiteLLM proxy (e.g. "gemini/gemini-2.5-flash"),
# the LangChain _type is "openai-chat" which incorrectly maps to "OpenAI".
# This map resolves the actual provider from the model prefix.
LITELLM_PREFIX_TO_PROVIDER_MAP = {
    "gemini": "Google",
    "anthropic": "Anthropic",
    "cohere": "Cohere",
    "mistral": "Mistral",
    "bedrock": "Bedrock",
    "vertex_ai": "Google",
    "azure": "Azure",
    "huggingface": "HuggingFace",
    "replicate": "Replicate",
    "together_ai": "Together_AI",
    "groq": "Groq",
    "deepseek": "DeepSeek",
    "fireworks_ai": "Fireworks_AI",
    "perplexity": "Perplexity",
    "ollama": "Ollama",
    "openai": "OpenAI",
}


if HAVE_LANGCHAIN:
    BaseCallbackHandlerClass = BaseCallbackHandler
    AsyncCallbackHandlerClass = AsyncCallbackHandler
else:
    BaseCallbackHandlerClass = object
    AsyncCallbackHandlerClass = object


class OpenlayerHandlerMixin:
    """Mixin class containing shared logic for both sync and async Openlayer
    handlers."""

    def __init__(self, **kwargs: Any) -> None:
        if not HAVE_LANGCHAIN:
            raise ImportError("LangChain library is not installed. Please install it with: pip install langchain-core")
        super().__init__()
        self.metadata: Dict[str, Any] = kwargs or {}
        self.steps: Dict[UUID, steps.Step] = {}
        self.root_steps: set[UUID] = set()  # Track which steps are root
        # Track standalone traces (consistent with async handler)
        self._traces_by_root: Dict[UUID, traces.Trace] = {}
        # Map every active run_id (root or nested) to the trace its step lives in,
        # so callbacks can find the right trace to update metadata on even when
        # nothing has set tracer._current_trace.
        self._run_id_to_trace: Dict[UUID, traces.Trace] = {}
        # Extract inference_id from kwargs if provided
        self._inference_id = kwargs.get("inference_id")
        # Extract metadata_transformer from kwargs if provided
        self._metadata_transformer = kwargs.get("metadata_transformer")
        # Opt-out flag: auto-map LangGraph metadata["thread_id"] to the trace's
        # session_id when the user has not explicitly provided one.
        self._map_thread_id_to_session = kwargs.get("map_thread_id_to_session", True)

    def _start_step(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        step_type: enums.StepType = enums.StepType.CHAT_COMPLETION,
        inputs: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **step_kwargs: Any,
    ) -> steps.Step:
        """Start a new step - use parent_run_id for proper nesting."""
        if run_id in self.steps:
            return self.steps[run_id]

        # Create the step with raw inputs and metadata
        step = steps.step_factory(
            step_type=step_type,
            name=name,
            inputs=inputs,
            metadata={**self.metadata, **(metadata or {})},
        )
        step.start_time = time.time()

        # Set step-specific attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Use parent_run_id to establish proper parent-child relationships
        if parent_run_id is not None and parent_run_id in self.steps:
            # This step has a parent - add it as a nested step
            parent_step = self.steps[parent_run_id]
            parent_step.add_nested_step(step)
            # Inherit the parent's owning trace so context lookups work.
            parent_trace = self._run_id_to_trace.get(parent_run_id)
            if parent_trace is not None:
                self._run_id_to_trace[run_id] = parent_trace
        else:
            # This is a root step - check if we're in an existing trace context
            current_step = tracer.get_current_step()
            current_trace = tracer.get_current_trace()

            if current_step is not None:
                # We're inside an existing step context - add as nested
                current_step.add_nested_step(step)
                if current_trace is not None:
                    self._run_id_to_trace[run_id] = current_trace
            elif current_trace is not None:
                # Existing trace but no current step - add to trace
                current_trace.add_step(step)
                self._run_id_to_trace[run_id] = current_trace
                # Don't track in _traces_by_root since we're using external trace
            else:
                # No existing context - create standalone trace
                trace = traces.Trace()
                trace.add_step(step)
                self._traces_by_root[run_id] = trace
                self._run_id_to_trace[run_id] = trace

            # Track root steps (those without parent_run_id)
            if parent_run_id is None:
                self.root_steps.add(run_id)
                # Override step ID with custom inference_id if provided
                if self._inference_id is not None:
                    step.id = self._inference_id

        self.steps[run_id] = step
        return step

    def _end_step(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        **step_kwargs: Any,
    ) -> None:
        """End a step and handle final processing."""
        if run_id not in self.steps:
            return

        step = self.steps.pop(run_id)
        self._run_id_to_trace.pop(run_id, None)
        is_root_step = run_id in self.root_steps

        if is_root_step:
            self.root_steps.remove(run_id)

        # Update step with final data
        if step.end_time is None:
            step.end_time = time.time()
        if step.latency is None:
            step.latency = (step.end_time - step.start_time) * 1000

        # Set raw output and additional attributes
        if output is not None:
            step.output = output  # Keep raw
        if error is not None:
            step.metadata = {**step.metadata, "error": error}

        # Set additional step attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Only upload if this is a standalone trace (not integrated with external trace)
        # If current_step is set, we're part of a larger trace and shouldn't upload
        if is_root_step and run_id in self._traces_by_root and tracer.get_current_step() is None:
            trace = self._traces_by_root.pop(run_id)
            if tracer._resolve("background_publish_enabled"):
                ctx = contextvars.copy_context()
                tracer._get_background_executor().submit(ctx.run, self._process_and_upload_trace, trace)
            else:
                self._process_and_upload_trace(trace)

    def _process_and_upload_trace(self, trace: traces.Trace) -> None:
        """Process and upload the completed trace (only for standalone root steps)."""
        if not trace:
            return

        # Convert all LangChain objects in the trace once at the end
        for step in trace.steps:
            self._convert_step_objects_recursively(step)

        trace_data, input_variable_names = tracer.post_process_trace(trace)

        config = dict(
            tracer.ConfigLlmData(
                output_column_name="output",
                input_variable_names=input_variable_names,
                latency_column_name="latency",
                cost_column_name="cost",
                timestamp_column_name="inferenceTimestamp",
                inference_id_column_name="inferenceId",
                num_of_token_column_name="tokens",
            )
        )

        # Add reserved column configurations for user context
        if "user_id" in trace_data:
            config.update({"user_id_column_name": "user_id"})
        if "session_id" in trace_data:
            config.update({"session_id_column_name": "session_id"})
        if "groundTruth" in trace_data:
            config.update({"ground_truth_column_name": "groundTruth"})
        if "context" in trace_data:
            config.update({"context_column_name": "context"})

        root_step = trace.steps[0] if trace.steps else None
        if (
            root_step
            and isinstance(root_step, steps.ChatCompletionStep)
            and root_step.inputs
            and "prompt" in root_step.inputs
        ):
            config.update({"prompt": utils.json_serialize(root_step.inputs["prompt"])})

        if tracer._publish:
            try:
                client = tracer._get_client()
                if client:
                    # Apply final JSON serialization to ensure everything is serializable
                    serialized_trace_data = utils.json_serialize(trace_data)
                    serialized_config = utils.json_serialize(config)

                    client.inference_pipelines.data.stream(
                        inference_pipeline_id=tracer.resolve_pipeline_id(),
                        rows=[serialized_trace_data],
                        config=serialized_config,
                    )
            except Exception as err:  # pylint: disable=broad-except
                tracer.logger.error("Could not stream data to Openlayer %s", err)

        # Reset trace context only for standalone traces
        tracer._current_trace.set(None)

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user-defined metadata transformation if provided."""
        if not metadata:
            return {}

        # First convert LangChain objects to JSON-serializable format
        converted_metadata = self._convert_langchain_objects(metadata)

        # Then apply custom transformer if provided
        if self._metadata_transformer:
            try:
                return self._metadata_transformer(converted_metadata)
            except Exception as e:
                # Log warning but continue with unconverted metadata
                tracer.logger.warning(f"Metadata transformer failed: {e}")
                return converted_metadata

        return converted_metadata

    def _convert_step_objects_recursively(self, step: steps.Step) -> None:
        """Convert all LangChain objects in a step and its nested steps."""
        # Convert step attributes
        if step.inputs is not None:
            step.inputs = self._convert_langchain_objects(step.inputs)
        if step.output is not None:
            # For outputs, first convert then serialize
            converted_output = self._convert_langchain_objects(step.output)
            step.output = utils.json_serialize(converted_output)
        if step.metadata is not None:
            step.metadata = self._process_metadata(step.metadata)

        # Convert nested steps recursively
        for nested_step in step.steps:
            self._convert_step_objects_recursively(nested_step)

    def _convert_langchain_objects(self, obj: Any) -> Any:
        """Recursively convert LangChain objects to JSON-serializable format."""
        # Explicit check for LangChain BaseMessage and its subclasses
        if HAVE_LANGCHAIN and isinstance(obj, langchain_schema.BaseMessage):
            return self._message_to_dict(obj)

        # Handle ChatPromptValue objects which contain messages
        if hasattr(obj, "messages") and hasattr(obj, "__class__") and "ChatPromptValue" in obj.__class__.__name__:
            return [self._convert_langchain_objects(msg) for msg in obj.messages]

        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._convert_langchain_objects(v) for k, v in obj.items()}

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_langchain_objects(item) for item in obj]

        # Handle objects with messages attribute
        if hasattr(obj, "messages"):
            return [self._convert_langchain_objects(m) for m in obj.messages]

        # Handle Pydantic model instances
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return self._convert_langchain_objects(obj.model_dump())
            except Exception:
                pass

        # Handle Pydantic model classes/metaclasses (type objects)
        if isinstance(obj, type):
            return str(obj.__name__ if hasattr(obj, "__name__") else obj)

        # Handle other LangChain objects with common attributes
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            # Many LangChain objects have a dict() method
            try:
                return self._convert_langchain_objects(obj.dict())
            except Exception:
                pass

        # Handle objects with content attribute
        if hasattr(obj, "content") and not isinstance(obj, langchain_schema.BaseMessage):
            return obj.content

        # Handle objects with value attribute
        if hasattr(obj, "value"):
            return self._convert_langchain_objects(obj.value)

        # Handle objects with kwargs attribute
        if hasattr(obj, "kwargs"):
            return self._convert_langchain_objects(obj.kwargs)

        # Return primitive types as-is
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # For everything else, convert to string
        return str(obj)

    def _message_to_dict(self, message: "langchain_schema.BaseMessage") -> Dict[str, Any]:
        """Convert a LangChain message to a JSON-serializable dictionary."""
        message_type = getattr(message, "type", "user")

        role = "user" if message_type == "human" else message_type
        if message_type == "ai":
            role = "assistant"
        elif message_type == "system":
            role = "system"

        content_str, content_blocks = self._normalize_content(message.content)
        result: Dict[str, Any] = {"role": role, "content": content_str}

        # With ``LC_OUTPUT_VERSION=v1`` the content can be a list of typed
        # blocks. Preserve any non-text blocks (reasoning, tool_call, image...)
        # structurally rather than stringifying the whole list.
        if content_blocks:
            result["content_blocks"] = content_blocks

        # Preserve tool calls on assistant messages (e.g. AIMessage.tool_calls).
        # Without this, agent turns that only emit tool calls would drop them
        # from the recorded inputs. Done defensively so it never raises on
        # messages that lack the attribute.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    @staticmethod
    def _normalize_content(content: Any) -> tuple:
        """Normalize message content into a (text, non_text_blocks) pair.

        Handles the three shapes ``message.content`` can take:
          * a plain string (returned unchanged, no blocks);
          * a list of strings (joined into the text);
          * a list of typed blocks (dicts with a ``type`` key) as produced
            under ``LC_OUTPUT_VERSION=v1`` -- text is extracted from
            ``text``-type blocks while non-text blocks are preserved
            structurally.

        Defensive: never raises on unexpected shapes.
        """
        if isinstance(content, str):
            return content, []

        if isinstance(content, list):
            text_parts: List[str] = []
            non_text_blocks: List[Any] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(str(block.get("text", "")))
                    else:
                        non_text_blocks.append(block)
                else:
                    non_text_blocks.append(block)
            return "".join(text_parts), non_text_blocks

        # Fallback for any other shape.
        return str(content), []

    def _messages_to_prompt_format(self, messages: List[List["langchain_schema.BaseMessage"]]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Openlayer prompt format using
        unified conversion."""
        prompt = []
        for message_batch in messages:
            for message in message_batch:
                prompt.append(self._message_to_dict(message))
        return prompt

    def _extract_model_info(
        self,
        serialized: Dict[str, Any],
        invocation_params: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract model information generically."""
        # Handle case where parameters can be None
        serialized = serialized or {}
        invocation_params = invocation_params or {}
        metadata = metadata or {}

        # Provider resolution order:
        #   1. metadata["ls_provider"] (standardized by LangChain v1)
        #   2. invocation_params["_type"] via the legacy _type map
        #   3. LiteLLM model prefix (handled below)
        ls_provider = metadata.get("ls_provider")
        if ls_provider:
            # Keep separators: cost is matched by lowercasing this value against a
            # llm-costs slug, and matching normalizes case but NOT separators. Turning
            # "vercel_ai_gateway" into "Vercel Ai Gateway" makes the row price at $0.
            provider = LS_PROVIDER_TO_OPENLAYER_MAP.get(ls_provider, str(ls_provider).title())
        else:
            provider = invocation_params.get("_type")
            if provider in LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP:
                provider = LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP[provider]

        model = (
            invocation_params.get("model_name")
            or invocation_params.get("model")
            or metadata.get("ls_model_name")
            or serialized.get("name")
        )

        # Strip the Google Gemini Developer API "models/" prefix
        # (e.g. "models/gemini-3.5-flash" -> "gemini-3.5-flash"). The cost table
        # stores bare Gemini names and no provider is named "models", so this is
        # safe. Runs before the LiteLLM prefix handling below so the remaining
        # name has no stray leading segment.
        if model and model.startswith("models/"):
            model = model[len("models/") :]

        # Handle LiteLLM model prefix (e.g. "gemini/gemini-2.5-flash"):
        # extract the actual provider and strip the prefix from the model name.
        if model and "/" in model:
            prefix, model_name = model.split("/", 1)
            litellm_provider = LITELLM_PREFIX_TO_PROVIDER_MAP.get(prefix)
            if litellm_provider:
                provider = litellm_provider
                model = model_name

        # Clean invocation params (remove internal LangChain params)
        clean_params = {k: v for k, v in invocation_params.items() if not k.startswith("_")}

        return {
            "provider": provider,
            "model": model,
            "model_parameters": clean_params,
        }

    def _extract_token_info(self, response: "langchain_schema.LLMResult") -> Dict[str, Any]:
        """Extract token information generically from LLM response.

        In LangChain v1, ``AIMessage.usage_metadata`` is the guaranteed,
        standardized token source, so it is read FIRST. The provider-specific
        ``llm_output`` / ``generation_info`` paths (Ollama ``prompt_eval_count``,
        Google ``usage_metadata`` in generation_info) remain as fallbacks so
        nothing regresses.

        When ``usage_metadata`` carries ``input_token_details`` /
        ``output_token_details`` (e.g. ``cache_creation``, ``cache_read``,
        ``reasoning``, ``audio``), those are surfaced under a ``token_details``
        key so cost is accurate for prompt-caching / reasoning models.
        """
        token_usage: Dict[str, Any] = {}
        token_details: Dict[str, Any] = {}

        # 1. usage_metadata on the message (standardized, v1-first).
        if response.generations:
            gen = response.generations[0][0]
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if usage:
                token_usage = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                input_details = usage.get("input_token_details")
                if input_details:
                    token_details["input_token_details"] = dict(input_details)
                output_details = usage.get("output_token_details")
                if output_details:
                    token_details["output_token_details"] = dict(output_details)

        # 2. Fall back to llm_output for providers that surface it there.
        if not token_usage:
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage") or llm_output.get("estimatedTokens") or {}

        # 3. Fall back to generation_info for providers like Ollama/Google.
        if not token_usage and response.generations:
            gen = response.generations[0][0]
            generation_info = gen.generation_info or {}

            # Ollama style
            if "prompt_eval_count" in generation_info:
                prompt_tokens = generation_info.get("prompt_eval_count", 0)
                completion_tokens = generation_info.get("eval_count", 0)
                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            # Google style
            elif "usage_metadata" in generation_info:
                usage = generation_info["usage_metadata"]
                token_usage = {
                    "prompt_tokens": usage.get("prompt_token_count", 0),
                    "completion_tokens": usage.get("candidates_token_count", 0),
                    "total_tokens": usage.get("total_token_count", 0),
                }

        result = {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "tokens": token_usage.get("total_tokens", 0),
        }
        if token_details:
            result["token_details"] = token_details
        return result

    @staticmethod
    def _build_usage_details(
        prompt_tokens: int,
        completion_tokens: int,
        input_token_details: Optional[Dict[str, int]] = None,
        output_token_details: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, int]]:
        """Build the per-category token map the cost backend prices by exact key.

        The backend prices a **non-overlapping** partition: it sums a price for
        each ``usageDetails`` key it recognizes. LangChain, however, reports the
        granular categories (``cache_read``, ``cache_creation``, ``audio``) as
        **subsets** of the ``input_tokens`` / ``output_tokens`` totals. So we
        break each granular category out under the key the cost table uses and
        subtract it from the input/output base, keeping the partition
        non-overlapping and its sum equal to the total token count.

        Key mapping (LangChain -> Openlayer cost table):
          input  ``cache_read``     -> ``cached_tokens``
          input  ``cache_creation`` -> ``cache_creation_tokens``
          input  ``audio``          -> ``audio_input_tokens``
          output ``audio``          -> ``audio_output_tokens``

        ``reasoning`` is intentionally left folded into ``output_tokens`` -- it
        is billed at the output rate and the cost table has no separate price for
        it. Zero-valued categories are omitted; returns ``None`` when there are no
        tokens so the ``usageDetails`` column is omitted rather than emitted empty.
        """
        input_token_details = input_token_details or {}
        output_token_details = output_token_details or {}

        cache_read = int(input_token_details.get("cache_read", 0) or 0)
        cache_creation = int(input_token_details.get("cache_creation", 0) or 0)
        audio_input = int(input_token_details.get("audio", 0) or 0)
        audio_output = int(output_token_details.get("audio", 0) or 0)

        # Base = total minus the granular categories broken out below.
        input_base = prompt_tokens - cache_read - cache_creation - audio_input
        output_base = completion_tokens - audio_output

        usage_details: Dict[str, int] = {}
        for key, value in (
            ("input_tokens", input_base),
            ("output_tokens", output_base),
            ("cached_tokens", cache_read),
            ("cache_creation_tokens", cache_creation),
            ("audio_input_tokens", audio_input),
            ("audio_output_tokens", audio_output),
        ):
            if value and value > 0:
                usage_details[key] = value
        return usage_details or None

    def _extract_output(self, response: "langchain_schema.LLMResult") -> str:
        """Extract output text from LLM response.

        When a generation carries no text (e.g. an agent turn whose model
        response is *only* tool calls, as in LangGraph / ``create_agent``),
        fall back to serializing the message's tool calls so the step output
        is not empty.
        """
        output = ""
        for generations in response.generations:
            for generation in generations:
                text = generation.text or ""
                if text:
                    output += text.replace("\n", " ")
                else:
                    output += self._tool_calls_to_str(generation)
        return output

    def _tool_calls_to_str(self, generation: Any) -> str:
        """Serialize tool calls from a generation's message, if any.

        Done defensively (getattr) so it never raises on generations or
        messages that lack a ``message`` / ``tool_calls`` attribute, and works
        whether or not langchain uses the v1 message shape.
        """
        message = getattr(generation, "message", None)
        if message is None:
            return ""

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return ""

        import json

        try:
            return json.dumps(tool_calls, default=str)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _safe_parse_json(self, input_str: str) -> Any:
        """Safely parse JSON string, returning the string if parsing fails."""
        try:
            import json

            return json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            return input_str

    # ---------------------- Common Callback Logic ---------------------- #

    def _handle_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for LLM start."""
        invocation_params = kwargs.get("invocation_params", {})
        model_info = self._extract_model_info(serialized, invocation_params, metadata or {})

        step_name = f"{model_info['provider'] or 'LLM'} Chat Completion"
        prompt = [{"role": "user", "content": text} for text in prompts]

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=step_name,
            step_type=enums.StepType.CHAT_COMPLETION,
            inputs={"prompt": prompt},
            metadata={"tags": tags} if tags else None,
            **model_info,
        )
        self._apply_thread_id_session(run_id, metadata)

    def _handle_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List["langchain_schema.BaseMessage"]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for chat model start."""
        invocation_params = kwargs.get("invocation_params", {})
        model_info = self._extract_model_info(serialized, invocation_params, metadata or {})

        # Always use provider-based name for chat completions (e.g. "Google Chat Completion")
        # rather than the run_name from the caller (e.g. "Language Model") which is generic.
        step_name = f"{model_info['provider'] or 'Chat Model'} Chat Completion"
        prompt = self._messages_to_prompt_format(messages)

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=step_name,
            step_type=enums.StepType.CHAT_COMPLETION,
            inputs={"prompt": prompt},
            metadata={"tags": tags} if tags else None,
            **model_info,
        )
        self._apply_thread_id_session(run_id, metadata)

    def _handle_llm_end(
        self,
        response: "langchain_schema.LLMResult",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for LLM end."""
        if run_id not in self.steps:
            return

        output = self._extract_output(response)

        # Only extract token info if it hasn't been set during streaming
        step = self.steps[run_id]
        token_info = {}
        if not (hasattr(step, "prompt_tokens") and step.prompt_tokens is not None and step.prompt_tokens > 0):
            token_info = self._extract_token_info(response)

        # ChatCompletionStep has no dedicated field for fine-grained token
        # details (cache_read/creation, reasoning, audio...), so surface them
        # under step metadata where cost-relevant breakdowns can be read.
        token_details = token_info.pop("token_details", None)
        if token_details:
            step.metadata = {**step.metadata, "token_details": token_details}

        # Also emit a priced, non-overlapping per-category usageDetails map so
        # the backend can populate costDetails (the token_details above are only
        # surfaced as informational metadata, not priced).
        usage_details = self._build_usage_details(
            token_info.get("prompt_tokens", 0),
            token_info.get("completion_tokens", 0),
            (token_details or {}).get("input_token_details"),
            (token_details or {}).get("output_token_details"),
        )
        if usage_details:
            token_info["usage_details"] = usage_details

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=output,
            **token_info,
        )

    def _handle_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for LLM error."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def _handle_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for chain start."""
        # Extract chain name from serialized data or use provided name
        # Handle case where serialized can be None
        serialized = serialized or {}
        metadata = metadata or {}

        # Resolve the chain's display name. Prefer the runnable's own ``name``
        # (e.g. a node's name like "callAgent", or "RunnableSequence"/"Prompt"
        # for nested LCEL runs), then fall back to LangGraph's injected
        # ``metadata["langgraph_node"]`` only when no name is available, then the
        # serialized id. This mirrors the TS handler's ``name ?? langgraph_node
        # ?? id`` precedence. ``langgraph_node`` is inherited by *every* run
        # nested inside a node, so preferring it over ``name`` would relabel all
        # of a node's internal LCEL runs (RunnableSequence/Prompt/...) — and even
        # nested sub-graphs — with the parent node's name. LangGraph already sets
        # ``name`` to the node name at node boundaries, so this keeps nodes
        # identifiable while preserving the inner structure's real names.
        langgraph_node = metadata.get("langgraph_node")
        chain_name = (
            name or langgraph_node or (serialized.get("id", [])[-1] if serialized.get("id") else None) or "Chain"
        )

        # Skip chains marked as hidden (e.g., internal LangGraph chains)
        if tags and "langsmith:hidden" in tags:
            return

        if chain_name == "LangGraph" and inputs.get("messages"):
            inputs = {
                "prompt": inputs.get("messages"),
            }

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=chain_name,
            step_type=enums.StepType.USER_CALL,
            inputs=inputs,
            metadata={
                "tags": tags,
                "serialized": serialized,
                **metadata,
                **kwargs,
            },
        )
        self._apply_thread_id_session(run_id, metadata)

    def _handle_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for chain end."""
        if run_id not in self.steps:
            return

        # Check if this is a ConversationalRetrievalChain with source documents
        if isinstance(outputs, dict) and "source_documents" in outputs:
            source_docs = outputs["source_documents"]
            if source_docs:
                # Extract content from source documents
                context_list = []
                for doc in source_docs:
                    if hasattr(doc, "page_content"):
                        context_list.append(doc.page_content)
                    else:
                        context_list.append(str(doc))

                if context_list:
                    current_trace = self._find_trace(run_id)
                    if current_trace:
                        current_trace.update_metadata(context=context_list)

        # Parse output for LangGraph
        step = self.steps[run_id]
        if step.name == "LangGraph" and outputs.get("messages"):
            if isinstance(outputs.get("messages"), list):
                if isinstance(outputs.get("messages")[-1], langchain_schema.BaseMessage):
                    outputs = outputs.get("messages")[-1].content

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=outputs,
        )

    def _handle_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for chain error."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def _handle_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for tool start."""
        # Handle case where serialized can be None
        serialized = serialized or {}
        tool_name = (
            name or serialized.get("name") or (serialized.get("id", [])[-1] if serialized.get("id") else None) or "Tool"
        )

        # Parse input - prefer structured inputs over string
        tool_input = inputs or self._safe_parse_json(input_str)

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=tool_name,
            step_type=enums.StepType.TOOL,
            inputs=tool_input,
            metadata={
                "tags": tags,
                "serialized": serialized,
                **(metadata or {}),
                **kwargs,
            },
            function_name=tool_name,
            arguments=tool_input,
        )

    def _find_trace(self, run_id: UUID) -> Optional["traces.Trace"]:
        """Find the trace that owns this run_id, preferring an external context.

        Falls back to the handler's own per-run_id map (covers pure-callback
        usage where nothing has set tracer._current_trace).
        """
        external = tracer.get_current_trace()
        if external is not None:
            return external
        return self._run_id_to_trace.get(run_id)

    def _extract_tool_output(self, output: Any) -> tuple:
        """Split a tool result into (content, artifact).

        Handles three shapes LangChain may pass to on_tool_end:
        - ToolMessage with .content/.artifact (response_format="content_and_artifact")
        - 2-tuple (content, artifact) (older LangChain versions)
        - plain string / anything else (no artifact)
        """
        if HAVE_LANGCHAIN and isinstance(output, langchain_schema.ToolMessage):
            return output.content, getattr(output, "artifact", None)
        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        return output, None

    def _is_document_list(self, value: Any) -> bool:
        """True iff value is a non-empty sequence of objects with page_content."""
        if not isinstance(value, (list, tuple)) or not value:
            return False
        return all(getattr(item, "page_content", None) is not None for item in value)

    def _handle_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for tool end."""
        if run_id not in self.steps:
            return

        content, artifact = self._extract_tool_output(output)

        if artifact is not None:
            step = self.steps[run_id]
            step.metadata["artifact"] = artifact

            if self._is_document_list(artifact):
                doc_contents = [doc.page_content for doc in artifact]
                current_trace = self._find_trace(run_id)
                if current_trace:
                    current_trace.update_metadata(context=doc_contents)

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=content,
        )

    def _handle_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for tool error."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def _handle_agent_action(
        self,
        action: "langchain_schema.AgentAction",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for agent action."""
        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=f"Agent Tool: {action.tool}",
            step_type=enums.StepType.AGENT,
            inputs={
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            },
            metadata={"agent_action": True, **kwargs},
            tool=action.tool,
            action=action,
            agent_type="langchain_agent",
        )

    def _handle_agent_finish(
        self,
        finish: "langchain_schema.AgentFinish",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for agent finish."""
        if run_id not in self.steps:
            return

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=finish.return_values,
        )

    def _handle_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for retriever start."""
        # Handle case where serialized can be None
        serialized = serialized or {}
        retriever_name = serialized.get("id", [])[-1] if serialized.get("id") else "Retriever"

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=retriever_name,
            step_type=enums.StepType.RETRIEVER,
            inputs={"query": query},
            metadata={
                "tags": tags,
                "serialized": serialized,
                **(metadata or {}),
                **kwargs,
            },
        )

    def _handle_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for retriever end."""
        if run_id not in self.steps:
            return

        # Extract document content
        doc_contents = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc_contents.append(doc.page_content)
            else:
                doc_contents.append(str(doc))

        # Populate the trace-level `context` so the reserved column is set.
        # Prefer an active (external) trace context; otherwise fall back to the
        # standalone trace this handler owns for the run. Without the fallback,
        # synchronous RAG pipelines (where the retriever is the root step and no
        # external @trace context exists) would silently drop the context.
        current_trace = self._find_trace(run_id)
        if current_trace:
            current_trace.update_metadata(context=doc_contents)

        # Update the step with RetrieverStep-specific attributes
        step = self.steps[run_id]
        if isinstance(step, steps.RetrieverStep):
            step.documents = doc_contents

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output={"documents": doc_contents, "count": len(documents)},
        )

    def _handle_retriever_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Common logic for retriever error."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def _handle_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Common logic for LLM new token."""
        # Safely check for chunk and usage_metadata
        chunk = kwargs.get("chunk")
        if chunk and hasattr(chunk, "message") and hasattr(chunk.message, "usage_metadata"):
            usage = chunk.message.usage_metadata

            # Only proceed if usage is not None
            if usage:
                # Extract run_id from kwargs (should be provided by LangChain)
                run_id = kwargs.get("run_id")
                if run_id and run_id in self.steps:
                    # Convert usage to the expected format like _extract_token_info does
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    token_info = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "tokens": usage.get("total_tokens", 0),
                    }
                    usage_details = self._build_usage_details(
                        prompt_tokens,
                        completion_tokens,
                        usage.get("input_token_details"),
                        usage.get("output_token_details"),
                    )
                    if usage_details:
                        token_info["usage_details"] = usage_details

                    # Update the step with token usage information
                    step = self.steps[run_id]
                    if isinstance(step, steps.ChatCompletionStep):
                        step.log(**token_info)
        return

    def _apply_thread_id_session(self, run_id: UUID, metadata: Optional[Dict[str, Any]]) -> None:
        """Map LangGraph ``metadata["thread_id"]`` to the trace's session_id.

        Any LangGraph app with a checkpointer injects ``thread_id`` into
        callback metadata. When ``map_thread_id_to_session`` is enabled (the
        default) and the user has not explicitly set a session via
        ``UserSessionContext``, the thread_id is written to the trace's
        ``session_id`` metadata so it is promoted to the ``session_id`` column
        by ``post_process_trace``.

        An explicitly-set session_id is never clobbered: ``post_process_trace``
        re-applies the ``UserSessionContext`` value after merging trace
        metadata, and this method additionally skips writing when a context
        session_id is present.
        """
        if not self._map_thread_id_to_session or not metadata:
            return

        thread_id = metadata.get("thread_id")
        if not thread_id:
            return

        # Do not clobber an explicitly-provided session_id.
        if tracer.UserSessionContext.get_session_id() is not None:
            return

        target_trace = self._find_trace(run_id)
        if target_trace is None:
            return

        existing = (target_trace.metadata or {}).get("session_id")
        if existing:
            return

        target_trace.update_metadata(session_id=str(thread_id))


class OpenlayerHandler(OpenlayerHandlerMixin, BaseCallbackHandlerClass):  # type: ignore[misc]
    """LangChain callback handler that logs to Openlayer."""

    def __init__(
        self,
        ignore_llm=False,
        ignore_chat_model=False,
        ignore_chain=False,
        ignore_retriever=False,
        ignore_agent=False,
        inference_id: Optional[Any] = None,
        metadata_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        map_thread_id_to_session: bool = True,
        **kwargs: Any,
    ) -> None:
        # Add both inference_id and metadata_transformer to kwargs so they get passed to mixin
        if inference_id is not None:
            kwargs["inference_id"] = inference_id
        if metadata_transformer is not None:
            kwargs["metadata_transformer"] = metadata_transformer
        kwargs["map_thread_id_to_session"] = map_thread_id_to_session
        super().__init__(**kwargs)
        # Store the ignore flags as instance variables
        self._ignore_llm = ignore_llm
        self._ignore_chat_model = ignore_chat_model
        self._ignore_chain = ignore_chain
        self._ignore_retriever = ignore_retriever
        self._ignore_agent = ignore_agent

    @property
    def ignore_llm(self) -> bool:
        return self._ignore_llm

    @property
    def ignore_chat_model(self) -> bool:
        return self._ignore_chat_model

    @property
    def ignore_chain(self) -> bool:
        return self._ignore_chain

    @property
    def ignore_retriever(self) -> bool:
        return self._ignore_retriever

    @property
    def ignore_agent(self) -> bool:
        return self._ignore_agent

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        if self.ignore_llm:
            return
        return self._handle_llm_start(serialized, prompts, **kwargs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List["langchain_schema.BaseMessage"]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        if self.ignore_chat_model:
            return
        return self._handle_chat_model_start(serialized, messages, **kwargs)

    def on_llm_end(self, response: "langchain_schema.LLMResult", **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self.ignore_llm:
            return
        return self._handle_llm_end(response, **kwargs)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""
        if self.ignore_llm:
            return
        return self._handle_llm_error(error, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        return self._handle_llm_new_token(token, **kwargs)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        if self.ignore_chain:
            return
        return self._handle_chain_start(serialized, inputs, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if self.ignore_chain:
            return
        return self._handle_chain_end(outputs, **kwargs)

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        if self.ignore_chain:
            return
        return self._handle_chain_error(error, **kwargs)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        if self.ignore_retriever:
            return
        return self._handle_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        if self.ignore_retriever:
            return
        return self._handle_tool_end(output, **kwargs)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when tool errors."""
        if self.ignore_retriever:
            return
        return self._handle_tool_error(error, **kwargs)

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any:
        """Run when retriever starts running."""
        if self.ignore_retriever:
            return
        return self._handle_retriever_start(serialized, query, **kwargs)

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> Any:
        """Run when retriever ends running."""
        if self.ignore_retriever:
            return
        return self._handle_retriever_end(documents, **kwargs)

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> Any:
        """Run when retriever errors."""
        if self.ignore_retriever:
            return
        return self._handle_retriever_error(error, **kwargs)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action: "langchain_schema.AgentAction", **kwargs: Any) -> Any:
        """Run on agent action."""
        if self.ignore_agent:
            return
        return self._handle_agent_action(action, **kwargs)

    def on_agent_finish(self, finish: "langchain_schema.AgentFinish", **kwargs: Any) -> Any:
        """Run on agent end."""
        if self.ignore_agent:
            return
        return self._handle_agent_finish(finish, **kwargs)


class AsyncOpenlayerHandler(OpenlayerHandlerMixin, AsyncCallbackHandlerClass):  # type: ignore[misc]
    """Async LangChain callback handler that logs to Openlayer."""

    def __init__(
        self,
        ignore_llm=False,
        ignore_chat_model=False,
        ignore_chain=False,
        ignore_retriever=False,
        ignore_agent=False,
        inference_id: Optional[Any] = None,
        metadata_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        map_thread_id_to_session: bool = True,
        **kwargs: Any,
    ) -> None:
        # Add both inference_id and metadata_transformer to kwargs so they get passed to mixin
        if inference_id is not None:
            kwargs["inference_id"] = inference_id
        if metadata_transformer is not None:
            kwargs["metadata_transformer"] = metadata_transformer
        kwargs["map_thread_id_to_session"] = map_thread_id_to_session
        super().__init__(**kwargs)
        # Store the ignore flags as instance variables
        self._ignore_llm = ignore_llm
        self._ignore_chat_model = ignore_chat_model
        self._ignore_chain = ignore_chain
        self._ignore_retriever = ignore_retriever
        self._ignore_agent = ignore_agent
        # For async: manage our own trace mapping since context vars are unreliable
        self._traces_by_root: Dict[UUID, traces.Trace] = {}
        # Detect if an external trace context exists at initialization time
        # If true, we'll create standalone traces for external system integration
        # instead of uploading them independently
        self._has_external_trace: bool = tracer.get_current_trace() is not None

    @property
    def ignore_llm(self) -> bool:
        return self._ignore_llm

    @property
    def ignore_chat_model(self) -> bool:
        return self._ignore_chat_model

    @property
    def ignore_chain(self) -> bool:
        return self._ignore_chain

    @property
    def ignore_retriever(self) -> bool:
        return self._ignore_retriever

    @property
    def ignore_agent(self) -> bool:
        return self._ignore_agent

    def _start_step(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        step_type: enums.StepType = enums.StepType.CHAT_COMPLETION,
        inputs: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **step_kwargs: Any,
    ) -> steps.Step:
        """Start a new step - async version with explicit trace management."""
        if run_id in self.steps:
            return self.steps[run_id]

        # Create the step
        step = steps.step_factory(
            step_type=step_type,
            name=name,
            inputs=inputs,
            metadata={**self.metadata, **(metadata or {})},
        )
        step.start_time = time.time()

        # Set step-specific attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Handle parent-child relationships
        if parent_run_id is not None and parent_run_id in self.steps:
            # This step has a parent - add as nested step
            parent_step = self.steps[parent_run_id]
            parent_step.add_nested_step(step)
            parent_trace = self._run_id_to_trace.get(parent_run_id)
            if parent_trace is not None:
                self._run_id_to_trace[run_id] = parent_trace
        else:
            # Check if we're in an existing trace context via ContextVars
            current_step = tracer.get_current_step()
            current_trace = tracer.get_current_trace()

            if current_step is not None:
                # We're inside an existing step context - add as nested
                current_step.add_nested_step(step)
                if current_trace is not None:
                    self._run_id_to_trace[run_id] = current_trace
            elif current_trace is not None:
                # Have trace but no current step
                # If it's an external trace, we should NOT add at root - external system will integrate
                # If it's a ContextVar trace with no current step, add to trace
                if not self._has_external_trace:
                    # ContextVar-detected trace - add directly
                    current_trace.add_step(step)
                    self._run_id_to_trace[run_id] = current_trace
                else:
                    # External trace without current step - create temp standalone for later integration
                    trace = traces.Trace()
                    trace.add_step(step)
                    self._traces_by_root[run_id] = trace
                    self._run_id_to_trace[run_id] = trace
            else:
                # No existing context - create standalone trace
                trace = traces.Trace()
                trace.add_step(step)
                self._traces_by_root[run_id] = trace
                self._run_id_to_trace[run_id] = trace

            # Track root steps
            if parent_run_id is None:
                self.root_steps.add(run_id)
                # Override step ID with custom inference_id if provided
                if self._inference_id is not None:
                    step.id = self._inference_id

        self.steps[run_id] = step
        return step

    def _end_step(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        **step_kwargs: Any,
    ) -> None:
        """End a step - async version with explicit upload logic."""
        if run_id not in self.steps:
            return

        step = self.steps.pop(run_id)
        self._run_id_to_trace.pop(run_id, None)
        is_root_step = run_id in self.root_steps

        if is_root_step:
            self.root_steps.remove(run_id)

        # Update step with final data
        if step.end_time is None:
            step.end_time = time.time()
        if step.latency is None:
            step.latency = (step.end_time - step.start_time) * 1000

        # Set output and error
        if output is not None:
            step.output = output
        if error is not None:
            step.metadata = {**step.metadata, "error": error}

        # Set additional step attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Only upload if this is a standalone trace (not integrated with external trace)
        has_standalone_trace = run_id in self._traces_by_root

        # Only upload if: root step + has standalone trace + not part of external trace
        if is_root_step and has_standalone_trace and not self._has_external_trace:
            trace = self._traces_by_root.pop(run_id)
            if tracer._resolve("background_publish_enabled"):
                ctx = contextvars.copy_context()
                tracer._get_background_executor().submit(ctx.run, self._process_and_upload_async_trace, trace)
            else:
                self._process_and_upload_async_trace(trace)

    def _process_and_upload_async_trace(self, trace: traces.Trace) -> None:
        """Process and upload trace for async handler."""
        # Convert all LangChain objects
        for step in trace.steps:
            self._convert_step_objects_recursively(step)

        # Use tracer's post-processing
        trace_data, input_variable_names = tracer.post_process_trace(trace)

        # Build config
        config = dict(
            tracer.ConfigLlmData(
                output_column_name="output",
                input_variable_names=input_variable_names,
                latency_column_name="latency",
                cost_column_name="cost",
                timestamp_column_name="inferenceTimestamp",
                inference_id_column_name="inferenceId",
                num_of_token_column_name="tokens",
            )
        )

        # Add reserved column configurations for user context
        if "user_id" in trace_data:
            config.update({"user_id_column_name": "user_id"})
        if "session_id" in trace_data:
            config.update({"session_id_column_name": "session_id"})
        if "groundTruth" in trace_data:
            config.update({"ground_truth_column_name": "groundTruth"})
        if "context" in trace_data:
            config.update({"context_column_name": "context"})

        root_step = trace.steps[0] if trace.steps else None
        if (
            root_step
            and isinstance(root_step, steps.ChatCompletionStep)
            and root_step.inputs
            and "prompt" in root_step.inputs
        ):
            config.update({"prompt": utils.json_serialize(root_step.inputs["prompt"])})

        # Upload to Openlayer
        if tracer._publish:
            try:
                client = tracer._get_client()
                if client:
                    # Apply final JSON serialization to ensure everything is serializable
                    serialized_trace_data = utils.json_serialize(trace_data)
                    serialized_config = utils.json_serialize(config)

                    client.inference_pipelines.data.stream(
                        inference_pipeline_id=tracer.resolve_pipeline_id(),
                        rows=[serialized_trace_data],
                        config=serialized_config,
                    )
            except Exception as err:
                tracer.logger.error("Could not stream data to Openlayer %s", err)

    # All callback methods remain the same - just delegate to mixin
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        if self.ignore_llm:
            return
        return self._handle_llm_start(serialized, prompts, **kwargs)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List["langchain_schema.BaseMessage"]],
        **kwargs: Any,
    ) -> Any:
        if self.ignore_chat_model:
            return
        return self._handle_chat_model_start(serialized, messages, **kwargs)

    async def on_llm_end(self, response: "langchain_schema.LLMResult", **kwargs: Any) -> Any:
        if self.ignore_llm:
            return
        return self._handle_llm_end(response, **kwargs)

    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        if self.ignore_llm:
            return
        return self._handle_llm_error(error, **kwargs)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        return self._handle_llm_new_token(token, **kwargs)

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.ignore_chain:
            return
        return self._handle_chain_start(serialized, inputs, **kwargs)

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        if self.ignore_chain:
            return
        return self._handle_chain_end(outputs, **kwargs)

    async def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        if self.ignore_chain:
            return
        return self._handle_chain_error(error, **kwargs)

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        if self.ignore_retriever:  # Note: tool events use ignore_retriever flag
            return
        return self._handle_tool_start(serialized, input_str, **kwargs)

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        if self.ignore_retriever:
            return
        return self._handle_tool_end(output, **kwargs)

    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        if self.ignore_retriever:
            return
        return self._handle_tool_error(error, **kwargs)

    async def on_text(self, text: str, **kwargs: Any) -> Any:
        pass

    async def on_agent_action(self, action: "langchain_schema.AgentAction", **kwargs: Any) -> Any:
        if self.ignore_agent:
            return
        return self._handle_agent_action(action, **kwargs)

    async def on_agent_finish(self, finish: "langchain_schema.AgentFinish", **kwargs: Any) -> Any:
        if self.ignore_agent:
            return
        return self._handle_agent_finish(finish, **kwargs)

    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> Any:
        if self.ignore_retriever:
            return
        return self._handle_retriever_start(serialized, query, **kwargs)

    async def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> Any:
        if self.ignore_retriever:
            return
        return self._handle_retriever_end(documents, **kwargs)

    async def on_retriever_error(self, error: Exception, **kwargs: Any) -> Any:
        if self.ignore_retriever:
            return
        return self._handle_retriever_error(error, **kwargs)

    async def on_retry(self, retry_state: Any, **kwargs: Any) -> Any:
        pass

    async def on_custom_event(self, name: str, data: Any, **kwargs: Any) -> Any:
        pass
