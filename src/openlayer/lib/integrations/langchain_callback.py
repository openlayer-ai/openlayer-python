"""Module with the Openlayer callback handler for LangChain."""

# pylint: disable=unused-argument
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain import schema as langchain_schema
from langchain.callbacks.base import BaseCallbackHandler

from ..tracing import tracer, steps, traces, enums
from .. import utils

LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP = {
    "openai-chat": "OpenAI",
    "chat-ollama": "Ollama",
    "vertexai": "Google",
}


class OpenlayerHandler(BaseCallbackHandler):
    """LangChain callback handler that logs to Openlayer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.metadata: Dict[str, Any] = kwargs or {}
        self.steps: Dict[UUID, steps.Step] = {}
        self.root_steps: set[UUID] = set()  # Track which steps are root

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
        else:
            # This is a root step - check if we're in an existing trace context
            current_step = tracer.get_current_step()
            current_trace = tracer.get_current_trace()

            if current_step is not None:
                # We're inside a @trace() decorated function - add as nested step
                current_step.add_nested_step(step)
            elif current_trace is not None:
                # There's an existing trace but no current step
                current_trace.add_step(step)
            else:
                # No existing trace - create new one (standalone mode)
                current_trace = traces.Trace()
                tracer._current_trace.set(current_trace)
                tracer._rag_context.set(None)
                current_trace.add_step(step)

            # Track root steps (those without parent_run_id)
            if parent_run_id is None:
                self.root_steps.add(run_id)

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

        # Only upload trace if this was a root step and we're not in a @trace() context
        if is_root_step and tracer.get_current_step() is None:
            self._process_and_upload_trace(step)

    def _process_and_upload_trace(self, root_step: steps.Step) -> None:
        """Process and upload the completed trace (only for standalone root steps)."""
        current_trace = tracer.get_current_trace()
        if not current_trace:
            return

        # Convert all LangChain objects in the trace once at the end
        self._convert_step_objects_recursively(root_step)
        for step in current_trace.steps:
            if step != root_step:  # Avoid converting root_step twice
                self._convert_step_objects_recursively(step)

        trace_data, input_variable_names = tracer.post_process_trace(current_trace)

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

        if "groundTruth" in trace_data:
            config.update({"ground_truth_column_name": "groundTruth"})
        if "context" in trace_data:
            config.update({"context_column_name": "context"})
        if (
            isinstance(root_step, steps.ChatCompletionStep)
            and root_step.inputs
            and "prompt" in root_step.inputs
        ):
            config.update({"prompt": root_step.inputs["prompt"]})

        if tracer._publish:
            try:
                tracer._client.inference_pipelines.data.stream(
                    inference_pipeline_id=utils.get_env_variable(
                        "OPENLAYER_INFERENCE_PIPELINE_ID"
                    ),
                    rows=[trace_data],
                    config=config,
                )
            except Exception as err:  # pylint: disable=broad-except
                tracer.logger.error("Could not stream data to Openlayer %s", err)

        # Reset trace context only for standalone traces
        tracer._current_trace.set(None)

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
            step.metadata = self._convert_langchain_objects(step.metadata)

        # Convert nested steps recursively
        for nested_step in step.steps:
            self._convert_step_objects_recursively(nested_step)

    def _convert_langchain_objects(self, obj: Any) -> Any:
        """Recursively convert LangChain objects to JSON-serializable format."""
        # Explicit check for LangChain BaseMessage and its subclasses
        if isinstance(obj, langchain_schema.BaseMessage):
            return self._message_to_dict(obj)

        # Handle ChatPromptValue objects which contain messages
        if (
            hasattr(obj, "messages")
            and hasattr(obj, "__class__")
            and "ChatPromptValue" in obj.__class__.__name__
        ):
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

        # Handle other LangChain objects with common attributes
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            # Many LangChain objects have a dict() method
            try:
                return self._convert_langchain_objects(obj.dict())
            except Exception:
                pass

        # Handle objects with content attribute
        if hasattr(obj, "content") and not isinstance(
            obj, langchain_schema.BaseMessage
        ):
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

    def _message_to_dict(self, message: langchain_schema.BaseMessage) -> Dict[str, str]:
        """Convert a LangChain message to a JSON-serializable dictionary."""
        message_type = getattr(message, "type", "user")

        role = "user" if message_type == "human" else message_type
        if message_type == "ai":
            role = "assistant"
        elif message_type == "system":
            role = "system"

        return {"role": role, "content": str(message.content)}

    def _messages_to_prompt_format(
        self, messages: List[List[langchain_schema.BaseMessage]]
    ) -> List[Dict[str, str]]:
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
        provider = invocation_params.get("_type")
        if provider in LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP:
            provider = LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP[provider]

        model = (
            invocation_params.get("model_name")
            or invocation_params.get("model")
            or metadata.get("ls_model_name")
            or serialized.get("name")
        )

        # Clean invocation params (remove internal LangChain params)
        clean_params = {
            k: v for k, v in invocation_params.items() if not k.startswith("_")
        }

        return {
            "provider": provider,
            "model": model,
            "model_parameters": clean_params,
        }

    def _extract_token_info(
        self, response: langchain_schema.LLMResult
    ) -> Dict[str, Any]:
        """Extract token information generically from LLM response."""
        llm_output = response.llm_output or {}

        # Try standard token_usage location first
        token_usage = (
            llm_output.get("token_usage") or llm_output.get("estimatedTokens") or {}
        )

        # Fallback to generation info for providers like Ollama/Google
        if not token_usage and response.generations:
            generation_info = response.generations[0][0].generation_info or {}

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

        return {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "tokens": token_usage.get("total_tokens", 0),
        }

    def _extract_output(self, response: langchain_schema.LLMResult) -> str:
        """Extract output text from LLM response."""
        output = ""
        for generations in response.generations:
            for generation in generations:
                output += generation.text.replace("\n", " ")
        return output

    # ---------------------- LangChain Callback Methods ---------------------- #

    def on_llm_start(
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
        """Run when LLM starts running."""
        invocation_params = kwargs.get("invocation_params", {})
        model_info = self._extract_model_info(
            serialized, invocation_params, metadata or {}
        )

        step_name = name or f"{model_info['provider'] or 'LLM'} Chat Completion"
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

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[langchain_schema.BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        invocation_params = kwargs.get("invocation_params", {})
        model_info = self._extract_model_info(
            serialized, invocation_params, metadata or {}
        )

        step_name = name or f"{model_info['provider'] or 'Chat Model'} Chat Completion"
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

    def on_llm_end(
        self,
        response: langchain_schema.LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""
        if run_id not in self.steps:
            return

        output = self._extract_output(response)
        token_info = self._extract_token_info(response)

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=output,
            **token_info,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_chain_start(
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
        """Run when chain starts running."""
        # Extract chain name from serialized data or use provided name
        chain_name = (
            name
            or (serialized.get("id", [])[-1] if serialized.get("id") else None)
            or "Chain"
        )

        # Skip chains marked as hidden (e.g., internal LangGraph chains)
        if tags and "langsmith:hidden" in tags:
            return

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=chain_name,
            step_type=enums.StepType.USER_CALL,
            inputs=inputs,
            metadata={
                "tags": tags,
                "serialized": serialized,
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        if run_id not in self.steps:
            return

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=outputs,  # Direct output - conversion happens at the end
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def on_tool_start(
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
        """Run when tool starts running."""
        tool_name = (
            name
            or (serialized.get("id", [])[-1] if serialized.get("id") else None)
            or "Tool"
        )

        # Parse input - prefer structured inputs over string
        tool_input = inputs or self._safe_parse_json(input_str)

        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=tool_name,
            step_type=enums.StepType.USER_CALL,
            inputs=tool_input,
            metadata={
                "tags": tags,
                "serialized": serialized,
                **(metadata or {}),
                **kwargs,
            },
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        if run_id not in self.steps:
            return

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=output,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        self._end_step(run_id=run_id, parent_run_id=parent_run_id, error=str(error))

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(
        self,
        action: langchain_schema.AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""
        self._start_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=f"Agent Tool: {action.tool}",
            step_type=enums.StepType.USER_CALL,
            inputs={
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            },
            metadata={"agent_action": True, **kwargs},
        )

    def on_agent_finish(
        self,
        finish: langchain_schema.AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        if run_id not in self.steps:
            return

        self._end_step(
            run_id=run_id,
            parent_run_id=parent_run_id,
            output=finish.return_values,
        )

    # ---------------------- Helper Methods ---------------------- #

    def _safe_parse_json(self, input_str: str) -> Any:
        """Safely parse JSON string, returning the string if parsing fails."""
        try:
            import json

            return json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            return input_str
