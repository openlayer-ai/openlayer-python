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
        self.context_tokens: Dict[UUID, Any] = {}  # Store context tokens for cleanup

    def _start_step(
        self,
        run_id: UUID,
        name: str,
        inputs: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **step_kwargs: Any,
    ) -> steps.ChatCompletionStep:
        """Start a new step."""
        if run_id in self.steps:
            return self.steps[run_id]

        # Create the step (same as create_step)
        step = steps.step_factory(
            step_type=enums.StepType.CHAT_COMPLETION,
            name=name,
            inputs=inputs,
            metadata={**self.metadata, **(metadata or {})},
        )
        step.start_time = time.time()

        # Set step-specific attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Mirror the exact logic from create_step
        parent_step = tracer.get_current_step()
        is_root_step = parent_step is None

        if parent_step is None:
            tracer.logger.debug("Starting a new trace...")
            current_trace = traces.Trace()
            tracer._current_trace.set(current_trace)
            tracer._rag_context.set(None)
            current_trace.add_step(step)
        else:
            tracer.logger.debug(
                "Adding step %s to parent step %s", name, parent_step.name
            )
            current_trace = tracer.get_current_trace()
            parent_step.add_nested_step(step)

        # Set current step context and store token for cleanup
        token = tracer._current_step.set(step)
        self.context_tokens[run_id] = (token, is_root_step)
        self.steps[run_id] = step
        return step

    def _end_step(
        self,
        run_id: UUID,
        output: Optional[Any] = None,
        error: Optional[str] = None,
        **step_kwargs: Any,
    ) -> None:
        """End a step."""
        if run_id not in self.steps:
            return

        step = self.steps.pop(run_id)
        token, is_root_step = self.context_tokens.pop(run_id)

        # Update step with final data
        if step.end_time is None:
            step.end_time = time.time()
        if step.latency is None:
            step.latency = (step.end_time - step.start_time) * 1000

        if output is not None:
            step.output = output
            step.raw_output = output
        if error is not None:
            step.metadata = {**step.metadata, "error": error}

        # Set additional step attributes
        for key, value in step_kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)

        # Mirror the exact cleanup logic from create_step
        tracer._current_step.reset(token)

        if is_root_step:
            tracer.logger.debug("Ending the trace...")
            current_trace = tracer.get_current_trace()
            if current_trace:
                trace_data, input_variable_names = tracer.post_process_trace(
                    current_trace
                )

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
                    isinstance(step, steps.ChatCompletionStep)
                    and step.inputs
                    and "prompt" in step.inputs
                ):
                    config.update({"prompt": step.inputs["prompt"]})

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
                        tracer.logger.error(
                            "Could not stream data to Openlayer %s", err
                        )
        else:
            tracer.logger.debug("Ending step %s", step.name)

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

    @staticmethod
    def _langchain_messages_to_prompt(
        messages: List[List[langchain_schema.BaseMessage]],
    ) -> List[Dict[str, str]]:
        """Convert LangChain messages to Openlayer prompt format."""
        prompt = []
        for message_batch in messages:
            for message in message_batch:
                role = "user" if message.type == "human" else message.type
                if message.type == "ai":
                    role = "assistant"
                prompt.append({"role": role, "content": message.content})
        return prompt

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
            name=step_name,
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
        prompt = self._langchain_messages_to_prompt(messages)

        self._start_step(
            run_id=run_id,
            name=step_name,
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
        self._end_step(run_id=run_id, error=str(error))

    # ---------------------- Unused Callback Methods ---------------------- #

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(
        self, action: langchain_schema.AgentAction, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        pass

    def on_agent_finish(
        self, finish: langchain_schema.AgentFinish, **kwargs: Any
    ) -> Any:
        """Run on agent end."""
        pass
