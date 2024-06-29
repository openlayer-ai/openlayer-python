"""Module with the Openlayer callback handler for LangChain."""

# pylint: disable=unused-argument
import time
from typing import Any, Dict, List, Optional, Union

from langchain import schema as langchain_schema
from langchain.callbacks.base import BaseCallbackHandler

from ..tracing import tracer

LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP = {"openai-chat": "OpenAI"}
PROVIDER_TO_STEP_NAME = {"OpenAI": "OpenAI Chat Completion"}


class OpenlayerHandler(BaseCallbackHandler):
    """LangChain callback handler that logs to Openlayer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        self.start_time: float = None
        self.end_time: float = None
        self.prompt: List[Dict[str, str]] = None
        self.latency: float = None
        self.provider: str = None
        self.model: Optional[str] = None
        self.model_parameters: Dict[str, Any] = None
        self.prompt_tokens: int = None
        self.completion_tokens: int = None
        self.total_tokens: int = None
        self.output: str = None
        self.metatada: Dict[str, Any] = kwargs or {}

    # noqa arg002
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        pass

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],  # noqa: ARG002
        messages: List[List[langchain_schema.BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self.model_parameters = kwargs.get("invocation_params", {})

        provider = self.model_parameters.get("_type", None)
        if provider in LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP:
            self.provider = LANGCHAIN_TO_OPENLAYER_PROVIDER_MAP[provider]
            self.model_parameters.pop("_type")

        self.model = self.model_parameters.get("model_name", None)
        self.output = ""
        self.prompt = self._langchain_messages_to_prompt(messages)
        self.start_time = time.time()

    @staticmethod
    def _langchain_messages_to_prompt(
        messages: List[List[langchain_schema.BaseMessage]],
    ) -> List[Dict[str, str]]:
        """Converts Langchain messages to the Openlayer prompt format (similar to
        OpenAI's.)"""
        prompt = []
        for message in messages:
            for m in message:
                if m.type == "human":
                    prompt.append({"role": "user", "content": m.content})
                elif m.type == "system":
                    prompt.append({"role": "system", "content": m.content})
                elif m.type == "ai":
                    prompt.append({"role": "assistant", "content": m.content})
        return prompt

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response: langchain_schema.LLMResult, **kwargs: Any) -> Any:  # noqa: ARG002, E501
        """Run when LLM ends running."""
        self.end_time = time.time()
        self.latency = (self.end_time - self.start_time) * 1000

        if response.llm_output and "token_usage" in response.llm_output:
            self.prompt_tokens = response.llm_output["token_usage"].get("prompt_tokens", 0)
            self.completion_tokens = response.llm_output["token_usage"].get("completion_tokens", 0)
            self.total_tokens = response.llm_output["token_usage"].get("total_tokens", 0)

        for generations in response.generations:
            for generation in generations:
                self.output += generation.text.replace("\n", " ")

        self._add_to_trace()

    def _add_to_trace(self) -> None:
        """Adds to the trace."""
        name = PROVIDER_TO_STEP_NAME.get(self.provider, "Chat Completion Model")
        tracer.add_chat_completion_step_to_trace(
            name=name,
            provider=self.provider,
            inputs={"prompt": self.prompt},
            output=self.output,
            tokens=self.total_tokens,
            latency=self.latency,
            start_time=self.start_time,
            end_time=self.end_time,
            model=self.model,
            model_parameters=self.model_parameters,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            metadata=self.metatada,
        )

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action: langchain_schema.AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_agent_finish(self, finish: langchain_schema.AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        pass
