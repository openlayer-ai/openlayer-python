"""Module with the different Step classes that can be used in a trace."""

import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from .. import utils
from . import enums
from .attachments import Attachment


class Step:
    """Step, defined as a single function call being traced.

    This is the base class for all the different types of steps that can be
    used in a trace. Steps can also contain nested steps, which represent
    function calls made within the parent step.
    """

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        self.name = name
        self.id = uuid.uuid4()
        self.inputs = inputs
        self.output = output
        self.metadata = metadata or {}

        self.step_type: enums.StepType = None
        self.start_time = time.time()
        self.end_time = None
        self.ground_truth = None
        self.latency = None

        self.steps = []

        # Attachments: unstructured data (images, audio, PDFs, etc.)
        self.attachments: List["Attachment"] = []

    def add_nested_step(self, nested_step: "Step") -> None:
        """Adds a nested step to the current step."""
        self.steps.append(nested_step)

    def attach(
        self,
        data: Union[bytes, str, Path, BinaryIO, "Attachment"],
        name: Optional[str] = None,
        media_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Attachment":
        """Attach unstructured data to this step.

        This method allows attaching binary content (images, audio, documents, etc.)
        to a trace step. The attachment will be uploaded to Openlayer storage
        when the trace is completed (if upload is enabled).

        Args:
            data: The data to attach. Can be:
                - bytes: Raw binary data
                - str/Path: File path to read from
                - File-like object: Will be read
                - Attachment: An existing Attachment object
            name: Display name for the attachment. If not provided, will be
                  inferred from the file path or set to "attachment".
            media_type: MIME type (e.g., "image/png", "audio/wav").
                        Auto-detected for file paths if not provided.
            metadata: Additional metadata dict (e.g., duration, dimensions).

        Returns:
            The created or added Attachment.

        Examples:
            >>> step.attach("/path/to/audio.wav")
            >>> step.attach(image_bytes, name="screenshot.png", media_type="image/png")
            >>> step.attach(pdf_file, name="document.pdf", media_type="application/pdf")
        """
        if isinstance(data, Attachment):
            attachment = data
        elif isinstance(data, bytes):
            attachment = Attachment.from_bytes(
                data=data,
                name=name or "attachment",
                media_type=media_type or "application/octet-stream",
            )
        elif isinstance(data, (str, Path)):
            attachment = Attachment.from_file(
                file_path=data,
                name=name,
                media_type=media_type,
            )
        else:
            # File-like object
            file_bytes = data.read()
            inferred_name = name or getattr(data, "name", None) or "attachment"
            attachment = Attachment.from_bytes(
                data=file_bytes,
                name=inferred_name,
                media_type=media_type or "application/octet-stream",
            )

        if metadata:
            attachment.metadata.update(metadata)

        self.attachments.append(attachment)
        return attachment

    def log(self, **kwargs: Any) -> None:
        """Logs step data."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the Step."""
        result = {
            "name": self.name,
            "id": str(self.id),
            "type": self.step_type.value,
            "inputs": utils.json_serialize(self.inputs),
            "output": utils.json_serialize(self.output),
            "groundTruth": utils.json_serialize(self.ground_truth),
            "metadata": utils.json_serialize(self.metadata),
            "steps": [nested_step.to_dict() for nested_step in self.steps],
            "latency": self.latency,
            "startTime": self.start_time,
            "endTime": self.end_time,
        }

        # Include valid attachments only (filter out ones with no data/reference)
        if self.attachments:
            valid_attachments = [
                attachment.to_dict()
                for attachment in self.attachments
                if attachment.is_valid()
            ]
            if valid_attachments:
                result["attachments"] = valid_attachments

        return result


class UserCallStep(Step):
    """User call step represents a generic user call in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = enums.StepType.USER_CALL


class ChatCompletionStep(Step):
    """Chat completion step represents an LLM chat completion in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)

        self.step_type = enums.StepType.CHAT_COMPLETION
        self.provider: str = None
        self.prompt_tokens: int = None
        self.completion_tokens: int = None
        self.tokens: int = None
        self.cost: float = None
        self.model: str = None
        self.model_parameters: Dict[str, Any] = None
        self.raw_output: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the ChatCompletionStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "provider": self.provider,
                "promptTokens": self.prompt_tokens,
                "completionTokens": self.completion_tokens,
                "tokens": self.tokens,
                "cost": self.cost,
                "model": self.model,
                "modelParameters": self.model_parameters,
                "rawOutput": self.raw_output,
            }
        )
        return step_dict


class AgentStep(Step):
    """Agent step represents an agent in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = enums.StepType.AGENT
        self.tool: str = None
        self.action: Any = None
        self.agent_type: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the AgentStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "tool": self.tool,
                "action": self.action,
                "agentType": self.agent_type,
            }
        )
        return step_dict


class RetrieverStep(Step):
    """Retriever step represents a retriever in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = enums.StepType.RETRIEVER
        self.documents: List[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the RetrieverStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "documents": self.documents,
            }
        )
        return step_dict


class ToolStep(Step):
    """Tool step represents a tool in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = enums.StepType.TOOL
        self.function_name: str = None
        self.arguments: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the ToolStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "functionName": self.function_name,
                "arguments": self.arguments,
            }
        )
        return step_dict


class HandoffStep(Step):
    """Handoff step represents a handoff in the trace."""

    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, any]] = None,
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = enums.StepType.HANDOFF
        self.from_component: str = None
        self.to_component: str = None
        self.handoff_data: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the HandoffStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "fromComponent": self.from_component,
                "toComponent": self.to_component,
                "handoffData": self.handoff_data,
            }
        )
        return step_dict


class GuardrailStep(Step):
    """Step for tracking guardrail execution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_type = enums.StepType.GUARDRAIL
        self.action: Optional[str] = None
        self.blocked_entities: Optional[List[str]] = None
        self.confidence_threshold: float = None
        self.reason: Optional[str] = None
        self.detected_entities: Optional[List[str]] = None
        self.redacted_entities: Optional[List[str]] = None
        self.block_strategy: Optional[str] = None
        self.data_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the GuardrailStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "action": self.action,
                "blockedEntities": self.blocked_entities,
                "confidenceThreshold": self.confidence_threshold,
                "reason": self.reason,
                "detectedEntities": self.detected_entities,
                "blockStrategy": self.block_strategy,
                "redactedEntities": self.redacted_entities,
                "dataType": self.data_type,
            }
        )
        return step_dict


# ----------------------------- Factory function ----------------------------- #
def step_factory(step_type: enums.StepType, *args, **kwargs) -> Step:
    """Factory function to create a step based on the step_type."""
    if step_type.value not in [item.value for item in enums.StepType]:
        raise ValueError(f"Step type {step_type.value} not recognized.")
    step_type_mapping = {
        enums.StepType.USER_CALL: UserCallStep,
        enums.StepType.CHAT_COMPLETION: ChatCompletionStep,
        enums.StepType.AGENT: AgentStep,
        enums.StepType.RETRIEVER: RetrieverStep,
        enums.StepType.TOOL: ToolStep,
        enums.StepType.HANDOFF: HandoffStep,
        enums.StepType.GUARDRAIL: GuardrailStep,
    }
    return step_type_mapping[step_type](*args, **kwargs)
