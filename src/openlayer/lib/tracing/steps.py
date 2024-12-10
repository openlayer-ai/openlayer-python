"""Module with the different Step classes that can be used in a trace."""

import time
import uuid
from typing import Any, Dict, Optional

from .. import utils
from . import enums


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

    def add_nested_step(self, nested_step: "Step") -> None:
        """Adds a nested step to the current step."""
        self.steps.append(nested_step)

    def log(self, **kwargs: Any) -> None:
        """Logs step data."""
        kwargs = utils.json_serialize(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the Step."""
        return {
            "name": self.name,
            "id": str(self.id),
            "type": self.step_type.value,
            "inputs": self.inputs,
            "output": self.output,
            "groundTruth": self.ground_truth,
            "metadata": self.metadata,
            "steps": [nested_step.to_dict() for nested_step in self.steps],
            "latency": self.latency,
            "startTime": self.start_time,
            "endTime": self.end_time,
        }


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


# ----------------------------- Factory function ----------------------------- #
def step_factory(step_type: enums.StepType, *args, **kwargs) -> Step:
    """Factory function to create a step based on the step_type."""
    if step_type.value not in [item.value for item in enums.StepType]:
        raise ValueError(f"Step type {step_type.value} not recognized.")
    step_type_mapping = {
        enums.StepType.USER_CALL: UserCallStep,
        enums.StepType.CHAT_COMPLETION: ChatCompletionStep,
    }
    return step_type_mapping[step_type](*args, **kwargs)
