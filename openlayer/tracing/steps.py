"""Module with the different Step classes that can be used in a trace."""

import time
from typing import Any, Dict, Optional


class Step:
    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Dict[str, any] = {},
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.output = output
        self.metadata = metadata

        self.step_type = None
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
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the Step."""
        return {
            "name": self.name,
            "type": self.step_type,
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
    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Dict[str, any] = {},
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)
        self.step_type = "user_call"


class OpenAIChatCompletionStep(Step):
    def __init__(
        self,
        name: str,
        inputs: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Dict[str, any] = {},
    ) -> None:
        super().__init__(name=name, inputs=inputs, output=output, metadata=metadata)

        self.step_type = "openai_chat_completion"
        self.prompt_tokens: int = None
        self.completion_tokens: int = None
        self.cost: float = None
        self.model: str = None
        self.model_parameters: Dict[str, Any] = None
        self.raw_output: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the OpenAIChatCompletionStep."""
        step_dict = super().to_dict()
        step_dict.update(
            {
                "promptTokens": self.prompt_tokens,
                "completionTokens": self.completion_tokens,
                "cost": self.cost,
                "model": self.model,
                "modelParameters": self.model_parameters,
                "rawOutput": self.raw_output,
            }
        )
        return step_dict


# ----------------------------- Factory function ----------------------------- #
def step_factory(step_type: str, *args, **kwargs) -> Step:
    """Factory function to create a step based on the step_type."""
    if step_type not in ["user_call", "openai_chat_completion"]:
        raise ValueError(f"Step type {step_type} not recognized.")
    step_type_mapping = {
        "user_call": UserCallStep,
        "openai_chat_completion": OpenAIChatCompletionStep,
    }
    return step_type_mapping[step_type](*args, **kwargs)
