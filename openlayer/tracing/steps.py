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

        self.step_type = "user_call"
        self.start_time = time.time()
        self.end_time = None
        self.ground_truth = None
        self.latency = None

        self.steps = []

    def add_nested_step(self, nested_step: "Step") -> None:
        """Adds a nested step to the current step."""
        self.steps.append(nested_step)

    def update_data(self, **kwargs: Any) -> None:
        """Updates the step data."""
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
