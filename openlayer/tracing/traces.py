"""Module with the Trace class."""

from typing import Any, Dict

from .steps import Step


class Trace:
    def __init__(self):
        self.steps = []
        self.current_step = None

    def add_step(self, step: Step) -> None:
        """Adds a step to the trace."""
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation of the Trace."""
        return {"rows": [step.to_dict() for step in self.steps]}
