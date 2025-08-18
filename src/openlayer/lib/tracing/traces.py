"""Module with the Trace class."""

from typing import Any, Dict, List, Optional

from .steps import Step


# Type definitions for metadata updates
LLMTestCase = Dict[str, Any]  # Placeholder for LLM test case data
Feedback = Dict[str, Any]     # Placeholder for feedback data


class Trace:
    """Trace, defined as a sequence of steps.

    Each step represents a function call being traced. Steps can also
    contain nested steps, which represent function calls made within the
    parent step."""

    def __init__(self):
        self.steps = []
        self.current_step = None
        
        # Enhanced trace metadata fields
        self.name: Optional[str] = None
        self.tags: Optional[List[str]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.thread_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.input: Optional[Any] = None
        self.output: Optional[Any] = None
        self.feedback: Optional[Feedback] = None
        self.test_case: Optional[LLMTestCase] = None

    def add_step(self, step: Step) -> None:
        """Adds a step to the trace."""
        self.steps.append(step)

    def update_metadata(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        feedback: Optional[Feedback] = None,
        test_case: Optional[LLMTestCase] = None,
    ) -> None:
        """Updates the trace metadata with the provided values."""
        if name is not None:
            self.name = name
        if tags is not None:
            self.tags = tags
        if metadata is not None:
            # Merge with existing metadata if it exists
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(metadata)
        if thread_id is not None:
            self.thread_id = thread_id
        if user_id is not None:
            self.user_id = user_id
        if input is not None:
            self.input = input
        if output is not None:
            self.output = output
        if feedback is not None:
            self.feedback = feedback
        if test_case is not None:
            self.test_case = test_case

    def to_dict(self) -> List[Dict[str, Any]]:
        """Dictionary representation of the Trace."""
        return [step.to_dict() for step in self.steps]
