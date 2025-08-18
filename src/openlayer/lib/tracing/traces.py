"""Module with the Trace class."""

from typing import Any, Dict, List, Optional

from .steps import Step


class Trace:
    """Trace, defined as a sequence of steps.

    Each step represents a function call being traced. Steps can also
    contain nested steps, which represent function calls made within the
    parent step."""

    def __init__(self):
        self.steps = []
        self.current_step = None
        self.metadata: Optional[Dict[str, Any]] = None

    def add_step(self, step: Step) -> None:
        """Adds a step to the trace."""
        self.steps.append(step)

    def update_metadata(self, **kwargs) -> None:
        """Updates the trace metadata with the provided values.
        
        All provided key-value pairs will be stored in self.metadata.
        Special handling for 'metadata' key which gets merged with existing metadata.
        """
        # Initialize metadata if it doesn't exist
        if self.metadata is None:
            self.metadata = {}
        
        # Handle special case for 'metadata' key - merge with existing
        if 'metadata' in kwargs:
            metadata_to_merge = kwargs.pop('metadata')
            if metadata_to_merge is not None:
                self.metadata.update(metadata_to_merge)
        
        # Add all other kwargs to metadata
        for key, value in kwargs.items():
            if value is not None:
                self.metadata[key] = value

    def to_dict(self) -> List[Dict[str, Any]]:
        """Dictionary representation of the Trace."""
        return [step.to_dict() for step in self.steps]
