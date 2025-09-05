"""Base classes and interfaces for guardrails system."""

import abc
import enum
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GuardrailAction(enum.Enum):
    """Actions that a guardrail can take."""

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


class BlockStrategy(enum.Enum):
    """Strategies for handling blocked requests."""

    RAISE_EXCEPTION = (
        "raise_exception"  # Raise GuardrailBlockedException (breaks pipeline)
    )
    RETURN_EMPTY = "return_empty"  # Return empty/None response (graceful)
    RETURN_ERROR_MESSAGE = "return_error_message"  # Return error message (graceful)
    SKIP_FUNCTION = "skip_function"  # Skip function execution, return None (graceful)


@dataclass
class GuardrailResult:
    """Result of applying a guardrail."""

    action: GuardrailAction
    modified_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    block_strategy: Optional[BlockStrategy] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate the result after initialization."""
        if self.action == GuardrailAction.MODIFY and self.modified_data is None:
            raise ValueError("modified_data must be provided when action is MODIFY")
        if self.action == GuardrailAction.BLOCK and self.block_strategy is None:
            self.block_strategy = (
                BlockStrategy.RAISE_EXCEPTION
            )  # Default to existing behavior


class GuardrailBlockedException(Exception):
    """Exception raised when a guardrail blocks execution."""

    def __init__(
        self,
        guardrail_name: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.guardrail_name = guardrail_name
        self.reason = reason
        self.metadata = metadata or {}
        super().__init__(f"Guardrail '{guardrail_name}' blocked execution: {reason}")


class BaseGuardrail(abc.ABC):
    """Base class for all guardrails."""

    def __init__(self, name: str, enabled: bool = True, **config):
        """Initialize the guardrail.

        Args:
            name: Human-readable name for this guardrail
            enabled: Whether this guardrail is active
            **config: Guardrail-specific configuration
        """
        self.name = name
        self.enabled = enabled
        self.config = config

    @abc.abstractmethod
    def check_input(self, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check and potentially modify function inputs.

        Args:
            inputs: Dictionary of function inputs (parameter_name -> value)

        Returns:
            GuardrailResult indicating the action to take
        """
        pass

    @abc.abstractmethod
    def check_output(self, output: Any, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check and potentially modify function output.

        Args:
            output: The function's output
            inputs: Dictionary of function inputs for context

        Returns:
            GuardrailResult indicating the action to take
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this guardrail is enabled."""
        return self.enabled

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this guardrail for trace logging."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "enabled": self.enabled,
            "config": self.config,
        }
