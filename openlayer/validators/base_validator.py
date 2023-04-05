"""Base validator interface.

The entry point for all validators. This is the interface that all validators
must implement.
"""
import logging
from abc import ABC, abstractmethod
from typing import List

import marshmallow as ma

# Validator logger
logger = logging.getLogger("validators")
logger.setLevel(logging.ERROR)

# Console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class BaseValidator(ABC):
    """Base validator interface."""

    def __init__(self, resource_display_name: str):
        self.resource_display_name = resource_display_name
        self.failed_validations = []

    def validate(self) -> List[str]:
        """Template method for validating a resource.

        Returns
        -------
        List[str]: A list of failed validations.
        """
        self._display_opening_message()
        self._validate()
        self._display_closing_message()

        return self.failed_validations

    def _display_opening_message(self) -> None:
        """Displays a message indicating that the validation of a
        resource has started."""
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                          %s validations                          ",
            self.resource_display_name.capitalize(),
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )

    @abstractmethod
    def _validate(self) -> None:
        """Validates the resource. This method should be implemented by
        child classes."""

    def _display_closing_message(self) -> None:
        """Displays a message that indicates the end of the validation of a
        resource. The message will be either a success or failure message."""
        if not self.failed_validations:
            self._display_success_message()
        else:
            self._display_failure_message()

    def _display_success_message(self) -> None:
        """Displays a message indicating that the validation of a resource
        has succeeded."""
        logger.info("✓ All %s validations passed!\n", self.resource_display_name)

    def _display_failure_message(self) -> None:
        """Displays the failed validations in a list format, with one failed
        validation per line."""
        logger.error("The following %s validations failed:", self.resource_display_name)
        for message in self.failed_validations:
            logger.error("* %s", message)
        logger.error("Please fix the issues and try again.\n")

    def _format_marshmallow_error_message(self, err: ma.ValidationError) -> str:
        """Formats the error messages from Marshmallow to conform to the expected
        list of strings format.

        Parameters
        ----------
        err : ma.ValidationError
            The error object returned by Marshmallow.

        Returns
        -------
        List[str]
            A list of strings, where each string is a failed validation.
        """
        error_message = []
        for input_data, msg in err.messages.items():
            if input_data == "_schema":
                temp_msg = "\n".join(msg)
                error_message.append(f"{temp_msg}")
            elif not isinstance(msg, dict):
                temp_msg = msg[0].lower()
                error_message.append(f"`{input_data}`: {temp_msg}")
            else:
                temp_msg = list(msg.values())[0][0].lower()
                error_message.append(
                    f"`{input_data}` contains items that are {temp_msg}"
                )

        return error_message
