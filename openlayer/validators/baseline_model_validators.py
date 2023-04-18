"""Implements the baseline model specific validation classes.
"""
import logging
import os
from typing import List, Optional

import marshmallow as ma
import yaml

from .. import schemas, tasks
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class BaseBaselineModelValidator(BaseValidator):
    """Validates the baseline model.

    Parameters
    ----------
    task_type : tasks.TaskType
        The task type.
    model_config_file_path : Optional[str], optional
        The path to the model config file, by default None
    """

    def __init__(
        self, task_type: tasks.TaskType, model_config_file_path: Optional[str] = None
    ):
        super().__init__(resource_display_name="baseline model")
        self.task_type = task_type
        self.model_config_file_path = model_config_file_path

    def _validate(self) -> List[str]:
        """Validates the baseline model.
        Returns
        -------
        List[str]
            The list of failed validations.
        """
        if self.model_config_file_path:
            self._validate_model_config()

    def _validate_model_config(self):
        """Validates the model config file."""
        # File existence check
        if self.model_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.model_config_file_path)):
                self.failed_validations.append(
                    f"File `{self.model_config_file_path}` does not exist."
                )
            else:
                with open(self.model_config_file_path, "r", encoding="UTF-8") as stream:
                    model_config = yaml.safe_load(stream)

        if model_config:
            baseline_model_schema = schemas.BaselineModelSchema()
            try:
                baseline_model_schema.load(
                    {"task_type": self.task_type.value, **model_config}
                )
            except ma.ValidationError as err:
                self.failed_validations.extend(
                    self._format_marshmallow_error_message(err)
                )


class TabularClassificationBaselineModelValidator(BaseBaselineModelValidator):
    """Baseline model validator for tabular classification."""

    pass


# ----------------------------- Factory function ----------------------------- #
def get_validator(
    task_type: tasks.TaskType,
    model_config_file_path: str,
) -> BaseBaselineModelValidator:
    """Factory function to get the correct baseline model validator.

    Parameters
    ----------
        task_type: The task type of the model.
        model_config_file_path: Path to the model config file.

    Returns
    -------
        The correct model validator.
    """
    if task_type == tasks.TaskType.TabularClassification:
        return TabularClassificationBaselineModelValidator(
            model_config_file_path=model_config_file_path,
            task_type=task_type,
        )
    else:
        raise ValueError(
            f"Task type `{task_type}` is not supported for baseline models."
        )
