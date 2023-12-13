"""Implements the project specific validation class.
"""
from typing import Dict

import marshmallow as ma

from ..schemas import project_schemas
from .base_validator import BaseValidator


class ProjectValidator(BaseValidator):
    """Validates the project.

    Parameters
    ----------
    project_config : Dict[str, str]
        The project configuration.
    """

    def __init__(
        self,
        project_config: Dict[str, str],
    ):
        super().__init__(resource_display_name="project")
        self.project_config = project_config

    def _validate(self):
        """Validates the project."""
        self._validate_project_config()

    def _validate_project_config(self):
        """Checks if the project configuration is valid."""
        project_schema = project_schemas.ProjectSchema()
        try:
            project_schema.load(
                {
                    "name": self.project_config.get("name"),
                    "description": self.project_config.get("description"),
                    "task_type": self.project_config.get("task_type").value,
                }
            )
        except ma.ValidationError as err:
            self.failed_validations.extend(self._format_marshmallow_error_message(err))
