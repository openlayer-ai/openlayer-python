"""Implements the inference pipeline validation class.
"""
from typing import Dict

import marshmallow as ma

from ..schemas import inference_pipeline_schemas
from .base_validator import BaseValidator


class InferencePipelineValidator(BaseValidator):
    """Validates the inference pipeline.

    Parameters
    ----------
    inference_pipeline_config : Dict[str, str]
        The inference pipeline configuration.
    """

    def __init__(
        self,
        inference_pipeline_config: Dict[str, str],
    ):
        super().__init__(resource_display_name="inference pipeline")
        self.inference_pipeline_config = inference_pipeline_config

    def _validate(self):
        """Validates the project."""
        self._validate_inference_pipeline_config()

    def _validate_inference_pipeline_config(self):
        """Checks if the inference pipeline configuration is valid."""
        inference_pipeline_schema = inference_pipeline_schemas.InferencePipelineSchema()
        try:
            inference_pipeline_schema.load(
                {
                    "name": self.inference_pipeline_config.get("name"),
                    "description": self.inference_pipeline_config.get("description"),
                }
            )
        except ma.ValidationError as err:
            self.failed_validations.extend(self._format_marshmallow_error_message(err))
