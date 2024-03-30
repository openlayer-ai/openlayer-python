"""Module for streaming data to the Openlayer platform.

Validates the arguments needed for data streaming and handles the streaming
process.
"""

import logging
from typing import Dict, Optional

import pandas as pd

import openlayer

from .. import inference_pipelines, tasks, utils

logger = logging.getLogger(__name__)


class DataStreamer:
    """Handles everything related to streaming data to the Openlayer platform,
    including creating and managing inference pipelines.
    """

    def __init__(
        self,
        openlayer_api_key: Optional[str] = None,
        openlayer_project_name: Optional[str] = None,
        openlayer_inference_pipeline_name: Optional[str] = None,
        openlayer_inference_pipeline_id: Optional[str] = None,
    ) -> None:
        self._openlayer_api_key = openlayer_api_key or utils.get_env_variable(
            "OPENLAYER_API_KEY"
        )
        self._openlayer_project_name = openlayer_project_name or utils.get_env_variable(
            "OPENLAYER_PROJECT_NAME"
        )
        self._openlayer_inference_pipeline_name = (
            openlayer_inference_pipeline_name
            or utils.get_env_variable("OPENLAYER_INFERENCE_PIPELINE_NAME")
            or "production"
        )
        self._openlayer_inference_pipeline_id = (
            openlayer_inference_pipeline_id
            or utils.get_env_variable("OPENLAYER_INFERENCE_PIPELINE_ID")
        )

        # Lazy load the inference pipeline
        self.inference_pipeline = None

    @property
    def openlayer_api_key(self) -> Optional[str]:
        """The Openlayer API key."""
        return self._get_openlayer_attribute("_openlayer_api_key", "OPENLAYER_API_KEY")

    @property
    def openlayer_project_name(self) -> Optional[str]:
        """The name of the project on Openlayer."""
        return self._get_openlayer_attribute(
            "_openlayer_project_name", "OPENLAYER_PROJECT_NAME"
        )

    @property
    def openlayer_inference_pipeline_name(self) -> Optional[str]:
        """The name of the inference pipeline on Openlayer."""
        return self._get_openlayer_attribute(
            "_openlayer_inference_pipeline_name", "OPENLAYER_INFERENCE_PIPELINE_NAME"
        )

    @property
    def openlayer_inference_pipeline_id(self) -> Optional[str]:
        """The id of the inference pipeline on Openlayer."""
        return self._get_openlayer_attribute(
            "_openlayer_inference_pipeline_id", "OPENLAYER_INFERENCE_PIPELINE_ID"
        )

    def _get_openlayer_attribute(
        self, attribute_name: str, env_variable: str
    ) -> Optional[str]:
        """A helper method to fetch an Openlayer attribute value.

        Args:
            attribute_name: The name of the attribute in this class.
            env_variable: The name of the environment variable to fetch.
        """
        attribute_value = getattr(self, attribute_name, None)
        if not attribute_value:
            attribute_value = utils.get_env_variable(env_variable)
            setattr(self, attribute_name, attribute_value)
        return attribute_value

    def _validate_attributes(self) -> None:
        """Granular validation of the arguments."""
        if not self.openlayer_api_key:
            logger.error(
                "An Openlayer API key is required for publishing."
                " Please set it as environment variable named OPENLAYER_API_KEY."
            )

        if (
            not self.openlayer_project_name
            and not self.openlayer_inference_pipeline_name
            and not self.openlayer_inference_pipeline_id
        ):
            logger.error(
                "You must provide more information about the project and"
                " inference pipeline on Openlayer to publish data."
                " Please provide either: "
                " - the project name and inference pipeline name, or"
                " - the inference pipeline id."
                " You can set them as environment variables named"
                " OPENLAYER_PROJECT_NAME, OPENLAYER_INFERENCE_PIPELINE_NAME, "
                "and OPENLAYER_INFERENCE_PIPELINE_ID."
            )

        if (
            self.openlayer_inference_pipeline_name
            and not self.openlayer_project_name
            and not self.openlayer_inference_pipeline_id
        ):
            logger.error(
                "You must provide the Openlayer project name where the inference"
                " pipeline is located."
                " Please set it as the environment variable"
                " OPENLAYER_PROJECT_NAME."
            )

    def stream_data(self, data: Dict[str, any], config: Dict[str, any]) -> None:
        """Stream data to the Openlayer platform.

        Args:
            data: The data to be streamed.
            config: The configuration for the data stream.
        """

        self._validate_attributes()
        self._check_inference_pipeline_ready()
        self.inference_pipeline.stream_data(stream_data=data, stream_config=config)
        logger.info("Data streamed to Openlayer.")

    def _check_inference_pipeline_ready(self) -> None:
        """Lazy load the inference pipeline and check if it is ready."""
        if self.inference_pipeline is None:
            self._load_inference_pipeline()
            if self.inference_pipeline is None:
                logger.error(
                    "No inference pipeline found. Please provide the inference pipeline"
                    " id or name."
                )

    def _load_inference_pipeline(self) -> None:
        """Load inference pipeline from the Openlayer platform.

        If no platform/project information is provided, it is set to None.
        """

        inference_pipeline = None
        try:
            client = openlayer.OpenlayerClient(
                api_key=self.openlayer_api_key, verbose=False
            )

            # Prioritize the inference pipeline id over the name
            if self.openlayer_inference_pipeline_id:
                inference_pipeline = inference_pipelines.InferencePipeline(
                    client=client,
                    upload=None,
                    json={
                        "id": self.openlayer_inference_pipeline_id,
                        "projectId": None,
                    },
                    task_type=tasks.TaskType.LLM,
                )
            elif self.openlayer_inference_pipeline_name:
                with utils.HidePrints():
                    project = client.create_project(
                        name=self.openlayer_project_name, task_type=tasks.TaskType.LLM
                    )
                    inference_pipeline = project.create_inference_pipeline(
                        name=self.openlayer_inference_pipeline_name
                    )
            if inference_pipeline:
                logger.info(
                    "Going to try to stream data to the inference pipeline with id %s.",
                    inference_pipeline.id,
                )
            else:
                logger.warning(
                    "No inference pipeline found. Data will not be streamed to "
                    "Openlayer."
                )
            self.inference_pipeline = inference_pipeline
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "An error occurred while trying to load the inference pipeline: %s", exc
            )

    def publish_batch_data(self, df: pd.DataFrame, config: Dict[str, any]) -> None:
        """Publish a batch of data to the Openlayer platform.

        Args:
            df: The data to be published.
            config: The configuration for the data stream.
        """
        self._check_inference_pipeline_ready()
        self.inference_pipeline.publish_batch_data(batch_df=df, batch_config=config)
        logger.info("Batch of data published to Openlayer.")
