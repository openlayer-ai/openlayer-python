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
        publish: bool = False,
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
        self.publish = publish

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
        if self.publish:
            if not self.openlayer_api_key:
                raise ValueError(
                    "An Openlayer API key is required for publishing."
                    " Please provide `openlayer_api_key` or set the"
                    " OPENLAYER_API_KEY environment variable."
                )

            if not self.openlayer_project_name:
                raise ValueError(
                    "You must specify the name of the project on Openlayer"
                    " that you want to publish to. Please provide"
                    " `openlayer_project_name` or set the OPENLAYER_PROJECT_NAME"
                    " environment variable."
                )

        if (
            not self.openlayer_inference_pipeline_id
            and not self.openlayer_inference_pipeline_name
        ):
            raise ValueError(
                "Either an inference pipeline id or name is required."
                " Please provide `openlayer_inference_pipeline_id` or"
                " `openlayer_inference_pipeline_name`, "
                "or set the OPENLAYER_INFERENCE_PIPELINE_ID or"
                " OPENLAYER_INFERENCE_PIPELINE_NAME environment variables."
            )
        logger.info(
            "Data will be streamed to Openlayer project %s and inference pipeline %s.",
            self.openlayer_project_name,
            (
                self.openlayer_inference_pipeline_id
                or self.openlayer_inference_pipeline_name
            ),
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
                raise ValueError(
                    "No inference pipeline found. Please provide the inference pipeline"
                    " id or name."
                )

    def _load_inference_pipeline(self) -> None:
        """Load inference pipeline from the Openlayer platform.

        If no platform/project information is provided, it is set to None.
        """
        inference_pipeline = None
        if self.openlayer_api_key:
            client = openlayer.OpenlayerClient(
                api_key=self.openlayer_api_key, verbose=False
            )
            if self.openlayer_inference_pipeline_id:
                # Load inference pipeline directly from the id
                inference_pipeline = inference_pipelines.InferencePipeline(
                    client=client,
                    upload=None,
                    json={
                        "id": self.openlayer_inference_pipeline_id,
                        "projectId": None,
                    },
                    task_type=tasks.TaskType.LLM,
                )
            else:
                if self.openlayer_project_name:
                    with utils.HidePrints():
                        project = client.create_project(
                            name=self.openlayer_project_name,
                            task_type=tasks.TaskType.LLM,
                        )
                        inference_pipeline = project.create_inference_pipeline(
                            name=self.openlayer_inference_pipeline_name
                        )

        self.inference_pipeline = inference_pipeline

    def publish_batch_data(self, df: pd.DataFrame, config: Dict[str, any]) -> None:
        """Publish a batch of data to the Openlayer platform.

        Args:
            df: The data to be published.
            config: The configuration for the data stream.
        """
        self._check_inference_pipeline_ready()
        self.inference_pipeline.publish_batch_data(batch_df=df, batch_config=config)
        logger.info("Batch of data published to Openlayer.")
