"""Module for the InferencePipeline class.
"""
from . import tasks


class InferencePipeline:
    """An object containing information about an inference pipeline
    on the Openlayer platform."""

    def __init__(self, json, upload, client):
        self._json = json
        self.id = json["id"]
        self.upload = upload
        self.client = client

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"InferencePipeline(id={self.id})"

    def __repr__(self):
        return f"InferencePipeline({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json

    def upload_reference_dataset(
        self,
        *args,
        **kwargs,
    ):
        """Uploads a reference dataset to the project."""
        return self.client.upload_reference_dataset(
            *args,
            inference_pipeline_id=self.id,
            task_type=tasks.TaskType(self.taskType),
            **kwargs,
        )

    def upload_reference_dataframe(
        self,
        *args,
        **kwargs,
    ):
        """Uploads a reference dataframe to the project."""
        return self.client.upload_reference_dataframe(
            *args,
            inference_pipeline_id=self.id,
            task_type=tasks.TaskType(self.taskType),
            **kwargs,
        )
