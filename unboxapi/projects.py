from unboxapi import tasks

from .datasets import Dataset
from .models import Model


class Project:
    """An object containing information about a project on the Unbox platform."""

    def __init__(self, json, upload, subscription_plan, client):
        self._json = json
        self.id = json["id"]
        self.upload = upload
        self.subscription_plan = subscription_plan
        self.client = client

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Project(id={self.id})"

    def __repr__(self):
        return f"Project({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json

    def add_model(
        self,
        *args,
        **kwargs,
    ) -> Model:
        return self.client.add_model(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataset(
        self,
        *args,
        **kwargs,
    ) -> Dataset:
        return self.client.add_dataset(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataframe(self, *args, **kwargs) -> Dataset:
        return self.client.add_dataframe(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )
