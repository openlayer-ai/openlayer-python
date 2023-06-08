"""Module for the Project class.
"""

from . import tasks


class Project:
    """An object containing information about a project on the Openlayer platform."""

    def __init__(self, json, upload, client, subscription_plan=None):
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
    ):
        """Adds a model to a project's staging area."""
        return self.client.add_model(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_baseline_model(
        self,
        *args,
        **kwargs,
    ):
        """Adds a baseline model to the project."""
        return self.client.add_baseline_model(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataset(
        self,
        *args,
        **kwargs,
    ):
        """Adds a dataset to a project's staging area (from a csv)."""
        return self.client.add_dataset(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataframe(self, *args, **kwargs):
        """Adds a dataset to a project's staging area (from a pandas DataFrame)."""
        return self.client.add_dataframe(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def commit(self, *args, **kwargs):
        """Adds a commit message to staged resources."""
        return self.client.commit(*args, project_id=self.id, **kwargs)

    def push(self, *args, **kwargs):
        """Pushes the commited resources to the platform."""
        return self.client.push(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def export(self, *args, **kwargs):
        """Exports the commited resources to a specified location."""
        return self.client.export(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def status(self, *args, **kwargs):
        """Shows the state of the staging area."""
        return self.client.status(*args, project_id=self.id, **kwargs)

    def restore(self, *args, **kwargs):
        """Removes the resource specified by ``resource_name`` from the staging area."""
        return self.client.restore(*args, project_id=self.id, **kwargs)
