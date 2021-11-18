from enum import Enum


class TaskType(Enum):
    """Task Type List"""

    TextClassification = "text-classification"
    TabularClassification = "tabular-classification"
    TabularRegression = "tabular-regression"


class Task:
    """Task class."""

    def __init__(self, json):
        self._json = json
        self.id = json["id"]

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Task(id={self.id})"

    def __repr__(self):
        return f"Task({self._json})"

    def to_dict(self):
        """Returns object properties as a dict
        Returns:
            Dict with object properties
        """
        return self._json
