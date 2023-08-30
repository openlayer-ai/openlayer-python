"""Module for the InferencePipeline class.
"""


class InferencePipeline:
    """An object containing information about an inference pipeline
    on the Openlayer platform."""

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
