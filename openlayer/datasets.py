# pylint: disable=invalid-name
"""This module contains structures relevant to interfacing with datasets on the Openlayer platform.

The DatasetType enum chooses between validation and training datasets. The Dataset object
contains information about a dataset on the Openlayer platform.

Typical usage example:

    validate=ma.validate.OneOf(
        [dataset_type.value for dataset_type in DatasetType],
        error="`label` not supported."
        + "The supported `labels` are 'training' and 'validation'."
    )

"""
from enum import Enum


class DatasetType(Enum):
    """The different dataset types that are supported by Openlayer.

    Used by the ``dataset_type`` argument of the :meth:`openlayer.OpenlayerClient.add_dataset` and
    :meth:`openlayer.OpenlayerClient.add_dataframe` methods."""

    #: For validation sets.
    Validation = "validation"
    #: For training sets.
    Training = "training"


class Dataset:
    """An object containing information about a dataset on the Openlayer platform."""

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
        return f"Dataset(id={self.id})"

    def __repr__(self):
        return f"Dataset({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json
