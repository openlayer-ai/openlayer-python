# pylint: disable=invalid-name
"""TaskTypes supported by Openlayer are defined here

TaskTypes enum chooses between the types of machine learning tasks supported by Openlayer.
Examples of these tasks are text classification, tabular classification, and tabular regression.

Typical usage example:

    task_type = ma.fields.Str(
        alidate=ma.validate.OneOf(
            [task_type.value for task_type in TaskType],
            error="`task_type` must be one of the supported tasks.",
        )
    )
"""
from enum import Enum


class TaskType(Enum):
    """A selection of machine learning tasks supported by Openlayer.

    .. note::
        Our `sample notebooks <https://github.com/openlayer-ai/openlayer-python/tree/main/examples>`_
        show you how to use each one of these task types with Openlayer.
    """

    #: For sequence classification tasks.
    TextClassification = "text-classification"
    #: For tabular classification tasks.
    TabularClassification = "tabular-classification"
    #: Coming soon!
    TabularRegression = "tabular-regression"
