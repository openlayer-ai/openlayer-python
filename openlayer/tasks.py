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
