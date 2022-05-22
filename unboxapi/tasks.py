from enum import Enum


class TaskType(Enum):
    """A selection of machine learning tasks supported by Unbox.

    .. note::
        Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/main/examples>`_
        show you how to use each one of these task types with Unbox.
    """

    #: For sequence classification tasks.
    TextClassification = "text-classification"
    #: For tabular classification tasks.
    TabularClassification = "tabular-classification"
    #: Coming soon!
    TabularRegression = "tabular-regression"
