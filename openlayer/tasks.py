# pylint: disable=invalid-name
"""TaskTypes supported by Openlayer are defined here

TaskTypes enum chooses between the types of machine learning tasks supported by
Openlayer. Examples of these tasks are text classification, tabular classification, and
tabular regression.
"""
from enum import Enum


class TaskType(Enum):
    """Enum for the AI/ML tasks types supported by Openlayer.

    The task type is used during project creation with the
    :meth:`openlayer.OpenlayerClient.create_project` method.

    It also determines the tests available on the platform and the information
    required to add models and datasets to the project.

    .. note::
        The `sample notebooks <https://github.com/openlayer-ai/openlayer-python/tree/main/examples>`_
        show you how to create projects for each of these task types.
    """

    #: For entity recognition tasks with LLMs.
    LLMNER = "llm-ner"
    #: For question answering tasks with LLMs.
    LLMQuestionAnswering = "llm-question-answering"
    #: For summarization tasks with LLMs.
    LLMSummarization = "llm-summarization"
    #: For translation tasks with LLMs.
    LLMTranslation = "llm-translation"
    #: For general LLM tasks (none of the above).
    LLM = "llm-base"
    #: For tabular classification tasks.
    TabularClassification = "tabular-classification"
    #: For tabular regression tasks.
    TabularRegression = "tabular-regression"
    #: For text classification tasks.
    TextClassification = "text-classification"
