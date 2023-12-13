# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module that contains structures relevant to interfacing models with Openlayer.

The ModelType enum chooses between different machine learning modeling frameworks.
The Model object contains information about a model on the Openlayer platform.
"""
import logging
from enum import Enum
from typing import Any, Dict

from . import exceptions, tasks, utils
from .model_runners import (
    base_model_runner,
    ll_model_runners,
    traditional_ml_model_runners,
)


class ModelType(Enum):
    """A selection of machine learning modeling frameworks supported by Openlayer.

    .. note::
        Our `sample notebooks <https://github.com/openlayer-ai/openlayer-python/tree/main/examples>`_
        show you how to use each one of these model types with Openlayer.
    """

    #: For custom built models.
    custom = "custom"
    #: For models built with `fastText <https://fasttext.cc/>`_.
    fasttext = "fasttext"
    #: For models built with `Keras <https://keras.io/>`_.
    keras = "keras"
    #: For large language models (LLMs), such as GPT
    llm = "llm"
    #: For models built with `PyTorch <https://pytorch.org/>`_.
    pytorch = "pytorch"
    #: For models built with `rasa <https://rasa.com/>`_.
    rasa = "rasa"
    #: For models built with `scikit-learn <https://scikit-learn.org/>`_.
    sklearn = "sklearn"
    #: For models built with `TensorFlow <https://www.tensorflow.org/>`_.
    tensorflow = "tensorflow"
    #: For models built with `Hugging Face transformers <https://huggingface.co/docs/transformers/index>`_.
    transformers = "transformers"
    #: For models built with `XGBoost <https://xgboost.readthedocs.io>`_.
    xgboost = "xgboost"


class Model:
    """An object containing information about a model on the Openlayer platform."""

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
        return f"Model(id={self.id})"

    def __repr__(self):
        return f"Model({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json


# --------- Function used by clients to get the correct model runner --------- #
def get_model_runner(
    **kwargs,
) -> base_model_runner.ModelRunnerInterface:
    """Factory function to get the correct model runner for the specified task type."""
    kwargs = utils.camel_to_snake_dict(kwargs)
    logger = kwargs.get("logger") or logging.getLogger("validators")
    model_package = kwargs.get("model_package")

    if model_package is not None:
        model_config = utils.camel_to_snake_dict(
            utils.read_yaml(f"{model_package}/model_config.yaml")
        )
        kwargs.update(model_config)

    return ModelRunnerFactory.create_model_runner(logger, **kwargs)


# --------------------- Factory method for model runners --------------------- #
class ModelRunnerFactory:
    """Factory class for creating model runners.

    The factory method `create_model_runner` takes in kwargs, which can include
    the `task_type` and returns the appropriate model runner.
    """

    # TODO: Create enum for LLM model providers
    _LLM_PROVIDERS = {
        "Anthropic": ll_model_runners.AnthropicModelRunner,
        "Cohere": ll_model_runners.CohereGenerateModelRunner,
        "OpenAI": ll_model_runners.OpenAIChatCompletionRunner,
        "SelfHosted": ll_model_runners.SelfHostedLLModelRunner,
        "HuggingFace": ll_model_runners.HuggingFaceModelRunner,
        "Google": ll_model_runners.GoogleGenAIModelRunner,
    }
    _MODEL_RUNNERS = {
        tasks.TaskType.TabularClassification.value: traditional_ml_model_runners.ClassificationModelRunner,
        tasks.TaskType.TabularRegression.value: traditional_ml_model_runners.RegressionModelRunner,
        tasks.TaskType.TextClassification.value: traditional_ml_model_runners.ClassificationModelRunner,
    }
    _LL_MODEL_RUNNERS = {
        tasks.TaskType.LLM.value: _LLM_PROVIDERS,
        tasks.TaskType.LLMNER.value: _LLM_PROVIDERS,
        tasks.TaskType.LLMQuestionAnswering.value: _LLM_PROVIDERS,
        tasks.TaskType.LLMSummarization.value: _LLM_PROVIDERS,
        tasks.TaskType.LLMTranslation.value: _LLM_PROVIDERS,
    }

    @staticmethod
    def create_model_runner(logger: logging.Logger, **kwargs: Dict[str, Any]):
        """Factory method for model runners.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger to use for logging the model runner runs.
        **kwargs : Dict[str, Any]
            Keyword arguments to pass to the model runner.
        """
        task_type = kwargs.pop("task_type", None)
        if isinstance(task_type, str):
            task_type = tasks.TaskType(task_type)

        if task_type is None:
            raise ValueError("Task type is required.")

        if task_type.value in ModelRunnerFactory._MODEL_RUNNERS:
            return ModelRunnerFactory._create_traditional_ml_model_runner(
                task_type=task_type, logger=logger, **kwargs
            )
        elif task_type.value in ModelRunnerFactory._LL_MODEL_RUNNERS:
            return ModelRunnerFactory._create_ll_model_runner(
                task_type=task_type, logger=logger, **kwargs
            )
        else:
            raise ValueError(f"Task type `{task_type}` is not supported.")

    @staticmethod
    def _create_traditional_ml_model_runner(
        task_type: tasks.TaskType, logger: logging.Logger, **kwargs
    ) -> base_model_runner.ModelRunnerInterface:
        """Factory method for traditional ML model runners."""
        model_runner_class = ModelRunnerFactory._MODEL_RUNNERS[task_type.value]
        return model_runner_class(logger=logger, **kwargs)

    @staticmethod
    def _create_ll_model_runner(
        task_type: tasks.TaskType, logger: logging.Logger, **kwargs
    ) -> base_model_runner.ModelRunnerInterface:
        """Factory method for LLM runners."""
        model_provider = kwargs.get("model_provider")

        if model_provider is None:
            raise ValueError("Model provider is required for LLM task types.")

        if model_provider not in ModelRunnerFactory._LLM_PROVIDERS:
            raise exceptions.OpenlayerUnsupportedLlmProvider(
                provider=model_provider,
                message="\nCurrently, the supported providers are: 'OpenAI', 'Cohere',"
                " 'Anthropic', 'SelfHosted', 'HuggingFace', and 'Google'."
                " Reach out if you'd like us to support your use case.",
            )

        model_runner_class = ModelRunnerFactory._LL_MODEL_RUNNERS[task_type.value][
            model_provider
        ]
        return model_runner_class(logger=logger, **kwargs)
