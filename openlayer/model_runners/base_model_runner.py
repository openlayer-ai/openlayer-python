# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module that defines the interface for all (concrete) model runners.
"""
import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .. import utils
from . import environment


class ModelRunnerInterface(ABC):
    """Interface for model runners."""

    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)

        model_package = kwargs.get("model_package")
        if model_package is not None:
            self.init_from_model_package(model_package)
        else:
            self.init_from_kwargs(**kwargs)

        self.validate_minimum_viable_config()

    def init_from_model_package(self, model_package: str) -> None:
        """Initializes the model runner from the model package.

        I.e., using the model_config.yaml file located in the model package
        directory.
        """
        self.model_package = model_package

        # Model config is originally a dict with camelCase keys
        self.model_config = utils.camel_to_snake_dict(
            utils.read_yaml(f"{model_package}/model_config.yaml")
        )

        self._conda_environment = None
        self.in_memory = True
        python_version_file_path = f"{model_package}/python_version"
        requirements_file_path = f"{model_package}/requirements.txt"
        if os.path.isfile(python_version_file_path) and os.path.isfile(
            requirements_file_path
        ):
            self.in_memory = False
            self._conda_environment = environment.CondaEnvironment(
                env_name=f"model-runner-env-{datetime.datetime.now().strftime('%m-%d-%H-%M-%S-%f')}",
                requirements_file_path=requirements_file_path,
                python_version_file_path=python_version_file_path,
                logger=self.logger,
            )

    def init_from_kwargs(self, **kwargs) -> None:
        """Initializes the model runner from the kwargs."""
        self.model_package = None
        self._conda_environment = None
        self.in_memory = True
        self.model_config = kwargs

    @abstractmethod
    def validate_minimum_viable_config(self) -> None:
        """Superficial validation of the minimum viable config needed to use
        the model runner.

        Each concrete model runner must implement this method.
        """
        pass

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model."""
        if self.in_memory:
            return self._run_in_memory(input_data)
        else:
            return self._run_in_conda(input_data)

    @abstractmethod
    def _run_in_memory(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the model in memory."""
        pass

    @abstractmethod
    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the model in a conda environment."""
        pass

    def __del__(self):
        if self._conda_environment is not None:
            self._conda_environment.delete()
