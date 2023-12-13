# pylint: disable=broad-exception-caught
"""Implements the model specific validation classes.
"""

import importlib
import logging
import os
import tarfile
import tempfile
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import marshmallow as ma
import numpy as np
import pandas as pd
import pkg_resources
import yaml

from .. import constants, models, tasks, utils
from ..schemas import model_schemas
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class BaseModelValidator(BaseValidator, ABC):
    """Base model validator.

    Parameters
    ----------
    model_config_file_path: str, optional
        Path to the model config file.
    model_config: Dict[str, any], optional
        Model config dictionary.
    task_type : tasks.TaskType
        Task type of the model.
    model_package_dir : str
        Path to the model package directory.
    sample_data : pd.DataFrame
        Sample data to be used for the model validation.
    """

    def __init__(
        self,
        task_type: tasks.TaskType,
        model_config_file_path: Optional[str] = None,
        model_config: Optional[Dict[str, any]] = None,
        use_runner: bool = False,
        model_package_dir: Optional[str] = None,
        sample_data: Optional[pd.DataFrame] = None,
    ):
        super().__init__(resource_display_name="model")
        if model_config_file_path and model_config:
            raise ValueError(
                "Both model_config_file_path and model_config are provided."
                " Please provide only one of them."
            )
        if not model_config_file_path and not model_config:
            raise ValueError(
                "Neither model_config_file_path nor model_config_file is provided."
                " Please provide one of them."
            )
        self.model_config_file_path = model_config_file_path
        self.model_config = model_config
        self.model_package_dir = model_package_dir
        self.sample_data = sample_data
        self._use_runner = use_runner
        self.task_type = task_type

        # Attributes to be set during validation
        self.model_config: Optional[Dict[str, any]] = None
        self.model_output: Optional[np.ndarray] = None

    def _validate(self) -> None:
        """Runs all model validations.

        At each stage, prints all the failed validations.

        Returns
        -------
        List[str]
            A list of all failed validations.
        """
        if self.model_package_dir:
            self._validate_model_package_dir()
            if self._use_runner:
                self._validate_model_runner()
            else:
                self._validate_requirements_file()
                self._validate_prediction_interface()
        self._validate_model_config()

    def _validate_model_package_dir(self):
        """Verifies the model package directory structure.

        The model package directory must follow the structure:

        model_package
          ├── artifacts.pkl  # potentially different name / format and multiple files
          ├── prediction_interface.py
          └── requirements.txt

        This method checks for the existence of the above files.
        """
        if not os.path.exists(self.model_package_dir):
            self.failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not exist."
            )

        if not os.path.isdir(self.model_package_dir):
            self.failed_validations.append(
                f"Model package directory `{self.model_package_dir}` is not a directory."
            )

        if self.model_package_dir == os.getcwd():
            self.failed_validations.append(
                f"Model package directory `{self.model_package_dir}` is the current "
                "working directory."
            )

        if not os.path.exists(
            os.path.join(self.model_package_dir, "prediction_interface.py")
        ):
            self.failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not contain the "
                "`prediction_interface.py` file."
            )

        if not os.path.exists(os.path.join(self.model_package_dir, "requirements.txt")):
            self.failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not contain the "
                "`requirements.txt` file."
            )

    def _validate_requirements_file(self):
        """Validates the requirements.txt file.

        Checks for the existence of the file and parses it to check for
        version discrepancies. Appends to the list of failed validations,
        if the file does not exist, and raises warnings in case of
        discrepancies.

        Beware of the order of the validations, as it is important.
        """
        # Path to the requirements.txt file
        requirements_txt_file = os.path.join(self.model_package_dir, "requirements.txt")

        # File existence check
        if not os.path.isfile(os.path.expanduser(requirements_txt_file)):
            self.failed_validations.append(
                f"File `{requirements_txt_file}` does not exist."
            )
        else:
            with open(requirements_txt_file, "r", encoding="UTF-8") as file:
                lines = file.readlines()

            # Parse the requirements file
            requirements = pkg_resources.parse_requirements(lines)

            for requirement in requirements:
                requirement = str(requirement)

                # Consistency checks between requirements and modules installed in the environment
                try:
                    pkg_resources.require(requirement)
                except pkg_resources.VersionConflict as err:
                    try:
                        warnings.warn(
                            "There is a version discrepancy between the current "
                            f"environment and the dependency `{requirement}`. \n"
                            f"`requirements.txt` specifies `{err.req}`, but the current "
                            f"environment contains `{err.dist}` installed. \n"
                            "There might be unexpected results once the model is in the platform. "
                            "Use at your own discretion.",
                            category=Warning,
                        )
                        return None
                    except AttributeError:
                        warnings.warn(
                            "There is a version discrepancy between the current "
                            f"environment and the dependency `{requirement}`. \n"
                            f"`requirements.txt` specifies `{requirement}`, but the current "
                            f"environment contains an incompatible version installed. \n"
                            "There might be unexpected results once the model is in the platform. "
                            "Use at your own discretion.",
                            category=Warning,
                        )
                        return None
                except pkg_resources.DistributionNotFound:
                    warnings.warn(
                        f"The dependency `{requirement}` specified in the `requirements.txt` "
                        "is not installed in the current environment. \n"
                        "There might be unexpected results once the model is in the platform. "
                        "Use at your own discretion.",
                        category=Warning,
                    )

    def _validate_model_config(self):
        """Checks whether the model_config.yaml file exists and is valid.

        Beware of the order of the validations, as it is important.
        """
        model_config_failed_validations = []

        # File existence check
        if self.model_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.model_config_file_path)):
                model_config_failed_validations.append(
                    f"File `{self.model_config_file_path}` does not exist."
                )
            else:
                with open(self.model_config_file_path, "r", encoding="UTF-8") as stream:
                    self.model_config = yaml.safe_load(stream)

        if self.model_config:
            model_schema = model_schemas.ModelSchema()
            try:
                model_schema.load(
                    {"task_type": self.task_type.value, **self.model_config}
                )
            except ma.ValidationError as err:
                model_config_failed_validations.extend(
                    self._format_marshmallow_error_message(err)
                )

        # Add the `model_config.yaml` failed validations to the list of all failed validations
        self.failed_validations.extend(model_config_failed_validations)

    def _validate_model_runner(self):
        """Validates the model using the model runner.

        This is mostly meant to be used by the platform, to validate the model. It will
        create the model's environment and use it to run the model.
        """
        model_runner = models.get_model_runner(
            task_type=self.task_type, model_package=self.model_package_dir
        )

        # Try to run some data through the runner
        # Will create the model environment if it doesn't exist
        try:
            model_runner.run(self.sample_data)
        except Exception as exc:
            self.failed_validations.append(f"{exc}")

    @abstractmethod
    def _validate_prediction_interface(self):
        """Validates the prediction interface.

        This method should be implemented by the child classes,
        since each task type has a different prediction interface.
        """
        pass


class ClassificationModelValidator(BaseModelValidator):
    """Implements specific validations for classification models,
    such as the prediction interface, model runner, etc.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _validate_prediction_interface(self):
        """Validates the implementation of the prediction interface.

        Checks for the existence of the file, the required functions, and
        runs test data through the model to ensure there are no implementation
        errors.

        Beware of the order of the validations, as it is important.
        """
        # Path to the prediction_interface.py file
        prediction_interface_file = os.path.join(
            self.model_package_dir, "prediction_interface.py"
        )

        # File existence check
        if not os.path.isfile(os.path.expanduser(prediction_interface_file)):
            self.failed_validations.append(
                f"File `{prediction_interface_file}` does not exist."
            )
        else:
            # Loading the module defined in the prediction_interface.py file
            module_spec = importlib.util.spec_from_file_location(
                "model_module", prediction_interface_file
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            # Check if the module contains the required functions
            if not hasattr(module, "load_model"):
                self.failed_validations.append(
                    "The `load_model` function is not defined in the `prediction_interface.py` "
                    "file."
                )
            else:
                # Test `load_model` function
                ml_model = None
                try:
                    ml_model = module.load_model()
                except Exception as exc:
                    self.failed_validations.append(
                        f"There is an error while loading the model: \n {exc}"
                    )

                if ml_model is not None:
                    # Check if the `predict_proba` method is part of the model object
                    if not hasattr(ml_model, "predict_proba"):
                        self.failed_validations.append(
                            "A `predict_proba` function is not defined in the model class "
                            "in the `prediction_interface.py` file."
                        )
                    else:
                        # Test `predict_proba` function
                        try:
                            with utils.HidePrints():
                                self.model_output = ml_model.predict_proba(
                                    self.sample_data
                                )
                        except Exception as exc:
                            exception_stack = utils.get_exception_stacktrace(exc)
                            self.failed_validations.append(
                                "The `predict_proba` function failed while running the test data. "
                                "It is failing with the following error message: \n"
                                f"\t {exception_stack}"
                            )

                        if self.model_output is not None:
                            self._validate_model_output()

    def _validate_model_output(self):
        """Validates the model output.

        Checks if the model output is an-array like object with shape (n_samples, n_classes)
        Also checks if the model output is a probability distribution.
        """
        # Check if the model output is an array-like object
        if not isinstance(self.model_output, np.ndarray):
            self.failed_validations.append(
                "The output of the `predict_proba` method in the `prediction_interface.py` "
                "file is not an array-like object. It should be a numpy array of shape "
                "(n_samples, n_classes)."
            )
        elif self.model_config is not None:
            # Check if the model output has the correct shape
            num_rows = len(self.sample_data)
            num_classes = len(self.model_config.get("classes"))
            if self.model_output.shape != (num_rows, num_classes):
                self.failed_validations.append(
                    "The output of the `predict_proba` method in the `prediction_interface.py` "
                    " has the wrong shape. It should be a numpy array of shape "
                    f"({num_rows}, {num_classes}). The current output has shape "
                    f"{self.model_output.shape}."
                )
            # Check if the model output is a probability distribution
            elif not np.allclose(self.model_output.sum(axis=1), 1, atol=0.05):
                self.failed_validations.append(
                    "The output of the `predict_proba` method in the `prediction_interface.py` "
                    "file is not a probability distribution. The sum of the probabilities for "
                    "each sample should be equal to 1."
                )


class RegressionModelValidator(BaseModelValidator):
    """Implements specific validations for classification models,
    such as the prediction interface, model runner, etc.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _validate_prediction_interface(self):
        """Validates the implementation of the prediction interface.

        Checks for the existence of the file, the required functions, and
        runs test data through the model to ensure there are no implementation
        errors.

        Beware of the order of the validations, as it is important.
        """
        # Path to the prediction_interface.py file
        prediction_interface_file = os.path.join(
            self.model_package_dir, "prediction_interface.py"
        )

        # File existence check
        if not os.path.isfile(os.path.expanduser(prediction_interface_file)):
            self.failed_validations.append(
                f"File `{prediction_interface_file}` does not exist."
            )
        else:
            # Loading the module defined in the prediction_interface.py file
            module_spec = importlib.util.spec_from_file_location(
                "model_module", prediction_interface_file
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            # Check if the module contains the required functions
            if not hasattr(module, "load_model"):
                self.failed_validations.append(
                    "The `load_model` function is not defined in the `prediction_interface.py` "
                    "file."
                )
            else:
                # Test `load_model` function
                ml_model = None
                try:
                    ml_model = module.load_model()
                except Exception as exc:
                    self.failed_validations.append(
                        f"There is an error while loading the model: \n {exc}"
                    )

                if ml_model is not None:
                    # Check if the `predict` method is part of the model object
                    if not hasattr(ml_model, "predict"):
                        self.failed_validations.append(
                            "A `predict` function is not defined in the model class "
                            "in the `prediction_interface.py` file."
                        )
                    else:
                        # Test `predict_proba` function
                        try:
                            with utils.HidePrints():
                                self.model_output = ml_model.predict(self.sample_data)
                        except Exception as exc:
                            exception_stack = utils.get_exception_stacktrace(exc)
                            self.failed_validations.append(
                                "The `predict` function failed while running the test data. "
                                "It is failing with the following error message: \n"
                                f"\t {exception_stack}"
                            )

                        if self.model_output is not None:
                            self._validate_model_output()

    def _validate_model_output(self):
        """Validates the model output.

        Checks if the model output is an-array like object with shape (n_samples,).
        """
        # Check if the model output is an array-like object
        if not isinstance(self.model_output, np.ndarray):
            self.failed_validations.append(
                "The output of the `predict` method in the `prediction_interface.py` "
                "file is not an array-like object. It should be a numpy array of shape "
                "(n_samples,)."
            )

        # Check if the model output has the correct shape
        num_rows = len(self.sample_data)
        if self.model_output.shape != (num_rows,):
            self.failed_validations.append(
                "The output of the `predict` method in the `prediction_interface.py` "
                " has the wrong shape. It should be a numpy array of shape "
                f"({num_rows},). The current output has shape "
                f"{self.model_output.shape}. "
                "If your array has one column, you can reshape it using "
                "`np.squeeze(arr, axis=1)` to remove the singleton dimension along "
                "the column axis."
            )


class TabularClassificationModelValidator(ClassificationModelValidator):
    """Tabular classification model validator."""

    pass


class TabularRegressionModelValidator(RegressionModelValidator):
    """Tabular regression model validator."""

    pass


class TextClassificationModelValidator(ClassificationModelValidator):
    """Text classification model validator."""

    pass


class LLMValidator(BaseModelValidator):
    """Agent validator.

    Parameters
    ----------
    model_config_file_path: str
        Path to the model config file.
    task_type : tasks.TaskType
        Task type of the model.
    model_package_dir : str
        Path to the model package directory.
    sample_data : pd.DataFrame
        Sample data to be used for the model validation.
    """

    def _validate(self) -> None:
        """Runs all agent validations.

        At each stage, prints all the failed validations.

        Returns
        -------
        List[str]
            A list of all failed validations.
        """
        if self.model_package_dir:
            self._validate_model_package_dir()
        self._validate_model_config()

    def _validate_model_package_dir(self):
        """Verifies that the agent directory is valid."""
        if not os.path.exists(self.model_package_dir):
            self.failed_validations.append(
                f"The agent directory `{self.model_package_dir}` does not exist."
            )

        if not os.path.isdir(self.model_package_dir):
            self.failed_validations.append(
                f"The agent directory `{self.model_package_dir}` is not a directory."
            )

        if self.model_package_dir == os.getcwd():
            self.failed_validations.append(
                f"The agent directory `{self.model_package_dir}` is the current "
                "working directory."
            )

        if dir_exceeds_size_limit(self.model_package_dir):
            self.failed_validations.append(
                f"The agent directory `{self.model_package_dir}` exceeds the size limit "
                f"of {constants.MAX_model_package_dir_SIZE_MB} MB."
            )

    def _validate_prediction_interface(self):
        """Validates the prediction interface for LLMs."""
        pass


# ----------------------------- Factory function ----------------------------- #
def get_validator(
    task_type: tasks.TaskType,
    model_config: Optional[Dict[str, any]] = None,
    model_config_file_path: Optional[str] = None,
    use_runner: bool = False,
    model_package_dir: Optional[str] = None,
    sample_data: Optional[pd.DataFrame] = None,
) -> BaseModelValidator:
    """Factory function to get the correct model validator.

    Parameters
    ----------
    task_type : :obj:`TaskType`
        The task type of the model.
    model_config : Dict[str, any], optional
        The model config dictionary, by default None.
    model_config_file_path : str, optional
        The path to the model config file.
    model_package_dir : Optional[str], optional
        The path to the model package directory, by default None.
    sample_data : Optional[pd.DataFrame], optional
        The sample data to use for validation, by default None.

    Returns
    -------
    ModelValidator
        The correct model validator for the ``task_type`` specified.


    Examples
    --------

    For example, to get the tabular model validator, you can do the following:

    >>> from openlayer.validators import model_validator
    >>> from openlayer.tasks import TaskType
    >>>
    >>> validator = model_validator.get_validator(
    >>>     task_type=TaskType.TabularClassification,
    >>>     model_config_file_path="model_config.yaml",
    >>>     model_package_dir="model_package",
    >>>     sample_data=x_val.iloc[:10, :]
    >>> )

    The ``validator`` object will be an instance of the
    :obj:`TabularClassificationModelValidator` class.

    Then, you can run the validations by calling the :obj:`validate` method:

    >>> validator.validate()

    If there are failed validations, they will be shown on the screen and a list
    of all failed validations will be returned.

    The same logic applies to the other task types.
    """
    if task_type == tasks.TaskType.TabularClassification:
        return TabularClassificationModelValidator(
            model_config=model_config,
            model_config_file_path=model_config_file_path,
            use_runner=use_runner,
            model_package_dir=model_package_dir,
            sample_data=sample_data,
            task_type=task_type,
        )
    elif task_type == tasks.TaskType.TabularRegression:
        return TabularRegressionModelValidator(
            model_config=model_config,
            model_config_file_path=model_config_file_path,
            use_runner=use_runner,
            model_package_dir=model_package_dir,
            sample_data=sample_data,
            task_type=task_type,
        )
    elif task_type == tasks.TaskType.TextClassification:
        return TextClassificationModelValidator(
            model_config=model_config,
            model_config_file_path=model_config_file_path,
            use_runner=use_runner,
            model_package_dir=model_package_dir,
            sample_data=sample_data,
            task_type=task_type,
        )
    elif task_type in [
        tasks.TaskType.LLM,
        tasks.TaskType.LLMNER,
        tasks.TaskType.LLMQuestionAnswering,
        tasks.TaskType.LLMSummarization,
        tasks.TaskType.LLMTranslation,
    ]:
        return LLMValidator(
            model_config=model_config,
            model_config_file_path=model_config_file_path,
            task_type=task_type,
        )
    else:
        raise ValueError(f"Task type `{task_type}` is not supported.")


# --------------- Helper functions used by multiple validators --------------- #
def dir_exceeds_size_limit(dir_path: str) -> bool:
    """Checks whether the tar version of the directory exceeds the maximim limit."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_file_path = os.path.join(tmp_dir, "tarfile")
        with tarfile.open(tar_file_path, mode="w:gz") as tar:
            tar.add(dir_path, arcname=os.path.basename(dir_path))
        tar_file_size = os.path.getsize(tar_file_path)

        return tar_file_size > constants.MAXIMUM_TAR_FILE_SIZE * 1024 * 1024
