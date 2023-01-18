import importlib
import os
import traceback
import warnings
from typing import List

import marshmallow as ma
import pandas as pd
import pkg_resources
import yaml

from . import schemas, utils


class ModelValidator:
    """Validates the model package's structure and files prior to the upload.

    Args:
        model_package_dir (str): path to the model package directory.
        sample_data (pd.DataFrame): sample data to be used for the model validation.
    """

    def __init__(
        self,
        model_package_dir: str,
        sample_data: pd.DataFrame,
    ):
        self.model_package_dir = model_package_dir

        if not isinstance(sample_data, pd.DataFrame):
            raise ValueError(
                "Test data must be a pandas DataFrame with at least 2 rows."
            )
        if len(sample_data) < 2:
            raise ValueError(
                f"Test data must contain at least 2 rows, but only {len(sample_data)} "
                "rows were provided."
            )
        self.sample_data = sample_data
        self.failed_validations = []

    def _validate_model_package_dir(self):
        """Verifies the model package directory structure.

        The model package directory must follow the structure:

        model_package
          ├── artifacts.pkl  # potentially different name / format and multiple files
          ├── model_config.yaml
          ├── prediction_interface.py
          └── requirements.txt

        This method checks for the existence of the above files.
        """
        model_package_failed_validations = []

        if not os.path.exists(self.model_package_dir):
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not exist."
            )

        if not os.path.isdir(self.model_package_dir):
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` is not a directory."
            )

        if self.model_package_dir == os.getcwd():
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` is the current "
                "working directory."
            )

        if not os.path.exists(
            os.path.join(self.model_package_dir, "prediction_interface.py")
        ):
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not contain the "
                "`prediction_interface.py` file."
            )

        if not os.path.exists(os.path.join(self.model_package_dir, "requirements.txt")):
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not contain the "
                "`requirements.txt` file."
            )

        if not os.path.exists(
            os.path.join(self.model_package_dir, "model_config.yaml")
        ):
            model_package_failed_validations.append(
                f"Model package directory `{self.model_package_dir}` does not contain the "
                "`model_config.yaml` file."
            )

        # Print results of the validation
        if model_package_failed_validations:
            print("Model package structure failed validations: \n")
            _list_failed_validation_messages(model_package_failed_validations)

        # Add the model package failed validations to the list of all failed validations
        self.failed_validations.extend(model_package_failed_validations)

    def _validate_requirements(self):
        """Validates the requirements.txt file.

        Checks for the existence of the file and parses it to check for
        version discrepancies. Appends to the list of failed validations,
        if the file does not exist, and raises warnings in case of
        discrepancies.

        Beware of the order of the validations, as it is important.
        """
        requirements_failed_validations = []

        # Path to the requirements.txt file
        requirements_txt_file = os.path.join(self.model_package_dir, "requirements.txt")

        # File existence check
        if not os.path.isfile(os.path.expanduser(requirements_txt_file)):
            requirements_failed_validations.append(
                f"File `{requirements_txt_file}` does not exist."
            )
        else:
            with open(requirements_txt_file, "r") as f:
                lines = f.readlines()

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
                except pkg_resources.DistributionNotFound as err:
                    warnings.warn(
                        f"The dependency `{requirement}` specified in the `requirements.txt` "
                        "is not installed in the current environment. \n"
                        "There might be unexpected results once the model is in the platform. "
                        "Use at your own discretion.",
                        category=Warning,
                    )

        # Print results of the validation
        if requirements_failed_validations:
            print("`requirements.txt` failed validations: \n")
            _list_failed_validation_messages(requirements_failed_validations)

        # Add the `requirements.txt` failed validations to the list of all failed validations
        self.failed_validations.extend(requirements_failed_validations)

    def _validate_model_config(self):
        """Checks whether the model_config.yaml file exists and is valid.

        Beware of the order of the validations, as it is important.
        """
        model_config_failed_validations = []

        # Path to the model_config.yaml file
        model_config_file = os.path.join(self.model_package_dir, "model_config.yaml")

        # File existence check
        if not os.path.isfile(os.path.expanduser(model_config_file)):
            model_config_failed_validations.append(
                f"File `{model_config_file}` does not exist."
            )
        else:
            with open(model_config_file, "r") as stream:
                model_config = yaml.safe_load(stream)

            model_schema = schemas.ModelSchema()
            try:
                model_schema.load(
                    {
                        "name": model_config.get("name"),
                        "model_type": model_config.get("model_type"),
                        "class_names": model_config.get("class_names"),
                        "feature_names": model_config.get("feature_names", []),
                        "categorical_feature_names": model_config.get(
                            "categorical_feature_names", []
                        ),
                    }
                )
            except ma.ValidationError as err:
                model_config_failed_validations.extend(
                    _format_marshmallow_error_message(err)
                )

        # Print results of the validation
        if model_config_failed_validations:
            print("`model_config.yaml` failed validations: \n")
            _list_failed_validation_messages(model_config_failed_validations)

        # Add the `model_config.yaml` failed validations to the list of all failed validations
        self.failed_validations.extend(model_config_failed_validations)

    def _validate_prediction_interface(self):
        """Validates the implementation of the prediction interface.

        Checks for the existence of the file, the required functions, and
        runs test data through the model to ensure there are no implementation
        errors.
        """
        prediction_interface_failed_validations = []

        # Path to the prediction_interface.py file
        prediction_interface_file = os.path.join(
            self.model_package_dir, "prediction_interface.py"
        )

        # File existence check
        if not os.path.isfile(os.path.expanduser(prediction_interface_file)):
            prediction_interface_failed_validations.append(
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
                prediction_interface_failed_validations.append(
                    "The `load_model` function is not defined in the `prediction_interface.py` "
                    "file."
                )
            else:
                # Test `load_model` function
                ml_model = None
                try:
                    ml_model = module.load_model()
                except Exception as e:
                    prediction_interface_failed_validations.append(
                        f"There is an error while loading the model: \n {e}"
                    )

                if ml_model is not None:
                    # Check if the `predict_proba` method is part of the model object
                    if not hasattr(ml_model, "predict_proba"):
                        prediction_interface_failed_validations.append(
                            "The `predict_proba` function is not defined in the model class."
                        )
                    else:
                        # Test `predict_proba` function
                        try:
                            with utils.HidePrints():
                                ml_model.predict_proba(self.sample_data)
                        except Exception as e:
                            exception_stack = "".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            )
                            prediction_interface_failed_validations.append(
                                "The `predict_proba` function failed while running the test data. "
                                "It is failing with the following error message: \n"
                                f"{exception_stack}"
                            )

        # Print results of the validation
        if prediction_interface_failed_validations:
            print("`prediction_interface.py` failed validations: \n")
            _list_failed_validation_messages(prediction_interface_failed_validations)

        # Add the `prediction_interface.py` failed validations to the list of all failed validations
        self.failed_validations.extend(prediction_interface_failed_validations)

    def validate(self) -> List[str]:
        """Runs all model validations.

        At each stage, prints all the failed validations.

        Rerturns:
            List[str]: a list of all failed validations.
        """
        self._validate_model_package_dir()
        self._validate_requirements()
        self._validate_model_config()
        self._validate_prediction_interface()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


# ----------------------------- Helper functions ----------------------------- #
def _format_marshmallow_error_message(err: ma.ValidationError) -> List[str]:
    """Formats the error messages from Marshmallow to conform to the expected
    list of strings format.

    Args:
        err (ma.ValidationError): the error object returned by Marshmallow.

    Returns:
        List[str]: a list of strings, where each string is a failed validation.
    """
    error_msg = []
    for input, msg in err.messages.items():
        if input == "_schema":
            temp_msg = "\n".join(msg)
            error_msg.append(f"{temp_msg}")
        elif not isinstance(msg, dict):
            temp_msg = msg[0].lower()
            error_msg.append(f"`{input}`: {temp_msg}")
        else:
            temp_msg = list(msg.values())[0][0].lower()
            error_msg.append(f"`{input}` contains items that are {temp_msg}")

    return error_msg


def _list_failed_validation_messages(failed_validations: List[str]):
    """Prints the failed validations in a list format, with one failed
    validation per line."""
    for msg in failed_validations:
        print(f"- {msg} \n")
