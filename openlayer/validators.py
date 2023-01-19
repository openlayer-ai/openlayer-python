import importlib
import os
import traceback
import warnings
from typing import Dict, List, Optional

import marshmallow as ma
import pandas as pd
import pkg_resources
import yaml

from . import schemas, utils


class DatasetValidator:
    """Validates the dataset and its arguments prior to the upload.

    Either the `dataset_file_path` or the `dataset_df` must be provided (not both).
    Either the `dataset_config_file_path` or the `dataset_config` must be provided (not both).

    Args:
        dataset_config_file_path (str): the path to the dataset_config.yaml file.
        dataset_config (dict): the dataset_config as a dictionary.
        dataset_file_path (str): the path to the dataset file.
        dataset_df (pd.DataFrame): the dataset to validate.
    """

    def __init__(
        self,
        dataset_config_file_path: Optional[str] = None,
        dataset_config: Optional[Dict] = None,
        dataset_file_path: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
    ):
        if dataset_df is not None and dataset_file_path:
            raise ValueError(
                "Both dataset_df and dataset_file_path are provided. Please provide only one of them."
            )
        elif dataset_df is None and not dataset_file_path:
            raise ValueError(
                "Neither dataset_df nor dataset_file_path is provided. Please provide one of them."
            )

        if dataset_config_file_path and dataset_config:
            raise ValueError(
                "Both dataset_config_file_path and dataset_config are provided. Please provide only one of them."
            )
        elif not dataset_config_file_path and not dataset_config:
            raise ValueError(
                "Neither dataset_config_file_path nor dataset_config is provided. Please provide one of them."
            )

        self.dataset_file_path = dataset_file_path
        self.dataset_df = dataset_df
        self.dataset_config_file_path = dataset_config_file_path
        self.dataset_config = dataset_config
        self.failed_validations = []

    def _validate_dataset_config(self):
        """Checks whether the dataset_config is valid.

        Beware of the order of the validations, as it is important.
        """
        dataset_config_failed_validations = []

        # File existence check
        if self.dataset_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.dataset_config_file_path)):
                dataset_config_failed_validations.append(
                    f"File `{self.dataset_config_file_path}` does not exist."
                )
            else:
                with open(self.dataset_config_file_path, "r") as stream:
                    self.dataset_config = yaml.safe_load(stream)

        if self.dataset_config:
            dataset_schema = schemas.DatasetSchema()
            try:
                dataset_schema.load(
                    {
                        "file_path": self.dataset_config.get("file_path"),
                        "class_names": self.dataset_config.get("class_names"),
                        "label_column_name": self.dataset_config.get(
                            "label_column_name"
                        ),
                        "dataset_type": self.dataset_config.get("dataset_type"),
                        "language": self.dataset_config.get("language", "en"),
                        "sep": self.dataset_config.get("sep", ","),
                        "feature_names": self.dataset_config.get("feature_names", []),
                        "text_column_name": self.dataset_config.get("text_column_name"),
                        "categorical_feature_names": self.dataset_config.get(
                            "categorical_feature_names", []
                        ),
                    }
                )
            except ma.ValidationError as err:
                dataset_config_failed_validations.extend(
                    _format_marshmallow_error_message(err)
                )

        # Print results of the validation
        if dataset_config_failed_validations:
            print("Dataset_config failed validations: \n")
            _list_failed_validation_messages(dataset_config_failed_validations)

        # Add the `dataset_config.yaml` failed validations to the list of all failed validations
        self.failed_validations.extend(dataset_config_failed_validations)

    def _validate_dataset_file(self):
        """Checks whether the dataset file exists and is valid.

        If it is valid, it loads the dataset file into the `self.dataset_df` attribute.

        Beware of the order of the validations, as it is important.
        """
        dataset_file_failed_validations = []

        # File existence check
        if not os.path.isfile(os.path.expanduser(self.dataset_file_path)):
            dataset_file_failed_validations.append(
                f"File `{self.dataset_file_path}` does not exist."
            )
        else:
            # File format (csv) check by loading it as a pandas df
            try:
                self.dataset_df = pd.read_csv(self.dataset_file_path)
            except Exception as err:
                dataset_file_failed_validations.append(
                    f"File `{self.dataset_file_path}` is not a valid .csv file."
                )

        # Print results of the validation
        if dataset_file_failed_validations:
            print("Dataset file failed validations: \n")
            _list_failed_validation_messages(dataset_file_failed_validations)

        # Add the dataset file failed validations to the list of all failed validations
        self.failed_validations.extend(dataset_file_failed_validations)

    def _validate_dataset_and_config_consistency(self):
        """Checks whether the dataset and its config are consistent.

        Beware of the order of the validations, as it is important.
        """
        dataset_and_config_consistency_failed_validations = []

        if self.dataset_config and self.dataset_df is not None:
            # Extract vars
            dataset_df = self.dataset_df
            class_names = self.dataset_config.get("class_names")
            label_column_name = self.dataset_config.get("label_column_name")
            feature_names = self.dataset_config.get("feature_names")
            text_column_name = self.dataset_config.get("text_column_name")

            if self.contains_null_values(dataset_df):
                dataset_and_config_consistency_failed_validations.append(
                    "The dataset contains null values, which are currently not supported. "
                    "Please provide a dataset without null values."
                )

            if self.contains_unsupported_dtypes(dataset_df):
                dataset_and_config_consistency_failed_validations.append(
                    "The dataset contains unsupported dtypes. The supported dtypes are "
                    "'float32', 'float64', 'int32', 'int64', 'object'. Please cast the columns "
                    "in your dataset to conform to these dtypes."
                )

            if label_column_name:
                if self.column_not_in_dataset_df(dataset_df, label_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The label column `{label_column_name}` specified as `label_column_name` "
                        "is not in the dataset."
                    )
                else:
                    if class_names:
                        if self.labels_not_in_class_names(
                            dataset_df, label_column_name, class_names
                        ):
                            dataset_and_config_consistency_failed_validations.append(
                                f"There are more labels in the dataset's column `{label_column_name}` "
                                "than specified in `class_names`. "
                                "Please specify all possible labels in the `class_names` list."
                            )
                        if self.labels_not_zero_indexed(
                            dataset_df, label_column_name, class_names
                        ):
                            dataset_and_config_consistency_failed_validations.append(
                                "The labels in the dataset are not zero-indexed. "
                                f"Make sure that the labels in the column `{label_column_name}` "
                                "are zero-indexed integers that match the list in `class_names`."
                            )

            # NLP-specific validations
            if text_column_name:
                if self.column_not_in_dataset_df(dataset_df, text_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The text column `{text_column_name}` specified as `text_column_name` "
                        "is not in the dataset."
                    )
                elif self.exceeds_character_limit(dataset_df, text_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The column `{text_column_name}` of the dataset contains rows that "
                        "exceed the 1000 character limit."
                    )

            # Tabular-specific validations
            if feature_names:
                if self.features_not_in_dataset_df(dataset_df, feature_names):
                    dataset_and_config_consistency_failed_validations.append(
                        f"There are features specified in `feature_names` which are "
                        "not in the dataset."
                    )

        # Print results of the validation
        if dataset_and_config_consistency_failed_validations:
            print("Inconsistencies between the dataset config and the dataset: \n")
            _list_failed_validation_messages(
                dataset_and_config_consistency_failed_validations
            )

        # Add the consistency failed validations to the list of all failed validations
        self.failed_validations.extend(
            dataset_and_config_consistency_failed_validations
        )

    @staticmethod
    def contains_null_values(dataset_df: pd.DataFrame) -> bool:
        """Checks whether the dataset contains null values."""
        return dataset_df.isnull().values.any()

    @staticmethod
    def labels_not_zero_indexed(
        dataset_df: pd.DataFrame, label_column_name: str, class_names: List[str]
    ) -> bool:
        """Checks whether the labels are zero-indexed."""
        unique_labels = set(dataset_df[label_column_name].unique())
        zero_indexed_set = set(range(len(class_names)))
        if unique_labels != zero_indexed_set:
            return True
        return False

    @staticmethod
    def contains_unsupported_dtypes(dataset_df: pd.DataFrame) -> bool:
        """Checks whether the dataset contains unsupported dtypes."""
        supported_dtypes = {"float32", "float64", "int32", "int64", "object"}
        dataset_df_dtypes = set([dtype.name for dtype in dataset_df.dtypes])
        unsupported_dtypes = dataset_df_dtypes - supported_dtypes
        if unsupported_dtypes:
            return True
        return False

    @staticmethod
    def column_not_in_dataset_df(dataset_df: pd.DataFrame, column_name: str) -> bool:
        """Checks whether the label column is in the dataset."""
        if column_name not in dataset_df.columns:
            return True
        return False

    @staticmethod
    def features_not_in_dataset_df(
        dataset_df: pd.DataFrame, feature_names: List[str]
    ) -> bool:
        """Checks whether the features are in the dataset."""
        if set(feature_names) - set(dataset_df.columns):
            return True
        return False

    @staticmethod
    def exceeds_character_limit(
        dataset_df: pd.DataFrame, text_column_name: str
    ) -> bool:
        """Checks whether the text column exceeds the character limit."""
        if dataset_df[text_column_name].str.len().max() > 1000:
            return True
        return False

    @staticmethod
    def labels_not_in_class_names(
        dataset_df: pd.DataFrame, label_column_name: str, class_names: List[str]
    ) -> bool:
        """Checks whether there are labels in the dataset which are not
        in the `class_names`."""
        num_classes = len(dataset_df[label_column_name].unique())
        if num_classes > len(class_names):
            return True
        return False

    def validate(self):
        """Runs all dataset validations.

        At each stage, prints all the failed validations.

        Rerturns:
            List[str]: a list of all failed validations.
        """
        self._validate_dataset_config()
        if self.dataset_file_path:
            self._validate_dataset_file()
        self._validate_dataset_and_config_consistency()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


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

        Beware of the order of the validations, as it is important.
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
