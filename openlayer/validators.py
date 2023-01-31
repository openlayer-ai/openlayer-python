"""Implements the series of validations needed to ensure that 
the objects uploaded to the platform are valid.

Typically, the validator object is created and then the `validate` method is called.
For example, to validate a model package:

>>> model_validator = ModelValidator(
...     model_package_dir="/path/to/model/package",
...     model_config_file_path="/path/to/model/config/file.yaml",
...     sample_data=df)
>>> model_validator.validate()
"""
import ast
import importlib
import os
import warnings
from typing import Any, Dict, List, Optional

import marshmallow as ma
import pandas as pd
import pkg_resources
import yaml

from . import schemas, utils


class CommitBundleValidator:
    """Validates the commit bundle prior to push.

    Parameters
    ----------
    bundle_path : str
        The path to the commit bundle (staging area, if for the Python API).
    """

    def __init__(self, bundle_path: str):
        self.bundle_path = bundle_path
        self._bundle_resources = self._list_resources_in_bundle()
        self.failed_validations = []

    def _validate_bundle_state(self):
        """Checks whether the bundle is in a valid state.

        This includes:
        - When a "model" is included, you always need to provide predictions for both
          "validation" and "training" (regardless of artifact or no artifact).
        - When a "model" is not included, you always need to NOT upload predictions with
          one exception:
            - "validation" set only in bundle, which means the predictions are for the
            previous model version.
        """
        bundle_state_failed_validations = []

        # Defining which datasets contain predictions
        training_predictions_column_name = None
        validation_predictions_column_name = None
        if "training" in self._bundle_resources:
            with open(
                f"{self.bundle_path}/training/dataset_config.yaml", "r"
            ) as stream:
                training_dataset_config = yaml.safe_load(stream)

            training_predictions_column_name = training_dataset_config.get(
                "predictionsColumnName"
            )

        if "validation" in self._bundle_resources:
            with open(
                f"{self.bundle_path}/validation/dataset_config.yaml", "r"
            ) as stream:
                validation_dataset_config = yaml.safe_load(stream)

            validation_predictions_column_name = validation_dataset_config.get(
                "predictionsColumnName"
            )

        if "model" in self._bundle_resources:
            if (
                training_predictions_column_name is None
                or validation_predictions_column_name is None
            ):
                bundle_state_failed_validations.append(
                    "To push a model to the platform, you must provide "
                    "training and a validation sets with predictions in the column "
                    "`predictions_column_name`."
                )
        else:
            if (
                "training" in self._bundle_resources
                and validation_predictions_column_name is not None
            ):
                bundle_state_failed_validations.append(
                    "A training set was provided alongside with a validation set with"
                    " predictions. Please either provide only a validation set with"
                    " predictions, or a model and both datasets with predictions"
                )
            elif training_predictions_column_name is not None:
                bundle_state_failed_validations.append(
                    "The training dataset contains predictions, but no model was"
                    " provided. To push a training set with predictions, please provide"
                    " a model and a validation set with predictions as well."
                )

        # Print results of the validation
        if bundle_state_failed_validations:
            print("Push failed validations: \n")
            _list_failed_validation_messages(bundle_state_failed_validations)

        # Add the bundle state failed validations to the list of all failed validations
        self.failed_validations.extend(bundle_state_failed_validations)

    def _validate_bundle_resources(self):
        """Runs the corresponding validations for each resource in the bundle."""
        bundle_resources_failed_validations = []

        if "training" in self._bundle_resources:
            training_set_validator = DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/training/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/training/dataset.csv",
            )
            bundle_resources_failed_validations.extend(
                training_set_validator.validate()
            )

        if "validation" in self._bundle_resources:
            validation_set_validator = DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/validation/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/training/dataset.csv",
            )
            bundle_resources_failed_validations.extend(
                validation_set_validator.validate()
            )

        if "model" in self._bundle_resources:
            model_files = os.listdir(f"{self.bundle_path}/model")
            # Shell model
            if len(model_files) == 1:
                model_validator = ModelValidator(
                    model_config_file_path=f"{self.bundle_path}/model/model_config.yaml"
                )
            # Model package
            else:
                # Use data from the validation as test data
                validation_dataset_df = self._load_dataset_from_bundle("validation")
                validation_dataset_config = self._load_dataset_config_from_bundle(
                    "validation"
                )

                sample_data = None
                if "textColumnName" in validation_dataset_config:
                    sample_data = validation_dataset_df[
                        validation_dataset_config["textColumnName"]
                    ].head()

                else:
                    sample_data = validation_dataset_df[
                        validation_dataset_config["featureNames"]
                    ].head()

                model_validator = ModelValidator(
                    model_config_file_path=f"{self.bundle_path}/model/model_config.yaml",
                    model_package_dir=f"{self.bundle_path}/model",
                    sample_data=sample_data,
                )
                bundle_resources_failed_validations.extend(model_validator.validate())

        # Print results of the validation
        if bundle_resources_failed_validations:
            print("Push failed validations: \n")
            _list_failed_validation_messages(bundle_resources_failed_validations)

        # Add the bundle resources failed validations to the list of all failed validations
        self.failed_validations.extend(bundle_resources_failed_validations)

    def _list_resources_in_bundle(self) -> List[str]:
        """Lists the resources in a commit bundle."""
        # TODO: factor out list of valid resources
        VALID_RESOURCES = ["model", "training", "validation"]

        resources = []

        for resource in os.listdir(self.bundle_path):
            if resource in VALID_RESOURCES:
                resources.append(resource)
        return resources

    def _load_dataset_from_bundle(self, label: str) -> pd.DataFrame:
        """Loads a dataset from a commit bundle.

        Parameters
        ----------
        label : str
            The type of the dataset. Can be either "training" or "validation".

        Returns
        -------
        pd.DataFrame
            The dataset.
        """
        dataset_file_path = f"{self.bundle_path}/{label}/dataset.csv"

        dataset_df = pd.read_csv(dataset_file_path)

        return dataset_df

    def _load_dataset_config_from_bundle(self, label: str) -> Dict[str, Any]:
        """Loads a dataset config from a commit bundle.

        Parameters
        ----------
        label : str
            The type of the dataset. Can be either "training" or "validation".

        Returns
        -------
        Dict[str, Any]
            The dataset config.
        """
        dataset_config_file_path = f"{self.bundle_path}/{label}/dataset_config.yaml"

        with open(dataset_config_file_path, "r") as stream:
            dataset_config = yaml.safe_load(stream)

        return dataset_config

    def validate(self) -> List[str]:
        """Validates the commit bundle.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        self._validate_bundle_state()
        self._validate_bundle_resources()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


class CommitValidator:
    """Validates the commit prior to the upload.

    Parameters
    ----------
    commit_message : str
        The commit message.
    """

    def __init__(
        self,
        commit_message: str,
    ):
        self.commit_message = commit_message
        self.failed_validations = []

    def _validate_commit_message(self):
        """Checks whether the commit message is valid."""
        commit_message_failed_validations = []

        commit_schema = schemas.CommitSchema()
        try:
            commit_schema.load({"commitMessage": self.commit_message})
        except ma.ValidationError as err:
            commit_message_failed_validations.extend(
                _format_marshmallow_error_message(err)
            )

        # Print results of the validation
        if commit_message_failed_validations:
            print("Commit failed validations: \n")
            _list_failed_validation_messages(commit_message_failed_validations)

        # Add the commit failed validations to the list of all failed validations
        self.failed_validations.extend(commit_message_failed_validations)

    def validate(self) -> List[str]:
        """Validates the commit.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        self._validate_commit_message()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


class DatasetValidator:
    """Validates the dataset and its arguments.

    Either the ``dataset_file_path`` or the ``dataset_df`` must be
    provided (not both).

    Either the ``dataset_config_file_path`` or the ``dataset_config``
    must be provided (not both).

    Parameters
    ----------
    dataset_config_file_path : str, optional
        The path to the dataset_config.yaml file.
    dataset_config : dict, optional
        The dataset_config as a dictionary.
    dataset_file_path : str, optional
        The path to the dataset file.
    dataset_df : pd.DataFrame, optional
        The dataset to validate.

    Examples
    --------

    Let's say we have a ``dataset_config.yaml`` file and a ``dataset.csv``
    file in the current directory.

    To ensure they are in the format the Openlayer platform expects to use the
    :meth:`openlayer.OpenlayerClient.add_dataset`, we can use the
    :class:`openlayer.DatasetValidator` class as follows:

    >>> from openlayer import DatasetValidator
    >>>
    >>> dataset_validator = DatasetValidator(
    ...     dataset_config_file_path="dataset_config.yaml",
    ...     dataset_file_path="dataset.csv",
    ... )
    >>> dataset_validator.validate()

    Alternatively, if we have a ``dataset_config.yaml`` file in the current
    directory and a ``dataset_df`` DataFrame, we can use the
    :class:`openlayer.DatasetValidator` class as follows:

    >>> from openlayer import DatasetValidator
    >>>
    >>> dataset_validator = DatasetValidator(datas
    ...     dataset_config_file_path="dataset_config.yaml",
    ...     dataset_df=dataset_df,
    ... )
    >>> dataset_validator.validate()
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
                "Both dataset_df and dataset_file_path are provided."
                " Please provide only one of them."
            )
        elif dataset_df is None and not dataset_file_path:
            raise ValueError(
                "Neither dataset_df nor dataset_file_path is provided."
                " Please provide one of them."
            )

        if dataset_config_file_path and dataset_config:
            raise ValueError(
                "Both dataset_config_file_path and dataset_config are provided."
                " Please provide only one of them."
            )
        elif not dataset_config_file_path and not dataset_config:
            raise ValueError(
                "Neither dataset_config_file_path nor dataset_config is provided."
                " Please provide one of them."
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
                dataset_schema.load(self.dataset_config)
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

        If it is valid, it loads the dataset file into the `self.dataset_df`
        attribute.

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
            class_names = self.dataset_config.get("classNames")
            column_names = self.dataset_config.get("columnNames")
            label_column_name = self.dataset_config.get("labelColumnName")
            feature_names = self.dataset_config.get("featureNames")
            text_column_name = self.dataset_config.get("textColumnName")
            predictions_column_name = self.dataset_config.get("predictionsColumnName")

            if self._contains_null_values(dataset_df):
                dataset_and_config_consistency_failed_validations.append(
                    "The dataset contains null values, which are currently not supported. "
                    "Please provide a dataset without null values."
                )

            if self._contains_unsupported_dtypes(dataset_df):
                dataset_and_config_consistency_failed_validations.append(
                    "The dataset contains unsupported dtypes. The supported dtypes are "
                    "'float32', 'float64', 'int32', 'int64', 'object'."
                    " Please cast the columns in your dataset to conform to these dtypes."
                )

            if self._columns_not_in_dataset_df(dataset_df, column_names):
                dataset_and_config_consistency_failed_validations.append(
                    "There are columns specified in the `columnNames` dataset config"
                    " which are not in the dataset."
                )

            if label_column_name:
                if self._column_not_in_dataset_df(dataset_df, label_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The label column `{label_column_name}` specified as `labelColumnName` "
                        "is not in the dataset."
                    )
                else:
                    if class_names:
                        if self._labels_not_in_class_names(
                            dataset_df, label_column_name, class_names
                        ):
                            dataset_and_config_consistency_failed_validations.append(
                                "There are more labels in the dataset's column"
                                f" `{label_column_name}` than specified in `classNames`. "
                                "Please specify all possible labels in the `classNames` list."
                            )
                        if self._labels_not_zero_indexed(
                            dataset_df, label_column_name, class_names
                        ):
                            dataset_and_config_consistency_failed_validations.append(
                                "The labels in the dataset are not zero-indexed. "
                                f"Make sure that the labels in the column `{label_column_name}` "
                                "are zero-indexed integers that match the list in `classNames`."
                            )

            # Predictions validations
            if predictions_column_name:
                if self._column_not_in_dataset_df(dataset_df, predictions_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The predictions column `{predictions_column_name}` specified as"
                        " `predictionsColumnName` is not in the dataset."
                    )
                else:
                    try:
                        # Getting prediction lists from strings saved in the csv
                        dataset_df[predictions_column_name] = dataset_df[
                            predictions_column_name
                        ].apply(ast.literal_eval)
                        if self._predictions_not_lists(
                            dataset_df, predictions_column_name
                        ):
                            dataset_and_config_consistency_failed_validations.append(
                                f"The predictions in the column `{predictions_column_name}` "
                                "are not lists. Please make sure that the predictions are "
                                "lists of floats."
                            )
                        else:
                            if self._prediction_lists_not_same_length(
                                dataset_df, predictions_column_name
                            ):
                                dataset_and_config_consistency_failed_validations.append(
                                    f"The prediction lists in the column `{predictions_column_name}` "
                                    "are not all of the same length. "
                                    "Please make sure that all prediction lists are of the same length."
                                )
                            else:
                                if self._predictions_not_class_probabilities(
                                    dataset_df, predictions_column_name
                                ):
                                    dataset_and_config_consistency_failed_validations.append(
                                        f"The predictions in the column `{predictions_column_name}` "
                                        "are not class probabilities. "
                                        "Please make sure that the predictions are lists of floats "
                                        "that sum to 1."
                                    )
                                elif class_names:
                                    if self._predictions_not_in_class_names(
                                        dataset_df, predictions_column_name, class_names
                                    ):
                                        dataset_and_config_consistency_failed_validations.append(
                                            f"There are predictions in the column `{predictions_column_name}` "
                                            "are not in `classNames`. "
                                            "Please make sure that the predictions are lists of floats "
                                            "that sum to 1 and that the classes in the predictions "
                                            "match the classes in `classNames`."
                                        )
                    except:
                        dataset_and_config_consistency_failed_validations.append(
                            f"The predictions in the column `{predictions_column_name}` are not lists. "
                            "Please make sure that the predictions are lists of floats."
                        )

            # NLP-specific validations
            if text_column_name:
                if self._column_not_in_dataset_df(dataset_df, text_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The text column `{text_column_name}` specified as `textColumnName` "
                        "is not in the dataset."
                    )
                elif self._exceeds_character_limit(dataset_df, text_column_name):
                    dataset_and_config_consistency_failed_validations.append(
                        f"The column `{text_column_name}` of the dataset contains rows that "
                        "exceed the 1000 character limit."
                    )

            # Tabular-specific validations
            if feature_names:
                if self._columns_not_in_dataset_df(dataset_df, feature_names):
                    dataset_and_config_consistency_failed_validations.append(
                        "There are features specified in `featureNames` which are "
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
    def _contains_null_values(dataset_df: pd.DataFrame) -> bool:
        """Checks whether the dataset contains null values."""
        return dataset_df.isnull().values.any()

    @staticmethod
    def _labels_not_zero_indexed(
        dataset_df: pd.DataFrame, label_column_name: str, class_names: List[str]
    ) -> bool:
        """Checks whether the labels are zero-indexed."""
        unique_labels = set(dataset_df[label_column_name].unique())
        zero_indexed_set = set(range(len(class_names)))
        if unique_labels != zero_indexed_set:
            return True
        return False

    @staticmethod
    def _contains_unsupported_dtypes(dataset_df: pd.DataFrame) -> bool:
        """Checks whether the dataset contains unsupported dtypes."""
        supported_dtypes = {"float32", "float64", "int32", "int64", "object"}
        dataset_df_dtypes = set([dtype.name for dtype in dataset_df.dtypes])
        unsupported_dtypes = dataset_df_dtypes - supported_dtypes
        if unsupported_dtypes:
            return True
        return False

    @staticmethod
    def _column_not_in_dataset_df(dataset_df: pd.DataFrame, column_name: str) -> bool:
        """Checks whether the label column is in the dataset."""
        if column_name not in dataset_df.columns:
            return True
        return False

    @staticmethod
    def _columns_not_in_dataset_df(
        dataset_df: pd.DataFrame, columns_list: List[str]
    ) -> bool:
        """Checks whether the columns are in the dataset."""
        if set(columns_list) - set(dataset_df.columns):
            return True
        return False

    @staticmethod
    def _exceeds_character_limit(
        dataset_df: pd.DataFrame, text_column_name: str
    ) -> bool:
        """Checks whether the text column exceeds the character limit."""
        if dataset_df[text_column_name].str.len().max() > 1000:
            return True
        return False

    @staticmethod
    def _labels_not_in_class_names(
        dataset_df: pd.DataFrame, label_column_name: str, class_names: List[str]
    ) -> bool:
        """Checks whether there are labels in the dataset which are not
        in the `class_names`."""
        num_classes = len(dataset_df[label_column_name].unique())
        if num_classes > len(class_names):
            return True
        return False

    @staticmethod
    def _predictions_not_lists(
        dataset_df: pd.DataFrame, predictions_column_name: str
    ) -> bool:
        """Checks whether all values in the column `predictions_column_name`
        are lists."""
        if not all(
            isinstance(predictions, list)
            for predictions in dataset_df[predictions_column_name]
        ):
            return True
        return False

    @staticmethod
    def _prediction_lists_not_same_length(
        dataset_df: pd.DataFrame, predictions_column_name: str
    ) -> bool:
        """Checks whether all the lists in the `predictions_column_name`
        have the same length."""
        if not len(set(dataset_df[predictions_column_name].str.len())) == 1:
            return True
        return False

    @staticmethod
    def _predictions_not_class_probabilities(
        dataset_df: pd.DataFrame, predictions_column_name: str
    ) -> bool:
        """Checks whether the predictions are class probabilities.
        Tolerate a 10% error margin."""
        if any(
            [
                (sum(predictions) < 0.9 or sum(predictions) > 1.1)
                for predictions in dataset_df[predictions_column_name]
            ]
        ):
            return True
        return False

    @staticmethod
    def _predictions_not_in_class_names(
        dataset_df: pd.DataFrame,
        predictions_column_name: str,
        class_names: List[str],
    ) -> bool:
        """Checks if the predictions map 1:1 to the `class_names` list."""
        num_classes_predicted = len(dataset_df[predictions_column_name].iloc[0])
        if num_classes_predicted != len(class_names):
            return True
        return False

    def validate(self) -> List[str]:
        """Runs all dataset validations.

        At each stage, prints all the failed validations.

        Returns
        -------
        List[str]
            List of all failed validations.
        """
        self._validate_dataset_config()
        if self.dataset_file_path:
            self._validate_dataset_file()
        self._validate_dataset_and_config_consistency()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


class ModelValidator:
    """Validates the model package's structure and files.

    Parameters
    ----------
    model_package_dir : str
        Path to the model package directory.
    sample_data : pd.DataFrame
        Sample data to be used for the model validation.

    Examples
    --------

    Let's say we have the prepared the model package and have some sample data expected
    by the model in a pandas DataFrame.

    To ensure the model package is in the format the Openlayer platform expects to use the
    :meth:`openlayer.OpenlayerClient.add_model` method, we can use the
    :class:`openlayer.ModelValidator` class as follows:,

    >>> from openlayer import ModelValidator
    >>>
    >>> model_validator = ModelValidator(
    ...     model_package_dir="/path/to/model/package",
    ...     sample_data=df,
    ... )
    >>> model_validator.validate()

    """

    def __init__(
        self,
        model_config_file_path: str,
        model_package_dir: Optional[str] = None,
        sample_data: Optional[pd.DataFrame] = None,
    ):
        self.model_config_file_path = model_config_file_path
        self.model_package_dir = model_package_dir
        self.sample_data = sample_data
        self.failed_validations = []

    def _validate_model_package_dir(self):
        """Verifies the model package directory structure.

        The model package directory must follow the structure:

        model_package
          ├── artifacts.pkl  # potentially different name / format and multiple files
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
            with open(requirements_txt_file, "r") as file:
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

        # File existence check
        if not os.path.isfile(os.path.expanduser(self.model_config_file_path)):
            model_config_failed_validations.append(
                f"File `{self.model_config_file_path}` does not exist."
            )
        else:
            with open(self.model_config_file_path, "r") as stream:
                model_config = yaml.safe_load(stream)

            if self.model_package_dir:
                model_schema = schemas.ModelSchema()
            else:
                model_schema = schemas.ShellModelSchema()
            try:
                model_schema.load(model_config)
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
                except Exception as exc:
                    prediction_interface_failed_validations.append(
                        f"There is an error while loading the model: \n {exc}"
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
                        except Exception as err:
                            exception_stack = utils.get_exception_stacktrace(err)
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

        Returns
        -------
        List[str]
            A list of all failed validations.
        """
        if self.model_package_dir:
            self._validate_model_package_dir()
            self._validate_requirements()
            self._validate_prediction_interface()
        self._validate_model_config()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


class ProjectValidator:
    """Validates the project.

    Parameters
    ----------
    project_config : Dict[str, str]
        The project configuration.
    """

    def __init__(
        self,
        project_config: Dict[str, str],
    ):
        self.project_config = project_config
        self.failed_validations = []

    def _validate_project_config(self):
        """Checks if the project configuration is valid."""
        project_config_failed_validations = []

        project_schema = schemas.ProjectSchema()
        try:
            project_schema.load(
                {
                    "name": self.project_config.get("name"),
                    "description": self.project_config.get("description"),
                    "task_type": self.project_config.get("task_type").value,
                }
            )
        except ma.ValidationError as err:
            project_config_failed_validations.extend(
                _format_marshmallow_error_message(err)
            )

        # Print results of the validation
        if project_config_failed_validations:
            print("Project config failed validations: \n")
            _list_failed_validation_messages(project_config_failed_validations)

        # Add the commit failed validations to the list of all failed validations
        self.failed_validations.extend(project_config_failed_validations)

    def validate(self):
        """Validates the project."""
        self._validate_project_config()

        if not self.failed_validations:
            print("All validations passed!")

        return self.failed_validations


# ----------------------------- Helper functions ----------------------------- #
def _format_marshmallow_error_message(err: ma.ValidationError) -> List[str]:
    """Formats the error messages from Marshmallow to conform to the expected
    list of strings format.

    Parameters
    ----------
    err : ma.ValidationError
        The error object returned by Marshmallow.

    Returns
    -------
    List[str]
        A list of strings, where each string is a failed validation.
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
