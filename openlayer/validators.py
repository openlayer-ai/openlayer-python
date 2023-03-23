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
import logging
import os
import warnings
from typing import Any, Dict, List, Optional

import marshmallow as ma
import pandas as pd
import pkg_resources
import yaml

from . import models, schemas, utils

# Validator logger
logger = logging.getLogger("validators")
logger.setLevel(logging.ERROR)

# Console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class BaselineModelValidator:
    """Validates the baseline model.

    Parameters
    ----------
    model_config_file_path : Optional[str], optional
        The path to the model config file, by default None
    """

    def __init__(self, model_config_file_path: Optional[str] = None):
        self.model_config_file_path = model_config_file_path
        self.failed_validations = []

    def validate(self) -> List[str]:
        """Validates the baseline model.

        Returns
        -------
        List[str]
            The list of failed validations.
        """
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                          Baseline model validations                          "
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        if self.model_config_file_path:
            self._validate_model_config()

        if not self.failed_validations:
            logger.info("✓ All baseline model validations passed! \n")

        return self.failed_validations

    def _validate_model_config(self):
        """Validates the model config file."""
        model_config_failed_validations = []

        # File existence check
        if self.model_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.model_config_file_path)):
                model_config_failed_validations.append(
                    f"File `{self.model_config_file_path}` does not exist."
                )
            else:
                with open(self.model_config_file_path, "r", encoding="UTF-8") as stream:
                    model_config = yaml.safe_load(stream)

        if model_config:
            baseline_model_schema = schemas.BaselineModelSchema()
            try:
                baseline_model_schema.load(model_config)
            except ma.ValidationError as err:
                model_config_failed_validations.extend(
                    _format_marshmallow_error_message(err)
                )

        # Print results of the validation
        if model_config_failed_validations:
            logger.error("Baseline model config failed validations:")
            _list_failed_validation_messages(model_config_failed_validations)

        # Add the `model_config.yaml` failed validations to the list of all failed validations
        self.failed_validations.extend(model_config_failed_validations)


class CommitBundleValidator:
    """Validates the commit bundle prior to push.

    Parameters
    ----------
    bundle_path : str
        The path to the commit bundle (staging area, if for the Python API).
    skip_model_validation : bool
        Whether to skip model validation, by default False
    skip_dataset_validation : bool
        Whether to skip dataset validation, by default False
    use_runner : bool
        Whether to use the runner to validate the model, by default False.
    """

    def __init__(
        self,
        bundle_path: str,
        skip_model_validation: bool = False,
        skip_dataset_validation: bool = False,
        use_runner: bool = False,
        log_file_path: Optional[str] = None,
    ):
        self.bundle_path = bundle_path
        self._bundle_resources = utils.list_resources_in_bundle(bundle_path)
        self._skip_model_validation = skip_model_validation
        self._skip_dataset_validation = skip_dataset_validation
        self._use_runner = use_runner
        self.failed_validations = []

        if log_file_path:
            bundle_file_handler = logging.FileHandler(log_file_path)
            bundle_formatter = logging.Formatter(
                "[%(asctime)s] - %(levelname)s - %(message)s"
            )
            bundle_file_handler.setFormatter(bundle_formatter)
            logger.addHandler(bundle_file_handler)

    def validate(self) -> List[str]:
        """Validates the commit bundle.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        logger.info(
            "                          Validating commit bundle                          \n"
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        self._validate_bundle_state()

        # Validate individual resources only if the bundle is in a valid state
        # TODO: improve the logic that determines whether to validate individual resources
        if not self.failed_validations:
            self._validate_bundle_resources()

        if not self.failed_validations:
            self._validate_resource_consistency()

        if not self.failed_validations:
            logger.info(
                "----------------------------------------------------------------------------\n"
            )
            logger.info(
                "                     All commit bundle validations passed!                   \n"
            )
            logger.info(
                "----------------------------------------------------------------------------\n"
            )
        else:
            logger.error("Please fix the all the issues above and push again.")

        return self.failed_validations

    def _validate_bundle_state(self):
        """Checks whether the bundle is in a valid state.

        This includes:
        - When a "model" (shell or full) is included, you always need to provide predictions for both
          "validation" and "training".
        - When a "baseline-model" is included, you always need to provide a "training"
          and "validation" set without predictions.
        - When a "model" nor a "baseline-model" are included, you always need to NOT
          upload predictions with one exception:
            - "validation" set only in bundle, which means the predictions are for the
            previous model version.
        """
        bundle_state_failed_validations = []

        # Defining which datasets contain predictions
        training_predictions_column_name = None
        validation_predictions_column_name = None
        if "training" in self._bundle_resources:
            with open(
                f"{self.bundle_path}/training/dataset_config.yaml",
                "r",
                encoding="UTF-8",
            ) as stream:
                training_dataset_config = yaml.safe_load(stream)

            training_predictions_column_name = training_dataset_config.get(
                "predictionsColumnName"
            )

        if "validation" in self._bundle_resources:
            with open(
                f"{self.bundle_path}/validation/dataset_config.yaml",
                "r",
                encoding="UTF-8",
            ) as stream:
                validation_dataset_config = yaml.safe_load(stream)

            validation_predictions_column_name = validation_dataset_config.get(
                "predictionsColumnName"
            )

        if "model" in self._bundle_resources:
            model_config = self._load_model_config_from_bundle()
            model_type = model_config.get("modelType")
            if (
                training_predictions_column_name is None
                or validation_predictions_column_name is None
            ) and model_type != "baseline":
                bundle_state_failed_validations.append(
                    "To push a model to the platform, you must provide "
                    "training and a validation sets with predictions in the column "
                    "specified by `predictionsColumnName`."
                )
            if model_type == "baseline":
                if (
                    "training" not in self._bundle_resources
                    or "validation" not in self._bundle_resources
                ):
                    bundle_state_failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets."
                    )
                elif (
                    training_predictions_column_name is not None
                    and validation_predictions_column_name is not None
                ):
                    bundle_state_failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets without predictions in the column "
                        "specified by `predictionsColumnName`."
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
            logger.info("Bundle state failed validations:")
            _list_failed_validation_messages(bundle_state_failed_validations)

        # Add the bundle state failed validations to the list of all failed validations
        self.failed_validations.extend(bundle_state_failed_validations)

    def _validate_bundle_resources(self):
        """Runs the corresponding validations for each resource in the bundle."""
        bundle_resources_failed_validations = []

        if "training" in self._bundle_resources and not self._skip_dataset_validation:
            training_set_validator = DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/training/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/training/dataset.csv",
            )
            bundle_resources_failed_validations.extend(
                training_set_validator.validate()
            )

        if "validation" in self._bundle_resources and not self._skip_dataset_validation:
            validation_set_validator = DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/validation/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/validation/dataset.csv",
            )
            bundle_resources_failed_validations.extend(
                validation_set_validator.validate()
            )

        if "model" in self._bundle_resources and not self._skip_model_validation:
            model_config_file_path = f"{self.bundle_path}/model/model_config.yaml"
            model_config = self._load_model_config_from_bundle()

            if model_config["modelType"] == "shell":
                model_validator = ModelValidator(
                    model_config_file_path=model_config_file_path
                )
            elif model_config["modelType"] == "full":
                # Use data from the validation as test data
                validation_dataset_df = self._load_dataset_from_bundle("validation")
                validation_dataset_config = self._load_dataset_config_from_bundle(
                    "validation"
                )

                sample_data = None
                if "textColumnName" in validation_dataset_config:
                    sample_data = validation_dataset_df[
                        [validation_dataset_config["textColumnName"]]
                    ].head()

                else:
                    sample_data = validation_dataset_df[
                        validation_dataset_config["featureNames"]
                    ].head()

                model_validator = ModelValidator(
                    model_config_file_path=model_config_file_path,
                    model_package_dir=f"{self.bundle_path}/model",
                    sample_data=sample_data,
                    use_runner=self._use_runner,
                )
            elif model_config["modelType"] == "baseline":
                model_validator = BaselineModelValidator(
                    model_config_file_path=model_config_file_path
                )
            else:
                raise ValueError(
                    f"Invalid model type: {model_config['modelType']}. "
                    "The model type must be one of 'shell', 'full' or 'baseline'."
                )
            bundle_resources_failed_validations.extend(model_validator.validate())

        # Add the bundle resources failed validations to the list of all failed validations
        self.failed_validations.extend(bundle_resources_failed_validations)

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

        with open(dataset_config_file_path, "r", encoding="UTF-8") as stream:
            dataset_config = yaml.safe_load(stream)

        return dataset_config

    def _load_model_config_from_bundle(self) -> Dict[str, Any]:
        """Loads a model config from a commit bundle.

        Returns
        -------
        Dict[str, Any]
            The model config.
        """
        model_config_file_path = f"{self.bundle_path}/model/model_config.yaml"

        with open(model_config_file_path, "r", encoding="UTF-8") as stream:
            model_config = yaml.safe_load(stream)

        return model_config

    def _validate_resource_consistency(self):
        """Validates that the resources in the bundle are consistent with each other.

        For example, if the `classNames` field on the dataset configs are consistent
        with the one on the model config.
        """
        resource_consistency_failed_validations = []

        if (
            "training" in self._bundle_resources
            and "validation" in self._bundle_resources
        ):
            # Loading the relevant configs
            model_config = {}
            if "model" in self._bundle_resources:
                model_config = self._load_model_config_from_bundle()
            training_dataset_config = self._load_dataset_config_from_bundle("training")
            validation_dataset_config = self._load_dataset_config_from_bundle(
                "validation"
            )
            model_feature_names = model_config.get("featureNames")
            model_class_names = model_config.get("classNames")
            training_feature_names = training_dataset_config.get("featureNames")
            training_class_names = training_dataset_config.get("classNames")
            validation_feature_names = validation_dataset_config.get("featureNames")
            validation_class_names = validation_dataset_config.get("classNames")

            # Validating the `featureNames` field
            if training_feature_names or validation_feature_names:
                if not self._feature_names_consistent(
                    model_feature_names=model_feature_names,
                    training_feature_names=training_feature_names,
                    validation_feature_names=validation_feature_names,
                ):
                    resource_consistency_failed_validations.append(
                        "The `featureNames` in the provided resources are inconsistent."
                        " The training and validation set feature names must have some overlap."
                        " Furthermore, if a model is provided, its feature names must be a subset"
                        " of the feature names in the training and validation sets."
                    )

            # Validating the `classNames` field
            if not self._class_names_consistent(
                model_class_names=model_class_names,
                training_class_names=training_class_names,
                validation_class_names=validation_class_names,
            ):
                resource_consistency_failed_validations.append(
                    "The `classNames` in the provided resources are inconsistent."
                    " The validation set's class names need to contain the training set's."
                    " Furthermore, if a model is provided, its class names must be contained"
                    " in the training and validation sets' class names."
                    " Note that the order of the items in the `classNames` list matters."
                )

        # Print results of the validation
        if resource_consistency_failed_validations:
            logger.error("Bundle resource consistency failed validations:")
            _list_failed_validation_messages(resource_consistency_failed_validations)

        # Add the bundle resource consistency failed validations to the list of all failed validations
        self.failed_validations.extend(resource_consistency_failed_validations)

    @staticmethod
    def _feature_names_consistent(
        model_feature_names: Optional[List[str]],
        training_feature_names: List[str],
        validation_feature_names: List[str],
    ) -> bool:
        """Checks whether the feature names in the training, validation and model
        configs are consistent.

        Parameters
        ----------
        model_feature_names : List[str]
            The feature names in the model config.
        training_feature_names : List[str]
            The feature names in the training dataset config.
        validation_feature_names : List[str]
            The feature names in the validation dataset config.

        Returns
        -------
        bool
            True if the feature names are consistent, False otherwise.
        """
        train_val_intersection = set(training_feature_names).intersection(
            set(validation_feature_names)
        )
        if model_feature_names is None:
            return len(train_val_intersection) != 0
        return set(model_feature_names).issubset(train_val_intersection)

    @staticmethod
    def _class_names_consistent(
        model_class_names: Optional[List[str]],
        training_class_names: List[str],
        validation_class_names: List[str],
    ) -> bool:
        """Checks whether the class names in the training and model configs
        are consistent.

        Parameters
        ----------
        model_class_names : List[str]
            The class names in the model config.
        training_class_names : List[str]
            The class names in the training dataset config.
        validation_class_names : List[str]
            The class names in the validation dataset config.

        Returns
        -------
        bool
            True if the class names are consistent, False otherwise.
        """
        if model_class_names is not None:
            num_model_classes = len(model_class_names)
            try:
                return (
                    training_class_names[:num_model_classes] == model_class_names
                    and validation_class_names[:num_model_classes] == model_class_names
                )
            except IndexError:
                return False
        num_training_classes = len(training_class_names)
        try:
            return validation_class_names[:num_training_classes] == training_class_names
        except IndexError:
            return False


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

    def validate(self) -> List[str]:
        """Validates the commit.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                         Commit message validations                         "
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        self._validate_commit_message()

        if not self.failed_validations:
            logger.info("✓ All commit validations passed!")

        return self.failed_validations

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
            logger.error("Commit failed validations:")
            _list_failed_validation_messages(commit_message_failed_validations)

        # Add the commit failed validations to the list of all failed validations
        self.failed_validations.extend(commit_message_failed_validations)


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

    Methods
    -------
    validate:
        Runs all dataset validations.

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

    def validate(self) -> List[str]:
        """Runs all dataset validations.

        At each stage, prints all the failed validations.

        Returns
        -------
        List[str]
            List of all failed validations.
        """
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                             Dataset validations                            "
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        self._validate_dataset_config()
        if self.dataset_file_path:
            self._validate_dataset_file()
        self._validate_dataset_and_config_consistency()

        if not self.failed_validations:
            logger.info(
                "✓ All %s dataset validations passed!\n", self.dataset_config["label"]
            )

        return self.failed_validations

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
                with open(
                    self.dataset_config_file_path, "r", encoding="UTF-8"
                ) as stream:
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
            logger.error("Dataset_config failed validations:")
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
            except Exception:
                dataset_file_failed_validations.append(
                    f"File `{self.dataset_file_path}` is not a valid .csv file."
                )

        # Print results of the validation
        if dataset_file_failed_validations:
            logger.error("Dataset file failed validations:")
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
            categorical_feature_names = self.dataset_config.get(
                "categoricalFeatureNames"
            )
            class_names = self.dataset_config.get("classNames")
            column_names = self.dataset_config.get("columnNames")
            label_column_name = self.dataset_config.get("labelColumnName")
            feature_names = self.dataset_config.get("featureNames")
            text_column_name = self.dataset_config.get("textColumnName")
            predictions_column_name = self.dataset_config.get("predictionsColumnName")

            if self._contains_unsupported_dtypes(dataset_df):
                dataset_and_config_consistency_failed_validations.append(
                    "The dataset contains unsupported dtypes. The supported dtypes are "
                    "'float32', 'float64', 'int32', 'int64', 'object'."
                    " Please cast the columns in your dataset to conform to these dtypes."
                )

            if self._columns_not_specified(dataset_df, column_names):
                dataset_and_config_consistency_failed_validations.append(
                    "Not all columns in the dataset are specified in `columnNames`."
                    " Please specify all dataset columns in `columnNames`."
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
                                    "The prediction lists in the column "
                                    f"`{predictions_column_name}` "
                                    "are not all of the same length. "
                                    "Please make sure that all prediction lists "
                                    "are of the same length."
                                )
                            else:
                                if self._predictions_not_class_probabilities(
                                    dataset_df, predictions_column_name
                                ):
                                    dataset_and_config_consistency_failed_validations.append(
                                        "The predictions in the column "
                                        f"`{predictions_column_name}` "
                                        "are not class probabilities. "
                                        "Please make sure that the predictions are lists "
                                        "of floats that sum to 1."
                                    )
                                elif class_names:
                                    if self._predictions_not_in_class_names(
                                        dataset_df, predictions_column_name, class_names
                                    ):
                                        dataset_and_config_consistency_failed_validations.append(
                                            f"The predictions in `{predictions_column_name}`"
                                            f" don't match the classes in `{class_names}`. "
                                            "Please make sure that the lists with predictions "
                                            "have the same length as the `classNames` list."
                                        )
                    except Exception:
                        dataset_and_config_consistency_failed_validations.append(
                            f"The predictions in the column `{predictions_column_name}` "
                            "are not lists. "
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
                if categorical_feature_names:
                    if self._columns_not_in_dataset_df(
                        dataset_df, categorical_feature_names
                    ):
                        dataset_and_config_consistency_failed_validations.append(
                            "There are categorical features specified in `categoricalFeatureNames` "
                            "which are not in the dataset."
                        )

        # Print results of the validation
        if dataset_and_config_consistency_failed_validations:
            logger.error("Inconsistencies between the dataset config and the dataset:")
            _list_failed_validation_messages(
                dataset_and_config_consistency_failed_validations
            )

        # Add the consistency failed validations to the list of all failed validations
        self.failed_validations.extend(
            dataset_and_config_consistency_failed_validations
        )

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
    def _columns_not_specified(
        dataset_df: pd.DataFrame, columns_list: List[str]
    ) -> bool:
        """Checks whether the columns are specified."""
        if set(columns_list) != set(dataset_df.columns):
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


class ModelValidator:
    """Validates the model package's structure and files.

    Parameters
    ----------
    model_config_file_path: str
        Path to the model config file.
    model_package_dir : str
        Path to the model package directory.
    sample_data : pd.DataFrame
        Sample data to be used for the model validation.

    Methods
    -------
    validate:
        Runs all model validations.

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
    ...     model_config_file_path="/path/to/model/config/file",
    ...     model_package_dir="/path/to/model/package",
    ...     sample_data=df,
    ... )
    >>> model_validator.validate()

    """

    def __init__(
        self,
        model_config_file_path: str,
        use_runner: bool = False,
        model_package_dir: Optional[str] = None,
        sample_data: Optional[pd.DataFrame] = None,
    ):
        self.model_config_file_path = model_config_file_path
        self.model_package_dir = model_package_dir
        self.sample_data = sample_data
        self._use_runner = use_runner
        self.failed_validations = []

    def validate(self) -> List[str]:
        """Runs all model validations.

        At each stage, prints all the failed validations.

        Returns
        -------
        List[str]
            A list of all failed validations.
        """
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                            Model validations                             "
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        if self.model_package_dir:
            self._validate_model_package_dir()
            if self._use_runner:
                self._validate_model_runner()
            else:
                self._validate_requirements_file()
                self._validate_prediction_interface()
        self._validate_model_config()

        if not self.failed_validations:
            logger.info("✓ All model validations passed! \n")

        return self.failed_validations

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
            logger.error("Model package structure failed validations:")
            _list_failed_validation_messages(model_package_failed_validations)

        # Add the model package failed validations to the list of all failed validations
        self.failed_validations.extend(model_package_failed_validations)

    def _validate_requirements_file(self):
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

        # Print results of the validation
        if requirements_failed_validations:
            logger.error("`requirements.txt` failed validations:")
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
            with open(self.model_config_file_path, "r", encoding="UTF-8") as stream:
                model_config = yaml.safe_load(stream)

            model_schema = schemas.ModelSchema()
            try:
                model_schema.load(model_config)
            except ma.ValidationError as err:
                model_config_failed_validations.extend(
                    _format_marshmallow_error_message(err)
                )

        # Print results of the validation
        if model_config_failed_validations:
            logger.error("`model_config.yaml` failed validations:")
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
                        except Exception as exc:
                            exception_stack = utils.get_exception_stacktrace(exc)
                            prediction_interface_failed_validations.append(
                                "The `predict_proba` function failed while running the test data. "
                                "It is failing with the following error message: \n"
                                f"\t {exception_stack}"
                            )

        # Print results of the validation
        if prediction_interface_failed_validations:
            logger.error("`prediction_interface.py` failed validations:")
            _list_failed_validation_messages(prediction_interface_failed_validations)

        # Add the `prediction_interface.py` failed validations to the list of all failed validations
        self.failed_validations.extend(prediction_interface_failed_validations)

    def _validate_model_runner(self):
        """Validates the model using the model runner.

        This is mostly meant to be used by the platform, to validate the model. It will
        create the model's environment and use it to run the model.
        """
        model_runner_failed_validations = []

        model_runner = models.ModelRunner(self.model_package_dir)

        # Try to run some data through the runner
        # Will create the model environment if it doesn't exist
        try:
            model_runner.run(self.sample_data)
        except Exception as exc:
            model_runner_failed_validations.append(f"{exc}")

        # Print results of the validation
        if model_runner_failed_validations:
            logger.error("Model runner failed validations:")
            _list_failed_validation_messages(model_runner_failed_validations)

        # Add the model runner failed validations to the list of all failed validations
        self.failed_validations.extend(model_runner_failed_validations)


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

    def validate(self):
        """Validates the project."""
        logger.info(
            "----------------------------------------------------------------------------"
        )
        logger.info(
            "                            Project validations                             "
        )
        logger.info(
            "----------------------------------------------------------------------------\n"
        )
        self._validate_project_config()

        if not self.failed_validations:
            logger.info("✓ All project validations passed!")

        return self.failed_validations

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
            logger.error("Project config failed validations:")
            _list_failed_validation_messages(project_config_failed_validations)

        # Add the commit failed validations to the list of all failed validations
        self.failed_validations.extend(project_config_failed_validations)


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
    for input_data, msg in err.messages.items():
        if input_data == "_schema":
            temp_msg = "\n".join(msg)
            error_msg.append(f"{temp_msg}")
        elif not isinstance(msg, dict):
            temp_msg = msg[0].lower()
            error_msg.append(f"`{input_data}`: {temp_msg}")
        else:
            temp_msg = list(msg.values())[0][0].lower()
            error_msg.append(f"`{input_data}` contains items that are {temp_msg}")

    return error_msg


def _list_failed_validation_messages(failed_validations: List[str]):
    """Prints the failed validations in a list format, with one failed
    validation per line."""
    for msg in failed_validations:
        logger.error("* %s", msg)
