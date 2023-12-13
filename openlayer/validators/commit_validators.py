"""Implements the commit bundle specific validation class.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import marshmallow as ma
import pandas as pd
import yaml

from .. import tasks, utils
from ..schemas import project_schemas as schemas
from . import baseline_model_validators, dataset_validators, model_validators
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class BaseCommitBundleValidator(BaseValidator, ABC):
    """Validates the commit bundle prior to push.

    Parameters
    ----------
    bundle_path : str
        The path to the commit bundle (staging area, if for the Python API).
    task_type : tasks.TaskType
        The task type.
    skip_model_validation : bool
        Whether to skip model validation, by default False
    skip_dataset_validation : bool
        Whether to skip dataset validation, by default False
    use_runner : bool
        Whether to use the runner to validate the model, by default False.
    log_file_path : Optional[str], optional
        The path to the log file, by default None
    """

    def __init__(
        self,
        bundle_path: str,
        task_type: tasks.TaskType,
        skip_model_validation: bool = False,
        skip_dataset_validation: bool = False,
        use_runner: bool = False,
        log_file_path: Optional[str] = None,
    ):
        super().__init__(resource_display_name="commit bundle")
        self.bundle_path = bundle_path
        self.task_type = task_type
        self._bundle_resources = utils.list_resources_in_bundle(bundle_path)
        self._skip_model_validation = skip_model_validation
        self._skip_dataset_validation = skip_dataset_validation
        self._use_runner = use_runner

        if log_file_path:
            bundle_file_handler = logging.FileHandler(log_file_path)
            bundle_formatter = logging.Formatter(
                "[%(asctime)s] - %(levelname)s - %(message)s"
            )
            bundle_file_handler.setFormatter(bundle_formatter)
            logger.addHandler(bundle_file_handler)

        self.model_config: Dict[str, any] = (
            utils.load_model_config_from_bundle(bundle_path=bundle_path)
            if "model" in self._bundle_resources
            else {}
        )
        if "training" in self._bundle_resources:
            self.training_dataset_config: Dict[
                str, any
            ] = utils.load_dataset_config_from_bundle(
                bundle_path=bundle_path, label="training"
            )
        elif "fine-tuning" in self._bundle_resources:
            self.training_dataset_config: Dict[
                str, any
            ] = utils.load_dataset_config_from_bundle(
                bundle_path=bundle_path, label="fine-tuning"
            )
        else:
            self.training_dataset_config = {}
        self.validation_dataset_config: Dict[str, any] = (
            utils.load_dataset_config_from_bundle(
                bundle_path=bundle_path, label="validation"
            )
            if "validation" in self._bundle_resources
            else {}
        )

    def _validate(self) -> List[str]:
        """Validates the commit bundle.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        self._validate_bundle_state()

        # Validate individual resources only if the bundle is in a valid state
        # TODO: improve the logic that determines whether to validate individual resources
        if not self.failed_validations:
            self._validate_bundle_resources()

        if not self.failed_validations:
            self._validate_resource_consistency()

    def _validate_bundle_state(self):
        """Checks whether the bundle is in a valid state.

        This includes:
        - When a "model" (shell or full) is included, you always need to
          provide predictions for both "validation" and "training".
        - When a "baseline-model" is included, you always need to provide a "training"
          and "validation" set without predictions.
        - When a "model" nor a "baseline-model" are included, you always need to NOT
          upload predictions with one exception:
            - "validation" set only in bundle, which means the predictions are for the
            previous model version.
        """

        # Defining which datasets contain outputs
        outputs_in_training_set = False
        outputs_in_validation_set = False
        if "training" in self._bundle_resources:
            outputs_in_training_set = self._dataset_contains_output(label="training")
        elif "fine-tuning" in self._bundle_resources:
            outputs_in_training_set = self._dataset_contains_output(label="fine-tuning")
        if "validation" in self._bundle_resources:
            outputs_in_validation_set = self._dataset_contains_output(
                label="validation"
            )

        # Check if flagged to compute the model outputs
        with open(
            f"{self.bundle_path}/commit.yaml", "r", encoding="UTF-8"
        ) as commit_file:
            commit = yaml.safe_load(commit_file)
        compute_outputs = commit.get("computeOutputs", False)

        if "model" in self._bundle_resources:
            model_type = self.model_config.get("modelType")

            if model_type == "baseline":
                if (
                    "training" not in self._bundle_resources
                ) or "validation" not in self._bundle_resources:
                    self.failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets."
                    )
                elif outputs_in_training_set and outputs_in_validation_set:
                    self.failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets without predictions in the columns "
                        "`predictionsColumnName` or  `predictionScoresColumnName`."
                    )
            else:
                if (
                    "training" not in self._bundle_resources
                    or "fine-tuning" not in self._bundle_resources
                ) and "validation" not in self._bundle_resources:
                    self.failed_validations.append(
                        "You are trying to push a model to the platform, but "
                        "you did not provide a training/fine-tuning or validation set. "
                        "To push a model to the platform, you must provide "
                        "either: \n"
                        "- training/fine-tuning and validation sets; or \n"
                        "- a validation set. \n"
                        "In any case, ensure that the model predictions are provided in the "
                        "datasets."
                    )
                elif (
                    "training" not in self._bundle_resources
                    or "fine-tuning" not in self._bundle_resources
                ) and ("validation" in self._bundle_resources):
                    if not outputs_in_validation_set and not compute_outputs:
                        self.failed_validations.append(
                            "You are trying to push a model and a validation set to the platform. "
                            "However, the validation set does not contain predictions. "
                            "Please provide predictions for the validation set. "
                        )
                elif (
                    "training" in self._bundle_resources
                    or "fine-tuning" in self._bundle_resources
                ) and "validation" not in self._bundle_resources:
                    self.failed_validations.append(
                        "You are trying to push a model and a training/fine-tuning set to the platform. "
                        "To push a model to the platform, you must provide "
                        "either: \n"
                        "- training/fine-tuning and validation sets; or \n"
                        "- a validation set. \n"
                        "In any case, ensure that the model predictions are provided in the "
                        "datasets."
                    )
                elif (
                    "training" in self._bundle_resources
                    or "fine-tuning" in self._bundle_resources
                ) and ("validation" in self._bundle_resources):
                    if (
                        not outputs_in_training_set or not outputs_in_validation_set
                    ) and not compute_outputs:
                        self.failed_validations.append(
                            "You are trying to push a model, a training/fine-tuning set and a validation "
                            "set to the platform. "
                            "However, the training/fine-tuning or the validation set do not contain model "
                            "predictions. Please provide predictions for both datasets."
                        )

        else:
            if (
                "training" in self._bundle_resources
                or "fine-tuning" in self._bundle_resources
            ) and ("validation" not in self._bundle_resources):
                if outputs_in_training_set:
                    self.failed_validations.append(
                        "The training/fine-tuning dataset contains predictions, but no model was"
                        " provided. To push a training/fine-tuning set with predictions, please provide"
                        " a model and a validation set with predictions as well."
                    )
            elif (
                "training" in self._bundle_resources
                or "fine-tuning" in self._bundle_resources
            ) and ("validation" in self._bundle_resources):
                if outputs_in_training_set or outputs_in_validation_set:
                    self.failed_validations.append(
                        "You are trying to push a training/fine-tuning set and a validation set to the platform. "
                        "However, the training/fine-tuning or the validation set contain predictions. "
                        "To push datasets with predictions, please provide a model as well."
                    )

    def _validate_bundle_resources(self):
        """Runs the corresponding validations for each resource in the bundle."""
        if "training" in self._bundle_resources and not self._skip_dataset_validation:
            training_set_validator = dataset_validators.get_validator(
                task_type=self.task_type,
                dataset_config_file_path=f"{self.bundle_path}/training/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/training/dataset.csv",
            )
            self.failed_validations.extend(training_set_validator.validate())

        if (
            "fine-tuning" in self._bundle_resources
            and not self._skip_dataset_validation
        ):
            fine_tuning_set_validator = dataset_validators.get_validator(
                task_type=self.task_type,
                dataset_config_file_path=f"{self.bundle_path}/fine-tuning/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/fine-tuning/dataset.csv",
            )
            self.failed_validations.extend(fine_tuning_set_validator.validate())

        if "validation" in self._bundle_resources and not self._skip_dataset_validation:
            validation_set_validator = dataset_validators.get_validator(
                task_type=self.task_type,
                dataset_config_file_path=f"{self.bundle_path}/validation/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/validation/dataset.csv",
            )
            self.failed_validations.extend(validation_set_validator.validate())

        if "model" in self._bundle_resources and not self._skip_model_validation:
            model_config_file_path = f"{self.bundle_path}/model/model_config.yaml"
            model_type = self.model_config.get("modelType")
            if model_type in ("shell", "api"):
                model_validator = model_validators.get_validator(
                    task_type=self.task_type,
                    model_config_file_path=model_config_file_path,
                )
            elif model_type == "full":
                sample_data = self._get_sample_input_data()

                model_validator = model_validators.get_validator(
                    task_type=self.task_type,
                    model_config_file_path=model_config_file_path,
                    model_package_dir=f"{self.bundle_path}/model",
                    sample_data=sample_data,
                    use_runner=self._use_runner,
                )
            elif model_type == "baseline":
                model_validator = baseline_model_validators.get_validator(
                    task_type=self.task_type,
                    model_config_file_path=model_config_file_path,
                )
            else:
                raise ValueError(
                    f"Invalid model type: {model_type}. "
                    "The model type must be one of 'api', 'shell', 'full' or 'baseline'."
                )
            self.failed_validations.extend(model_validator.validate())

    def _validate_resource_consistency(self):
        """Validates that the resources in the bundle are consistent with each other.

        For example, if the `classNames` field on the dataset configs are consistent
        with the one on the model config.
        """
        if (
            "training" in self._bundle_resources
            and "validation" in self._bundle_resources
        ):
            self._validate_input_consistency()
            self._validate_output_consistency()

    @abstractmethod
    def _dataset_contains_output(self, label: str) -> bool:
        """Checks whether the dataset contains output.

        I.e., predictions, for classification, sequences, for s2s, etc.
        """
        pass

    @abstractmethod
    def _get_sample_input_data(self) -> Optional[pd.DataFrame]:
        """Gets a sample of the input data from the bundle.

        This is the data that will be used to validate the model.
        """
        pass

    @abstractmethod
    def _validate_input_consistency(self):
        """Verifies that the input data is consistent across the bundle."""
        pass

    @abstractmethod
    def _validate_output_consistency(self):
        """Verifies that the output data is consistent across the bundle."""
        pass


class TabularCommitBundleValidator(BaseCommitBundleValidator):
    """Tabular commit bundle validator.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _get_sample_input_data(self) -> Optional[pd.DataFrame]:
        """Gets a sample of tabular data input."""
        # Use data from the validation as test data
        sample_data = None
        validation_dataset_df = utils.load_dataset_from_bundle(
            bundle_path=self.bundle_path, label="validation"
        )
        if validation_dataset_df is not None:
            sample_data = validation_dataset_df[
                self.validation_dataset_config["featureNames"]
            ].head()

        return sample_data

    def _validate_input_consistency(self):
        """Verifies that the feature names across the bundle are consistent."""
        # Extracting the relevant vars
        model_feature_names = self.model_config.get("featureNames", [])
        training_feature_names = self.training_dataset_config.get("featureNames", [])
        validation_feature_names = self.validation_dataset_config.get(
            "featureNames", []
        )

        # Validating the `featureNames` field
        if training_feature_names or validation_feature_names:
            if not self._feature_names_consistent(
                model_feature_names=model_feature_names,
                training_feature_names=training_feature_names,
                validation_feature_names=validation_feature_names,
            ):
                self.failed_validations.append(
                    "The `featureNames` in the provided resources are inconsistent."
                    " The training and validation set feature names must have some overlap."
                    " Furthermore, if a model is provided, its feature names must be a subset"
                    " of the feature names in the training and validation sets."
                )

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


class TextCommitBundleValidator(BaseCommitBundleValidator):
    """Text commit bundle validator.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _get_sample_input_data(self) -> Optional[pd.DataFrame]:
        """Gets a sample of text data input."""
        # Use data from the validation as test data
        sample_data = None
        validation_dataset_df = utils.load_dataset_from_bundle(
            bundle_path=self.bundle_path, label="validation"
        )
        if validation_dataset_df is not None:
            sample_data = validation_dataset_df[
                [self.validation_dataset_config["textColumnName"]]
            ].head()

        return sample_data

    def _validate_input_consistency(self):
        """Currently, there are no input consistency checks for text
        bundles."""
        pass


class ClassificationCommitBundleValidator(BaseCommitBundleValidator):
    """Classification commit bundle validator.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _dataset_contains_output(self, label: str) -> bool:
        """Checks whether the dataset contains predictions.

        Parameters
        ----------
        label : str
            The label of the dataset to check.

        Returns
        -------
        bool
            Whether the dataset contains predictions.
        """
        dataset_config = utils.load_dataset_config_from_bundle(
            bundle_path=self.bundle_path, label=label
        )
        predictions_column_name = dataset_config.get("predictionsColumnName")
        prediction_scores_column_name = dataset_config.get("predictionScoresColumnName")
        return (
            predictions_column_name is not None
            or prediction_scores_column_name is not None
        )

    def _validate_output_consistency(self):
        """Verifies that the output data (class names) is consistent across the bundle."""

        # Extracting the relevant vars
        model_class_names = self.model_config.get("classNames", [])
        training_class_names = self.training_dataset_config.get("classNames", [])
        validation_class_names = self.validation_dataset_config.get("classNames", [])

        # Validating the `classNames` field
        if not self._class_names_consistent(
            model_class_names=model_class_names,
            training_class_names=training_class_names,
            validation_class_names=validation_class_names,
        ):
            self.failed_validations.append(
                "The `classNames` in the provided resources are inconsistent."
                " The validation set's class names need to contain the training set's."
                " Furthermore, if a model is provided, its class names must be contained"
                " in the training and validation sets' class names."
                " Note that the order of the items in the `classNames` list matters."
            )

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


class RegressionCommitBundleValidator(BaseCommitBundleValidator):
    """Regression commit bundle validator.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    def _dataset_contains_output(self, label: str) -> bool:
        """Checks whether the dataset contains predictions.

        Parameters
        ----------
        label : str
            The label of the dataset to check.

        Returns
        -------
        bool
            Whether the dataset contains predictions.
        """
        dataset_config = utils.load_dataset_config_from_bundle(
            bundle_path=self.bundle_path, label=label
        )
        predictions_column_name = dataset_config.get("predictionsColumnName")
        return predictions_column_name is not None

    def _validate_output_consistency(self):
        """Currently, there are no output consistency checks for regression
        bundles."""
        pass


class LLMCommitBundleValidator(BaseCommitBundleValidator):
    """LLM commit bundle validator."""

    def _dataset_contains_output(self, label: str) -> bool:
        """Checks whether the dataset contains predictions.

        Parameters
        ----------
        label : str
            The label of the dataset to check.

        Returns
        -------
        bool
            Whether the dataset contains predictions.
        """
        dataset_config = utils.load_dataset_config_from_bundle(
            bundle_path=self.bundle_path, label=label
        )
        output_column_name = dataset_config.get("outputColumnName")
        return output_column_name is not None

    def _get_sample_input_data(self) -> Optional[pd.DataFrame]:
        """Gets a sample of the input data from the bundle.

        This is the data that will be used to validate the model.
        """
        pass

    def _validate_input_consistency(self):
        """Verifies that the input data is consistent across the bundle."""
        pass

    def _validate_output_consistency(self):
        """Verifies that the output data is consistent across the bundle."""
        pass


class TabularClassificationCommitBundleValidator(
    TabularCommitBundleValidator, ClassificationCommitBundleValidator
):
    """Tabular classification commit bundle validator."""

    pass


class TabularRegressionCommitBundleValidator(
    TabularCommitBundleValidator, RegressionCommitBundleValidator
):
    """Tabular regression commit bundle validator."""

    pass


class TextClassificationCommitBundleValidator(
    TextCommitBundleValidator, ClassificationCommitBundleValidator
):
    """Text classification commit bundle validator."""

    pass


# ----------------------------- Factory function ----------------------------- #
def get_validator(
    bundle_path: str,
    task_type: tasks.TaskType,
    skip_model_validation: bool = False,
    skip_dataset_validation: bool = False,
    use_runner: bool = False,
    log_file_path: Optional[str] = None,
):
    """Returns a commit bundle validator based on the task type.

    Parameters
    ----------
    bundle_path : str
        The path to the bundle.
    task_type : tasks.TaskType
        The task type.
    skip_model_validation : bool, optional
        Whether to skip model validation, by default False
    skip_dataset_validation : bool, optional
        Whether to skip dataset validation, by default False
    use_runner : bool, optional
        Whether to use the runner to validate the model, by default False
    log_file_path : Optional[str], optional
        The path to the log file, by default None

    Returns
    -------
    BaseCommitBundleValidator
        The commit bundle validator.
    """
    if task_type == tasks.TaskType.TabularClassification:
        return TabularClassificationCommitBundleValidator(
            task_type=task_type,
            bundle_path=bundle_path,
            skip_model_validation=skip_model_validation,
            skip_dataset_validation=skip_dataset_validation,
            use_runner=use_runner,
            log_file_path=log_file_path,
        )
    elif task_type == tasks.TaskType.TabularRegression:
        return TabularRegressionCommitBundleValidator(
            task_type=task_type,
            bundle_path=bundle_path,
            skip_model_validation=skip_model_validation,
            skip_dataset_validation=skip_dataset_validation,
            use_runner=use_runner,
            log_file_path=log_file_path,
        )
    elif task_type == tasks.TaskType.TextClassification:
        return TextClassificationCommitBundleValidator(
            task_type=task_type,
            bundle_path=bundle_path,
            skip_model_validation=skip_model_validation,
            skip_dataset_validation=skip_dataset_validation,
            use_runner=use_runner,
            log_file_path=log_file_path,
        )
    elif task_type in [
        tasks.TaskType.LLM,
        tasks.TaskType.LLMNER,
        tasks.TaskType.LLMQuestionAnswering,
        tasks.TaskType.LLMSummarization,
        tasks.TaskType.LLMTranslation,
    ]:
        return LLMCommitBundleValidator(
            task_type=task_type,
            bundle_path=bundle_path,
            skip_model_validation=skip_model_validation,
            skip_dataset_validation=skip_dataset_validation,
            use_runner=use_runner,
            log_file_path=log_file_path,
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")


class CommitValidator(BaseValidator):
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
        super().__init__(resource_display_name="commit message")
        self.commit_message = commit_message

    def _validate(self) -> List[str]:
        """Validates the commit.

        Returns
        -------
        List[str]
            A list of failed validations.
        """
        self._validate_commit_message()

    def _validate_commit_message(self):
        """Checks whether the commit message is valid."""
        commit_schema = schemas.CommitSchema()
        try:
            commit_schema.load({"commitMessage": self.commit_message})
        except ma.ValidationError as err:
            self.failed_validations.extend(self._format_marshmallow_error_message(err))
