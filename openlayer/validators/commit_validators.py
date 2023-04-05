"""Implements the commit bundle specific validation class.
"""
import logging
from typing import List, Optional

import marshmallow as ma

from .. import schemas, utils
from . import dataset_validators, model_validators
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class CommitBundleValidator(BaseValidator):
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
        super().__init__(resource_display_name="commit bundle")
        self.bundle_path = bundle_path
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

        # Defining which datasets contain predictions
        preds_in_training_set = False
        preds_in_validation_set = False
        if "training" in self._bundle_resources:
            preds_in_training_set = self._dataset_contains_predictions(label="training")

        if "validation" in self._bundle_resources:
            preds_in_validation_set = self._dataset_contains_predictions(
                label="validation"
            )

        if "model" in self._bundle_resources:
            model_config = utils.load_model_config_from_bundle(
                bundle_path=self.bundle_path
            )
            model_type = model_config.get("modelType")
            if (
                not preds_in_training_set or not preds_in_validation_set
            ) and model_type != "baseline":
                self.failed_validations.append(
                    "To push a model to the platform, you must provide "
                    "training and a validation sets with predictions. "
                    "The predictions can be specified in the column `predictionsColumnName` "
                    "as integers and/or in the column `predictionScoresColumnName` as "
                    "lists of class probabilities."
                )
            if model_type == "baseline":
                if (
                    "training" not in self._bundle_resources
                    or "validation" not in self._bundle_resources
                ):
                    self.failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets."
                    )
                elif preds_in_training_set and preds_in_validation_set:
                    self.failed_validations.append(
                        "To push a baseline model to the platform, you must provide "
                        "training and validation sets without predictions in the columns "
                        "`predictionsColumnName` or  `predictionScoresColumnName`."
                    )
        else:
            if "training" in self._bundle_resources and preds_in_validation_set:
                self.failed_validations.append(
                    "A training set was provided alongside with a validation set with"
                    " predictions. Please either provide only a validation set with"
                    " predictions, or a model and both datasets with predictions."
                )
            elif preds_in_training_set:
                self.failed_validations.append(
                    "The training dataset contains predictions, but no model was"
                    " provided. To push a training set with predictions, please provide"
                    " a model and a validation set with predictions as well."
                )

    def _dataset_contains_predictions(self, label: str) -> bool:
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

    def _validate_bundle_resources(self):
        """Runs the corresponding validations for each resource in the bundle."""
        if "training" in self._bundle_resources and not self._skip_dataset_validation:
            training_set_validator = dataset_validators.DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/training/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/training/dataset.csv",
            )
            self.failed_validations.extend(training_set_validator.validate())

        if "validation" in self._bundle_resources and not self._skip_dataset_validation:
            validation_set_validator = dataset_validators.DatasetValidator(
                dataset_config_file_path=f"{self.bundle_path}/validation/dataset_config.yaml",
                dataset_file_path=f"{self.bundle_path}/validation/dataset.csv",
            )
            self.failed_validations.extend(validation_set_validator.validate())

        if "model" in self._bundle_resources and not self._skip_model_validation:
            model_config_file_path = f"{self.bundle_path}/model/model_config.yaml"
            model_config = utils.load_model_config_from_bundle(
                bundle_path=self.bundle_path
            )

            if model_config["modelType"] == "shell":
                model_validator = model_validators.ModelValidator(
                    model_config_file_path=model_config_file_path
                )
            elif model_config["modelType"] == "full":
                # Use data from the validation as test data
                validation_dataset_df = utils.load_dataset_from_bundle(
                    bundle_path=self.bundle_path, label="validation"
                )
                validation_dataset_config = utils.load_dataset_config_from_bundle(
                    bundle_path=self.bundle_path, label="validation"
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

                model_validator = model_validators.ModelValidator(
                    model_config_file_path=model_config_file_path,
                    model_package_dir=f"{self.bundle_path}/model",
                    sample_data=sample_data,
                    use_runner=self._use_runner,
                )
            elif model_config["modelType"] == "baseline":
                model_validator = model_validators.BaselineModelValidator(
                    model_config_file_path=model_config_file_path
                )
            else:
                raise ValueError(
                    f"Invalid model type: {model_config['modelType']}. "
                    "The model type must be one of 'shell', 'full' or 'baseline'."
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
            # Loading the relevant configs
            model_config = {}
            if "model" in self._bundle_resources:
                model_config = utils.load_model_config_from_bundle(
                    bundle_path=self.bundle_path
                )
            training_dataset_config = utils.load_dataset_config_from_bundle(
                bundle_path=self.bundle_path, label="training"
            )
            validation_dataset_config = utils.load_dataset_config_from_bundle(
                bundle_path=self.bundle_path, label="validation"
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
                    self.failed_validations.append(
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
                self.failed_validations.append(
                    "The `classNames` in the provided resources are inconsistent."
                    " The validation set's class names need to contain the training set's."
                    " Furthermore, if a model is provided, its class names must be contained"
                    " in the training and validation sets' class names."
                    " Note that the order of the items in the `classNames` list matters."
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
