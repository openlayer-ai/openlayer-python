# pylint: disable=bare-except
"""Implements the dataset specific validation classes.
"""
import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import marshmallow as ma
import pandas as pd
import yaml

from .. import constants, tasks
from ..datasets import DatasetType
from ..schemas import dataset_schemas
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class BaseDatasetValidator(BaseValidator, ABC):
    """Validates the dataset and its arguments.

    Either the ``dataset_file_path`` or the ``dataset_df`` must be
    provided (not both).

    Either the ``dataset_config_file_path`` or the ``dataset_config``
    must be provided (not both).

    Parameters
    ----------
    task_type : tasks.TaskType, optional
        The task type of the dataset.
    dataset_config_file_path : str, optional
        The path to the dataset_config.yaml file.
    dataset_config : dict, optional
        The dataset_config as a dictionary.
    dataset_file_path : str, optional
        The path to the dataset file.
    dataset_df : pd.DataFrame, optional
        The dataset to validate.
    log_file_path : str, optional
        The path to the log file.
    """

    def __init__(
        self,
        task_type: tasks.TaskType,
        dataset_config_file_path: Optional[str] = None,
        dataset_config: Optional[Dict] = None,
        dataset_file_path: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        log_file_path: Optional[str] = None,
    ):
        super().__init__(resource_display_name="dataset")

        if log_file_path:
            bundle_file_handler = logging.FileHandler(log_file_path)
            bundle_formatter = logging.Formatter(
                "[%(asctime)s] - %(levelname)s - %(message)s"
            )
            bundle_file_handler.setFormatter(bundle_formatter)
            logger.addHandler(bundle_file_handler)

        if dataset_df is not None and dataset_file_path:
            raise ValueError(
                "Both dataset_df and dataset_file_path are provided."
                " Please provide only one of them."
            )
        if dataset_df is None and not dataset_file_path:
            raise ValueError(
                "Neither dataset_df nor dataset_file_path is provided."
                " Please provide one of them."
            )
        if dataset_config_file_path and dataset_config:
            raise ValueError(
                "Both dataset_config_file_path and dataset_config are provided."
                " Please provide only one of them."
            )
        if not dataset_config_file_path and not dataset_config:
            raise ValueError(
                "Neither dataset_config_file_path nor dataset_config is provided."
                " Please provide one of them."
            )

        self.dataset_file_path = dataset_file_path
        self.dataset_df = dataset_df
        self.dataset_config_file_path = dataset_config_file_path
        self.dataset_config = dataset_config
        self.task_type = task_type

    def _validate(self) -> List[str]:
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

        # Update the resource_display_name with the dataset label
        label = self.dataset_config.get("label")
        if label:
            self.resource_display_name = (
                self.dataset_config["label"] + " " + self.resource_display_name
            )

    def _validate_dataset_config(self):
        """Checks whether the dataset_config is valid.

        Beware of the order of the validations, as it is important.
        """
        self._validate_file_existence()
        self._load_dataset_config()
        self._validate_dataset_schema()

    def _validate_file_existence(self):
        """Checks whether the dataset_config_file_path exists."""
        # File existence check
        if self.dataset_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.dataset_config_file_path)):
                self.failed_validations.append(
                    f"File `{self.dataset_config_file_path}` does not exist."
                )

    def _load_dataset_config(self):
        """Loads the dataset_config_file_path into the `self.dataset_config`
        attribute."""
        if self.dataset_config_file_path:
            try:
                with open(
                    self.dataset_config_file_path, "r", encoding="UTF-8"
                ) as stream:
                    self.dataset_config = yaml.safe_load(stream)
            # pylint: disable=broad-exception-caught
            except Exception:
                self.failed_validations.append(
                    f"File `{self.dataset_config_file_path}` is not a valid .yaml file."
                )

    def _validate_dataset_schema(self):
        """Checks whether the dataset schema is valid."""
        if self.dataset_config:
            label = self.dataset_config.get("label")
            if label in [
                DatasetType.Training.value,
                DatasetType.Validation.value,
                DatasetType.FineTuning.value,
            ]:
                dataset_schema = dataset_schemas.DatasetSchema()
            elif label == DatasetType.Reference.value:
                dataset_schema = dataset_schemas.ReferenceDatasetSchema()
            elif label == DatasetType.Production.value:
                dataset_schema = dataset_schemas.ProductionDataSchema()
            else:
                self.failed_validations.append(
                    f"The dataset label `{label}` is not supported. "
                    "The supported dataset labels are 'training', 'validation', "
                    "'fine-tuning', 'reference', and 'production'."
                )
                return

            try:
                dataset_schema.load(
                    {"task_type": self.task_type.value, **self.dataset_config}
                )
            except ma.ValidationError as err:
                self.failed_validations.extend(
                    self._format_marshmallow_error_message(err)
                )

    def _validate_dataset_file(self):
        """Checks whether the dataset file exists and is valid.

        If it is valid, it loads the dataset file into the `self.dataset_df`
        attribute.

        Beware of the order of the validations, as it is important.
        """
        # File existence check
        if not os.path.isfile(os.path.expanduser(self.dataset_file_path)):
            self.failed_validations.append(
                f"File `{self.dataset_file_path}` does not exist."
            )
        else:
            # File format (csv) check by loading it as a pandas df
            try:
                self.dataset_df = pd.read_csv(self.dataset_file_path)
            # pylint: disable=broad-exception-caught
            except Exception:
                self.failed_validations.append(
                    f"File `{self.dataset_file_path}` is not a valid .csv file."
                )

    def _validate_dataset_and_config_consistency(self):
        """Checks whether the dataset and its config are consistent.

        Beware of the order of the validations, as it is important.
        """

        if self.dataset_config and self.dataset_df is not None:
            # Dataset-wide validations
            self._validate_dataset_dtypes()

            # Timestamps, id, and latency validations
            if self.dataset_config.get("timestampColumnName"):
                self._validate_timestamps()
            if self.dataset_config.get("inferenceIdColumnName"):
                self._validate_inference_ids()
            if self.dataset_config.get("latencyColumnName"):
                self._validate_latencies()

            self._validate_inputs()
            self._validate_outputs()

    def _validate_dataset_dtypes(self):
        """Checks whether the dataset contains unsupported dtypes."""
        supported_dtypes = {"bool", "float32", "float64", "int32", "int64", "object"}
        dataset_df_dtypes = {dtype.name for dtype in self.dataset_df.dtypes}
        unsupported_dtypes = dataset_df_dtypes - supported_dtypes
        if unsupported_dtypes:
            self.failed_validations.append(
                "The dataset contains unsupported dtypes. The supported dtypes are "
                "'bool', 'float32', 'float64', 'int32', 'int64', 'object'. "
                f"The dataset contains the following unsupported dtypes: {unsupported_dtypes}"
                " Please cast the columns in your dataset to conform to these dtypes."
            )

    def _validate_timestamps(self):
        """Checks whether the timestamps are in the correct format."""
        timestamp_column_name = self.dataset_config.get("timestampColumnName")
        if timestamp_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The timestamp column `{timestamp_column_name}` specified as "
                "`timestampColumnName` is not in the dataset."
            )
        else:
            # Validate if values in the timestamp column are UNIX epoch/time in seconds
            if not self._timestamps_are_unix_epoch_seconds(
                self.dataset_df, timestamp_column_name
            ):
                self.failed_validations.append(
                    f"The timestamps in the column `{timestamp_column_name}` specified"
                    " as `timestampColumnName` are not in the correct format. "
                    "Please make sure that the timestamps are UNIX epoch/time in"
                    " seconds."
                )
            elif not self._timestamps_within_valid_range(
                self.dataset_df, timestamp_column_name
            ):
                self.failed_validations.append(
                    f"The timestamps in the column `{timestamp_column_name}` specified"
                    " as `timestampColumnName` are not within the allowed range. "
                    "The allowed range is from 2 years ago to 2 years into the future. "
                    "Please make sure that the timestamps are within the allowed range."
                )

    @staticmethod
    def _timestamps_are_unix_epoch_seconds(
        dataset_df: pd.DataFrame, timestamp_column_name: str
    ) -> bool:
        """Checks whether the timestamps are UNIX epoch/time in seconds."""
        try:
            # Note the unit="s" argument
            pd.to_datetime(dataset_df[timestamp_column_name], unit="s")
        # pylint: disable=broad-exception-caught
        except Exception:
            return False
        return True

    @staticmethod
    def _timestamps_within_valid_range(
        dataset_df: pd.DataFrame, timestamp_column_name: str
    ) -> bool:
        """Checks whether the timestamps are within the allowed range."""
        # Note the unit="s" argument
        timestamps = pd.to_datetime(
            dataset_df[timestamp_column_name], utc=True, unit="s"
        )
        now = pd.Timestamp.utcnow()
        two_years_ago = now - pd.Timedelta(days=365 * 2)
        two_years_from_now = now + pd.Timedelta(days=365 * 2)
        if any(
            (timestamp < two_years_ago) or (timestamp > two_years_from_now)
            for timestamp in timestamps
        ):
            return False
        return True

    def _validate_inference_ids(self):
        """Checks whether the inference ids are in the correct format."""
        inference_id_column_name = self.dataset_config.get("inferenceIdColumnName")
        if inference_id_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The inference id column `{inference_id_column_name}` specified as "
                "`inferenceIdColumnName` is not in the dataset."
            )
        else:
            num_unique_ids = len(self.dataset_df[inference_id_column_name].unique())
            if num_unique_ids != len(self.dataset_df):
                self.failed_validations.append(
                    f"The inference ids in the column `{inference_id_column_name}`"
                    " specified as `inferenceIdColumnName` are not unique. "
                    "This means that more than one inference has the same id. "
                    "Please make sure that the inference ids are unique."
                )

    def _validate_latencies(self):
        """Checks if the latencies are in the correct format."""
        latency_column_name = self.dataset_config.get("latencyColumnName")
        if latency_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The latency column `{latency_column_name}` specified as "
                "`latencyColumnName` is not in the dataset."
            )
        else:
            # Validate if values in the latency column are numbers (ints or floats)
            if not self._values_are_numbers(self.dataset_df, latency_column_name):
                self.failed_validations.append(
                    f"The latencies in the column `{latency_column_name}` specified"
                    " as `latencyColumnName` are not in the correct format. "
                    "Please make sure that the dtype of the column with the latencies "
                    "is one of int32, int64, float32, or float64."
                )

    def _values_are_numbers(self, dataset_df: pd.DataFrame, column_name: str) -> bool:
        """Checks whether the values in the column are numbers (ints or floats)."""
        if dataset_df[column_name].dtype.name in (
            "int64",
            "int32",
            "float32",
            "float64",
        ):
            return True
        return False

    @abstractmethod
    def _validate_inputs(self):
        """To be implemented by InputValidator child classes."""
        pass

    @abstractmethod
    def _validate_outputs(self):
        """To be implemented by OutputValidator child classes."""
        pass


# ----------------------------- Input validators ----------------------------- #
class LLInputValidator(BaseDatasetValidator):
    """Validates LLM inputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    input_variable_names: Optional[List[str]] = None
    context_column_name: Optional[str] = None
    question_column_name: Optional[str] = None

    def _validate_inputs(self):
        """Validates LLM inputs."""
        # Setting the attributes needed for the validations
        self.input_variable_names = self.dataset_config.get("inputVariableNames")
        self.context_column_name = self.dataset_config.get("contextColumnName")
        self.question_column_name = self.dataset_config.get("questionColumnName")

        if self.input_variable_names:
            self._validate_input_variables()
        if self.context_column_name:
            self._validate_context()
        if self.question_column_name:
            self._validate_question()

    def _validate_input_variables(self):
        """Validates the data in the input variables columns."""
        if columns_not_in_df(self.dataset_df, self.input_variable_names):
            self.failed_validations.append(
                "Not all input variables specified in `inputVariableNames` are in "
                "the dataset. Please make sure that the dataset contains one column "
                "for each input variable specified in `inputVariableNames`."
            )
        elif self._input_variables_not_castable_to_str(
            dataset_df=self.dataset_df, input_variable_names=self.input_variable_names
        ):
            self.failed_validations.append(
                "Not all input variables are castable to string. Please make sure that "
                "all input variables specified in `inputVariableNames` can be "
                "cast to string."
            )
        else:
            for input_variable in self.input_variable_names:
                if exceeds_character_limit(self.dataset_df, input_variable):
                    self.failed_validations.append(
                        f"Input variable `{input_variable}` exceeds the maximum "
                        f"character limit of {constants.MAXIMUM_CHARACTER_LIMIT} characters. "
                        "Please make sure that all input variables specified in "
                        "`inputVariableNames` do not exceed the maximum character limit."
                    )

    def _validate_context(self):
        """Validations on the ground truth column."""
        if self.context_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The context column `{self.context_column_name}` specified as"
                " `contextColumnName` is not in the dataset."
            )
        elif not hasattr(self.dataset_df[self.context_column_name], "str"):
            self.failed_validations.append(
                f"The context column `{self.context_column_name}` specified as"
                " `contextColumnName` is not a string column."
            )
        elif exceeds_character_limit(self.dataset_df, self.context_column_name):
            self.failed_validations.append(
                f"The ground truth column `{self.context_column_name}` specified as"
                " `contextColumnName` contains strings that exceed the "
                f" {constants.MAXIMUM_CHARACTER_LIMIT} character limit."
            )

    def _validate_question(self):
        """Validations on the ground truth column."""
        if self.question_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The question column `{self.question_column_name}` specified as"
                " `questionColumnName` is not in the dataset."
            )
        elif not hasattr(self.dataset_df[self.question_column_name], "str"):
            self.failed_validations.append(
                f"The question column `{self.question_column_name}` specified as"
                " `questionColumnName` is not a string column."
            )
        elif exceeds_character_limit(self.dataset_df, self.question_column_name):
            self.failed_validations.append(
                f"The ground truth column `{self.question_column_name}` specified as"
                " `questionColumnName` contains strings that exceed the "
                f" {constants.MAXIMUM_CHARACTER_LIMIT} character limit."
            )

    @staticmethod
    def _input_variables_not_castable_to_str(
        dataset_df: pd.DataFrame,
        input_variable_names: List[str],
    ) -> bool:
        """Checks whether the input variables can be cast to string."""
        for input_variable in input_variable_names:
            try:
                dataset_df[input_variable].astype(str)
            except ValueError:
                return True
        return False


class TabularInputValidator(BaseDatasetValidator):
    """Validates tabular inputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    categorical_feature_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None

    def _validate_inputs(self):
        """Validates tabular inputs."""
        # Setting the attributes needed for the validations
        self.categorical_feature_names = self.dataset_config.get(
            "categoricalFeatureNames"
        )
        self.feature_names = self.dataset_config.get("featureNames")

        if self.feature_names:
            self._validate_features()

    def _validate_features(self):
        """Validates the data in the features and categorical features columns."""
        if columns_not_in_df(self.dataset_df, self.feature_names):
            self.failed_validations.append(
                "There are features specified in `featureNames` which are "
                "not in the dataset."
            )
        if self.categorical_feature_names:
            if columns_not_in_df(self.dataset_df, self.categorical_feature_names):
                self.failed_validations.append(
                    "There are categorical features specified in `categoricalFeatureNames` "
                    "which are not in the dataset."
                )


class TextInputValidator(BaseDatasetValidator):
    """Validates text inputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    text_column_name: Optional[str] = None

    def _validate_inputs(self):
        """Validates text inputs."""
        # Setting the attributes needed for the validations
        self.text_column_name = self.dataset_config.get("textColumnName")

        if self.text_column_name:
            self._validate_text()

    def _validate_text(self):
        """Validates the data in the text column."""
        if self.text_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The text column `{self.text_column_name}` specified as `textColumnName` "
                "is not in the dataset."
            )
        elif self._text_column_not_string_or_nans(
            self.dataset_df, self.text_column_name
        ):
            self.failed_validations.append(
                f"The column `{self.text_column_name}` specified as `textColumnName` "
                "contains values  that are not strings.  "
                "Please make sure that the column contains only strings or NaNs."
            )
        elif exceeds_character_limit(self.dataset_df, self.text_column_name):
            self.failed_validations.append(
                f"The column `{self.text_column_name}` of the dataset contains rows that "
                f"exceed the {constants.MAXIMUM_CHARACTER_LIMIT} character limit."
            )

    @staticmethod
    def _text_column_not_string_or_nans(
        dataset_df: pd.DataFrame, text_column_name: str
    ) -> bool:
        """Checks whether the text column contains only strings
        and NaNs."""
        for text in dataset_df[text_column_name]:
            if not isinstance(text, str) and not pd.isna(text):
                return True
        return False


# ----------------------------- Output validators ---------------------------- #
class ClassificationOutputValidator(BaseDatasetValidator):
    """Validates classification outputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    class_names: Optional[List[str]] = None
    label_column_name: Optional[str] = None
    predictions_column_name: Optional[str] = None
    prediction_scores_column_name: Optional[str] = None

    def _validate_outputs(self):
        """Validates the classification outputs (i.e., predictions and classes)."""
        self.class_names = self.dataset_config.get("classNames")
        self.label_column_name = self.dataset_config.get("labelColumnName")
        self.predictions_column_name = self.dataset_config.get("predictionsColumnName")
        self.prediction_scores_column_name = self.dataset_config.get(
            "predictionScoresColumnName"
        )
        # Label validations
        if self.label_column_name:
            self._validate_labels()

        # Predictions validations
        if self.predictions_column_name:
            self._validate_predictions()

        # Prediction scores validations
        if self.prediction_scores_column_name:
            self._validate_prediction_scores()

    def _validate_labels(self):
        """Validates the data in the label column."""
        if self.label_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The label column `{self.label_column_name}` specified as `labelColumnName` "
                "is not in the dataset."
            )
        else:
            if self.class_names:
                self._validate_categories_zero_indexed(
                    column_name=self.label_column_name
                )
            if self.predictions_column_name:
                self._validate_label_and_predictions_columns_different()

    def _validate_categories_zero_indexed(self, column_name: str):
        """Checks whether the categories are zero-indexed in the dataset's `column_name`."""
        if self.dataset_df[column_name].dtype.name not in ("int64", "int32"):
            self.failed_validations.append(
                f"The classes in the dataset column `{column_name}` must be integers. "
                f"Make sure that the column `{column_name}` is of dtype `int32` or `int64`."
            )
        else:
            max_class = self.dataset_df[column_name].max()

            if max_class > len(self.class_names) - 1:
                self.failed_validations.append(
                    "The classes in the dataset are not zero-indexed. "
                    f"The column `{column_name}` contains classes up to {max_class}, "
                    f"but the list of classes provided in `classNames` contains only "
                    f"{len(self.class_names)} elements. "
                    f"Make sure that the classes in the column `{column_name}` "
                    "are zero-indexed integers that match the list in `classNames`. "
                    "Note that the index of the first class should be 0, not 1, so "
                    f"if the maximum class is {max_class}, the `classNames` list "
                    f"should contain {max_class + 1} elements."
                )

    def _validate_label_and_predictions_columns_different(self):
        """Checks whether the predictions and label columns are different."""
        if self.label_column_name == self.predictions_column_name:
            self.failed_validations.append(
                "The predictions column and the label column are the same. "
                "Please specify different columns for the predictions and the label."
            )

    def _validate_predictions(self):
        """Validates the data in the predictions column."""
        if self.predictions_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The predictions column `{self.predictions_column_name}` specified as "
                "`predictionsColumnName` is not in the dataset."
            )
        else:
            if self.class_names:
                self._validate_categories_zero_indexed(
                    column_name=self.predictions_column_name
                )

    def _validate_prediction_scores(self):
        """Validates the data in the prediction scores column."""
        if self.prediction_scores_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The predictions column `{self.prediction_scores_column_name}` specified as"
                " `predictionScoresColumnName` is not in the dataset."
            )
        else:
            try:
                # Getting prediction lists from strings saved in the csv
                self.dataset_df[self.prediction_scores_column_name] = self.dataset_df[
                    self.prediction_scores_column_name
                ].apply(ast.literal_eval)

                if self._predictions_not_lists(
                    self.dataset_df, self.prediction_scores_column_name
                ):
                    self.failed_validations.append(
                        f"There are predictions in the column `{self.prediction_scores_column_name}` "
                        " that are not lists. Please make sure that all the predictions are "
                        "lists of floats."
                    )
                else:
                    if self._prediction_lists_not_same_length(
                        self.dataset_df, self.prediction_scores_column_name
                    ):
                        self.failed_validations.append(
                            "There are prediction lists in the column "
                            f"`{self.prediction_scores_column_name}` "
                            "that have different lengths. "
                            "Please make sure that all prediction lists "
                            "are of the same length."
                        )
                    else:
                        if self._predictions_not_class_probabilities(
                            self.dataset_df, self.prediction_scores_column_name
                        ):
                            self.failed_validations.append(
                                "The predictions in the column "
                                f"`{self.prediction_scores_column_name}` "
                                "are not class probabilities. "
                                "Please make sure that the predictions are lists "
                                "of floats that sum to 1."
                            )
                        elif self.class_names:
                            if self._predictions_not_in_class_names(
                                self.dataset_df,
                                self.prediction_scores_column_name,
                                self.class_names,
                            ):
                                self.failed_validations.append(
                                    f"There are predictions in `{self.prediction_scores_column_name}`"
                                    f" that don't match the classes in `{self.class_names}`. "
                                    "Please make sure that all the lists with predictions "
                                    "have the same length as the `classNames` list."
                                )
            # pylint: disable=broad-exception-caught
            except Exception:
                self.failed_validations.append(
                    f"The predictions in the column `{self.prediction_scores_column_name}` "
                    "are not lists. "
                    "Please make sure that the predictions are lists of floats."
                )

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
            sum(predictions) < 0.9 or sum(predictions) > 1.1
            for predictions in dataset_df[predictions_column_name]
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


class LLMOutputValidator(BaseDatasetValidator):
    """Validates LLM outputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    ground_truth_column_name: Optional[str] = None
    output_column_name: Optional[str] = None

    def _validate_outputs(self):
        """Validates the LLM outputs (i.e., ground truth and output)."""
        self.ground_truth_column_name = self.dataset_config.get("groundTruthColumnName")
        self.output_column_name = self.dataset_config.get("outputColumnName")
        self.num_of_token_column_name = self.dataset_config.get("numOfTokenColumnName")

        if self.ground_truth_column_name:
            self._validate_ground_truth()

        if self.output_column_name:
            self._validate_output()

        if self.ground_truth_column_name and self.output_column_name:
            self._validate_ground_truth_and_output_columns_different()

        if self.num_of_token_column_name:
            self._validate_num_of_token()

    def _validate_ground_truth(self):
        """Validations on the ground truth column."""
        if self.ground_truth_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The ground truth column `{self.ground_truth_column_name}` specified as"
                " `groundTruthColumnName` is not in the dataset."
            )
        elif not hasattr(self.dataset_df[self.ground_truth_column_name], "str"):
            self.failed_validations.append(
                f"The ground truth column `{self.ground_truth_column_name}` specified as"
                " `groundTruthColumnName` is not a string column."
            )
        elif exceeds_character_limit(self.dataset_df, self.ground_truth_column_name):
            self.failed_validations.append(
                f"The ground truth column `{self.ground_truth_column_name}` specified as"
                " `groundTruthColumnName` contains strings that exceed the "
                f" {constants.MAXIMUM_CHARACTER_LIMIT} character limit."
            )

    def _validate_output(self):
        """Validations on the output column."""
        if self.output_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The output column `{self.output_column_name}` specified as"
                " `outputColumnName` is not in the dataset."
            )
        elif not hasattr(self.dataset_df[self.output_column_name], "str"):
            self.failed_validations.append(
                f"The output column `{self.output_column_name}` specified as"
                " `outputColumnName` is not a string column."
            )
        elif exceeds_character_limit(self.dataset_df, self.output_column_name):
            self.failed_validations.append(
                f"The output column `{self.output_column_name}` specified as"
                " `outputColumnName` contains strings that exceed the "
                f" {constants.MAXIMUM_CHARACTER_LIMIT} character limit."
            )

    def _validate_ground_truth_and_output_columns_different(self):
        """Validates that the ground truth and output columns are different."""
        if self.ground_truth_column_name == self.output_column_name:
            self.failed_validations.append(
                "The output column and the ground truth column are the same. "
                "Please specify different columns for the output and the ground truths."
            )

    def _validate_num_of_token(self):
        """Validates the number of tokens column."""
        if self.num_of_token_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The number of tokens column `{self.num_of_token_column_name}` "
                "specified as `numOfTokenColumnName` is not in the dataset."
            )
        elif not self._values_are_numbers(
            self.dataset_df, self.num_of_token_column_name
        ):
            self.failed_validations.append(
                f"The number of tokens in the column `{self.num_of_token_column_name}`"
                " specified as `numOfTokenColumnName` are not in the correct format. "
                "Please make sure that the dtype of the column with the number of"
                " tokens is one of int32, int64, float32, or float64."
            )


class RegressionOutputValidator(BaseDatasetValidator):
    """Validates regression outputs.

    This is not a complete implementation of the abstract class. This is a
    partial implementation used to compose the full classes.
    """

    target_column_name: Optional[str] = None
    predictions_column_name: Optional[str] = None

    def _validate_outputs(self):
        """Validates the classification outputs (i.e., predictions and classes)."""
        self.target_column_name = self.dataset_config.get("targetColumnName")
        self.predictions_column_name = self.dataset_config.get("predictionsColumnName")

        if self.target_column_name:
            self._validate_targets()

        if self.predictions_column_name:
            self._validate_predictions()

        if self.target_column_name and self.predictions_column_name:
            self._validate_targets_and_predictions_columns_different()

    def _validate_targets(self):
        """Checks whether the target column is in the dataset and
        if the targets are floats."""
        if self.target_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The target column `{self.target_column_name}` specified as "
                "`targetColumnName` is not in the dataset."
            )
        else:
            self._validate_values_are_floats(column_name=self.target_column_name)

    def _validate_predictions(self):
        """Checks whether the predictions column is in the dataset and
        if the values are floats."""
        if self.predictions_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The prediction column `{self.predictions_column_name}` specified as "
                "`predictionsColumnName` is not in the dataset."
            )
        else:
            self._validate_values_are_floats(column_name=self.predictions_column_name)

    def _validate_values_are_floats(self, column_name: str):
        """Checks whether the targets are floats."""
        if not all(isinstance(value, float) for value in self.dataset_df[column_name]):
            self.failed_validations.append(
                f"There are values in the column `{column_name}` that "
                "are not floats. Please make sure that all values in the column "
                "are floats."
            )

    def _validate_targets_and_predictions_columns_different(self):
        """Checks whether the predictions and targets columns are different."""
        if self.target_column_name == self.predictions_column_name:
            self.failed_validations.append(
                "The target column and the predictions column are the same. "
                "Please specify different columns for the predictions and the target."
            )


# ------------------------ Complete dataset validators ----------------------- #
class LLMDatasetValidator(LLInputValidator, LLMOutputValidator):
    """Validates an LLM dataset."""

    pass


class TabularClassificationDatasetValidator(
    TabularInputValidator, ClassificationOutputValidator
):
    """Validates a tabular classification dataset."""

    pass


class TabularRegressionDatasetValidator(
    TabularInputValidator, RegressionOutputValidator
):
    """Validates a tabular regression dataset."""

    pass


class TextClassificationDatasetValidator(
    TextInputValidator, ClassificationOutputValidator
):
    """Validates a text classification dataset."""

    pass


# ----------------------------- Factory function ----------------------------- #
def get_validator(
    task_type: tasks.TaskType,
    dataset_config_file_path: Optional[str] = None,
    dataset_config: Optional[Dict] = None,
    dataset_file_path: Optional[str] = None,
    dataset_df: Optional[pd.DataFrame] = None,
    log_file_path: Optional[str] = None,
) -> BaseDatasetValidator:
    """Factory function to get the correct dataset validator for the task type.

    Parameters
    ----------
    task_type: :obj:`TaskType`
        The task type of the dataset.
    dataset_config_file_path : str, optional
        The path to the dataset_config.yaml file.
    dataset_config : dict, optional
        The dataset_config as a dictionary.
    dataset_file_path : str, optional
        The path to the dataset file.
    dataset_df : pd.DataFrame, optional
        The dataset to validate.
    log_file_path : str, optional
        The path to the log file.

    Returns
    -------
    DatasetValidator :
        The correct dataset validator for the ``task_type`` specified.

    Examples
    --------

    For example, to get the tabular dataset validator, you can do the following:

    >>> from openlayer.validators import dataset_validators
    >>> from openlayer.tasks import TaskType
    >>>
    >>> validator = dataset_validators.get_validator(
    >>>     task_type=TaskType.TabularClassification,
    >>>     dataset_config_file_path="dataset_config.yaml",
    >>>     dataset_file_path="dataset.csv",
    >>> )

    The ``validator`` object will be an instance of the
    :obj:`TabularClassificationDatasetValidator` class.

    Then, you can run the validations by calling the :obj:`validate` method:

    >>> validator.validate()

    If there are failed validations, they will be shown on the screen and a list
    of all failed validations will be returned.

    The same logic applies to the other task types.

    """
    if task_type == tasks.TaskType.TabularClassification:
        return TabularClassificationDatasetValidator(
            dataset_config_file_path=dataset_config_file_path,
            dataset_config=dataset_config,
            dataset_file_path=dataset_file_path,
            dataset_df=dataset_df,
            task_type=task_type,
            log_file_path=log_file_path,
        )
    elif task_type == tasks.TaskType.TabularRegression:
        return TabularRegressionDatasetValidator(
            dataset_config_file_path=dataset_config_file_path,
            dataset_config=dataset_config,
            dataset_file_path=dataset_file_path,
            dataset_df=dataset_df,
            task_type=task_type,
            log_file_path=log_file_path,
        )
    elif task_type == tasks.TaskType.TextClassification:
        return TextClassificationDatasetValidator(
            dataset_config_file_path=dataset_config_file_path,
            dataset_config=dataset_config,
            dataset_file_path=dataset_file_path,
            dataset_df=dataset_df,
            task_type=task_type,
            log_file_path=log_file_path,
        )
    elif task_type in [
        tasks.TaskType.LLM,
        tasks.TaskType.LLMNER,
        tasks.TaskType.LLMQuestionAnswering,
        tasks.TaskType.LLMSummarization,
        tasks.TaskType.LLMTranslation,
    ]:
        return LLMDatasetValidator(
            dataset_config_file_path=dataset_config_file_path,
            dataset_config=dataset_config,
            dataset_file_path=dataset_file_path,
            dataset_df=dataset_df,
            task_type=task_type,
            log_file_path=log_file_path,
        )
    else:
        raise ValueError(f"Task type `{task_type}` is not supported.")


# --------------- Helper functions used by multiple validators --------------- #
def columns_not_in_df(df: pd.DataFrame, columns_list: List[str]) -> bool:
    """Checks whether the columns are in the dataset."""
    if set(columns_list) - set(df.columns):
        return True
    return False


def exceeds_character_limit(df: pd.DataFrame, column: str) -> bool:
    """Checks whether the column exceeds the character limit."""
    if not hasattr(df[column], "str"):
        return False
    if df[column].str.len().max() > constants.MAXIMUM_CHARACTER_LIMIT:
        return True
    return False
