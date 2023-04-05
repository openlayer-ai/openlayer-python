"""Implements the commit bundle specific validation class.
"""
import ast
import logging
import os
from typing import Dict, List, Optional

import marshmallow as ma
import pandas as pd
import yaml

from .. import schemas
from .base_validator import BaseValidator

logger = logging.getLogger("validators")


class DatasetValidator(BaseValidator):
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
    :class:`DatasetValidator` class as follows:

    >>> from openlayer.validators import dataset_validators
    >>>
    >>> dataset_validator = dataset_validators.DatasetValidator(
    ...     dataset_config_file_path="dataset_config.yaml",
    ...     dataset_file_path="dataset.csv",
    ... )
    >>> dataset_validator.validate()

    Alternatively, if we have a ``dataset_config.yaml`` file in the current
    directory and a ``dataset_df`` DataFrame, we can use the
    :class:`DatasetValidator` class as follows:

    >>> from openlayer.validators import dataset_validators
    >>>
    >>> dataset_validator = dataset_validators.DatasetValidator(
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
        super().__init__(resource_display_name="dataset")
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

        # Attributes to be set during validation
        self.categorical_feature_names: Optional[List[str]] = None
        self.class_names: Optional[List[str]] = None
        self.column_names: Optional[List[str]] = None
        self.label_column_name: Optional[str] = None
        self.feature_names: Optional[List[str]] = None
        self.text_column_name: Optional[str] = None
        self.predictions_column_name: Optional[str] = None
        self.prediction_scores_column_name: Optional[str] = None

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
        # File existence check
        if self.dataset_config_file_path:
            if not os.path.isfile(os.path.expanduser(self.dataset_config_file_path)):
                self.failed_validations.append(
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
            except Exception:
                self.failed_validations.append(
                    f"File `{self.dataset_file_path}` is not a valid .csv file."
                )

    def _validate_dataset_and_config_consistency(self):
        """Checks whether the dataset and its config are consistent.

        Beware of the order of the validations, as it is important.
        """

        if self.dataset_config and self.dataset_df is not None:
            # Setting the attributes needed for the validations
            self.categorical_feature_names = self.dataset_config.get(
                "categoricalFeatureNames"
            )
            self.class_names = self.dataset_config.get("classNames")
            self.column_names = self.dataset_config.get("columnNames")
            self.label_column_name = self.dataset_config.get("labelColumnName")
            self.feature_names = self.dataset_config.get("featureNames")
            self.text_column_name = self.dataset_config.get("textColumnName")
            self.predictions_column_name = self.dataset_config.get(
                "predictionsColumnName"
            )
            self.prediction_scores_column_name = self.dataset_config.get(
                "predictionScoresColumnName"
            )

            # Dataset-wide validations
            self._validate_dataset_dtypes()
            self._validate_dataset_columns()

            # Label validations
            if self.label_column_name:
                self._validate_labels()

            # Predictions validations
            if self.predictions_column_name:
                self._validate_predictions()

            # Prediction scores validations
            if self.prediction_scores_column_name:
                self._validate_prediction_scores()

            # NLP-specific validations
            if self.text_column_name:
                self._validate_text()

            # Tabular-specific validations
            if self.feature_names:
                self._validate_features()

    def _validate_dataset_dtypes(self):
        """Checks whether the dataset contains unsupported dtypes."""
        supported_dtypes = {"float32", "float64", "int32", "int64", "object"}
        dataset_df_dtypes = {dtype.name for dtype in self.dataset_df.dtypes}
        unsupported_dtypes = dataset_df_dtypes - supported_dtypes
        if unsupported_dtypes:
            self.failed_validations.append(
                "The dataset contains unsupported dtypes. The supported dtypes are "
                "'float32', 'float64', 'int32', 'int64', 'object'. "
                f"The dataset contains the following unsupported dtypes: {unsupported_dtypes}"
                " Please cast the columns in your dataset to conform to these dtypes."
            )

    def _validate_dataset_columns(self):
        """Checks whether all columns in the dataset are specified in `columnNames`."""
        dataset_columns = set(self.dataset_df.columns)
        column_names = set(self.column_names)
        if dataset_columns != column_names:
            self.failed_validations.append(
                "Not all columns in the dataset are specified in `columnNames`. "
                "Please specify all dataset columns in `columnNames`."
            )

    def _validate_labels(self):
        """Validates the data in the label column."""
        if self.label_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The label column `{self.label_column_name}` specified as `labelColumnName` "
                "is not in the dataset."
            )
        else:
            if self.class_names:
                self._validate_all_categories_in_class_names(
                    column_name=self.label_column_name
                )
                self._validate_categories_zero_indexed(
                    column_name=self.label_column_name
                )

    def _validate_all_categories_in_class_names(self, column_name: str):
        """Checks whether there are categories in the dataset's `column_name` which are not
        in the `class_names`."""
        num_classes = len(self.dataset_df[column_name].unique())
        if num_classes > len(self.class_names):
            self.failed_validations.append(
                "There are more classes in the dataset's column"
                f" `{column_name}` than specified in `classNames`. "
                "Please specify all possible labels in the `classNames` list."
            )

    def _validate_categories_zero_indexed(self, column_name: str):
        """Checks whether the categories are zero-indexed in the dataset's `column_name`."""
        unique_labels = set(self.dataset_df[column_name].unique())
        zero_indexed_set = set(range(len(self.class_names)))
        if unique_labels != zero_indexed_set:
            self.failed_validations.append(
                "The classes in the dataset are not zero-indexed. "
                f"Make sure that the classes in the column `{column_name}` "
                "are zero-indexed integers that match the list in `classNames`."
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
                self._validate_all_categories_in_class_names(
                    column_name=self.predictions_column_name
                )
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
                        f"The predictions in the column `{self.prediction_scores_column_name}` "
                        "are not lists. Please make sure that the predictions are "
                        "lists of floats."
                    )
                else:
                    if self._prediction_lists_not_same_length(
                        self.dataset_df, self.prediction_scores_column_name
                    ):
                        self.failed_validations.append(
                            "The prediction lists in the column "
                            f"`{self.prediction_scores_column_name}` "
                            "are not all of the same length. "
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
                                    f"The predictions in `{self.prediction_scores_column_name}`"
                                    f" don't match the classes in `{self.class_names}`. "
                                    "Please make sure that the lists with predictions "
                                    "have the same length as the `classNames` list."
                                )

            except Exception:
                self.failed_validations.append(
                    f"The predictions in the column `{self.prediction_scores_column_name}` "
                    "are not lists. "
                    "Please make sure that the predictions are lists of floats."
                )

    def _validate_text(self):
        """Validates the data in the text column."""
        if self.text_column_name not in self.dataset_df.columns:
            self.failed_validations.append(
                f"The text column `{self.text_column_name}` specified as `textColumnName` "
                "is not in the dataset."
            )
        elif self._exceeds_character_limit(self.dataset_df, self.text_column_name):
            self.failed_validations.append(
                f"The column `{self.text_column_name}` of the dataset contains rows that "
                "exceed the 1000 character limit."
            )

    def _validate_features(self):
        """Validates the data in the features and categorical features columns."""
        if self._columns_not_in_dataset_df(self.dataset_df, self.feature_names):
            self.failed_validations.append(
                "There are features specified in `featureNames` which are "
                "not in the dataset."
            )
        if self.categorical_feature_names:
            if self._columns_not_in_dataset_df(
                self.dataset_df, self.categorical_feature_names
            ):
                self.failed_validations.append(
                    "There are categorical features specified in `categoricalFeatureNames` "
                    "which are not in the dataset."
                )

    @staticmethod
    def _exceeds_character_limit(
        dataset_df: pd.DataFrame, text_column_name: str
    ) -> bool:
        """Checks whether the text column exceeds the character limit."""
        if dataset_df[text_column_name].str.len().max() > 1000:
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

    @staticmethod
    def _columns_not_in_dataset_df(
        dataset_df: pd.DataFrame, columns_list: List[str]
    ) -> bool:
        """Checks whether the columns are in the dataset."""
        if set(columns_list) - set(dataset_df.columns):
            return True
        return False
