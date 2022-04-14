import csv
import os
import shutil
import tarfile
import tempfile
import uuid
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .api import Api
from .datasets import Dataset
from .exceptions import UnboxException, UnboxInvalidRequest
from .models import Model, ModelType, create_template_model
from .tasks import Task, TaskType


class DeploymentType(Enum):
    ONPREM = 1
    AWS = 2
    GCP = 3


DEPLOYMENT = DeploymentType.ONPREM
# DEPLOYMENT = DeploymentType.AWS
# DEPLOYMENT = DeploymentType.GCP


class UnboxClient(object):
    """ Client class that interacts with the Unbox Platform. """

    def __init__(self, api_key: str):
        self.api = Api(api_key)
        self.subscription_plan = self.api.get_request("users/subscriptionPlan")

        if DEPLOYMENT == DeploymentType.AWS:
            self.upload = self.api.upload_blob_s3
        elif DEPLOYMENT == DeploymentType.GCP:
            self.upload = self.api.upload_blob_gcs
        else:
            self.upload = self.api.transfer_blob

    def add_model(
        self,
        function,
        model,
        model_type: ModelType,
        task_type: TaskType,
        class_names: List[str],
        name: str,
        description: str = None,
        requirements_txt_file: Optional[str] = None,
        setup_script: Optional[str] = None,
        custom_model_code: Optional[str] = None,
        dependent_dir: Optional[str] = None,
        feature_names: List[str] = [],
        train_sample_df: pd.DataFrame = None,
        train_sample_label_column_name: str = None,
        categorical_features_map: Dict[str, List[str]] = {},
        **kwargs,
    ) -> Model:
        """Uploads a model.

        Args:
            function:
                Prediction function object in expected format
            model:
                Model object
            model_type (ModelType):
                Model framework type of model
                ex. `ModelType.sklearn`
            task_type (TaskType):
                Type of ML task
                ex. `TaskType.TextClassification`
            class_names (List[str]):
                List of class names corresponding to outputs of predict function
            name (str):
                Name of model
            description (str):
                Description of model
            requirements_txt_file (Optional[str]):
                Path to a requirements file containing dependencies needed by the predict function
            setup_script (Optional[str]):
                Path to a bash script executing any commands necessary to run before loading the model
            custom_model_code (Optional[str]):
                Custom code needed to initialize the model. Model object must be none in this case.
            dependent_dir (Optional[str]):
                Path to a dir of file dependencies needed to load the model
            feature_names (List[str]):
                List of input feature names. Required for tabular classification.
            train_sample_df (pd.DataFrame):
                A random sample of >= 100 rows from your training dataset. Required for tabular classification.
                This is used to support explainability features.
            train_sample_label_column_name (str):
                Column header in train_sample_df containing the labels
            categorical_features_map (Dict[str, List[str]]):
                A dict containing a list of category names for each feature that is categorical.
                ex. {'Weather': ['Hot', 'Cold']}

        Returns:
            Model:
                Returns uploaded model
        """
        if custom_model_code:
            assert (
                model_type is ModelType.custom
            ), "model_type must be ModelType.custom if specifying custom_model_code"
        if task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
            required_fields = [
                (feature_names, "feature_names"),
                (train_sample_df, "train_sample_df"),
                (train_sample_label_column_name, "train_sample_label_column_name"),
            ]
            for value, field in required_fields:
                if value is None:
                    raise UnboxException(
                        f"Must specify {field} for TabularClassification"
                    )
            if len(train_sample_df.index) < 100:
                raise UnboxException("train_sample_df must have at least 100 rows")
            train_sample_df = train_sample_df.sample(
                min(3000, len(train_sample_df.index))
            )
            try:
                headers = train_sample_df.columns.tolist()
                [
                    headers.index(name)
                    for name in feature_names + [train_sample_label_column_name]
                ]
            except ValueError:
                raise UnboxException(
                    "Feature / label column names not in train_sample_df"
                )
            self._validate_categorical_features(
                train_sample_df, categorical_features_map
            )

        with TempDirectory() as dir:
            bento_service = create_template_model(
                model_type,
                task_type,
                dir,
                requirements_txt_file,
                setup_script,
                custom_model_code,
            )
            if model_type is ModelType.transformers:
                if "tokenizer" not in kwargs:
                    raise UnboxException(
                        "Must specify tokenizer in kwargs when using a transformers model"
                    )
                bento_service.pack(
                    "model", {"model": model, "tokenizer": kwargs["tokenizer"]}
                )
                kwargs.pop("tokenizer")
            elif model_type not in [ModelType.custom, ModelType.rasa]:
                bento_service.pack("model", model)

            bento_service.pack("function", function)
            bento_service.pack("kwargs", kwargs)

            with TempDirectory() as temp_dir:
                print("Bundling model and artifacts...")
                _write_bento_content_to_dir(bento_service, temp_dir)

                if model_type is ModelType.rasa:
                    dependent_dir = model.model_metadata.model_dir

                # Add dependent directory to bundle
                if dependent_dir is not None:
                    dependent_dir = os.path.abspath(dependent_dir)
                    if dependent_dir == os.getcwd():
                        raise UnboxException("dependent_dir can't be working directory")
                    shutil.copytree(
                        dependent_dir,
                        os.path.join(
                            temp_dir,
                            f"TemplateModel/{os.path.basename(dependent_dir)}",
                        ),
                    )

                # Add sample of training data to bundle
                if task_type in [
                    TaskType.TabularClassification,
                    TaskType.TabularRegression,
                ]:
                    train_sample_df.to_csv(
                        os.path.join(temp_dir, f"TemplateModel/train_sample.csv"),
                        index=False,
                    )

                # Tar the model bundle with its artifacts and upload
                with TempDirectory() as tarfile_dir:
                    tarfile_path = f"{tarfile_dir}/model"

                    with tarfile.open(tarfile_path, mode="w:gz") as tar:
                        tar.add(temp_dir, arcname=bento_service.name)

                    endpoint = "models"
                    payload = dict(
                        name=name,
                        description=description,
                        classNames=class_names,
                        taskType=task_type.value,
                        type=model_type.name,
                        kwargs=list(kwargs.keys()),
                        featureNames=feature_names,
                        categoricalFeaturesMap=categorical_features_map,
                        trainSampleLabelColumnName=train_sample_label_column_name,
                    )
                    print("Uploading model to Unbox...")
                    modeldata = self.upload(endpoint, tarfile_path, payload)
        os.remove("template_model.py")
        return Model(modeldata)

    def add_dataset(
        self,
        file_path: str,
        task_type: TaskType,
        class_names: List[str],
        name: str,
        label_column_name: str,
        text_column_name: Optional[str] = None,
        description: Optional[str] = None,
        tag_column_name: Optional[str] = None,
        language: str = "en",
        sep: str = ",",
        feature_names: List[str] = [],
        categorical_features_map: Dict[str, List[str]] = {},
    ) -> Dataset:
        """Uploads a dataset from a csv.

        Args:
            file_path (str):
                Path to the dataset csv
            task_type (TaskType):
                Type of ML task
                ex. `TaskType.TextClassification`
            class_names (List[str]):
                List of class names indexed by label integer in the dataset
                ex. `[negative, positive]` when `[0, 1]` are labels in the csv
            name (str):
                Name of dataset
            label_column_name (str):
                Column header in the csv containing the labels
            text_column_name (Optional[str]):
                For TextClassification - Column header in the csv containing the input text
            description (Optional[str]):
                Description of dataset
            tag_column_name (Optional[str]):
                Column header in the csv containing any pre-computed tags
            language (str):
                The language of the dataset in ISO 639-1 (alpha-2 code) format
            sep (str):
                Delimiter to use
            feature_names (List[str]):
                List of input feature names. Required for tabular classification.
            categorical_features_map (Dict[str, List[str]]):
                A dict containing a list of category names for each feature that is categorical.
                ex. {'Weather': ['Hot', 'Cold']}

        Raises:
            UnboxException:
                If the file doesn't exist or the label / text / tag column names
                are not in the dataset

        Returns:
            Dataset:
                Returns uploaded dataset
        """
        file_path = os.path.expanduser(file_path)
        if not os.path.isfile(file_path):
            raise UnboxException("File path does not exist.")
        if task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
            if feature_names is None:
                raise UnboxException(
                    "Must specify feature_names for TabularClassification"
                )
            self._validate_categorical_features(
                pd.read_csv(file_path, sep=sep), categorical_features_map
            )
        else:
            feature_names = []

        with open(file_path, "rt") as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
            row_count = sum(1 for _ in reader)
        if row_count > self.subscription_plan["datasetSize"]:
            raise UnboxException(
                f"Dataset contains {row_count} rows, which exceeds your plan's"
                f" limit of {self.subscription_plan['datasetSize']}."
            )
        try:
            headers.index(label_column_name)
            if text_column_name:
                feature_names.append(text_column_name)
            if tag_column_name:
                headers.index(tag_column_name)
            for feature_name in feature_names:
                headers.index(feature_name)
        except ValueError:
            raise UnboxException(
                "Label / text / feature / tag column names not in dataset."
            )
        endpoint = "datasets"
        payload = dict(
            name=name,
            description=description,
            taskType=task_type.value,
            classNames=class_names,
            labelColumnName=label_column_name,
            tagColumnName=tag_column_name,
            language=language,
            sep=sep,
            featureNames=feature_names,
            categoricalFeaturesMap=categorical_features_map,
        )
        return Dataset(self.upload(endpoint, file_path, payload))

    def add_dataframe(
        self,
        df: pd.DataFrame,
        task_type: TaskType,
        class_names: List[str],
        name: str,
        label_column_name: str,
        text_column_name: Optional[str] = None,
        description: Optional[str] = None,
        tag_column_name: Optional[str] = None,
        language: str = "en",
        feature_names: List[str] = [],
        categorical_features_map: Dict[str, List[str]] = {},
    ) -> Dataset:
        """Uploads a dataset from a dataframe.

        Args:
            df (pd.DataFrame):
                Dataframe object
            task_type (TaskType):
                Type of ML task
                ex. `TaskType.TextClassification`
            class_names (List[str]):
                List of class names indexed by label integer in the dataset
                ex. `[negative, positive]` when `[0, 1]` are labels in the csv
            name (str):
                Name of dataset
            label_column_name (str):
                Column header in the dataframe containing the labels
            text_column_name (Optional[str]):
                Column header in the datafrmae containing the input text
            description (Optional[str]):
                Description of dataset
            tag_column_name (Optional[str]):
                Column header in the dataframe containing any pre-computed tags
            language (str):
                The language of the dataset in ISO 639-1 (alpha-2 code) format
            feature_names (List[str]):
                List of input feature names. Required for tabular classification.
            categorical_features_map (Dict[str, List[str]]):
                A dict containing a list of category names for each feature that is categorical.
                ex. {'Weather': ['Hot', 'Cold']}

        Raises:
            UnboxException:
                If the file doesn't exist or the label or text column names
                are not in the dataset

        Returns:
            Dataset:
                Returns uploaded dataset
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(file_path, index=False)
            return self.add_dataset(
                file_path=file_path,
                task_type=task_type,
                class_names=class_names,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                name=name,
                description=description,
                tag_column_name=tag_column_name,
                language=language,
                feature_names=feature_names,
                categorical_features_map=categorical_features_map,
            )

    @staticmethod
    def _validate_categorical_features(
        df: pd.DataFrame, categorical_features_map: Dict[str, List[str]]
    ):
        for feature, options in categorical_features_map.items():
            if len(df[feature].unique()) > len(options):
                raise UnboxInvalidRequest(
                    f"Feature '{feature}' contains more options in the df than provided "
                    "for it in `categorical_features_map`"
                )
