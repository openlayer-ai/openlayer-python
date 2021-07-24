import csv
import os
import shutil
import tarfile
import tempfile
import uuid
from enum import Enum
from typing import List, Optional

import pandas as pd
from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .api import Api
from .datasets import Dataset
from .exceptions import UnboxException
from .models import Model, ModelType, create_template_model, create_rasa_model


class DeploymentType(Enum):
    ONPREM = 1
    AWS = 2


DEPLOYMENT = DeploymentType.ONPREM


class UnboxClient(object):
    """ Client class that interacts with the Unbox Platform. """

    def __init__(self, api_key: str):
        self.api = Api(api_key)
        self.subscription_plan = self.api.get_request("users/subscriptionPlan")

        if DEPLOYMENT == DeploymentType.AWS:
            self.upload = self.api.upload_blob
        else:
            self.upload = self.api.transfer_blob

    def add_model(
        self,
        function,
        model,
        model_type: ModelType,
        class_names: List[str],
        name: str,
        description: str = None,
        requirements_txt_file: Optional[str] = None,
        setup_script: Optional[str] = None,
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
            class_names (List[str]):
                List of class names corresponding to outputs of predict function
            name (str):
                Name of model
            description (str):
                Description of model
            requirements_txt_file (Optional[str]):
                Path to a requirements file containing dependencies needed by the predict function
            setup_script (Optional[str]):
                Path to a setup script executing any commands necessary to run before loading the model

        Returns:
            Model:
                Returns uploaded model
        """
        with TempDirectory() as dir:

            if model_type == ModelType.rasa:
                bento_service = create_rasa_model(
                    dir, requirements_txt_file, setup_script
                )
            else:
                bento_service = create_template_model(
                    model_type, dir, requirements_txt_file, setup_script
                )
                if model_type == ModelType.transformers:
                    if "tokenizer" not in kwargs:
                        raise UnboxException(
                            "Must specify tokenizer in kwargs when using a transformers model"
                        )
                    bento_service.pack(
                        "model", {"model": model, "tokenizer": kwargs["tokenizer"]}
                    )
                    kwargs.pop("tokenizer")
                else:
                    bento_service.pack("model", model)

            bento_service.pack("function", function)
            bento_service.pack("kwargs", kwargs)

            with TempDirectory() as temp_dir:
                _write_bento_content_to_dir(bento_service, temp_dir)

                if model_type == ModelType.rasa:
                    destination_path = os.path.join(temp_dir, "TemplateModel/nlu")
                    shutil.copytree(model.model_metadata.model_dir, destination_path)

                with TempDirectory() as tarfile_dir:
                    tarfile_path = f"{tarfile_dir}/model"

                    with tarfile.open(tarfile_path, mode="w:gz") as tar:
                        tar.add(temp_dir, arcname=bento_service.name)

                    print("Connecting to Unbox server")
                    endpoint = "models"
                    payload = dict(
                        name=name,
                        description=description,
                        classNames=class_names,
                        type=model_type.name,
                    )
                    modeldata = self.upload(endpoint, tarfile_path, payload)
        os.remove("template_model.py")
        return Model(modeldata)

    def add_dataset(
        self,
        file_path: str,
        class_names: List[str],
        label_column_name: str,
        text_column_name: str,
        name: str,
        description: str = None,
    ) -> Dataset:
        """Uploads a dataset from a csv.

        Args:
            file_path (str):
                Path to the dataset csv
            class_names (List[str]):
                List of class names indexed by label integer in the dataset
                ex. `[negative, positive]` when `[0, 1]` are labels in the csv
            label_column_name (str):
                Column header in the csv containing the labels
            text_column_name (str):
                Column header in the csv containing the input text
            name (str):
                Name of dataset
            description (str):
                Description of dataset

        Raises:
            UnboxException:
                If the file doesn't exist or the label or text column names
                are not in the dataset

        Returns:
            Dataset:
                Returns uploaded dataset
        """
        if not os.path.isfile(file_path):
            raise UnboxException("File path does not exist.")

        with open(file_path, "rt") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row_count = sum(1 for _ in reader)
        if row_count > self.subscription_plan["datasetSize"]:
            raise UnboxException(
                f"Dataset contains {row_count} rows, which exceeds your plan's"
                f" limit of {self.subscription_plan['datasetSize']}."
            )
        try:
            label_column_index = headers.index(label_column_name)
            text_column_index = headers.index(text_column_name)
        except ValueError:
            raise UnboxException(
                "Label column and/or text column names not in dataset."
            )
        endpoint = "datasets"
        payload = dict(
            name=name,
            description=description,
            classNames=class_names,
            labelColumnName=label_column_name,
            textColumnName=text_column_name,
            labelColumnIndex=label_column_index,
            textColumnIndex=text_column_index,
        )
        return Dataset(self.upload(endpoint, file_path, payload))

    def add_dataframe(
        self,
        df: pd.DataFrame,
        class_names: List[str],
        label_column_name: str,
        text_column_name: str,
        name: str,
        description: str,
    ) -> Dataset:
        """Uploads a dataset from a dataframe.

        Args:
            df (pd.DataFrame):
                Dataframe object
            class_names (List[str]):
                List of class names indexed by label integer in the dataset
                ex. `[negative, positive]` when `[0, 1]` are labels in the csv
            label_column_name (str):
                Column header in the csv containing the labels
            text_column_name (str):
                Column header in the csv containing the input text
            name (str):
                Name of dataset
            description (str):
                Description of dataset

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
                file_path,
                class_names,
                label_column_name,
                text_column_name,
                name,
                description,
            )
