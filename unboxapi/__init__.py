import csv
import os
import pandas as pd
import tarfile
import tempfile
import uuid
from typing import List

from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .lib.network import UnboxAPI
from .template import create_template_model


class UnboxClient(object):

    # Public functions
    def __init__(self, email: str = None, password: str = None):
        self.unbox_api = UnboxAPI(email=email, password=password)

    def add_model(
        self,
        function,
        model,
        class_names: List[str],
        name: str,
        description: str,
        model_type: str = "sklearn",
    ):
        bento_service = create_template_model(model_type)
        bento_service.pack("model", model)
        bento_service.pack("function", function)

        with TempDirectory() as temp_dir:
            _write_bento_content_to_dir(bento_service, temp_dir)

            with TempDirectory() as tarfile_dir:
                tarfile_path = f"{tarfile_dir}/model"

                with tarfile.open(tarfile_path, mode="w:gz") as tar:
                    tar.add(temp_dir, arcname=bento_service.name)

                print("Connecting to Unbox server")
                # Upload the model and metadata to our Flask API
                response = self.unbox_api.upload_model(
                    name,
                    description,
                    class_names,
                    tarfile_path,
                )
        return response

    def add_dataset(
        self,
        file_path: str,
        name: str,
        description: str,
        class_names: List[str],
        label_column_name: str,
        text_column_name: str,
    ):
        with open(file_path, "rt") as f:
            reader = csv.reader(f)
            headers = next(reader)
        try:
            label_column_index = headers.index(label_column_name)
            text_column_index = headers.index(text_column_name)
            # Upload dataset to our Flask API
            response = self.unbox_api.upload_dataset(
                name,
                description,
                class_names,
                label_column_name,
                text_column_name,
                label_column_index,
                text_column_index,
                file_path,
            )
            return response
        except ValueError:
            raise ValueError(f"Label column and/or text column names not in dataset.")

    def add_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        class_names: List[str],
        label_column_name: str,
        text_column_name: str,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(dataset_file_path, index=False)
            return self.add_dataset(
                dataset_file_path,
                name,
                description,
                class_names,
                label_column_name,
                text_column_name,
            )
