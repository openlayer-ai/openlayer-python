import os
import pandas as pd
import tarfile
import tempfile
import uuid
from typing import List
from modAL.models import ActiveLearner

from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .lib.network import UnboxAPI
from .template import create_template_model


class EstimatorHelper:

    def __init__(self, callback, model, tokenizer=None, vocab=None):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.callback = callback

    def predict_proba(self, text_list):

        if self.tokenizer:
            if not self.vocab:
                return self.callback(self.model, self.tokenizer, text_list)[0]
            else:
                return self.callback(self.model, self.tokenizer, self.vocab, text_list)[0]
        else:
            return self.callback(self.model, text_list)[0]


def active_learning_function(text_list, n_instances, callback, model, tokenizer=None, vocab=None):
    est = EstimatorHelper(callback, model, tokenizer, vocab)

    learner = ActiveLearner(
        estimator=est
    )

    return learner.query(text_list, n_instances=n_instances)


class UnboxClient(object):

    def __init__(self, email: str = None, password: str = None):
        if email and password:
            self.unbox_api = UnboxAPI(email=email, password=password)
        else:
            print("User is not logged in.")

    def add_model(
            self, function, model, tokenizer=None, vocab=None, name: str = "TemplateModel", description: str = "", model_type: str = "sklearn",
            local_imports: List[str] = []
    ):
        local_imports = "\n".join([" ".join(["import", s.strip()]) for s in local_imports])
        bento_service = create_template_model(model_type, name, local_imports, bool(tokenizer))

        if model_type == "transformers":
            bento_service.pack("model", {"model": model, "tokenizer": tokenizer})
        else:
            bento_service.pack("model", model)
        bento_service.pack("function", function)
        if tokenizer and model_type != "transformers":
            bento_service.pack("tokenizer", tokenizer)
            bento_service.pack("vocab", vocab)
        bento_service.pack("active_learning_function", active_learning_function)

        with TempDirectory() as temp_dir:
            _write_bento_content_to_dir(bento_service, temp_dir)
            print("Packaged bento content")

            with TempDirectory() as tarfile_dir:
                tarfile_path = f"{tarfile_dir}/model"

                with tarfile.open(tarfile_path, mode="w:gz") as tar:
                    tar.add(temp_dir, arcname=bento_service.name)

                print("Connecting to Unbox server")
                # Upload the model and metadata to our Flask API
                response = self.unbox_api.upload_model(
                    name,
                    description,
                    tarfile_path,
                )
        return response

    @staticmethod
    def pack_model(
            function, model, tokenizer=None, vocab=None, model_name: str = "TemplateModel", model_type: str = "sklearn",
            local_imports: List[str] = []
    ):
        local_imports = "\n".join([" ".join(["import", s.strip()]) for s in local_imports])
        bento_service = create_template_model(model_type, model_name, local_imports, bool(tokenizer))

        if model_type == "transformers":
            bento_service.pack("model", {"model": model, "tokenizer": tokenizer})
        else:
            bento_service.pack("model", model)
        bento_service.pack("function", function)
        if tokenizer and model_type != "transformers":
            bento_service.pack("tokenizer", tokenizer)
            bento_service.pack("vocab", vocab)
        bento_service.pack("active_learning_function", active_learning_function)

        saved_path = bento_service.save()

        return saved_path

    def add_dataset(
            self,
            file_path: str,
            name: str,
            description: str,
            label_column_name: str,
            text_column_name: str,
    ):
        # Upload dataset to our Flask API
        response = self.unbox_api.upload_dataset(
            name,
            description,
            label_column_name,
            text_column_name,
            file_path,
        )
        return response

    def add_dataframe(
            self,
            df: pd.DataFrame,
            name: str,
            description: str,
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
                label_column_name,
                text_column_name,
            )
