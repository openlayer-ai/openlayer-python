import bentoml
import getpass
import os
import pandas as pd
import tarfile
import tempfile
import uuid

from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .lib.network import FlaskAPI, FirebaseAPI
from .template import create_template_model


class UnboxClient(object):

    # Public functions

    def __init__(self, email: str = None, password: str = None):
        self.flask_api = FlaskAPI()
        self.firebase_api = FirebaseAPI(email=email, password=password)

    def add_model(self, function, model):
        bento_service = create_template_model('sklearn', 'text')
        bento_service.pack('model', model)
        bento_service.pack('function', function)

        with TempDirectory() as temp_dir:
            _write_bento_content_to_dir(bento_service, temp_dir)

            with TempDirectory() as tarfile_dir:
                model_id = str(uuid.uuid1())
                tarfile_path = f'{tarfile_dir}/{model_id}'

                with tarfile.open(tarfile_path, mode='w:gz') as tar:
                    tar.add(temp_dir, arcname=bento_service.name)

                user_id = self.firebase_api.user['localId']
                remote_path = f'users/{user_id}/models/{model_id}'
                self.firebase_api.upload(remote_path, tarfile_path)

    def add_dataset(self, file_path: str, name: str):
        # For now, let's upload straight to Firebase Storage from here
        user_id = self.firebase_api.user['localId']
        dataset_id = str(uuid.uuid1())
        remote_path = f'users/{user_id}/datasets/{dataset_id}'
        self.firebase_api.upload(remote_path, file_path)

        # And then set the metadata via request to our Flask API
        id_token = self.firebase_api.user['idToken']
        response = self.flask_api.upload_dataset_metadata(user_id,
                                                          dataset_id,
                                                          name,
                                                          id_token)
        return response.json()

    def add_dataframe(self, df: pd.DataFrame, name: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(dataset_file_path, index=False)
            return self.add_dataset(dataset_file_path, name)
