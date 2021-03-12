import bentoml
import getpass
import os
import pandas as pd
import tarfile
import tempfile
import uuid
from typing import List

from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .lib.network import FlaskAPI, FirebaseAPI
from .template import create_template_model


class UnboxClient(object):

    # Public functions

    def __init__(self, email: str = None, password: str = None):
        self.flask_api = FlaskAPI()
        self.firebase_api = FirebaseAPI(email=email, password=password)

    def _test_associate(self, model_id: str, dataset_id: str):
        ''' This is just for testing.'''
        print(self.flask_api._test_associate_model_dataset(
            id_token=self.firebase_api.user['idToken'],
            model_id=model_id,
            dataset_id=dataset_id))

    def add_model(self, function, model, name: str, description: str, model_type: str = 'sklearn'):
        bento_service = create_template_model(model_type)
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

                # Now set the metadata via request to our Flask API
                id_token = self.firebase_api.user['idToken']
                response = self.flask_api.upload_model_metadata(user_id,
                                                                model_id,
                                                                name,
                                                                description,
                                                                id_token)
        return response

    def add_dataset(self, file_path: str, name: str, description: str, label_column_name: str, text_column_name: str):
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
                                                          description,
                                                          label_column_name,
                                                          text_column_name,
                                                          id_token)
        return response.json()

    def add_dataframe(self, df: pd.DataFrame, name: str, description: str, label_column_name: str, text_column_name: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(dataset_file_path, index=False)
            return self.add_dataset(dataset_file_path, name, description, label_column_name, text_column_name)

    def add_slice(self, dataset_id: str, slice_name: str, indices: List[int], slice_id: str = uuid.uuid1()):
        # For now, let's upload straight to Firebase Storage from here
        user_id = self.firebase_api.user['localId']

        # And then set the metadata via request to our Flask API
        id_token = self.firebase_api.user['idToken']
        response = self.flask_api.upload_slice_metadata(user_id,
                                                        dataset_id,
                                                        slice_id,
                                                        slice_name,
                                                        indices,
                                                        id_token)
        return response.json()

    def update_slice_indices(self, indices: List[int], slice_id: str):
        # For now, let's upload straight to Firebase Storage from here
        user_id = self.firebase_api.user['localId']

        # And then set the metadata via request to our Flask API
        id_token = self.firebase_api.user['idToken']
        response = self.flask_api.upload_slice_new_indices(user_id,
                                                        slice_id,
                                                        indices,
                                                        id_token)
        return response.json()
