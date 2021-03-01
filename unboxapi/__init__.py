import os
import getpass
import tarfile
import pyrebase
import bentoml
from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .template import create_template_model


class UnboxClient(object):
    def __init__(self):
        self.authenticate()

    def add(self, function, model):
        bento_service = create_template_model("sklearn", "text")
        bento_service.pack('model', model)
        bento_service.pack('function', function)

        with TempDirectory() as temp_dir:
            _write_bento_content_to_dir(bento_service, temp_dir)
            with TempDirectory() as tarfile_dir:
                file_name = f'{bento_service.name}.tar'
                tarfile_path = f'{tarfile_dir}/{file_name}'
                with tarfile.open(tarfile_path, mode="w:gz") as tar:
                    tar.add(temp_dir, arcname=bento_service.name)
                self.upload(
                    f"users/{self.user['localId']}/models/{file_name}", tarfile_path)

    def upload(self, remote_path, file_path):
        storage = self.firebase.storage()
        storage.child(remote_path).put(file_path, self.user['idToken'])

    def authenticate(self):
        config = {
            "apiKey": "AIzaSyAKlGQOmXTjPQhL1Uvj-Jr-_jUtNWmpOgs",
            "authDomain": "unbox-ai.firebaseapp.com",
            "databaseURL": "https://unbox-ai.firebaseio.com",
            "storageBucket": "unbox-ai.appspot.com"
        }

        self.firebase = pyrebase.initialize_app(config)

        # Get a reference to the auth service
        auth = self.firebase.auth()

        # Log the user in
        email = input("What is your Unbox email?")
        password = getpass.getpass("What is your Unbox password?")
        self.user = auth.sign_in_with_email_and_password(email, password)
