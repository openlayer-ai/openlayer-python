import pyrebase
import requests
import getpass


class FlaskAPI:
    def __init__(self):
        self.url = "http://0.0.0.0:8080"

    def post(self, id_token: str, endpoint: str = "/", data: any = None):
        return requests.post(
            self.url + endpoint,
            json=data,
            headers={"Authorization": id_token},
        )

    def post_file(
        self, id_token: str, endpoint: str = "/", data: any = None, files: any = None
    ):
        return requests.post(
            self.url + endpoint,
            data=data,
            files=files,
            headers={"Authorization": id_token},
        )

    def upload_dataset(
        self,
        user_id: str,
        dataset_id: str,
        name: str,
        description: str,
        label_column_name: str,
        text_column_name: str,
        file_path: str,
        id_token: str,
    ):
        data = {
            "datasetId": dataset_id,
            "name": name,
            "description": description,
            "labelColumnName": label_column_name,
            "textColumnName": text_column_name,
            "userId": user_id,
        }
        files = {"file": open(file_path, "rb")}
        return self.post_file(id_token, "/api/dataset/upload", data, files)

    def upload_model(
        self,
        user_id: str,
        model_id: str,
        name: str,
        description: str,
        file_path: str,
        id_token: str,
    ):
        data = {
            "modelId": model_id,
            "name": name,
            "description": description,
            "userId": user_id,
        }
        files = {"file": open(file_path, "rb")}
        return self.post_file(id_token, "/api/model/upload", data, files)


class FirebaseAPI:
    def __init__(self, email: str = None, password: str = None):
        if not email or not password:
            email = input("What is your Unbox email?")
            password = getpass.getpass("What is your Unbox password?")

        config = {
            "apiKey": "AIzaSyAKlGQOmXTjPQhL1Uvj-Jr-_jUtNWmpOgs",
            "authDomain": "unbox-ai.firebaseapp.com",
            "databaseURL": "https://unbox-ai.firebaseio.com",
            "storageBucket": "unbox-ai.appspot.com",
        }

        # Initialize Pyrebase instance
        self.firebase = pyrebase.initialize_app(config)
        # Get a reference to the auth service
        auth = self.firebase.auth()
        # Login
        self.user = auth.sign_in_with_email_and_password(email, password)
