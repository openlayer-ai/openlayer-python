import pyrebase
import requests
import getpass


class FlaskAPI:

    def __init__(self):
        self.url = 'http://localhost:5000'

    def post(self, endpoint: str = '/', data: any = None, file: any = None):
        return requests.post(self.url + endpoint,
                             json=data,
                             files=file and {'file': file})

    def upload_dataset_metadata(self,
                                user_id: str,
                                dataset_id: str,
                                name: str,
                                description: str,
                                label_column_name: str,
                                text_column_name: str,
                                id_token: str):
        data = {
            'datasetId': dataset_id,
            'idToken': id_token,
            'name': name,
            'description': description,
            'labelColumnName': label_column_name,
            'textColumnName': text_column_name,
            'userId': user_id
        }
        return self.post(endpoint='/api/dataset/upload_metadata', data=data)

    def upload_model_metadata(self,
                              user_id: str,
                              model_id: str,
                              name: str,
                              description: str,
                              id_token: str):
        data = {
            'modelId': model_id,
            'idToken': id_token,
            'name': name,
            'description': description,
            'userId': user_id
        }
        return self.post(endpoint='/api/model/upload_metadata', data=data)

    # def upload_dataset(self, file_path: str, name: str, id_token: str):
    #     data = {'name': name, 'idToken': id_token}
    #     file = open(file_path, 'rb')
    #     return self.post(endpoint='/api/dataset/upload_dataset', data=data, file=file)

    def _test_associate_model_dataset(self,
                                      id_token: str,
                                      model_id: str,
                                      dataset_id: str,
                                      user_id: str):
        data = {
            'idToken': id_token,
            'userId': user_id,
            'modelId': model_id,
            'datasetId': dataset_id
        }
        return self.post(endpoint='/api/run/create_run', data=data)


class FirebaseAPI:

    def __init__(self, email: str = None, password: str = None):
        if not email or not password:
            email = input('What is your Unbox email?')
            password = getpass.getpass('What is your Unbox password?')

        config = {
            'apiKey': 'AIzaSyAKlGQOmXTjPQhL1Uvj-Jr-_jUtNWmpOgs',
            'authDomain': 'unbox-ai.firebaseapp.com',
            'databaseURL': 'https://unbox-ai.firebaseio.com',
            'storageBucket': 'unbox-ai.appspot.com'
        }

        # Initialize Pyrebase instance
        self.firebase = pyrebase.initialize_app(config)

        # Get a reference to the auth service
        auth = self.firebase.auth()

        # Login
        self.user = auth.sign_in_with_email_and_password(email, password)

    def upload(self, remote_path: str, file_path: str):
        storage = self.firebase.storage()
        id_token = self.user['idToken']
        storage.child(remote_path).put(file_path, id_token)
