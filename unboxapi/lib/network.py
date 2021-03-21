import requests
from typing import Dict


class UnboxAPI:
    def __init__(self, id_token: str = None, email: str = None, password: str = None):
        self.url = "http://0.0.0.0:8080"
        if id_token:
            self.id_token = id_token
        else:
            self.id_token = requests.get(
                self.url + "/tokens", auth=(email, password)
            ).json()["token"]

    def post(self, endpoint: str, data: Dict[str, str], files):
        return requests.post(
            self.url + endpoint,
            data=data,
            files=files,
            headers={"Authorization": f"Token {self.id_token}"},
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
        return self.post("/api/datasets", data, files)

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
        return self.post("/api/models", data, files)
