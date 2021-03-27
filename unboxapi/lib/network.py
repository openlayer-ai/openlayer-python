import requests
from typing import Dict


class UnboxAPI:
    def __init__(self, id_token: str = None, email: str = None, password: str = None):
        self.url = "http://0.0.0.0:8080"
        if id_token:
            self.id_token = id_token
        else:
            response = requests.get(self.url + "/api/tokens", auth=(email, password))
            if response.ok:
                self.id_token = response.json()["token"]
            else:
                print("Failed to retrieve a token for the email / password provided.")

    def post(self, endpoint: str, data: Dict[str, str], files):
        return requests.post(
            self.url + endpoint,
            data=data,
            files=files,
            headers={"Authorization": f"Bearer {self.id_token}"},
        )

    def upload_dataset(
        self,
        name: str,
        description: str,
        label_column_name: str,
        text_column_name: str,
        file_path: str,
    ):
        data = {
            "name": name,
            "description": description,
            "labelColumnName": label_column_name,
            "textColumnName": text_column_name,
        }
        files = {"file": open(file_path, "rb")}
        return self.post("/api/datasets", data, files)

    def upload_model(self, name: str, description: str, file_path: str):
        data = {"name": name, "description": description}
        files = {"file": open(file_path, "rb")}
        return self.post("/api/models", data, files)
