import requests
import os
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from typing import Dict


class UnboxAPI:
    def __init__(self, id_token: str = None, email: str = None, password: str = None):
        self.url = "http://0.0.0.0:8080"
        # self.url = "https://unbox-flask-server-qpvun7qfdq-uw.a.run.app"
        if id_token:
            self.id_token = id_token
        else:
            response = requests.get(self.url + "/api/tokens", auth=(email, password))
            if response.ok:
                self.id_token = response.json()["token"]
            else:
                print("Failed to retrieve a token for the email / password provided.")

    def upload(self, endpoint: str, data: Dict[str, str], file_path):
        response = requests.get(
            self.url + endpoint, headers={"Authorization": f"Bearer {self.id_token}"}
        )
        if response.ok and "url" in response.json():
            storage_url = response.json()["url"]
            object_id = response.json()["id"]
            file_size = os.stat(file_path).st_size
            with open(file_path, "rb") as f:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as t:
                    wrapped_file = CallbackIOWrapper(t.update, f, "read")
                    response = requests.put(
                        storage_url,
                        data=wrapped_file,
                        headers={"Content-Type": "application/x-gzip"},
                    )
            if response.ok:
                return requests.post(
                    f"{self.url}{endpoint}/{object_id}",
                    json=data,
                    headers={"Authorization": f"Bearer {self.id_token}"},
                )
            else:
                print("Failed to upload object.")
        else:
            print("Failed to upload object.")

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
        return self.upload("/api/datasets", data, file_path)

    def upload_model(self, name: str, description: str, file_path: str):
        data = {"name": name, "description": description}
        return self.upload("/api/models", data, file_path)
