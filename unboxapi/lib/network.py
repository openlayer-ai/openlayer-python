import requests
import os
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from typing import Dict, List


class UnboxAPI:
    def __init__(self, id_token: str = None, email: str = None, password: str = None):
        # self.url = "http://0.0.0.0:8080"
        self.url = "https://dev.tryunbox.ai"
        if id_token:
            self.id_token = id_token
        else:
            response = requests.get(self.url + "/api/tokens", auth=(email, password))
            if response.ok:
                self.id_token = response.json()["accessToken"]
            else:
                print("Failed to retrieve a token for the email / password provided.")

    def upload(self, endpoint: str, data: Dict[str, str], file_path):
        response = requests.get(
            self.url + endpoint, headers={"Authorization": f"Bearer {self.id_token}"}
        )
        if response.ok and "url" in response.json():
            response_data = response.json()
            storage_url = response_data["url"]
            object_id = response_data["id"]
            fields = response_data["fields"]
            file_size = os.stat(file_path).st_size
            with open(file_path, "rb") as f:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as t:
                    wrapped_file = CallbackIOWrapper(t.update, f, "read")
                    files = {"file": (object_id, wrapped_file)}
                    response = requests.post(
                        storage_url,
                        data=fields,
                        files=files,
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
        class_names: List[str],
        label_column_name: str,
        text_column_name: str,
        label_column_index: str,
        text_column_index: str,
        file_path: str,
    ):
        data = {
            "name": name,
            "description": description,
            "classNames": class_names,
            "labelColumnName": label_column_name,
            "textColumnName": text_column_name,
            "labelColumnIndex": label_column_index,
            "textColumnIndex": text_column_index,
        }
        return self.upload("/api/datasets", data, file_path)

    def upload_model(
        self, name: str, description: str, class_names: List[str], file_path: str
    ):
        data = {"name": name, "description": description, "classNames": class_names}
        return self.upload("/api/models", data, file_path)
