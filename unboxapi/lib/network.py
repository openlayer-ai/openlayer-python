import requests


class FlaskAPIRequest:

    def __init__(self):
        self.url = "http://localhost:5000"

    def post(self, file, data, endpoint="/"):
        return requests.post(
            self.url + endpoint,
            files={"file": file},
            data=data,
        )

    def upload_dataset(self, filename):
        file = (filename, open(filename, 'rb'))
        return self.post(file, {}, endpoint="/upload_dataset")
