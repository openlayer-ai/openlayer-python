import os
import bentoml

from .template import create_template_model


class UnboxClient(object):
    def __init__(self, api_key):
        self.api_key = api_key

    def add(self, function, model):
        bento_service = create_template_model("sklearn", "text")
        bento_service.pack('model', model)
        bento_service.pack('function', function)

        self.upload(bento_service, "Tg0T6QuoZ5baPz5Hbw3fvz8tvGw1")

    @staticmethod
    def upload(bento_service, user_id):
        # TODO: Authenticate with the user provided API key
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/rishabramanathan/projects/unbox/serviceAccount.json'

        bentoml.save_to_dir(bento_service,
                            f'gs://unbox-ai.appspot.com/{user_id}/models')
