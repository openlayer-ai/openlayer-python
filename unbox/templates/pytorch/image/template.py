from typing import BinaryIO, List
from PIL import Image
from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import FileInput


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('model'), PickleArtifact('function')])
class PytorchImageTemplateModel(BentoService):

    @api(input=FileInput(), batch=True)
    def batch(self, file_streams: List[BinaryIO]):
        input_images = []
        for fs in file_streams:
            img = Image.open(fs)
            input_images.append(img)

        return self.artifacts.function(
            self.artifacts.model,
            input_images
        )

    @api(input=FileInput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]):
        input_images = []
        for fs in file_streams:
            img = Image.open(fs)
            input_images.append(img)

        return self.artifacts.function(
            self.artifacts.model,
            input_images
        )