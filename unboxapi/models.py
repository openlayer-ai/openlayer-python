import textwrap
from enum import Enum


class ModelType(Enum):
    """Task Type List"""

    fasttext = "FasttextModelArtifact"
    sklearn = "SklearnModelArtifact"
    pytorch = "PytorchModelArtifact"
    tensorflow = "TensorflowSavedModelArtifact"
    transformers = "TransformersModelArtifact"


class Model:
    """Model class."""

    def __init__(self, json):
        self._json = json
        self.id = json["id"]

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Model(id={self.id})"

    def __repr__(self):
        return f"Model({self._json})"

    def to_dict(self):
        """Returns object properties as a dict
        Returns:
            Dict with object properties
        """
        return self._json


def _predict_function(model_type: ModelType):
    if model_type == ModelType.transformers:
        return "results = self.artifacts.function(self.artifacts.model.get('model'), text, tokenizer=self.artifacts.model.get('tokenizer'), **self.artifacts.kwargs)"
    else:
        return "results = self.artifacts.function(self.artifacts.model, text, **self.artifacts.kwargs)"


def create_template_model(model_type: ModelType):
    with open("template_model.py", "w") as python_file:
        file_contents = f"""\
        import json
        import os
        import pandas as pd
        from typing import List

        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type.name} import {model_type.value}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput
        from bentoml.types import JsonSerializable
        from bentoml.utils.tempdir import TempDirectory


        @env(infer_pip_packages=True)
        @artifacts([{model_type.value}('model'), PickleArtifact('function'), PickleArtifact('kwargs')])
        class TemplateModel(BentoService):
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                text = parsed_json['text']
                {_predict_function(model_type)}
                return results
            
            @api(input=JsonInput())
            def predict_from_path(self, parsed_json: JsonSerializable):
                input_path = parsed_json["input_path"]
                output_path = parsed_json["output_path"]
                text = pd.read_csv(input_path)['text'].tolist()
                {_predict_function(model_type)}
                with open(output_path, 'w') as f:
                    json.dump(results.tolist(), f)
                return "Success"
        """
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel

    return TemplateModel()
