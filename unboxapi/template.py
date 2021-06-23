import textwrap


modelTypes = {
    "sklearn": "SklearnModelArtifact",
    "pytorch": "PytorchModelArtifact",
    "tensorflow": "TensorflowSavedModelArtifact",
    "transformers": "TransformersModelArtifact",
}


def create_template_model(model_type: str):
    with open("template_model.py", "w") as python_file:
        file_contents = f"""\
        import json
        import os
        import pandas as pd
        from typing import List

        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type} import {modelTypes[model_type]}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput
        from bentoml.types import JsonSerializable
        from bentoml.utils.tempdir import TempDirectory


        @env(infer_pip_packages=True)
        @artifacts([{modelTypes[model_type]}('model'), PickleArtifact('function')])
        class TemplateModel(BentoService):
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                return self.artifacts.function(
                    self.artifacts.model,
                    parsed_json['text']
                )
            
            @api(input=JsonInput())
            def predict_from_path(self, parsed_json: JsonSerializable):
                input_path = parsed_json["input_path"]
                output_path = parsed_json["output_path"]
                df = pd.read_csv(input_path)
                results = self.artifacts.function(
                    self.artifacts.model,
                    df['text'].tolist()
                )
                with open(output_path, 'w') as f:
                    json.dump(results.tolist(), f)
                return "Success"
        """
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel

    return TemplateModel()
