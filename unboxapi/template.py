import textwrap


modelTypes = {
    "sklearn": "SklearnModelArtifact",
    "pytorch": "PytorchModelArtifact",
    "tensorflow": "TensorflowSavedModelArtifact",
    "transformers": "TransformersModelArtifact",
}


# TODO: in dire need of cleanup
def create_template_model(model_type: str):
    with open("template_model.py", "w") as python_file:
        file_contents = f"""\
        from typing import List
        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type} import {modelTypes[model_type]}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput, StringInput
        from bentoml.types import JsonSerializable

        @env(infer_pip_packages=True)
        @artifacts([{modelTypes[model_type]}('model'), PickleArtifact('function')])
        class TemplateModel(BentoService):
            @api(input=JsonInput(), batch=False)
            def predict(self, parsed_json: JsonSerializable):
                text = []
                if 'text' in parsed_json:
                    text = parsed_json['text']

                return self.artifacts.function(
                    self.artifacts.model,
                    text
                )
        """
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel

    return TemplateModel()
