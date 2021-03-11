import textwrap


modelTypes = {
    'sklearn': 'SklearnModelArtifact',
    'pytorch': 'PytorchModelArtifact',
    'tensorflow': 'TensorflowSavedModelArtifact',
    'transformers': 'TransformersModelArtifact'
}


# TODO: in dire need of cleanup
def create_template_model(model_type: str):
    with open('template_model.py', 'w') as python_file:
        file_contents = f'''\
        from typing import List
        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type} import {modelTypes[model_type]}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput, StringInput
        from bentoml.types import JsonSerializable, InferenceTask

        @env(infer_pip_packages=True)
        @artifacts([{modelTypes[model_type]}('model'), PickleArtifact('function')])
        class TemplateModel(BentoService):

            @api(input=StringInput(), batch=True)
            def predict_str(self, text):
                return self.artifacts.function(
                    self.artifacts.model,
                    text
                )

            @api(input=JsonInput(), batch=True)
            def predict(self, parsed_json_list: List[JsonSerializable]):
                text = []
                for json in parsed_json_list:
                    if 'text' in json:
                        text.append(json['text'])
                    else:
                        task.discard(http_status=400,
                                    err_msg='input json must contain `text` field')

                prediction_probs, class_names, _ = self.artifacts.function(
                    self.artifacts.model,
                    text
                )

                return [
                    {{class_names[i]: prob for i,
                        prob in enumerate(probs)}}
                    for probs in prediction_probs
                ]
        '''
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel
    return TemplateModel()
