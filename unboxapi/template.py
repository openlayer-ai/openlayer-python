import textwrap
from unboxapi.model_strings import only_model_string, transformers_string, tokenizer_string

modelTypes = {
    'sklearn': 'SklearnModelArtifact',
    'pytorch': 'PytorchModelArtifact',
    'tensorflow': 'TensorflowSavedModelArtifact',
    'transformers': 'TransformersModelArtifact',
    'fasttext': 'FasttextModelArtifact'
}


# TODO: in dire need of cleanup
def create_template_model(model_type: str, local_imports: str, has_tokenizer: bool):

    artifact_name = modelTypes[model_type]

    if model_type == 'transformers':
        bento_specs = transformers_string.format(artifact_name)
    elif has_tokenizer:
        bento_specs = tokenizer_string.format(artifact_name)
    else:
        bento_specs = only_model_string.format(artifact_name)

    with open('template_model.py', 'w') as python_file:
        file_contents = f'''\
        from typing import List
        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type} import {modelTypes[model_type]}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput, StringInput
        from bentoml.types import JsonSerializable, InferenceTask

        {local_imports}

        @env(infer_pip_packages=True)
        {bento_specs}
        '''
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel
    return TemplateModel()
