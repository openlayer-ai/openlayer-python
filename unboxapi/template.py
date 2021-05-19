import textwrap
from unboxapi.model_strings import only_model_string, transformers_string, tokenizer_and_vocab_string

modelTypes = {
    'sklearn': 'SklearnModelArtifact',
    'pytorch': 'PytorchModelArtifact',
    'tensorflow': 'TensorflowSavedModelArtifact',
    'transformers': 'TransformersModelArtifact',
    'fasttext': 'FasttextModelArtifact'
}


def get_bento_specs(model_type, artifact_name, class_name, has_tokenizer):

    if model_type == 'transformers':
        bento_specs = transformers_string.format(artifact_name, class_name)
    elif has_tokenizer:
        bento_specs = tokenizer_and_vocab_string.format(artifact_name, class_name)
    else:
        bento_specs = only_model_string.format(artifact_name, class_name)

    return bento_specs


# TODO: in dire need of cleanup
def create_template_model(model_type: str, class_name: str, local_imports: str, has_tokenizer: bool):

    artifact_name = modelTypes[model_type]
    bento_specs = get_bento_specs(model_type, artifact_name, class_name, has_tokenizer)

    with open('template_model.py', 'w') as python_file:
        file_contents = f'''\
        from typing import List
        from bentoml import env, artifacts, api, BentoService
        from bentoml.frameworks.{model_type} import {modelTypes[model_type]}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput, StringInput
        from bentoml.types import JsonSerializable

        {local_imports}

        @env(infer_pip_packages=True)
        {bento_specs}
        '''
        python_file.write(textwrap.dedent(file_contents))

    mod = __import__('template_model', fromlist=[class_name])
    template_class = getattr(mod, class_name)
    return template_class()
