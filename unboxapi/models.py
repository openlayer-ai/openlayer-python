import os
import shutil
import textwrap
from enum import Enum
from typing import Optional

import bentoml

from .tasks import TaskType


class ModelType(Enum):
    """A selection of machine learning modeling frameworks supported by Unbox.

    .. note::
        Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/main/examples>`_
        show you how to use each one of these model types with Unbox.
    """

    #: For models built with `fastText <https://fasttext.cc/>`_.
    fasttext = "FasttextModelArtifact"
    #: For models built with `scikit-learn <https://scikit-learn.org/>`_.
    sklearn = "SklearnModelArtifact"
    #: For models built with `PyTorch <https://pytorch.org/>`_.
    pytorch = "PytorchModelArtifact"
    #: For models built with `TensorFlow <https://www.tensorflow.org/>`_.
    tensorflow = "TensorflowSavedModelArtifact"
    #: For models built with `XGBoost <https://xgboost.readthedocs.io>`_.
    xgboost = "XgboostModelArtifact"
    #: For models built with `Hugging Face transformers <https://huggingface.co/docs/transformers/index>`_.
    transformers = "TransformersModelArtifact"
    #: For models built with `Keras <https://keras.io/>`_.
    keras = "KerasModelArtifact"
    #: For models built with `rasa <https://rasa.com/>`_.
    rasa = "Rasa"
    #: For custom built models.
    custom = "Custom"


class Model:
    """An object containing information about a model on the Unbox platform."""

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
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json


def _predict_function(model_type: ModelType) -> str:
    if model_type is ModelType.transformers:
        return "results = self.artifacts.function(self.artifacts.model.get('model'), input, tokenizer=self.artifacts.model.get('tokenizer'), **self.artifacts.kwargs)"
    elif model_type in [ModelType.rasa, ModelType.custom]:
        return (
            "results = self.artifacts.function(model, input, **self.artifacts.kwargs)"
        )
    else:
        return "results = self.artifacts.function(self.artifacts.model, input, **self.artifacts.kwargs)"


def _extract_input_from_json(task_type: TaskType, from_csv_path: bool = False) -> str:
    if task_type is TaskType.TextClassification:
        if from_csv_path:
            return "input = pd.read_csv(input_path).iloc[:, 0].tolist()"
        return "input = parsed_json['input']"
    elif task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
        if from_csv_path:
            return "input = pd.read_csv(input_path).to_numpy()"
        return "input = np.array(parsed_json['input'], dtype='O')"


def _model(model_type: ModelType) -> str:
    if model_type is ModelType.custom or model_type is ModelType.rasa:
        return ""
    else:
        return f"from bentoml.frameworks.{model_type.name} import {model_type.value}"


def _artifacts(model_type: ModelType) -> str:
    if model_type is ModelType.custom or model_type is ModelType.rasa:
        return "@artifacts([PickleArtifact('function'), PickleArtifact('kwargs')])"
    else:
        return f"@artifacts([{model_type.value}('model'), PickleArtifact('function'), PickleArtifact('kwargs')])"


def _format_custom_code(custom_model_code: Optional[str]) -> str:
    if custom_model_code is None:
        return ""
    return textwrap.indent(
        textwrap.dedent("\n" + custom_model_code), prefix="            "
    )


def _env_dependencies(
    tmp_dir: str, requirements_txt_file: Optional[str], setup_script: Optional[str]
):
    unbox_req_file = f"{tmp_dir}/requirements.txt"
    env_wrapper_str = ""
    if not requirements_txt_file:
        env_wrapper_str += "@env(infer_pip_packages=True"
    else:
        shutil.copy(requirements_txt_file, unbox_req_file)
        # Add required dependencies
        deps = [f"bentoml=={bentoml.__version__}", "pandas"]
        with open(unbox_req_file, "a") as f:
            f.write("\n")
            [f.write(f"{dep}\n") for dep in deps]
        env_wrapper_str += f"@env(requirements_txt_file='{unbox_req_file}'"

    # Add a user defined setup script to execute on startup
    if setup_script:
        env_wrapper_str += f", setup_sh='{setup_script}')"
    else:
        env_wrapper_str += ")"
    return env_wrapper_str


def create_template_model(
    model_type: ModelType,
    task_type: TaskType,
    tmp_dir: str,
    requirements_txt_file: Optional[str],
    setup_script: Optional[str],
    custom_model_code: Optional[str],
):
    if model_type is ModelType.rasa:
        custom_model_code = """
        from rasa.nlu.model import Interpreter
        model = Interpreter.load("nlu")
        """
    if custom_model_code:
        # Set a flag to prevent wasted memory when importing the script
        os.environ["UNBOX_DO_NOT_LOAD_MODEL"] = "True"
        assert (
            "model = " in custom_model_code
        ), "custom_model_code must intialize a `model` var"
    with open(f"template_model.py", "w") as python_file:
        file_contents = f"""\
        import json
        import os
        import numpy as np
        import pandas as pd

        from bentoml import env, artifacts, api, BentoService
        {_model(model_type)}
        from bentoml.service.artifacts.common import PickleArtifact
        from bentoml.adapters import JsonInput
        from bentoml.types import JsonSerializable
        
        if not os.getenv("UNBOX_DO_NOT_LOAD_MODEL"):
            cwd = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            {_format_custom_code(custom_model_code)}
            os.chdir(cwd)

        {_env_dependencies(tmp_dir, requirements_txt_file, setup_script)}
        {_artifacts(model_type)}
        class TemplateModel(BentoService):
            @api(input=JsonInput())
            def predict(self, parsed_json: JsonSerializable):
                {_extract_input_from_json(task_type, from_csv_path=False)}
                {_predict_function(model_type)}
                return results
            
            @api(input=JsonInput())
            def predict_from_path(self, parsed_json: JsonSerializable):
                input_path = parsed_json["input_path"]
                output_path = parsed_json["output_path"]
                {_extract_input_from_json(task_type, from_csv_path=True)}
                {_predict_function(model_type)}
                with open(output_path, 'w') as f:
                    if type(results) not in [list, np.ndarray]:
                        raise TypeError(f"Wrong return type from predict function: {{type(results)}}")
                    results = np.array(results)
                    json.dump(results.tolist(), f)
                return "Success"

            @api(input=JsonInput())
            def tokenize(self, parsed_json: JsonSerializable):
                {_extract_input_from_json(task_type, from_csv_path=False)}
                results = None
                if "tokenizer" in self.artifacts.kwargs:
                    results = self.artifacts.kwargs["tokenizer"](input)
                return results

            @api(input=JsonInput())
            def tokenize_from_path(self, parsed_json: JsonSerializable):
                input_path = parsed_json["input_path"]
                output_path = parsed_json["output_path"]
                {_extract_input_from_json(task_type, from_csv_path=True)}
                if "tokenizer" in self.artifacts.kwargs:
                    results = self.artifacts.kwargs["tokenizer"](input)
                with open(output_path, 'w') as f:
                    if type(results) == list:
                        json.dump(results, f)
                    else:
                        json.dump(results.tolist(), f)
                return "Success"
        """
        python_file.write(textwrap.dedent(file_contents))

    from template_model import TemplateModel

    return TemplateModel()
