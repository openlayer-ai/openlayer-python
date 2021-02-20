from typing import List
from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput, DataframeInput
from bentoml.types import JsonSerializable, InferenceTask


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model'), PickleArtifact('function')])
class SklearnTextTemplateModel(BentoService):

    @api(input=DataframeInput(
        orient="records",
        columns=["text"],
        dtype={"text": "str"},
    ), batch=True)
    def batch(self, df):
        text = df['text'].tolist()
        return self.artifacts.function(
            self.artifacts.model,
            text
        )

    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_json_list: List[JsonSerializable], tasks: List[InferenceTask]):
        text = []
        for json, task in zip(parsed_json_list, tasks):
            if "text" in json:
                text.append(json['text'])
            else:
                task.discard(http_status=400, err_msg="input json must contain `text` field")

        return self.artifacts.function(
            self.artifacts.model,
            text
        )