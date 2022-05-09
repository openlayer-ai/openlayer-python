import os
import shutil
import textwrap
from typing import Optional

import bentoml

from .tasks import TaskType


class ModelType:
    """A selection of machine learning modeling frameworks supported by Unbox. """

    @property
    def fasttext(self) -> str:
        """For models built with `fastText <https://fasttext.cc/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebook <https://github.com/unboxai/unboxapi-python-client/tree/main/examples/text-classification/fasttext>`_ and
            `NLP tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you have trained a ``fasttext`` model that performs text classification. Your training pipeline might look like this:

        >>> import fasttext
        >>> model = fasttext.train_supervised(input="training_set")

        You must next define a ``predict_proba`` function that adheres to the following signature:

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model.predict(text_list, k=num_classes)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        You can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.fasttext,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=['Negative', 'Positive'],
        ...    name='My Fast Text model',
        ...    description='this is my fasttext model',
        ... )
        >>> model.to_dict()
        """
        return "FasttextModelArtifact"

    @property
    def sklearn(self) -> str:
        """For models built with `scikit-learn <https://scikit-learn.org/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/main/examples/>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        **If your task type is tabular classification...**

        Let's say your dataset looks like the following:

        >>> df
            CreditScore  Geography    Balance  Churned
        0           618     France     321.92        1
        1           714    Germany  102001.22        0
        2           604      Spain   12333.15        0
        ..          ...        ...        ...      ...

        The first set of variables needed by Unbox are:

        >>> from unboxapi.tasks import TaskType
        >>>
        >>> task_type = TaskType.TabularClassification
        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> categorical_features_map = {'CreditScore': ['France', 'Germany', 'Spain']}

        Now let's say you've trained a simple ``scikit-learn`` model on data that looks like the above.

        You must next define a ``predict_proba`` function that adheres to the following signature:

        >>> def predict_proba(model, input_features: np.ndarray, **kwargs):
        ...     # Optional pre-processing of input_features
        ...     preds = model.predict_proba(input_features)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``input_features`` arg must be a 2D numpy array
        containing a batch of features that will be passed to the model as inputs.

        You can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the ``client.add_model`` function call when you upload the model.

        Here's an example of the ``predict_proba`` function in action:

        >>> x_train = df[feature_names]
        >>> y_train = df['Churned']

        >>> sklearn_model = LogisticRegression(random_state=1300)
        >>> sklearn_model.fit(x_train, y_train)
        >>>
        >>> input_features = x_train.to_numpy()
        array([[618, 'France', 321.92],
               [714, 'Germany', 102001.22],
               [604, 'Spain', 12333.15], ...], dtype=object)

        >>> predict_proba(sklearn_model, input_features)
        array([[0.21735231, 0.78264769],
               [0.66502929, 0.33497071],
               [0.81455616, 0.18544384], ...])

        The other model-specific variables needed by Unbox are:

        >>> from unboxapi.models import ModelType
        >>>
        >>> model_type = ModelType.sklearn
        >>> train_sample_df = df.sample(5000)
        >>> train_sample_label_column_name = 'Churned'

        .. important::
            For tabular classification models, Unbox needs a representative sample of your training
            dataset, so it can effectively explain your model's predictions.

        You can now upload this dataset to Unbox:

        >>> model = client.add_model(
        ...     name='Churn Classifier',
        ...     task_type=task_type,
        ...     function=predict_proba,
        ...     model=sklearn_model,
        ...     model_type=model_type,
        ...     class_names=class_names,
        ...     feature_names=feature_names,
        ...     train_sample_df=train_sample_df,
        ...     train_sample_label_column_name=train_sample_label_column_name,
        ...     categorical_features_map=categorical_features_map,
        ... )
        >>> model.to_dict()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        >>> df
                                      Text  Sentiment
        0    I have had a long weekend              0
        1    I'm in a fantastic mood today          1
        2    Things are looking up                  1
        ..                             ...        ...

        The first set of variables needed by Unbox are:

        >>> from unboxapi.tasks import TaskType
        >>>
        >>> task_type = TaskType.TextClassification
        >>> class_names = ['Negative', 'Positive']

        Now let's say you've trained a simple ``scikit-learn`` model on data that looks like the above.

        You must next define a ``predict_proba`` function that adheres to the following signature:

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model.predict_proba(text_list)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        You can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the ``client.add_model`` function call when you upload the model.

        Here's an example of the ``predict_proba`` function in action:

        >>> x_train = df['Text']
        >>> y_train = df['Sentiment']

        >>> sentiment_lr = Pipeline(
        ...     [
        ...         (
        ...             "count_vect",
        ...             CountVectorizer(min_df=100, ngram_range=(1, 2), stop_words="english"),
        ...         ),
        ...         ("lr", LogisticRegression()),
        ...     ]
        ... )
        >>> sklearn_model.fit(x_train, y_train)

        >>> text_list = ['good', 'bad']
        >>> predict_proba(sentiment_lr, text_list)
        array([[0.30857194, 0.69142806],
               [0.71900947, 0.28099053]])

        The other model-specific variables needed by Unbox are:

        >>> from unboxapi.models import ModelType
        >>>
        >>> model_type = ModelType.sklearn

        You can now upload this dataset to Unbox:

        >>> model = client.add_model(
        ...     name='Churn Classifier',
        ...     task_type=task_type,
        ...     function=predict_proba,
        ...     model=sklearn_model,
        ...     model_type=model_type,
        ...     class_names=class_names,
        ... )
        >>> model.to_dict()
        """
        return "SklearnModelArtifact"

    @property
    def pytorch(self) -> str:
        """For models built with `PyTorch <https://pytorch.org/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/cid/api-docs-improvements/examples/text-classification/tensorflow>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you have trained a ``torch`` model that performs text classification. Your training pipeline might look like this:

        >>> import tensorflow as tf
        >>> from tensorflow import keras
        >>>
        >>> model.compile(optimizer='adam',
        ...     loss='binary_crossentropy',
        ...     metrics=['accuracy'])
        >>>
        >>> model.fit(X_train, y_train, epochs=30, batch_size=512)

        You must next define a ``predict_proba`` function that adheres to the signature defined below.

        **If your task type is text classification...**

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model(text_list)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        **If your task type is tabular classification...**

        >>> def predict_proba(model, input_features: np.ndarray, **kwargs):
        ...     # Optional pre-processing of input_features
        ...     preds = model(input_features)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``input_features`` arg must be a 2D numpy array
        containing a batch of features that will be passed to the model as inputs.

        On both cases, you can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        **If your task type is text classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.tensorflow,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=['Negative', 'Positive'],
        ...    name='My Tensorflow model',
        ...    description='this is my tensorflow model',
        ... )
        >>> model.to_dict()

        **If your task type is tabular classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.tensorflow,
        ...    task_type=TaskType.TabularClassification,
        ...    class_names=['Exited', 'Retained'],
        ...    name='My Tensorflow model',
        ...    description='this is my tensorflow model',
        ... )
        >>> model.to_dict()
        """
        return "PytorchModelArtifact"

    @property
    def tensorflow(self) -> str:
        """For models built with `TensorFlow <https://www.tensorflow.org/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/cid/api-docs-improvements/examples/text-classification/tensorflow>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you have trained a ``tensorflow`` binary classifier. Your training pipeline might look like this:

        >>> import tensorflow as tf
        >>> from tensorflow import keras
        >>>
        >>> model.compile(optimizer='adam',
        ...     loss='binary_crossentropy',
        ...     metrics=['accuracy'])
        >>>
        >>> model.fit(X_train, y_train, epochs=30, batch_size=512)

        You must next define a ``predict_proba`` function that adheres to the signature defined below.

        **If your task type is text classification...**

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model(text_list)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        **If your task type is tabular classification...**

        >>> def predict_proba(model, input_features: np.ndarray, **kwargs):
        ...     # Optional pre-processing of input_features
        ...     preds = model(input_features)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``input_features`` arg must be a 2D numpy array
        containing a batch of features that will be passed to the model as inputs.

        On both cases, you can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        **If your task type is text classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.tensorflow,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=['Negative', 'Positive'],
        ...    name='My Tensorflow model',
        ...    description='this is my tensorflow model',
        ... )
        >>> model.to_dict()

        **If your task type is tabular classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.tensorflow,
        ...    task_type=TaskType.TabularClassification,
        ...    class_names=['Exited', 'Retained'],
        ...    name='My Tensorflow model',
        ...    description='this is my tensorflow model',
        ... )
        >>> model.to_dict()
        """
        return "TensorflowSavedModelArtifact"

    @property
    def transformers(self) -> str:
        """For models built with `Hugging Face transformers <https://huggingface.co/docs/transformers/index>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebook <https://github.com/unboxai/unboxapi-python-client/tree/main/examples/text-classification/transformers>`_ and
            `NLP tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you have trained a ``transformer`` model that performs text classification. The process of loading the transformer and the tokenizer
        might look like this:

        >>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
        >>> import torch
        >>> tokenizer = AutoTokenizer.from_pretrained(
        ...     "distilbert-base-uncased-finetuned-sst-2-english"
        ... )
        >>> model = AutoModelForSequenceClassification.from_pretrained(
        ...     "distilbert-base-uncased-finetuned-sst-2-english"
        ... )

        You must next define a ``predict_proba`` function such as the following:

        >>> def predict_proba(model, texts, tokenizer):
        >>>     batch = tokenizer(
        ...         texts,
        ...         padding=True,
        ...         truncation=True,
        ...         max_length=512,
        ...         return_tensors="pt"
        ...     )
        >>>     with torch.no_grad():
        >>>         outputs = model(**batch)
        >>>         predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        >>>         probs = predictions.detach().numpy().tolist()
        >>>         return probs

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        .. important:: For ``transformers``, the tokenizer is required for the ``predict_proba`` function.

        You can optionally include other kwargs in the function, including variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.transformers,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=['Negative', 'Positive'],
        ...    name='My transformer model',
        ...    description='this is my transformer model',
        ...    tokenizer=tokenizer
        ...    requirements_txt_file='./requirements.txt'
        ... )
        >>> model.to_dict()
        """
        return "TransformersModelArtifact"

    @property
    def keras(self) -> str:
        """For models built with `Keras <https://keras.io/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/cid/api-docs-improvements/examples/text-classification/tensorflow>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you have trained a ``keras`` binary classifier. Your training pipeline might look like this:

        >>> from tensorflow import keras
        >>>
        >>> model.compile(optimizer='adam',
        ...     loss='binary_crossentropy',
        ...     metrics=['accuracy'])
        >>>
        >>> model.fit(X_train, y_train, epochs=30, batch_size=512)

        You must next define a ``predict_proba`` function that adheres to the signature defined below.

        **If your task type is text classification...**

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model(text_list)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        **If your task type is tabular classification...**

        >>> def predict_proba(model, input_features: np.ndarray, **kwargs):
        ...     # Optional pre-processing of input_features
        ...     preds = model(input_features)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the ``input_features`` arg must be a 2D numpy array
        containing a batch of features that will be passed to the model as inputs.

        On both cases, you can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        **If your task type is text classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.keras,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=['Negative', 'Positive'],
        ...    name='My Keras model',
        ...    description='this is my keras model',
        ... )
        >>> model.to_dict()

        **If your task type is tabular classification...**

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.keras,
        ...    task_type=TaskType.TabularClassification,
        ...    class_names=['Exited', 'Retained'],
        ...    name='My keras model',
        ...    description='this is my keras model',
        ... )
        >>> model.to_dict()
        """
        return "KerasModelArtifact"

    @property
    def rasa(self) -> str:
        """For models built with `rasa <https://rasa.com/>`_.

        Examples
        --------
        .. seealso::
            Our `sample notebook <https://github.com/unboxai/unboxapi-python-client/tree/main/examples/text-classification/rasa>`_ and
            `NLP tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Let's say you are using a ``rasa`` model that performs text intent classification.
        Loading such a model might look like this:

        >>> from rasa.nlu.model import Interpreter
        >>> model = Interpreter.load("nlu")

        Furthermore, the intents you wish to classify are the following:

        >>> intents = ["greet", "goodbye", "bot_challenge", "password_reset", "inform",
        ...     "thank", "help", "problem_email", "open_incident", "incident_status",
        ...     "out_of_scope", "restart", "affirm", "deny", "trigger_handoff",
        ...     "human_handoff", "handoff"]

        You must next define a ``predict_proba`` function that looks similar to this:

        >>> def predict_proba(model, text_list):
        >>>     results = []
        >>>
        >>>     labels = ["greet", "goodbye", "bot_challenge", "password_reset", "inform",
        ...         "thank", "help", "problem_email", "open_incident", "incident_status",
        ...         "out_of_scope", "restart", "affirm", "deny", "trigger_handoff",
        ...         "human_handoff", "handoff"]
        >>>
        >>>     for text in text_list:
        >>>         probs = []
        >>>         output = model.parse(text, only_output_properties=False)
        >>>         confs = {d['name']:d['confidence'] for d in output['intent_ranking'] if "id" in d}
        >>>         for label in labels:
        >>>             prob = 0.0
        >>>             if label in confs:
        >>>                 prob = confs[label]
        >>>             probs.append(prob)
        >>>         results.append(probs)
        >>>
        >>>     return results

        The ``model`` arg must be the actual trained model object, and the ``text_list`` arg must be a list of
        strings.

        You can optionally include other kwargs in the function, including tokenizers, variables, encoders etc.
        You simply pass those kwargs to the :meth:`unboxapi.UnboxClient.add_model` function call when you upload the model.

        To upload the model to Unbox, first instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Now, you can use the ``client.add_model()`` method:

        >>> model = client.add_model(
        ...    function=predict_proba,
        ...    model=model,
        ...    model_type=ModelType.rasa,
        ...    task_type=TaskType.TextClassification,
        ...    class_names=intents,
        ...    name='My rasa model',
        ...    description='this is my rasa model',
        ... )
        >>> model.to_dict()
        """
        return "Rasa"

    @property
    def custom(self) -> str:
        """For custom built models."""
        return "Custom"


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

        Returns
        -------
        Dict with object properties
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
