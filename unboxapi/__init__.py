import csv
import os
import shutil
import tarfile
import tempfile
import uuid
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from bentoml.saved_bundle.bundler import _write_bento_content_to_dir
from bentoml.utils.tempdir import TempDirectory

from .api import Api
from .datasets import Dataset
from .exceptions import UnboxException, UnboxInvalidRequest
from .models import Model, ModelType, create_template_model
from .tasks import TaskType
from .version import __version__

from .schemas import DatasetSchema, ModelSchema
from marshmallow import ValidationError


class DeploymentType(Enum):
    """Specify the storage medium being used by your Unbox deployment."""

    ONPREM = 1
    AWS = 2
    GCP = 3
    AZURE = 4


# NOTE: Don't modify this unless you are deploying on-prem.
DEPLOYMENT = DeploymentType.ONPREM


class UnboxClient(object):
    """Client class that interacts with the Unbox Platform.

    Parameters
    ----------
    api_key : str
        Your API key. Retrieve it from the web app.

    Examples
    --------
    Instantiate a client with your api key

    >>> import unboxapi
    >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')
    """

    def __init__(self, api_key: str = None):
        self.api = Api(api_key)
        self.subscription_plan = self.api.get_request("users/subscriptionPlan")

        if DEPLOYMENT == DeploymentType.AWS:
            self.upload = self.api.upload_blob_s3
        elif DEPLOYMENT == DeploymentType.GCP:
            self.upload = self.api.upload_blob_gcs
        elif DEPLOYMENT == DeploymentType.AZURE:
            self.upload = self.api.upload_blob_azure
        else:
            self.upload = self.api.transfer_blob

    def add_model(
        self,
        name: str,
        task_type: TaskType,
        function,
        model,
        model_type: ModelType,
        class_names: List[str],
        requirements_txt_file: Optional[str] = None,
        feature_names: List[str] = [],
        categorical_feature_names: List[str] = [],
        train_sample_df: pd.DataFrame = None,
        train_sample_label_column_name: str = None,
        setup_script: Optional[str] = None,
        custom_model_code: Optional[str] = None,
        dependent_dir: Optional[str] = None,
        description: str = None,
        **kwargs,
    ) -> Model:
        """Uploads a model to the Unbox platform.

        Parameters
        ----------
        name : str
            Name of your model.
        task_type : :obj:`TaskType`
            Type of ML task. E.g. :obj:`TaskType.TextClassification`.
        function :
            Prediction function object in expected format. Scroll down for examples.

            .. note::
                On the Unbox platform, running inference with the model corresponds to calling ``function``. Therefore,
                expect the latency of model calls in the platform to be similar to that of calling ``function`` on a CPU.
                Preparing ``function`` to work with batches of data can improve latency.
        model :
            The Python object for your model loaded into memory. This will get pickled now and later loaded and
            passed to your ``predict_proba`` function to compute run reports, test reports, or conduct what-if analysis.
        model_type : :obj:`ModelType`
            Model framework. E.g. :obj:`ModelType.sklearn`.
        class_names : List[str]
            List of class names corresponding to the outputs of your predict function. E.g. `['positive', 'negative']`.
        requirements_txt_file : str, default None
            Path to a requirements.txt file containing Python dependencies needed by your predict function.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        categorical_features_map : Dict[str], default {}
            A dict containing a list of category names for each feature that is categorical. E.g. `{'Weather': ['Hot', 'Cold']}`.
            Only applicable if your ``task_type`` is :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        train_sample_df : pd.DataFrame, default None
            A random sample of >= 100 rows from your training dataset. This is used to support explainability features.
            Only applicable if your ``task_type`` is :obj:`TaskType.TabularClassification`
            or :obj:`TaskType.TabularRegression`.
        train_sample_label_column_name : str, default None
            Column header in train_sample_df containing the labels. Only applicable if your ``task_type``
            is :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        setup_script : str, default None
            Path to a bash script executing any commands necessary to run before loading the model. This is run after installing
            python requirements.

            .. note::
                This is useful for installing custom libraries, downloading NLTK corpora etc.
        custom_model_code : str, default None
            Code needed to initialize the model. Model object must be ``None`` in this case. Required, and only applicable if your
            ``model_type`` is :obj:`ModelType.custom`.
        dependent_dir : str, default None
            Path to a dir of file dependencies needed to load the model. Required if your ``model_type``
            is :obj:`ModelType.custom`.
        description : str, default None
            Commit message for this version.
        **kwargs
            Any additional keyword args you would like to pass to your ``predict_proba`` function.

        Returns
        -------
        :obj:`Model`
            An object containing information about your uploaded model.

        Examples
        --------

        .. seealso::
            Our `sample notebooks <https://github.com/unboxai/unboxapi-python-client/tree/main/examples>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        Instantiate the client:

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

        >>> from unboxapi import TaskType
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

        >>> from unboxapi import ModelType
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
        ...     categorical_features_map=categorical_features_map,
        ...     train_sample_df=train_sample_df,
        ...     train_sample_label_column_name=train_sample_label_column_name,
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

        >>> from unboxapi import TaskType
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

        >>> from unboxapi import ModelType
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
        # --------------------------- Resource validations --------------------------- #

        # ---------------------------- Schema validations ---------------------------- #
        SCHEMA_VALIDATION_CONTEXT = (
            "There are issues with the data being passed as argument. \n"
        )
        SCHEMA_VALIDATION_MITIGATION = (
            "Make sure to respect the datatypes and constraints specified above."
        )

        model_schema = ModelSchema()
        try:
            model_schema.load(
                {
                    "name": name,
                    "function": function,
                    "description": description,
                    "task_type": task_type.value,
                    "model_type": model_type,
                    "class_names": class_names,
                    "requirements_txt_file": requirements_txt_file,
                    "train_sample_label_column_name": train_sample_label_column_name,
                    "feature_names": feature_names,
                    "categorical_feature_names": categorical_feature_names,
                    "setup_script": setup_script,
                    "custom_model_code": custom_model_code,
                    "dependent_dir": dependent_dir,
                }
            )
        except ValidationError as err:
            error_msg = ""
            for input, msg in err.messages.items():
                error_msg += f"- {msg[0]} \n"
            raise UnboxException(
                SCHEMA_VALIDATION_CONTEXT
                + f"{error_msg}"
                + SCHEMA_VALIDATION_MITIGATION
            )

        # ----------------- Resource-schema consistency validations ---------------- #
        INCONSISTENCY_CONTEXT = "There is an inconsistency between the dataset and the"
        INCONSITENCY_MITIGATION = " Make sure that the value specified in this argument is a column header in the dataframe or csv being uploaded."

        # --------------------- Subscription plan validations ---------------------- #
        SUBSCRIPTION_PLAN_CONTEXT = "There was a problem uploading the dataset due to your subscription plan. \n"
        SUBSCRIPTION_PLAN_MITIGATION = (
            "To upgrade your plan, visit our website https://unbox.ai"
        )
            )

        # if custom_model_code:
        #     assert (
        #         model_type is ModelType.custom
        #     ), "model_type must be ModelType.custom if specifying custom_model_code"
        # if model_type is ModelType.custom:
        #     assert (
        #         custom_model_code is not None
        #     ), "Must specify custom_model_code when using ModelType.custom"
        #     assert (
        #         dependent_dir is not None
        #     ), "Must specify dependent_dir when using ModelType.custom"
        #     assert (
        #         requirements_txt_file is not None
        #     ), "Must specify requirements_txt_file when using ModelType.custom"
        #     assert model is None, "model must be None when using ModelType.custom"

        # Validate predict_proba extra args
        user_args = function.__code__.co_varnames[: function.__code__.co_argcount][2:]
        kwarg_keys = tuple(kwargs)
        assert (
            user_args == kwarg_keys
        ), f"Your function's additional args {user_args} must match the kwargs you specifed {kwarg_keys}"
        
        if task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
        #     required_fields = [
        #         (feature_names, "feature_names"),
        #         (train_sample_df, "train_sample_df"),
        #         (train_sample_label_column_name, "train_sample_label_column_name"),
        #     ]
        #     for value, field in required_fields:
        #         if value is None:
        #             raise UnboxException(
        #                 f"Must specify {field} for TabularClassification"
        #             )
            if len(train_sample_df.index) < 100:
                raise UnboxException("train_sample_df must have at least 100 rows")
            train_sample_df = train_sample_df.sample(
                min(3000, len(train_sample_df.index))
            )
            try:
                headers = train_sample_df.columns.tolist()
                [
                    headers.index(name)
                    for name in feature_names + [train_sample_label_column_name]
                ]
            except ValueError:
                raise UnboxException(
                    "Feature / label column names not in train_sample_df"
                )

            # TODO: replace validation
            # self._validate_categorical_features(
            #     train_sample_df, categorical_features_map
            # )

        with TempDirectory() as dir:
            bento_service = create_template_model(
                model_type,
                task_type,
                dir,
                requirements_txt_file,
                setup_script,
                custom_model_code,
            )
            if model_type is ModelType.transformers:
                if "tokenizer" not in kwargs:
                    raise UnboxException(
                        "Must specify tokenizer in kwargs when using a transformers model"
                    )
                bento_service.pack(
                    "model", {"model": model, "tokenizer": kwargs["tokenizer"]}
                )
                kwargs.pop("tokenizer")
            elif model_type not in [ModelType.custom, ModelType.rasa]:
                bento_service.pack("model", model)

            bento_service.pack("function", function)
            bento_service.pack("kwargs", kwargs)

            with TempDirectory() as temp_dir:
                print("Bundling model and artifacts...")
                _write_bento_content_to_dir(bento_service, temp_dir)

                if model_type is ModelType.rasa:
                    dependent_dir = model.model_metadata.model_dir

                # Add dependent directory to bundle
                if dependent_dir is not None:
                    dependent_dir = os.path.abspath(dependent_dir)
                    if dependent_dir == os.getcwd():
                        raise UnboxException("dependent_dir can't be working directory")
                    shutil.copytree(
                        dependent_dir,
                        os.path.join(
                            temp_dir,
                            f"TemplateModel/{os.path.basename(dependent_dir)}",
                        ),
                    )

                # Add sample of training data to bundle
                if task_type in [
                    TaskType.TabularClassification,
                    TaskType.TabularRegression,
                ]:
                    train_sample_df.to_csv(
                        os.path.join(temp_dir, f"TemplateModel/train_sample.csv"),
                        index=False,
                    )

                # Tar the model bundle with its artifacts and upload
                with TempDirectory() as tarfile_dir:
                    tarfile_path = f"{tarfile_dir}/model"

                    with tarfile.open(tarfile_path, mode="w:gz") as tar:
                        tar.add(temp_dir, arcname=bento_service.name)

                    endpoint = "models"
                    payload = dict(
                        name=name,
                        description=description,
                        classNames=class_names,
                        taskType=task_type.value,
                        type=model_type.name,
                        kwargs=list(kwargs.keys()),
                        featureNames=feature_names,
                        categoricalFeatureNames=categorical_feature_names,
                        trainSampleLabelColumnName=train_sample_label_column_name,
                    )
                    print("Uploading model to Unbox...")
                    modeldata = self.upload(
                        endpoint=endpoint,
                        file_path=tarfile_path,
                        object_name="tarfile",
                        body=payload,
                    )
        os.remove("template_model.py")
        return Model(modeldata)

    def add_dataset(
        self,
        name: str,
        task_type: TaskType,
        file_path: str,
        class_names: List[str],
        label_column_name: str,
        feature_names: List[str] = [],
        text_column_name: Optional[str] = None,
        categorical_feature_names: List[str] = [],
        tag_column_name: Optional[str] = None,
        language: str = "en",
        sep: str = ",",
        description: Optional[str] = None,
    ) -> Dataset:
        r"""Uploads a dataset to the Unbox platform (from a csv).

        Parameters
        ----------
        name : str
            Name of your dataset.
        task_type : :obj:`TaskType`
            Type of ML task. E.g. :obj:`TaskType.TextClassification`.
        file_path : str
            Path to the csv file containing the dataset.
        class_names : List[str]
            List of class names indexed by label integer in the dataset.
            E.g. `[negative, positive]` when `[0, 1]` are in your label column.
        label_column_name : str
            Column header in the csv containing the labels.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        text_column_name : str, default None
            Column header in the csv containing the input text. Only applicable if your ``task_type`` is
            :obj:`TaskType.TextClassification`.
        categorical_features_map : Dict[str, List[str]], default {}
            A dict containing a list of category names for each feature that is categorical. E.g. `{'Weather': ['Hot', 'Cold']}`.
            Only applicable if your ``task_type`` is :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        tag_column_name : str, default None
            Column header in the csv containing tags you want pre-populated in Unbox.

            .. important::
                Each cell in this column must be either empty or contain a list of strings.

                .. csv-table::
                    :header: ..., Tags

                    ..., "['sample']"
                    ..., "['tag_one', 'tag_two']"
        language : str, default 'en'
            The language of the dataset in ISO 639-1 (alpha-2 code) format.
        sep : str, default ','
            Delimiter to use. E.g. `'\\t'`.
        description : str, default None
            Commit message for this version.

        Returns
        -------
        :obj:`Dataset`
            An object containing information about your uploaded dataset.

        Notes
        -----
        - Please ensure your input features are strings, ints or floats.
        - Please ensure your label column name is not contained in ``feature_names``.

        Examples
        --------
        Instantiate the client:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        **If your task type is tabular classification...**

        Let's say your dataset looks like the following:

        .. csv-table::
            :header: CreditScore, Geography, Balance, Churned

            618, France, 321.92, 1
            714, Germany, 102001.22, 0
            604, Spain, 12333.15, 0

        .. important::
            The labels in your csv **must** be integers that correctly index into the ``class_names`` array
            that you define (as shown below). E.g. 0 => 'Retained', 1 => 'Churned'

        The variables are needed by Unbox are:

        >>> from unboxapi import TaskType
        >>>
        >>> task_type = TaskType.TabularClassification
        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_features_map = {'CreditScore': ['France', 'Germany', 'Spain']}

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     name='Churn Validation',
        ...     task_type=task_type,
        ...     file_path='/path/to/dataset.csv',
        ...     class_names=class_names,
        ...     label_column_name=label_column_name,
        ...     feature_names=feature_names,
        ...     categorical_features_map=categorical_map,
        ... )
        >>> dataset.to_dict()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        .. csv-table::
            :header: Text, Sentiment

            I have had a long weekend, 0
            I'm in a fantastic mood today, 1
            Things are looking up, 1

        The variables are needed by Unbox are:

        >>> from unboxapi import TaskType
        >>>
        >>> task_type = TaskType.TextClassification
        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     name='Churn Validation',
        ...     task_type=task_type,
        ...     file_path='/path/to/dataset.csv',
        ...     class_names=class_names,
        ...     label_column_name=label_column_name,
        ...     text_column_name=text_column_name,
        ... )
        >>> dataset.to_dict()
        """
        # --------------------------- Resource validations --------------------------- #
        exp_file_path = os.path.expanduser(file_path)
        object_name = "original.csv"
        if not os.path.isfile(exp_file_path):
            raise UnboxException(
                "There is a problem with the specified file path."
                f"The file path {exp_file_path} does not contain a file."
                "Make sure that the `file_path` contains the dataset csv file."
            )

        with open(exp_file_path, "rt") as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
            row_count = sum(1 for _ in reader)

        # ---------------------------- Schema validations ---------------------------- #
        SCHEMA_VALIDATION_CONTEXT = (
            "There are issues with the data being passed as argument. \n"
        )
        SCHEMA_VALIDATION_MITIGATION = (
            "Make sure to respect the datatypes and constraints specified above."
        )

        dataset_schema = DatasetSchema()
        try:
            dataset_schema.load(
                {
                    "name": name,
                    "file_path": file_path,
                    "description": description,
                    "task_type": task_type.value,
                    "class_names": class_names,
                    "label_column_name": label_column_name,
                    "tag_column_name": tag_column_name,
                    "language": language,
                    "sep": sep,
                    "feature_names": feature_names,
                    "text_column_name": text_column_name,
                    "categorical_feature_names": categorical_feature_names,
                }
            )
        except ValidationError as err:
            error_msg = ""
            for input, msg in err.messages.items():
                error_msg += f"- {msg[0]} \n"
            raise UnboxException(
                SCHEMA_VALIDATION_CONTEXT
                + f"{error_msg}"
                + SCHEMA_VALIDATION_MITIGATION
            )

        # ----------------- Resource-schema consistency validations ---------------- #
        INCONSISTENCY_CONTEXT = "There is an inconsistency between the dataset and the"
        INCONSITENCY_MITIGATION = " Make sure that the value specified in this argument is a column header in the dataframe or csv being uploaded."

        df = pd.read_csv(file_path, sep=sep)

        # Label column validations
        try:
            headers.index(label_column_name)
        except ValueError:
            raise UnboxException(
                INCONSISTENCY_CONTEXT
                + " `label_column_name`. \n"
                + f"The column {label_column_name} is not on the dataset. \n"
                + INCONSITENCY_MITIGATION
            )

        dataset_classes = list(df[label_column_name].unique())
        if len(dataset_classes) > len(class_names):
            raise UnboxException(
                INCONSISTENCY_CONTEXT
                + " `class_names`. \n"
                + f"There are {len(dataset_classes)} classes represented on the dataset, but there are only {len(class_names)} items on the `class_names` list. \n"
                f"Make sure that there are at most {len(class_names)} classes in your dataset."
            )

        # Feature validations
        try:
            if text_column_name:
                feature_names.append(text_column_name)
            for feature_name in feature_names:
                headers.index(feature_name)
        except ValueError:
            if text_column_name:
                raise UnboxException(
                    INCONSISTENCY_CONTEXT
                    + " `text_column`. \n"
                    + f"The column `{text_column_name}` is not on the dataset. \n"
                    + INCONSITENCY_MITIGATION
                )
            else:
                features_not_in_dataset = [
                    feature for feature in feature_names if feature not in headers
                ]
                raise UnboxException(
                    INCONSISTENCY_CONTEXT
                    + " `feature_names` \n."
                    + f"The features `{features_not_in_dataset}` is not on the dataset. \n"
                    + INCONSITENCY_MITIGATION
                )

        # Tag column validation
        try:
            if tag_column_name:
                headers.index(tag_column_name)
        except ValueError:
            raise UnboxException(
                INCONSISTENCY_CONTEXT
                + " `tag_column_name`. \n"
                + f"The column `{tag_column_name}` is not on the dataset. \n"
                + INCONSITENCY_MITIGATION
            )

        # --------------------- Subscription plan validations ---------------------- #
        SUBSCRIPTION_PLAN_CONTEXT = "There was a problem uploading the dataset due to your subscription plan. \n"
        SUBSCRIPTION_PLAN_MITIGATION = (
            "To upgrade your plan, visit our website https://unbox.ai"
        )

        if row_count > self.subscription_plan["datasetSize"]:
            raise UnboxException(
                SUBSCRIPTION_PLAN_CONTEXT
                + f"The dataset your are trying to upload contains {row_count} rows, which exceeds your plan's"
                f" limit of {self.subscription_plan['datasetSize']}. \n"
                + SUBSCRIPTION_PLAN_MITIGATION
            )

        endpoint = "datasets"
        payload = dict(
            name=name,
            description=description,
            taskType=task_type.value,
            classNames=class_names,
            labelColumnName=label_column_name,
            tagColumnName=tag_column_name,
            language=language,
            sep=sep,
            featureNames=feature_names,
            categoricalFeatureNames=categorical_feature_names,
        )
        return Dataset(
            self.upload(
                endpoint=endpoint,
                file_path=file_path,
                object_name=object_name,
                body=payload,
            )
        )

    def add_dataframe(
        self,
        name: str,
        task_type: TaskType,
        df: pd.DataFrame,
        class_names: List[str],
        label_column_name: str,
        feature_names: List[str] = [],
        text_column_name: Optional[str] = None,
        categorical_feature_names: List[str] = [],
        description: Optional[str] = None,
        tag_column_name: Optional[str] = None,
        language: str = "en",
    ) -> Dataset:
        r"""Uploads a dataset to the Unbox platform (from a pandas DataFrame).

        Parameters
        ----------
        name : str
            Name of your dataset.
        task_type : :obj:`TaskType`
            Type of ML task. E.g. :obj:`TaskType.TextClassification`.
        df : pd.DataFrame
            Dataframe containing your dataset.
        class_names : List[str]
            List of class names indexed by label integer in the dataset.
            E.g. `[negative, positive]` when `[0, 1]` are in your label column.
        label_column_name : str
            Column header in the csv containing the labels.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        text_column_name : str, default None
            Column header in the csv containing the input text. Only applicable if your ``task_type`` is
            :obj:`TaskType.TextClassification`.
        categorical_features_map : Dict[str, List[str]], default {}
            A dict containing a list of category names for each feature that is categorical. E.g. `{'Weather': ['Hot', 'Cold']}`.
            Only applicable if your ``task_type`` is :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        description : str, default None
            Commit message for this version.
        tag_column_name : str, default None
            Column header in the csv containing tags you want pre-populated in Unbox.

            .. important::
                Each cell in this column must be either empty or contain a list of strings.

                .. csv-table::
                    :header: ..., Tags

                    ..., "['sample']"
                    ..., "['tag_one', 'tag_two']"
        language : str, default 'en'
            The language of the dataset in ISO 639-1 (alpha-2 code) format.

        Returns
        -------
        :obj:`Dataset`
            An object containing information about your uploaded dataset.

        Notes
        -----
        - Please ensure your input features are strings, ints or floats.
        - Please ensure your label column name is not contained in ``feature_names``.

        Examples
        --------
        Instantiate the client

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        **If your task type is tabular classification...**

        Let's say your dataframe looks like the following:

        >>> df
            CreditScore  Geography    Balance  Churned
        0           618     France     321.92        1
        1           714    Germany  102001.22        0
        2           604      Spain   12333.15        0

        .. important::
            The labels in your dataframe **must** be integers that correctly index into the ``class_names`` array
            that you define (as shown below). E.g. 0 => 'Retained', 1 => 'Churned'

        The variables are needed by Unbox are:

        >>> from unboxapi import TaskType
        >>>
        >>> task_type = TaskType.TabularClassification
        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_features_map = {'CreditScore': ['France', 'Germany', 'Spain']}

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     name='Churn Validation',
        ...     task_type=task_type,
        ...     df=df,
        ...     class_names=class_names,
        ...     feature_names=feature_names,
        ...     label_column_name=label_column_name,
        ...     categorical_features_map=categorical_map,
        ... )
        >>> dataset.to_dict()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        >>> df
                                      Text  Sentiment
        0    I have had a long weekend              0
        1    I'm in a fantastic mood today          1
        2    Things are looking up                  1

        The variables are needed by Unbox are:

        >>> from unboxapi import TaskType
        >>>
        >>> task_type = TaskType.TextClassification
        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     name='Churn Validation',
        ...     task_type=task_type,
        ...     df=df,
        ...     class_names=class_names,
        ...     text_column_name=text_column_name,
        ...     label_column_name=label_column_name,
        ... )
        >>> dataset.to_dict()
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(file_path, index=False)
            return self.add_dataset(
                file_path=file_path,
                task_type=task_type,
                class_names=class_names,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                name=name,
                description=description,
                tag_column_name=tag_column_name,
                language=language,
                feature_names=feature_names,
                categorical_feature_names=categorical_feature_names,
            )

    @staticmethod
    def _validate_categorical_features(
        df: pd.DataFrame, categorical_features_map: Dict[str, List[str]]
    ):
        for feature, options in categorical_features_map.items():
            if len(df[feature].unique()) > len(options):
                raise UnboxInvalidRequest(
                    f"Feature '{feature}' contains more options in the df than provided "
                    "for it in `categorical_features_map`"
                )
