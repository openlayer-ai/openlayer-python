import csv
import os
import shutil
import tarfile
import tempfile
import traceback
import uuid
from typing import Callable, List, Optional

import marshmallow as ma
import pandas as pd
from bentoml.saved_bundle import bundler
from bentoml.utils import tempdir

from . import api, exceptions, schemas, utils
from .datasets import Dataset
from .models import Model, ModelType, create_template_model
from .projects import Project
from .tasks import TaskType
from .version import __version__  # noqa: F401


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
        self.api = api.Api(api_key)
        self.subscription_plan = self.api.get_request("me/subscription-plan")

    def create_project(
        self, name: str, task_type: TaskType, description: Optional[str] = None
    ) -> Project:
        """Creates a project on the Unbox platform.

        Parameters
        ----------
        name : str
            Name of your project.

            .. important::
                The project name must be unique in a user's collection of projects.

        task_type : :obj:`TaskType`
            Type of ML task. E.g. :obj:`TaskType.TabularClassification` or
            :obj:`TaskType.TextClassification`.

        description : str
            Project description.

        Returns
        -------
        Project
            An object that is used to upload models and datasets to the Unbox platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and create the project:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')
        >>>
        >>> from unboxapi.tasks import TaskType
        >>> project = client.create_project(
        ...     name="Churn prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first error analysis playground",
        ... )

        With the Project object created, you are able to start uploading models and
        datasets to the platform. Refer to :obj:`add_model` and obj:`add_dataset` or
        obj:`add_dataframe` for detailed examples.
        """
        # ----------------------------- Schema validation ---------------------------- #
        project_schema = schemas.ProjectSchema()
        try:
            project_schema.load({"name": name, "description": description})
        except ma.ValidationError as err:
            raise exceptions.UnboxValidationError(
                self._format_error_message(err)
            ) from None

        endpoint = "projects"
        payload = dict(name=name, description=description, taskType=task_type.value)
        project_data = self.api.post_request(endpoint, body=payload)

        project = Project(project_data, self.api.upload, self.subscription_plan, self)
        print(f"Created your project. Navigate to {project.links['app']} to see it.")
        return project

    def load_project(self, name: str) -> Project:
        """Loads an existing project from the Unbox platform.

        Parameters
        ----------
        name : str
            Name of the project to be loaded. The name of the project is the one
            displayed on the Unbox platform.

            .. note::
                If you haven't created the project yet, you should use the
                :obj:`create_project` method.

        Returns
        -------
        Project
            An object that is used to upload models and datasets to the Unbox platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and load the project:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")

        With the Project object loaded, you are able to upload models and datasets to
        the platform. Refer to :obj:`add_model` and obj:`add_dataset` or
        obj:`add_dataframe` for detailed examples.
        """
        endpoint = f"me/projects/{name}"
        project_data = self.api.get_request(endpoint)
        project = Project(project_data, self.api.upload, self.subscription_plan, self)
        print(f"Found your project. Navigate to {project.links['app']} to see it.")
        return project

    def create_or_load_project(
        self, name: str, task_type: TaskType, description: Optional[str] = None
    ) -> Project:
        """Helper function that returns a project given a name. Creates a new project
        if no project with the name exists.

        Parameters
        ----------
        name : str
            Name of your project.

            .. important::
                The project name must be unique in a user's collection of projects.

        task_type : :obj:`TaskType`
            Type of ML task. E.g. :obj:`TaskType.TabularClassification` or
            :obj:`TaskType.TextClassification`.

        description : str
            Project description.

        Returns
        -------
        Project
            An object that is used to upload models and datasets to the Unbox platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and create or load the project:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')
        >>>
        >>> from unboxapi.tasks import TaskType
        >>> project = client.create_or_load_project(
        ...     name="Churn prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first error analysis playground",
        ... )

        With the Project object, you are able to start uploading models and
        datasets to the platform. Refer to :obj:`add_model` and obj:`add_dataset` or
        obj:`add_dataframe` for detailed examples.
        """
        try:
            return self.create_project(
                name=name, task_type=task_type, description=description
            )
        except exceptions.UnboxDuplicateTask:
            return self.load_project(name)

    def add_model(
        self,
        name: str,
        task_type: TaskType,
        function,
        model,
        model_type: ModelType,
        class_names: List[str],
        requirements_txt_file: str,
        feature_names: List[str] = [],
        categorical_feature_names: List[str] = [],
        train_sample_df: pd.DataFrame = None,
        train_sample_label_column_name: str = None,
        setup_script: Optional[str] = None,
        custom_model_code: Optional[str] = None,
        dependent_dir: Optional[str] = None,
        commit_message: str = None,
        project_id: str = None,
        **kwargs,
    ) -> Model:
        """Uploads a model to the Unbox platform.

        Parameters
        ----------
        name : str
            Name of your model.

            .. important::
                Versioning models on the Unbox platform happens via the ``name``
                argument. If ``add_model`` is called with a ``name`` that still
                does not exist inside the project, Unbox treats it as the
                **first version** of a new model lineage. On the other hand, if
                a model with the specified ``name`` already exists inside the project,
                Unbox treats it as a **new version** of an existing model lineage.
        function :
            Prediction function object in expected format. Scroll down for examples.

            .. note::
                On the Unbox platform, running inference with the model corresponds
                to calling ``function``. Therefore, expect the latency of model calls
                in the platform to be similar to that of calling ``function`` on a CPU.
                Preparing ``function`` to work with batches of data can improve latency.
        model :
            The Python object for your model loaded into memory. This will get pickled
            now and later loaded and passed to your ``predict_proba`` function to
            compute run reports, test reports, or conduct what-if analysis.
        model_type : :obj:`ModelType`
            Model framework. E.g. :obj:`ModelType.sklearn`.
        class_names : List[str]
            List of class names corresponding to the outputs of your predict function.
            E.g. `['positive', 'negative']`.
        requirements_txt_file : str, default None
            Path to a requirements.txt file containing Python dependencies needed by
            your predict function.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        categorical_feature_names : List[str], default []
            A list containing the names of all categorical features used by the model.
            E.g. `["Gender", "Geography"]`. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        train_sample_df : pd.DataFrame, default None
            A random sample of >= 100 rows from your training dataset. This is used to
            support explainability features. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        train_sample_label_column_name : str, default None
            Column header in train_sample_df containing the labels. Only applicable if
            your ``task_type`` is :obj:`TaskType.TabularClassification` or
            :obj:`TaskType.TabularRegression`.
        setup_script : str, default None
            Path to a bash script executing any commands necessary to run before
            loading the model. This is run after installing python requirements.

            .. note::
                This is useful for installing custom libraries, downloading NLTK
                corpora etc.
        custom_model_code : str, default None
            Code needed to initialize the model. Model object must be ``None`` in
            this case. Required, and only applicable if your ``model_type`` is
            :obj:`ModelType.custom`.
        dependent_dir : str, default None
            Path to a dir of file dependencies needed to load the model.
            Required if your ``model_type`` is :obj:`ModelType.custom`.
        commit_message : str, default None
            Commit message for this version.
        **kwargs
            Any additional keyword args you would like to pass to your
            ``predict_proba`` function.

            .. note::
                If you include `tokenizer` as part of your ``predict_proba``'s kwargs,
                it will also be used by our explainability techniques.

        Returns
        -------
        :obj:`Model`
            An object containing information about your uploaded model.

        Examples
        --------

        .. seealso::
            Our `sample notebooks
            <https://github.com/unboxai/unboxapi-python-client/tree/main/examples>`_ and
            `tutorials <https://unbox.readme.io/docs/overview-of-tutorial-tracks>`_.

        First, instantiate the client:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from unboxapi.tasks import TaskType
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        **If your project's task type is tabular classification...**

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
        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> categorical_feature_names = ['Geography']

        Now let's say you've trained a simple ``scikit-learn`` model on data that looks
        like the above.

        You must next define a ``predict_proba`` function that adheres to the
        following signature:

        >>> def predict_proba(model, input_features: np.ndarray, **kwargs):
        ...     # Optional pre-processing of input_features
        ...     preds = model.predict_proba(input_features)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the
        ``input_features`` arg must be a 2D numpy array containing a batch of features
        that will be passed to the model as inputs.

        You can optionally include other kwargs in the function, including variables,
        encoders etc. You simply pass those kwargs to the ``project.add_model``
        function call when you upload the model.

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
        >>> requirements_txt_file = "requirements.txt"  # path to requirements.txt

        .. important::
            For tabular classification models, Unbox needs a representative sample
            of your training dataset, so it can effectively explain your
            model's predictions.

        You can now upload this model to Unbox:

        >>> model = project.add_model(
        ...     name='Linear classifier',
        ...     commit_message='First iteration of vanilla logistic regression',
        ...     function=predict_proba,
        ...     model=sklearn_model,
        ...     model_type=model_type,
        ...     class_names=class_names,
        ...     feature_names=feature_names,
        ...     categorical_feature_names=categorical_feature_names,
        ...     train_sample_df=train_sample_df,
        ...     train_sample_label_column_name=train_sample_label_column_name,
        ...     requirements_txt_file=requirements_txt_file,
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

        The first variable needed by Unbox is:

        >>> class_names = ['Negative', 'Positive']

        Now let's say you've trained a simple ``scikit-learn`` model on data that
        looks like the above.

        You must next define a ``predict_proba`` function that adheres to the
        following signature:

        >>> def predict_proba(model, text_list: List[str], **kwargs):
        ...     # Optional pre-processing of text_list
        ...     preds = model.predict_proba(text_list)
        ...     # Optional re-weighting of preds
        ...     return preds

        The ``model`` arg must be the actual trained model object, and the
        ``text_list`` arg must be a list of strings.

        You can optionally include other kwargs in the function, including tokenizers,
        variables, encoders etc. You simply pass those kwargs to the
        ``project.add_model`` function call when you upload the model.

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
        >>> requirements_txt_file = "requirements.txt"  # path to requirements.txt


        You can now upload this dataset to Unbox:

        >>> model = project.add_model(
        ...     name='Linear classifier',
        ...     commit_message='First iteration of vanilla logistic regression',
        ...     function=predict_proba,
        ...     model=sklearn_model,
        ...     model_type=model_type,
        ...     class_names=class_names,
        ...     requirements_txt_file=requirements_txt_file,
        ... )
        >>> model.to_dict()

        .. note::
            If inside the given project the ``add_model`` method is called with
            ``name='Linear classifier'`` for the first time, a new model lineage
            will be created with ``Linear classifier`` as a name and ``description``
            will be the first commit on that new tree. In the future, if you'd like
            to commit a new version to that same lineage, you can simply call
            `add_model` using ``name='Linear classifier'`` again and use
            ``description`` with the new commit message. If you'd like to start a
            new separate lineage inside that project, you can call the ``add_model``
            method with a different ``name``. E.g., ``name ='Nonlinear classifier'``.
        """
        # ---------------------------- Schema validations ---------------------------- #
        if task_type not in [
            TaskType.TabularClassification,
            TaskType.TextClassification,
        ]:
            raise exceptions.UnboxValidationError(
                "`task_type` must be either TaskType.TabularClassification or "
                "TaskType.TextClassification. \n"
            ) from None
        if model_type not in [model_framework for model_framework in ModelType]:
            raise exceptions.UnboxValidationError(
                "`model_type` must be one of the supported ModelTypes. Check out "
                "our API reference for a full list "
                "https://reference.unbox.ai/reference/api/unboxapi.ModelType.html. \n"
            ) from None
        model_schema = schemas.ModelSchema()
        try:
            model_schema.load(
                {
                    "name": name,
                    "commit_message": commit_message,
                    "task_type": task_type.value,
                    "model_type": model_type.value,
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
        except ma.ValidationError as err:
            raise exceptions.UnboxValidationError(
                self._format_error_message(err)
            ) from None

        # --------------------------- Resource validations --------------------------- #
        # Requirements check
        if requirements_txt_file and not os.path.isfile(
            os.path.expanduser(requirements_txt_file)
        ):
            raise exceptions.UnboxResourceError(
                f"File `{requirements_txt_file}` does not exist. \n"
            ) from None

        # Setup script
        if setup_script and not os.path.isfile(os.path.expanduser(setup_script)):
            raise exceptions.UnboxResourceError(
                f"File `{setup_script}` does not exist. \n"
            ) from None

        # Dependent dir
        if dependent_dir and dependent_dir == os.getcwd():
            raise exceptions.UnboxResourceError(
                "`dependent_dir` cannot be the working directory. \n",
                mitigation="Make sure that the specified `dependent_dir` is different "
                f"from `{os.getcwd()}`.",
            ) from None

        # Training set
        if task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
            if len(train_sample_df.index) < 100:
                raise exceptions.UnboxResourceError(
                    context="There's an issue with the specified `train_sample_df`. \n",
                    message=f"Only {len(train_sample_df.index)} rows were found. \n",
                    mitigation="Make sure to upload a training sample with 100+ rows.",
                ) from None
            if train_sample_df.isnull().values.any():
                raise exceptions.UnboxResourceError(
                    context="There's an issue with the specified `train_sample_df`. \n",
                    message=f"The `train_sample_df` contains null values, which is "
                    "currently not supported. \n",
                    mitigation="Make sure to upload a training sample without "
                    "null values.",
                ) from None

            train_sample_df = train_sample_df.sample(
                min(3000, len(train_sample_df.index))
            )

        # predict_proba
        if not isinstance(function, Callable):
            raise exceptions.UnboxValidationError(
                f"- `{function}` specified as `function` is not callable. \n"
            ) from None

        user_args = function.__code__.co_varnames[: function.__code__.co_argcount][2:]
        kwarg_keys = tuple(kwargs)
        if user_args != kwarg_keys:
            raise exceptions.UnboxResourceError(
                context="There's an issue with the speficied `function`. \n",
                message=f"Your function's additional args {user_args} do not match the "
                f"kwargs you specified {kwarg_keys}. \n",
                mitigation=f"Make sure to include all of the kwargs required "
                "to run your inference `function`.",
            ) from None

        if model_type != ModelType.custom:
            try:
                if task_type in [
                    TaskType.TabularClassification,
                    TaskType.TabularRegression,
                ]:
                    test_input = train_sample_df[:3][feature_names].to_numpy()
                    with utils.HidePrints():
                        function(model, test_input, **kwargs)
                else:
                    test_input = [
                        "Unbox is great!",
                        "Let's see if this function is ready for some error analysis",
                    ]
                    with utils.HidePrints():
                        function(model, test_input, **kwargs)
            except Exception as e:
                exception_stack = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                raise exceptions.UnboxResourceError(
                    context="There's an issue with the specified `function`. \n",
                    message=f"It is failing with the following error: \n"
                    f"{exception_stack}",
                    mitigation="Make sure your function receives the model and the "
                    "input as arguments, plus the additional kwargs. You may find it "
                    "helpful to test your function out before uploading your model.",
                ) from None

        # Transformers resources
        if model_type is ModelType.transformers:
            if "tokenizer" not in kwargs:
                raise exceptions.UnboxResourceError(
                    context="There's a missing kwarg for the specified model type. \n",
                    message="`tokenizer` must be specified in kwargs when using a "
                    "transformers model. \n",
                    mitigation="Make sure to specify the `tokenizer`.",
                ) from None

        # ------------------ Resource-schema consistency validations ----------------- #
        # Feature validations
        if task_type in [TaskType.TabularClassification, TaskType.TabularRegression]:
            try:
                headers = train_sample_df.columns.tolist()
                [
                    headers.index(name)
                    for name in feature_names + [train_sample_label_column_name]
                ]
            except ValueError:
                features_not_in_dataset = [
                    feature
                    for feature in feature_names + [train_sample_label_column_name]
                    if feature not in headers
                ]
                raise exceptions.UnboxDatasetInconsistencyError(
                    f"Features {features_not_in_dataset} specified in `feature_names` "
                    "are not on the training sample. \n"
                ) from None

            required_fields = [
                (feature_names, "feature_names"),
                (train_sample_df, "train_sample_df"),
                (train_sample_label_column_name, "train_sample_label_column_name"),
            ]
            for value, field in required_fields:
                if value is None:
                    raise exceptions.UnboxDatasetInconsistencyError(
                        message=f"TabularClassification task missing `{field}`.\n",
                        mitigation=f"Make sure to specify `{field}` for tabular "
                        "classification tasks.",
                    ) from None

        with tempdir.TempDirectory() as dir:
            bento_service = create_template_model(
                model_type,
                task_type,
                dir,
                requirements_txt_file,
                setup_script,
                custom_model_code,
            )
            if model_type is ModelType.transformers:
                bento_service.pack(
                    "model", {"model": model, "tokenizer": kwargs["tokenizer"]}
                )
                kwargs.pop("tokenizer")
            elif model_type not in [ModelType.custom, ModelType.rasa]:
                bento_service.pack("model", model)

            bento_service.pack("function", function)
            bento_service.pack("kwargs", kwargs)

            with tempdir.TempDirectory() as temp_dir:
                print("Bundling model and artifacts...")
                bundler._write_bento_content_to_dir(bento_service, temp_dir)

                if model_type is ModelType.rasa:
                    dependent_dir = model.model_metadata.model_dir

                # Add dependent directory to bundle
                if dependent_dir is not None:
                    dependent_dir = os.path.abspath(dependent_dir)
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
                with tempdir.TempDirectory() as tarfile_dir:
                    tarfile_path = f"{tarfile_dir}/model"

                    with tarfile.open(tarfile_path, mode="w:gz") as tar:
                        tar.add(temp_dir, arcname=bento_service.name)

                    endpoint = f"projects/{project_id}/ml-models"
                    payload = dict(
                        name=name,
                        taskType=task_type.value,
                        commitMessage=commit_message,
                        classNames=class_names,
                        architectureType=model_type.name,
                        kwargs=list(kwargs.keys()),
                        featureNames=feature_names,
                        categoricalFeatureNames=categorical_feature_names,
                        trainSampleLabelColumnName=train_sample_label_column_name,
                    )

                    modeldata = self.api.upload(
                        endpoint=endpoint,
                        file_path=tarfile_path,
                        object_name="tarfile",
                        body=payload,
                    )
        os.remove("template_model.py")

        print(f"Adding your model to Unbox! Check out the project page to have a look.")
        return Model(modeldata)

    def add_dataset(
        self,
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
        commit_message: Optional[str] = None,
        project_id: str = None,
    ) -> Dataset:
        r"""Uploads a dataset to the Unbox platform (from a csv).

        Parameters
        ----------
        file_path : str
            Path to the csv file containing the dataset.
        class_names : List[str]
            List of class names indexed by label integer in the dataset.
            E.g. `[negative, positive]` when `[0, 1]` are in your label column.
        label_column_name : str
            Column header in the csv containing the labels.

            .. important::
                The labels in this column must be zero-indexed integer values.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        text_column_name : str, default None
            Column header in the csv containing the input text. Only applicable if your
            ``task_type`` is :obj:`TaskType.TextClassification`.
        categorical_feature_names : List[str], default []
            A list containing the names of all categorical features in the dataset.
            E.g. `["Gender", "Geography"]`. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        tag_column_name : str, default None
            Column header in the csv containing tags you want pre-populated in Unbox.

            .. important::
                Each cell in this column must be either empty or contain a list of
                strings.

                .. csv-table::
                    :header: ..., Tags

                    ..., "['sample']"
                    ..., "['tag_one', 'tag_two']"
        language : str, default 'en'
            The language of the dataset in ISO 639-1 (alpha-2 code) format.
        sep : str, default ','
            Delimiter to use. E.g. `'\\t'`.
        commit_message : str, default None
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

        First, instantiate the client:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from unboxapi.tasks import TaskType
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        **If your project's task type is tabular classification...**

        Let's say your dataset looks like the following:

        .. csv-table::
            :header: CreditScore, Geography, Balance, Churned

            618, France, 321.92, 1
            714, Germany, 102001.22, 0
            604, Spain, 12333.15, 0

        .. important::
            The labels in your csv **must** be integers that correctly index into the
            ``class_names`` array that you define (as shown below).
            E.g. 0 => 'Retained', 1 => 'Churned'

        The variables are needed by Unbox are:

        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_feature_names = ['Geography']

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     commit_message="First commit!",
        ...     class_names=class_names,
        ...     label_column_name=label_column_name,
        ...     feature_names=feature_names,
        ...     categorical_feature_names=categorical_feature_names,
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

        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     commit_message="First commit!",
        ...     class_names=class_names,
        ...     label_column_name=label_column_name,
        ...     text_column_name=text_column_name,
        ... )
        >>> dataset.to_dict()
        """
        # ---------------------------- Schema validations ---------------------------- #
        if task_type not in [
            TaskType.TabularClassification,
            TaskType.TextClassification,
        ]:
            raise exceptions.UnboxValidationError(
                "`task_type` must be either TaskType.TabularClassification or "
                "TaskType.TextClassification. \n"
            ) from None
        dataset_schema = schemas.DatasetSchema()
        try:
            dataset_schema.load(
                {
                    "file_path": file_path,
                    "task_type": task_type.value,
                    "commit_message": commit_message,
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
        except ma.ValidationError as err:
            raise exceptions.UnboxValidationError(
                self._format_error_message(err)
            ) from None

        # --------------------------- Resource validations --------------------------- #
        exp_file_path = os.path.expanduser(file_path)
        object_name = "original.csv"
        if not os.path.isfile(exp_file_path):
            raise exceptions.UnboxResourceError(
                f"File at path `{file_path}` does not contain the dataset. \n"
            ) from None

        with open(exp_file_path, "rt") as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
            row_count = sum(1 for _ in reader)

        df = pd.read_csv(file_path, sep=sep)

        if df.isnull().values.any():
            raise exceptions.UnboxResourceError(
                context="There's an issue with the specified dataset. \n",
                message="The dataset contains null values, which is currently "
                "not supported. \n",
                mitigation="Make sure to upload a dataset without null values.",
            ) from None

        # ------------------ Resource-schema consistency validations ----------------- #
        # Label column validations
        try:
            headers.index(label_column_name)
        except ValueError:
            raise exceptions.UnboxDatasetInconsistencyError(
                f"`{label_column_name}` specified as `label_column_name` is not "
                "in the dataset. \n"
            ) from None

        dataset_classes = list(df[label_column_name].unique())
        if len(dataset_classes) > len(class_names):
            raise exceptions.UnboxDatasetInconsistencyError(
                f"There are {len(dataset_classes)} classes represented in the dataset, "
                f"but only {len(class_names)} items in your `class_names`. \n",
                mitigation=f"Make sure that there are at most {len(class_names)} "
                "classes in your dataset.",
            ) from None

        # Feature validations
        try:
            if text_column_name:
                feature_names = [text_column_name]
            for feature_name in feature_names:
                headers.index(feature_name)
        except ValueError:
            if text_column_name:
                raise exceptions.UnboxDatasetInconsistencyError(
                    f"`{text_column_name}` specified as `text_column_name` is not in "
                    "the dataset. \n"
                ) from None
            else:
                features_not_in_dataset = [
                    feature for feature in feature_names if feature not in headers
                ]
                raise exceptions.UnboxDatasetInconsistencyError(
                    f"Features {features_not_in_dataset} specified in `feature_names` "
                    "are not in the dataset. \n"
                ) from None
        # Tag column validation
        try:
            if tag_column_name:
                headers.index(tag_column_name)
        except ValueError:
            raise exceptions.UnboxDatasetInconsistencyError(
                f"`{tag_column_name}` specified as `tag_column_name` is not in "
                "the dataset. \n"
            ) from None

        # ----------------------- Subscription plan validations ---------------------- #
        if row_count > self.subscription_plan["datasetRowCount"]:
            raise exceptions.UnboxSubscriptionPlanException(
                f"The dataset your are trying to upload contains {row_count} rows, "
                "which exceeds your plan's limit of "
                f"{self.subscription_plan['datasetRowCount']}. \n"
            ) from None
        if task_type == TaskType.TextClassification:
            max_text_size = df[text_column_name].str.len().max()
            if max_text_size > 1000:
                raise exceptions.UnboxSubscriptionPlanException(
                    "The dataset you are trying to upload contains rows with "
                    f"{max_text_size} characters, which exceeds the 1000 character "
                    "limit."
                ) from None

        endpoint = f"projects/{project_id}/datasets"
        payload = dict(
            commitMessage=commit_message,
            taskType=task_type.value,
            classNames=class_names,
            labelColumnName=label_column_name,
            tagColumnName=tag_column_name,
            language=language,
            sep=sep,
            featureNames=feature_names,
            categoricalFeatureNames=categorical_feature_names,
        )
        print(
            f"Adding your dataset to Unbox! Check out the project page to have a look."
        )
        return Dataset(
            self.api.upload(
                endpoint=endpoint,
                file_path=file_path,
                object_name=object_name,
                body=payload,
            )
        )

    def add_dataframe(
        self,
        task_type: TaskType,
        df: pd.DataFrame,
        class_names: List[str],
        label_column_name: str,
        feature_names: List[str] = [],
        text_column_name: Optional[str] = None,
        categorical_feature_names: List[str] = [],
        commit_message: Optional[str] = None,
        tag_column_name: Optional[str] = None,
        language: str = "en",
        project_id: str = None,
    ) -> Dataset:
        r"""Uploads a dataset to the Unbox platform (from a pandas DataFrame).

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing your dataset.
        class_names : List[str]
            List of class names indexed by label integer in the dataset.
            E.g. `[negative, positive]` when `[0, 1]` are in your label column.
        label_column_name : str
            Column header in the dataframe containing the labels.

            .. important::
                The labels in this column must be zero-indexed integer values.
        feature_names : List[str], default []
            List of input feature names. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        text_column_name : str, default None
            Column header in the dataframe containing the input text. Only applicable if your
            ``task_type`` is :obj:`TaskType.TextClassification`.
        categorical_feature_names : List[str], default []
            A list containing the names of all categorical features in the dataframe.
            E.g. `["Gender", "Geography"]`. Only applicable if your ``task_type`` is
            :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
        commit_message : str, default None
            Commit message for this version.
        tag_column_name : str, default None
            Column header in the dataframe containing tags you want pre-populated in Unbox.

            .. important::
                Each cell in this column must be either empty or contain a list of
                strings.

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

        First, instantiate the client:

        >>> import unboxapi
        >>> client = unboxapi.UnboxClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from unboxapi.tasks import TaskType
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        **If your project's task type is tabular classification...**

        Let's say your dataframe looks like the following:

        >>> df
            CreditScore  Geography    Balance  Churned
        0           618     France     321.92        1
        1           714    Germany  102001.22        0
        2           604      Spain   12333.15        0

        .. important::
            The labels in your dataframe **must** be integers that correctly index into
            the ``class_names`` array that you define (as shown below).
            E.g. 0 => 'Retained', 1 => 'Churned'.

        The variables are needed by Unbox are:

        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_feature_names = ['Geography']

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     df=df,
        ...     commit_message="First commit!",
        ...     class_names=class_names,
        ...     feature_names=feature_names,
        ...     label_column_name=label_column_name,
        ...     categorical_feature_names=categorical_feature_names,
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

        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Unbox:

        >>> dataset = client.add_dataset(
        ...     df=df,
        ...     commit_message="First commit!",
        ...     class_names=class_names,
        ...     text_column_name=text_column_name,
        ...     label_column_name=label_column_name,
        ... )
        >>> dataset.to_dict()
        """
        # --------------------------- Resource validations --------------------------- #
        if not isinstance(df, pd.DataFrame):
            raise exceptions.UnboxValidationError(
                f"- `df` is a `{type(df)}`, but it must be of type `pd.DataFrame`. \n"
            ) from None
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            df.to_csv(file_path, index=False)
            return self.add_dataset(
                file_path=file_path,
                task_type=task_type,
                class_names=class_names,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
                commit_message=commit_message,
                tag_column_name=tag_column_name,
                language=language,
                feature_names=feature_names,
                categorical_feature_names=categorical_feature_names,
                project_id=project_id,
            )

    @staticmethod
    def _format_error_message(err) -> str:
        """Formats the error messaeges from Marshmallow"""
        error_msg = ""
        for input, msg in err.messages.items():
            if input == "_schema":
                temp_msg = "\n- ".join(msg)
                error_msg += f"- {temp_msg} \n"
            elif not isinstance(msg, dict):
                temp_msg = msg[0].lower()
                error_msg += f"- `{input}` {temp_msg} \n"
            else:
                temp_msg = list(msg.values())[0][0].lower()
                error_msg += f"- `{input}` contains items that are {temp_msg} \n"
        return error_msg
