import csv
import os
import tarfile
import tempfile
import uuid
from typing import List, Optional

import marshmallow as ma
import pandas as pd
import yaml

from . import api, exceptions, schemas, utils, validators
from .datasets import Dataset
from .models import Model
from .projects import Project
from .tasks import TaskType
from .version import __version__  # noqa: F401


class OpenlayerClient(object):
    """Client class that interacts with the Openlayer Platform.

    Parameters
    ----------
    api_key : str
        Your API key. Retrieve it from the web app.

    Examples
    --------
    Instantiate a client with your api key

    >>> import openlayer
    >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
    """

    def __init__(self, api_key: str = None):
        self.api = api.Api(api_key)
        self.subscription_plan = self.api.get_request("me/subscription-plan")

    def create_project(
        self, name: str, task_type: TaskType, description: Optional[str] = None
    ) -> Project:
        """Creates a project on the Openlayer platform.

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
            An object that is used to upload models and datasets to the Openlayer platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and create the project:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> from openlayer.tasks import TaskType
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
            raise exceptions.OpenlayerValidationError(
                self._format_error_message(err)
            ) from None

        endpoint = "projects"
        payload = dict(name=name, description=description, taskType=task_type.value)
        project_data = self.api.post_request(endpoint, body=payload)

        project = Project(project_data, self.api.upload, self.subscription_plan, self)
        print(f"Created your project. Navigate to {project.links['app']} to see it.")
        return project

    def load_project(self, name: str) -> Project:
        """Loads an existing project from the Openlayer platform.

        Parameters
        ----------
        name : str
            Name of the project to be loaded. The name of the project is the one
            displayed on the Openlayer platform.

            .. note::
                If you haven't created the project yet, you should use the
                :obj:`create_project` method.

        Returns
        -------
        Project
            An object that is used to upload models and datasets to the Openlayer platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and load the project:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
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
            An object that is used to upload models and datasets to the Openlayer platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and create or load the project:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> from openlayer.tasks import TaskType
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
        except exceptions.OpenlayerDuplicateTask:
            return self.load_project(name)

    def add_model(
        self,
        model_package_dir: str,
        task_type: TaskType,
        sample_data: pd.DataFrame = None,
        project_id: str = None,
        **kwargs,
    ) -> Model:
        """Uploads a model to the Openlayer platform.

        Parameters
        ----------
        model_package_dir : str
            Path to the directory containing the model package. For instructions on
            how to create a model package, refer to the documentation.
        sample_data : pd.DataFrame
            Sample data that can be run through the model. This data is used to ensure
            the model's prediction interface is compatible with the Openlayer platform.

        Returns
        -------
        :obj:`Model`
            An object containing information about your uploaded model.

        Examples
        --------

        .. seealso::
            Our `sample notebooks
            <https://github.com/openlayer-ai/openlayer-python/tree/main/examples>`_ and
            `tutorials <https://docs.openlayer.com/docs/overview-of-tutorial-tracks>`_.

        First, instantiate the client:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        **TODO: complete examples.**

        **If your project's task type is tabular classification...**

        Let's say your dataset looks like the following:

        >>> df
            CreditScore  Geography    Balance  Churned
        0           618     France     321.92        1
        1           714    Germany  102001.22        0
        2           604      Spain   12333.15        0
        ..          ...        ...        ...      ...


        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        >>> df
                                      Text  Sentiment
        0    I have had a long weekend              0
        1    I'm in a fantastic mood today          1
        2    Things are looking up                  1
        ..                             ...        ...

        """

        # ------------------------- Model package validations ------------------------ #
        model_package_validator = validators.ModelValidator(
            model_package_dir=model_package_dir,
            sample_data=sample_data,
        )
        failed_validations = model_package_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                context="There are issues with the model package, as specified above. \n",
                mitigation="Make sure to fix all of them before uploading the model.",
            ) from None

        # ------ Start of temporary workaround for the arguments in the payload ------ #
        model_config_file = os.path.join(model_package_dir, "model_config.yaml")

        with open(model_config_file, "r") as config_file:
            model_config = yaml.safe_load(config_file)

        name = model_config.get("name")
        model_type = model_config.get("model_type")
        class_names = model_config.get("class_names")
        feature_names = model_config.get("feature_names") or model_config.get(
            "text_column_name"
        )
        categorical_feature_names = model_config.get("categorical_feature_names")
        # ------- End of temporary workaround for the arguments in the payload ------- #

        # Prepare tar for upload
        with utils.TempDirectory() as tarfile_dir:
            tarfile_path = f"{tarfile_dir}/model"

            # Augment model package with the current env's Python version
            utils.write_python_version(model_package_dir)

            with tarfile.open(tarfile_path, mode="w:gz") as tar:
                tar.add(model_package_dir, arcname="model_package")

            # Remove the Python version file after tarring
            utils.remove_python_version(model_package_dir)

            # Make sure the resulting model package is less than 2 GB
            if float(os.path.getsize("model")) / 1e9 > 2:
                raise exceptions.OpenlayerResourceError(
                    context="There's an issue with the specified `model_package_dir`. \n",
                    message=f"The model package is too large. \n",
                    mitigation="Make sure to upload a model package with size less than 2 GB.",
                ) from None

            endpoint = f"projects/{project_id}/ml-models"

            # TODO: Re-define arguments in the payload
            payload = dict(
                name=name,
                commitMessage="Initial commit",
                architectureType=model_type,
                taskType=task_type.value,
                classNames=class_names,
                featureNames=feature_names,
                categoricalFeatureNames=categorical_feature_names,
            )

            modeldata = self.api.upload(
                endpoint=endpoint,
                file_path=tarfile_path,
                object_name="tarfile",
                body=payload,
            )

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
        r"""Uploads a dataset to the Openlayer platform (from a csv).

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
            Column header in the csv containing tags you want pre-populated in Openlayer.

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

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
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

        The variables are needed by Openlayer are:

        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_feature_names = ['Geography']

        You can now upload this dataset to Openlayer:

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

        The variables are needed by Openlayer are:

        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Openlayer:

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
            raise exceptions.OpenlayerValidationError(
                "`task_type` must be either TaskType.TabularClassification or "
                "TaskType.TextClassification. \n"
            ) from None
        dataset_schema = schemas.DatasetSchema()
        try:
            dataset_schema.load(
                {
                    "file_path": file_path,
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
            raise exceptions.OpenlayerValidationError(
                self._format_error_message(err)
            ) from None

        # --------------------------- Resource validations --------------------------- #
        exp_file_path = os.path.expanduser(file_path)
        object_name = "original.csv"
        if not os.path.isfile(exp_file_path):
            raise exceptions.OpenlayerResourceError(
                f"File at path `{file_path}` does not contain the dataset. \n"
            ) from None

        with open(exp_file_path, "rt") as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
            row_count = sum(1 for _ in reader)

        df = pd.read_csv(file_path, sep=sep)

        # Checking for null values
        if df.isnull().values.any():
            raise exceptions.OpenlayerResourceError(
                context="There's an issue with the specified dataset. \n",
                message="The dataset contains null values, which is currently "
                "not supported. \n",
                mitigation="Make sure to upload a dataset without null values.",
            ) from None

        # Validating if the labels are zero indexed ints
        unique_labels = set(df[label_column_name].unique())
        zero_indexed_set = set(range(len(class_names)))
        if unique_labels != zero_indexed_set:
            raise exceptions.OpenlayerResourceError(
                context=f"There's an issue with values in the column "
                f"`{label_column_name}` of the dataset. \n",
                message=f"The labels in `{label_column_name}` must be "
                "zero-indexed integer values. \n",
                mitigation="Make sure to upload a dataset with zero-indexed "
                "integer labels that match the list in `class_names`. "
                f"For example, the class `{class_names[0]}` should be "
                "represented as a 0 in the dataset, the class "
                f"`{class_names[1]}` should be a 1, and so on.",
            ) from None

        # Validating the column dtypes
        supported_dtypes = {"float32", "float64", "int32", "int64", "object"}
        error_msg = ""
        for col in df:
            dtype = df[col].dtype.name
            if dtype not in supported_dtypes:
                error_msg += f"- Column `{col}` is of dtype {dtype}. \n"
        if error_msg:
            raise exceptions.OpenlayerResourceError(
                context="There is an issue with some of the columns dtypes.\n",
                message=error_msg,
                mitigation=f"The supported dtypes are {supported_dtypes}. "
                "Make sure to cast the above columns to a supported dtype.",
            ) from None
        # ------------------ Resource-schema consistency validations ----------------- #
        # Label column validations
        try:
            headers.index(label_column_name)
        except ValueError:
            raise exceptions.OpenlayerDatasetInconsistencyError(
                f"`{label_column_name}` specified as `label_column_name` is not "
                "in the dataset. \n"
            ) from None

        if len(unique_labels) > len(class_names):
            raise exceptions.OpenlayerDatasetInconsistencyError(
                f"There are {len(unique_labels)} classes represented in the dataset, "
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
                raise exceptions.OpenlayerDatasetInconsistencyError(
                    f"`{text_column_name}` specified as `text_column_name` is not in "
                    "the dataset. \n"
                ) from None
            else:
                features_not_in_dataset = [
                    feature for feature in feature_names if feature not in headers
                ]
                raise exceptions.OpenlayerDatasetInconsistencyError(
                    f"Features {features_not_in_dataset} specified in `feature_names` "
                    "are not in the dataset. \n"
                ) from None
        # Tag column validation
        try:
            if tag_column_name:
                headers.index(tag_column_name)
        except ValueError:
            raise exceptions.OpenlayerDatasetInconsistencyError(
                f"`{tag_column_name}` specified as `tag_column_name` is not in "
                "the dataset. \n"
            ) from None

        # ----------------------- Subscription plan validations ---------------------- #
        if row_count > self.subscription_plan["datasetRowCount"]:
            raise exceptions.OpenlayerSubscriptionPlanException(
                f"The dataset your are trying to upload contains {row_count} rows, "
                "which exceeds your plan's limit of "
                f"{self.subscription_plan['datasetRowCount']}. \n"
            ) from None
        if task_type == TaskType.TextClassification:
            max_text_size = df[text_column_name].str.len().max()
            if max_text_size > 1000:
                raise exceptions.OpenlayerSubscriptionPlanException(
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
            f"Adding your dataset to Openlayer! Check out the project page to have a look."
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
        r"""Uploads a dataset to the Openlayer platform (from a pandas DataFrame).

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
            Column header in the dataframe containing tags you want pre-populated in Openlayer.

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

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
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

        The variables are needed by Openlayer are:

        >>> class_names = ['Retained', 'Churned']
        >>> feature_names = ['CreditScore', 'Geography', 'Balance']
        >>> label_column_name = 'Churned'
        >>> categorical_feature_names = ['Geography']

        You can now upload this dataset to Openlayer:

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

        The variables are needed by Openlayer are:

        >>> class_names = ['Negative', 'Positive']
        >>> text_column_name = 'Text'
        >>> label_column_name = 'Sentiment'

        You can now upload this dataset to Openlayer:

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
            raise exceptions.OpenlayerValidationError(
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
        """Formats the error messages from Marshmallow"""
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
