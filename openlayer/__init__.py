import os
import shutil
import tarfile
import tempfile
import time
import uuid
from typing import List, Optional

import pandas as pd
import yaml

from . import api, exceptions, utils, validators
from .projects import Project
from .schemas import DatasetSchema, ModelSchema, ShellModelSchema
from .tasks import TaskType
from .version import __version__  # noqa: F401

OPENLAYER_DIR = os.path.join(os.path.expanduser("~"), ".openlayer")


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
        # self.subscription_plan = self.api.get_request("me/subscription-plan")

        if not os.path.exists(OPENLAYER_DIR):
            os.makedirs(OPENLAYER_DIR)

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
            An object that is used to add models and datasets to the Openlayer platform
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

        With the Project object created, you are able to start adding models and
        datasets to it. Refer to :obj:`add_model` and :obj:`add_dataset` or
        :obj:`add_dataframe` for detailed examples.
        """
        # Validate project
        project_config = {
            "name": name,
            "description": description,
            "task_type": task_type,
        }
        project_validator = validators.ProjectValidator(project_config=project_config)
        failed_validations = project_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the project. \n"
                "Make sure to fix all of the issues listed above before creating it.",
            ) from None

        endpoint = "projects"
        payload = dict(name=name, description=description, taskType=task_type.value)
        project_data = self.api.post_request(endpoint, body=payload)

        project = Project(project_data, self.api.upload, self)

        # Check if the staging area exists
        project_dir = os.path.join(OPENLAYER_DIR, f"{project.id}/staging")
        os.makedirs(project_dir)

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
            An object that is used to add models and datasets to the Openlayer platform
            that also contains information about the project.

        Examples
        --------
        Instantiate the client and load the project:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")

        With the Project object loaded, you are able to add models and datasets to
        the it. Refer to :obj:`add_model` and :obj:`add_dataset` or
        :obj:`add_dataframe` for detailed examples.
        """
        endpoint = f"projects?name={name}"
        project_data = self.api.get_request(endpoint)
        if len(project_data["items"]) == 0:
            raise exceptions.OpenlayerResourceNotFound(
                f"Project with name {name} not found."
            )
        project = Project(project_data["items"][0], self.api.upload, self)

        # Create the project staging area, if it doesn't yet exist
        project_dir = os.path.join(OPENLAYER_DIR, f"{project.id}/staging")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

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
            An object that is used to add models and datasets to the Openlayer platform
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

        With the Project object, you are able to start adding models and
        datasets to it. Refer to :obj:`add_model` and :obj:`add_dataset` or
        :obj:`add_dataframe` for detailed examples.
        """
        try:
            return self.create_project(
                name=name, task_type=task_type, description=description
            )
        except exceptions.OpenlayerDuplicateTask:
            return self.load_project(name)

    def add_model(
        self,
        model_config_file_path: str,
        task_type: TaskType,
        model_package_dir: Optional[str] = None,
        sample_data: Optional[pd.DataFrame] = None,
        force: bool = False,
        project_id: str = None,
    ):
        """Adds a model to a project's staging area.

        Parameters
        ----------
        model_config_file_path : str
            Path to the model configuration YAML file.

            .. admonition:: What's on the model config file?

                The content of the YAML file depends on whether you are adding a shell
                model or a model package.

                **If you are adding a shell model**, the model configuration file
                must contain the following fields:

                - ``name`` : str
                    Name of the model.
                - ``architectureType`` : str
                    The model's framework. Must be one of the supported frameworks
                    on :obj:`ModelType`.
                - ``metadata`` : Dict[str, any], default {}
                    Dictionary containing metadata about the model. This is the
                    metadata that will be displayed on the Openlayer platform.

                **Alternatively, if you are adding a model package** (i.e., with model
                artifacts and prediction interface), the model configuration file must
                contain the following fields:

                - ``name`` : str
                    Name of the model.
                - ``architectureType`` : str
                    The model's framework. Must be one of the supported frameworks
                    on :obj:`ModelType`.
                - ``classNames`` : List[str]
                    List of class names corresponding to the outputs of your predict function.
                    E.g. ``['positive', 'negative']``.
                - ``featureNames`` : List[str], default []
                    List of input feature names. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``categoricalFeatureNames`` : List[str], default []
                    A list containing the names of all categorical features used by the model.
                    E.g. ``["Gender", "Geography"]``. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``metadata`` : Dict[str, any], default {}
                    Dictionary containing metadata about the model. This is the metadata that
                    will be displayed on the Openlayer platform.


        model_package_dir : str, default None
            Path to the directory containing the model package. **Only needed if you are
            interested in adding the model's artifacts.**

            .. admonition:: What's inside `model_package_dir`?

                The model package directory must contain the following files:

                - ``prediction_interface.py``
                    The prediction interface file.
                - ``model artifacts``
                    The model artifacts. This can be a single file or a directory containing
                    multiple files. The model artifacts must be compatible with the
                    prediction interface file.
                - ``requirements.txt``
                    The requirements file. This file contains the dependencies needed to run
                    the prediction interface file.

                For instructions on how to create a model package, refer to
                the documentation.

        sample_data : pd.DataFrame, default None
            Sample data that can be run through the model. **Only needed if  model_package_dir
            is not None**. This data is used to ensure
            the model's prediction interface is compatible with the Openlayer platform.

            .. important::
                The sample_data must be a dataframe with at least two rows.
        force : bool
            If :obj:`add_model` is called when there is already a model in the staging area,
            when ``force=True``, the existing staged model will be overwritten by the new
            one. When ``force=False``, the user will be prompted to confirm the
            overwrite.

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

        **If your project's task type is tabular classification...**

        Let's say your model expects to receive a dataset looks like
        the following as input:

        >>> df
            CreditScore  Geography    Balance
        0           618     France     321.92
        1           714    Germany  102001.22
        2           604      Spain   12333.15
        ..          ...        ...        ...

        Then, you can add the model to the project with:

        >>> project.add_model(
        ...     model_config_file_path="path/to/model/config",
        ...     model_package_dir="path/to/model/package")
        ...     sample_data=df.iloc[:5, :],
        ... )

        After adding the model to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the model
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial model commit.")
        >>> project.push()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        >>> df
                                      Text
        0    I have had a long weekend
        1    I'm in a fantastic mood today
        2    Things are looking up
        ..                             ...

        Then, you can add the model to the project with:

        >>> project.add_model(
        ...     model_config_file_path="path/to/model/config",
        ...     model_package_dir="path/to/model/package")
        ...     sample_data=df.iloc[:5, :],
        ... )

        After adding the model to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the model
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial model commit.")
        >>> project.push()
        """
        # Basic argument combination checks
        if (model_package_dir is not None and sample_data is None) or (
            model_package_dir is None and sample_data is not None
        ):
            raise ValueError(
                "Both `model_package_dir` and `sample_data` must be provided together to"
                " add a model with its artifacts to the platform."
            )
        if sample_data is not None:
            if not isinstance(sample_data, pd.DataFrame):
                raise ValueError(
                    "The sample data must be a pandas DataFrame with at least 2 rows."
                )
            elif len(sample_data) < 2:
                raise ValueError(
                    "The sample data must contain at least 2 rows, but only"
                    f"{len(sample_data)} rows were provided."
                )

        # Validate model package
        model_package_validator = validators.ModelValidator(
            model_package_dir=model_package_dir,
            model_config_file_path=model_config_file_path,
            sample_data=sample_data,
        )
        failed_validations = model_package_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the model package. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        # Load model config and augment with defaults
        model_config = utils.read_yaml(model_config_file_path)
        if model_package_dir:
            model_data = ModelSchema().load(model_config)
        else:
            model_data = ShellModelSchema().load(model_config)

        # Copy relevant resources to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            if model_package_dir:
                shutil.copytree(model_package_dir, temp_dir, dirs_exist_ok=True)
                utils.write_python_version(temp_dir)

            utils.write_yaml(model_data, f"{temp_dir}/model_config.yaml")

            self._stage_resource(
                resource_name="model",
                resource_dir=temp_dir,
                project_id=project_id,
                force=force,
            )

    def add_dataset(
        self,
        file_path: str,
        dataset_config_file_path: str,
        task_type: TaskType,
        project_id: str = None,
        force: bool = False,
    ):
        r"""Adds a dataset to a project's staging area (from a csv).

        Parameters
        ----------
        file_path : str
            Path to the csv file containing the dataset.
        dataset_config_file_path : str
            Path to the dataset configuration YAML file.

            .. admonition:: What's on the dataset config file?

                The YAML file with the dataset config must have the following fields:

                - ``columnNames`` : List[str]
                    List of the dataset's column names.
                - ``classNames`` : List[str]
                    List of class names indexed by label integer in the dataset.
                    E.g. ``[negative, positive]`` when ``[0, 1]`` are in your label column.
                - ``labelColumnName`` : str
                    Column header in the csv containing the labels.

                    .. important::
                        The labels in this column must be zero-indexed integer values.
                - ``label`` : str
                    Type of dataset. E.g. ``'training'`` or
                    ``'validation'``.
                - ``featureNames`` : List[str], default []
                    List of input feature names. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``textColumnName`` : str, default None
                    Column header in the csv containing the input text. Only applicable if
                    your ``task_type`` is :obj:`TaskType.TextClassification`.
                - ``predictionsColumnName`` : str, default None
                    Column header in the csv containing the predictions. Only applicable if you
                    are uploading a model as well with the :obj:`add_model` method.

                    .. important::
                        Each cell in this column must contain a list of
                        class probabilities. For example, for a binary classification
                        task, the cell values should look like this:
                        .. csv-table::
                            :header: ..., predictions
                            ..., "[0.6650292861587155, 0.3349707138412845]"
                            ..., "[0.8145561636482788, 0.18544383635172124]"

                - ``categoricalFeatureNames`` : List[str], default []
                    A list containing the names of all categorical features in the dataset.
                    E.g. ``["Gender", "Geography"]``. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``language`` : str, default 'en'
                    The language of the dataset in ISO 639-1 (alpha-2 code) format.
                - ``sep`` : str, default ','
                    Delimiter to use. E.g. `'\\t'`.
        force : bool
            If :obj:`add_dataset` is called when there is already a dataset of the same type
            in the staging area, when ``force=True``, the existing staged dataset will be overwritten
            by the new one. When ``force=False``, the user will be prompted to confirm the overwrite.

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

        Write the dataset config YAML file with the variables are needed by Openlayer:

        >>> import yaml
        >>>
        >> dataset_config = {
        ...     'columnNames': ['CreditScore', 'Geography', 'Balance', 'Churned'],
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'label': 'training',  # or 'validation'
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }
        >>>
        >>> with open('/path/to/dataset_config.yaml', 'w') as f:
        ...     yaml.dump(dataset_config, f)

        You can now add this dataset to your project with:

        >>> project.add_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     dataset_config_file_path='/path/to/dataset_config.yaml',
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the dataset
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        .. csv-table::
            :header: Text, Sentiment

            I have had a long weekend, 0
            I'm in a fantastic mood today, 1
            Things are looking up, 1

        Write the dataset config YAML file with the variables are needed by Openlayer:

        >>> import yaml
        >>>
        >> dataset_config = {
        ...     'columnNames': ['Text', 'Sentiment'],
        ...     'classNames': ['Negative', 'Positive'],
        ...     'labelColumnName': 'Sentiment',
        ...     'label': 'training',  # or 'validation'
        ...     'textColumnName': 'Text',
        ... }
        >>>
        >>> with open('/path/to/dataset_config.yaml', 'w') as f:
        ...     yaml.dump(dataset_config, f)

        You can now add this dataset to your project with:

        >>> project.add_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     dataset_config_file_path='/path/to/dataset_config.yaml',
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the dataset
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()
        """
        # Validate dataset
        dataset_validator = validators.DatasetValidator(
            dataset_config_file_path=dataset_config_file_path,
            dataset_file_path=file_path,
        )
        failed_validations = dataset_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the dataset and its config. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        # Load dataset config and augment with defaults
        dataset_config = utils.read_yaml(dataset_config_file_path)
        dataset_data = DatasetSchema().load(dataset_config)

        # Copy relevant resources to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy(file_path, f"{temp_dir}/dataset.csv")
            utils.write_yaml(dataset_data, f"{temp_dir}/dataset_config.yaml")

            self._stage_resource(
                resource_name=dataset_data.get("label"),
                resource_dir=temp_dir,
                project_id=project_id,
                force=force,
            )

    def add_dataframe(
        self,
        df: pd.DataFrame,
        dataset_config_file_path: str,
        task_type: TaskType,
        project_id: str = None,
        force: bool = False,
    ):
        r"""Adds a dataset to a project's staging area (from a pandas DataFrame).

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing your dataset.
        dataset_config_file_path : str
            Path to the dataset configuration YAML file.

            .. admonition:: What's on the dataset config file?

                The YAML file with the dataset config must have the following fields:

                - ``columnNames`` : List[str]
                    List of the dataset's column names.
                - ``classNames`` : List[str]
                    List of class names indexed by label integer in the dataset.
                    E.g. ``[negative, positive]`` when ``[0, 1]`` are in your label column.
                - ``labelColumnName`` : str
                    Column header in the csv containing the labels.

                    .. important::
                        The labels in this column must be zero-indexed integer values.
                - ``label`` : str
                    Type of dataset. E.g. ``'training'`` or
                    ``'validation'``.
                - ``featureNames`` : List[str], default []
                    List of input feature names. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``textColumnName`` : str, default None
                    Column header in the csv containing the input text. Only applicable if your
                    ``task_type`` is :obj:`TaskType.TextClassification`.
                - ``predictionsColumnName`` : str, default None
                    Column header in the csv containing the predictions. Only applicable if you
                    are uploading a model as well with the :obj:`add_model` method.

                    .. important::
                        Each cell in this column must contain a list of
                        class probabilities. For example, for a binary classification
                        task, the cell values should look like this:
                        .. csv-table::
                            :header: ..., predictions
                            ..., "[0.6650292861587155, 0.3349707138412845]"
                            ..., "[0.8145561636482788, 0.18544383635172124]"

                - ``categoricalFeatureNames`` : List[str], default []
                    A list containing the names of all categorical features in the dataset.
                    E.g. ``["Gender", "Geography"]``. Only applicable if your ``task_type`` is
                    :obj:`TaskType.TabularClassification` or :obj:`TaskType.TabularRegression`.
                - ``language`` : str, default 'en'
                    The language of the dataset in ISO 639-1 (alpha-2 code) format.
                - ``sep`` : str, default ','
                    Delimiter to use. E.g. `'\\t'`.
        force : bool
            If :obj:`add_dataframe` is called when there is already a dataset of the same type in the
            staging area, when ``force=True``, the existing staged dataset will be overwritten
            by the new one. When ``force=False``, the user will be prompted to confirm the overwrite.

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

        Write the dataset config YAML file with the variables are needed by Openlayer:

        >>> import yaml
        >>>
        >> dataset_config = {
        ...     'columnNames': ['CreditScore', 'Geography', 'Balance', 'Churned'],
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'label': 'training',  # or 'validation'
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }
        >>>
        >>> with open('/path/to/dataset_config.yaml', 'w') as f:
        ...     yaml.dump(dataset_config, f)

        You can now add this dataset to your project with:

        >>> project.add_dataframe(
        ...     df=df,
        ...     dataset_config_file_path='/path/to/dataset_config.yaml',
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the dataset
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()

        **If your task type is text classification...**

        Let's say your dataset looks like the following:

        >>> df
                                      Text  Sentiment
        0    I have had a long weekend              0
        1    I'm in a fantastic mood today          1
        2    Things are looking up                  1

        Write the dataset config YAML file with the variables are needed by Openlayer:

        >>> import yaml
        >>>
        >> dataset_config = {
        ...     'columnNames': ['Text', 'Sentiment'],
        ...     'classNames': ['Negative', 'Positive'],
        ...     'labelColumnName': 'Sentiment',
        ...     'label': 'training',  # or 'validation'
        ...     'textColumnName': 'Text',
        ... }
        >>>
        >>> with open('/path/to/dataset_config.yaml', 'w') as f:
        ...     yaml.dump(dataset_config, f)

        You can now add this dataset to your project with:

        >>> project.add_dataframe(
        ...     df=df,
        ...     dataset_config_file_path='/path/to/dataset_config.yaml',
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform. You can check what's on
        your staging area with :obj:`status`. If you want to push the dataset
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()
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
                project_id=project_id,
                dataset_config_file_path=dataset_config_file_path,
                force=force,
            )

    def commit(self, message: str, project_id: int, force: bool = False):
        """Adds a commit message to staged resources.

        Parameters
        ----------
        message : str
            The commit message, between 1 and 140 characters.
        force : bool
            If :obj:`commit` is called when there is already a commit message,
            when ``force=True``, the existing message will be overwritten by the new
            one. When ``force=False``, the user will be prompted to confirm the overwrite.

        Notes
        -----
        - To use this method, you must first add a model and/or dataset to the staging area using one of the ``add`` methods (e.g., :obj:`add_model`, :obj:`add_dataset`, :obj:`add_dataframe`).

        Examples
        --------
        A commit message is associated with a project version. We have a new project version every time
        any of its resources (namely, model and/or dataset) are updated. The commit message is supposed
        to be a short description of the changes from one version to the next.

        Let's say you have a project with a model and a dataset staged. You can confirm these resources
        are indeed in the staging area using the :obj:`status` method:

        >>> project.status()

        Now, you can add a commit message to the staged resources.

        >>> project.commit("Initial commit.")

        After adding the commit message, the resources are ready to be pushed to the platform. You use
        the :obj:`push` method to do so:

        >>> project.push()
        """
        # Validate commit
        commit_validator = validators.CommitValidator(commit_message=message)
        failed_validations = commit_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the commit message specified. \n"
                "Make sure to fix all of the issues listed above before committing.",
            ) from None

        project_dir = f"{OPENLAYER_DIR}/{project_id}/staging"

        if not os.listdir(project_dir):
            print(
                "There is nothing staged to commit. Please add model and/or datasets"
                " first before committing."
            )
            return

        if os.path.exists(f"{project_dir}/commit.yaml"):
            print("Found a previous commit that was not pushed to the platform.")
            overwrite = "n"

            if not force:
                with open(f"{project_dir}/commit.yaml", "r") as commit_file:
                    commit = yaml.safe_load(commit_file)
                    print(
                        f"\t - Commit message: `{commit['message']}` \n \t - Date: {commit['date']}"
                    )
                overwrite = input(
                    "Do you want to overwrite it with the current message? [y/n]: "
                )
            if overwrite.lower() == "y" or force:
                print("Overwriting commit message...")
                os.remove(f"{project_dir}/commit.yaml")

            else:
                print("Keeping the existing commit message.")
                return

        commit = dict(message=message, date=time.ctime())
        with open(f"{project_dir}/commit.yaml", "w") as commit_file:
            yaml.dump(commit, commit_file)

        print("Committed!")

    def push(self, project_id: int):
        """Pushes the commited resources to the platform.

        Notes
        -----
        - To use this method, you must first have committed your changes with the :obj:`commit`
        method.

        Examples
        --------

        Let's say you have a project with a model and a dataset staged and committed. You can
        confirm these resources are indeed in the staging area using the :obj:`status` method:

        >>> project.status()

        You should see the staged resources as well as the commit message associated with them.

        Now, you can push the resources to the platform with:

        >>> project.push()
        """
        project_dir = f"{OPENLAYER_DIR}/{project_id}/staging"

        if not os.listdir(project_dir):
            print(
                "The staging area is clean and there is nothing committed to push. "
                "Please add model and/or datasets first, and then commit before pushing."
            )
            return

        if not os.path.exists(f"{project_dir}/commit.yaml"):
            print(
                "There are resources staged, but you haven't committed them yet. "
                "Please commit before pushing"
            )
            return

        with open(f"{project_dir}/commit.yaml", "r") as commit_file:
            commit = yaml.safe_load(commit_file)

        # Validate bundle resources
        commit_bundle_validator = validators.CommitBundleValidator(
            bundle_path=project_dir
        )
        failed_validations = commit_bundle_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the staged resources. \n"
                "Make sure to fix all of the issues listed above before pushing.",
            ) from None

        print(
            "Pushing changes to the platform with the commit message: \n"
            f"\t - Message: {commit['message']} \n"
            f"\t - Date: {commit['date']}"
        )

        # Tar the project's staging area
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file_path = os.path.join(tmp_dir, "staging")
            with tarfile.open(tar_file_path, mode="w:gz") as tar:
                tar.add(project_dir, arcname=os.path.basename(project_dir))

            # Upload the tar file
            payload = {"commit": {"message": commit["message"]}}
            self.api.upload(
                endpoint=f"projects/{project_id}/versions",
                file_path=tar_file_path,
                object_name="tarfile",
                body=payload,
            )

        # Clean up the staging area
        shutil.rmtree(project_dir)
        os.makedirs(project_dir, exist_ok=True)

        print("Pushed!")

    def status(self, project_id: int):
        """Shows the state of the staging area.

        Examples
        --------

        You can use the :obj:`status` method to check the state of the staging area.

        >>> project.status()

        The staging area can be in one of three states.

        You can have a clean staging area, which is the initial state as well as the state
        after you have pushed your changes to the platform (with the :obj:`push` method).

        You can have a staging area with different resources staged (e.g., models and datasets
        added with the :obj:`add_model`, :obj:`add_dataset`, and :obj:`add_dataframe` mehtods).

        Finally, you can have a staging area with resources staged and committed (with the
        :obj:`commit` method).
        """
        project_dir = f"{OPENLAYER_DIR}/{project_id}/staging"
        valid_resource_names = ["model", "training", "validation"]

        if not os.listdir(project_dir):
            print(
                "The staging area is clean. You can stage models and/or datasets by"
                " using the corresponding `add` methods."
            )
            return

        if not os.path.exists(f"{project_dir}/commit.yaml"):
            print("The following resources are staged, waiting to be committed:")
            for file in os.listdir(project_dir):
                if file in valid_resource_names:
                    print(f"\t - {file}")
            print("Use the `commit` method to add a commit message to your changes.")
            return

        with open(f"{project_dir}/commit.yaml", "r") as commit_file:
            commit = yaml.safe_load(commit_file)
        print("The following resources are committed, waiting to be pushed:")
        for file in os.listdir(project_dir):
            if file != "commit.yaml":
                print(f"\t - {file}")
        print(f"Commit message from {commit['date']}:")
        print(f"\t {commit['message']}")
        print("Use the `push` method to push your changes to the platform.")

    def restore(self, resource_name: str, project_id: int):
        """Removes the resource specified by ``resource_name`` from the staging area.

        Parameters
        ----------
        resource_name : str
            The name of the resource to restore. Can be one of ``"model"``, ``"training"``,
            or ``"validation"``.

            .. important::
                To see the names of the resources staged, use the :obj:`status` method.

        Examples
        --------
        Let's say you have initially used the :obj:`add_model` method to add a model to the
        staging area.

        >>> project.add_model(
        ...     model_package_dir="/path/to/model/package",
        ...     sample_data=df
        ... )

        You can see the model staged with the :obj:`status` method:

        >>> project.status()

        You can then remove the model from the staging area with the :obj:`restore` method:

        >>> project.restore(resource_name="model")
        """
        project_dir = f"{OPENLAYER_DIR}/{project_id}/staging"

        if not os.path.exists(f"{project_dir}/{resource_name}"):
            print(
                f"There's no resource named `{resource_name}` in the staging area. "
                "Make sure that you are trying to restore a staged resource. "
                "To see the names of the resources staged, use the `status` method."
            )
            return

        shutil.rmtree(f"{project_dir}/{resource_name}")
        print(f"Removed resource `{resource_name}` from the staging area.")

        # Remove commit if there are no more resources staged
        if len(os.listdir(project_dir)) == 1 and os.path.exists(
            f"{project_dir}/commit.yaml"
        ):
            os.remove(f"{project_dir}/commit.yaml")

    def _stage_resource(
        self, resource_name: str, resource_dir: str, project_id: int, force: bool
    ):
        """Adds the resource specified by `resource_name` to the project's staging directory.

        Parameters
        ----------
        resource_name : str
            The name of the resource to stage. Can be one of "model", "training",
            or "validation".
        resource_dir : str
            The path from which to copy the resource.
        project_id : int
            The id of the project to which the resource should be added.
        force : bool
            Whether to overwrite the resource if it already exists in the staging area.
        """
        if resource_name not in ["model", "training", "validation"]:
            raise ValueError(
                f"Resource name must be one of 'model', 'training', or 'validation',"
                f" but got {resource_name}."
            )

        staging_dir = f"{OPENLAYER_DIR}/{project_id}/staging/{resource_name}"

        # Append 'dataset' to the end of the resource name for the prints
        if resource_name in ["training", "validation"]:
            resource_name += " dataset"

        if os.path.exists(staging_dir):
            print(f"Found an existing {resource_name} staged.")
            overwrite = "n"

            if not force:
                overwrite = input("Do you want to overwrite it? [y/n] ")
            if overwrite.lower() == "y" or force:
                print(f"Overwriting previously staged {resource_name}...")
                shutil.rmtree(staging_dir)
            else:
                print(f"Keeping the existing {resource_name} staged.")
                return

        shutil.copytree(resource_dir, staging_dir)

        print(f"Staged the {resource_name}!")
