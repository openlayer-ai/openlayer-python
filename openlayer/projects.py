"""Module for the Project class.
"""

from . import tasks


class Project:
    """An object containing information about a project on the Openlayer platform."""

    def __init__(self, json, upload, client, subscription_plan=None):
        self._json = json
        self.id = json["id"]
        self.upload = upload
        self.subscription_plan = subscription_plan
        self.client = client

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Project(id={self.id})"

    def __repr__(self):
        return f"Project({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json

    def add_model(
        self,
        *args,
        **kwargs,
    ):
        """Adds a model to a project's staging area.

        This is the method for every model upload, regardless of whether you want to add a shell model,
        a full model, or a direct-to-API model (for LLMs-only).

        Refer to the `Knowledge base guide on model upload <https://docs.openlayer.com/docs/knowledge-base/development/versioning#adding-models>`_ to
        learn more about the differences between these options.

        Parameters
        ----------
        model_config : Dict[str, any]
            Dictionary containing the model configuration. This is not needed if
            ``model_config_file_path`` is provided.

            .. admonition:: What's in the model config dict?

                The model configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write model configs <https://docs.openlayer.com/docs/how-to-guides/write-model-configs>`_
                guide for details.

        model_config_file_path : str
            Path to the model configuration YAML file. This is not needed if
            ``model_config`` is provided.

            .. admonition:: What's in the model config file?

                The model configuration YAML depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write model configs <https://docs.openlayer.com/docs/how-to-guides/write-model-configs>`_
                guide for details.

        model_package_dir : str, default None
            Path to the directory containing the model package. **Only needed if you are
            interested in adding a full model.**

            .. admonition:: What's in the `model_package_dir`?

                The model package directory must contain the following files:

                - ``prediction_interface.py``
                    The prediction interface file.
                - ``model artifacts``
                    The model artifacts. This can be a single file, multiple files or a directory.
                    The model artifacts must be compatible with the
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
                The ``sample_data`` must be a dataframe with at least two rows.
        force : bool
            If :obj:`add_model` is called when there is already a model in the staging area,
            when ``force=True``, the existing staged model will be overwritten by the new
            one. When ``force=False``, the user will be prompted to confirm the
            overwrite.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        First, instantiate the client:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
        >>>
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        Let’s say you have a tabular classification project and your dataset looks
        like the following:

        >>> df
            CreditScore  Geography    Balance    PredictionScores
        0           618     France     321.92      [0.1, 0.9]
        1           714    Germany  102001.22      [0.7, 0.3]
        2           604      Spain   12333.15      [0.2, 0.8]
        ..          ...        ...        ...

        **If you want to add a shell model...**

        Prepare the model config:

        >>> model_config = {
        ...     "metadata": {  # Can add anything here, as long as it is a dict
        ...         "model_type": "Gradient Boosting Classifier",
        ...         "regularization": "None",
        ...         "encoder_used": "One Hot",
        ...     },
        ...     "classNames": class_names,
        ...     "featureNames": feature_names,
        ...     "categoricalFeatureNames": categorical_feature_names,
        ... }

        .. admonition:: What's in the model config?

                The model configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write model configs guides <https://docs.openlayer.com/docs/how-to-guides/write-model-configs>`_
                for details.

        Then, you can add the model to the project with:

        >>> project.add_model(
        ...     model_config=model_config,
        ... )

        **If you want to add a full model...**

        Prepare the model config and the model package directory. Refer to the
        `Examples gallery GitHub repository for code examples <https://github.com/openlayer-ai/examples-gallery>`_.

        You can then add the model to the project with:

        Then, you can add the model to the project with:

        >>> project.add_model(
        ...     model_config=model_config,
        ...     model_package_dir="path/to/model/package")
        ...     sample_data=df.loc[:5],
        ... )

        After adding the model to the project, it is staged, waiting to
        be committed and pushed to the platform.

        You can check what's on
        your staging area with :obj:`status`. If you want to push the model
        right away with a commit message, you can use the :obj:`commit` and
        :obj:`push` methods:

        >>> project.commit("Initial model commit.")
        >>> project.push()
        """
        return self.client.add_model(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_baseline_model(
        self,
        *args,
        **kwargs,
    ):
        """Adds a baseline model to the project."""
        return self.client.add_baseline_model(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataset(
        self,
        *args,
        **kwargs,
    ):
        r"""Adds a dataset (csv file) to a project's staging area.

        Parameters
        ----------
        file_path : str
            Path to the dataset csv file.
        dataset_config: Dict[str, any]
            Dictionary containing the dataset configuration. This is not needed if
            ``dataset_config_file_path`` is provided.

            .. admonition:: What's in the dataset config?

                The dataset configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        dataset_config_file_path : str
            Path to the dataset configuration YAML file. This is not needed if
            ``dataset_config`` is provided.

            .. admonition:: What's in the dataset config file?

                The dataset configuration YAML depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        force : bool
            If :obj:`add_dataset` is called when there is already a dataset of the same
            type in the staging area, when ``force=True``, the existing staged dataset
            will be overwritten by the new one. When ``force=False``, the user will
            be prompted to confirm the overwrite first.

        Notes
        -----
        **Your dataset is in a pandas dataframe?** You can use the
        :obj:`add_dataframe` method instead.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        First, instantiate the client:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
        >>>
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        Let's say you have a tabular classification project and your dataset looks like
        the following:

        .. csv-table::
            :header: CreditScore, Geography, Balance, Churned

            618, France, 321.92, 1
            714, Germany, 102001.22, 0
            604, Spain, 12333.15, 0

        Prepare the dataset config:

        >>> dataset_config = {
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'label': 'training',  # or 'validation'
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }

        .. admonition:: What's in the dataset config?

                The dataset configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        You can now add this dataset to your project with:

        >>> project.add_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     dataset_config=dataset_config,
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform.

        You can check what's on your staging area with :obj:`status`. If you want to
        push the dataset right away with a commit message, you can use the
        :obj:`commit` and :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()
        """
        return self.client.add_dataset(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def add_dataframe(self, *args, **kwargs):
        r"""Adds a dataset (Pandas dataframe) to a project's staging area.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Dataframe with your dataset.
        dataset_config: Dict[str, any]
            Dictionary containing the dataset configuration. This is not needed if
            ``dataset_config_file_path`` is provided.

            .. admonition:: What's in the dataset config?

                The dataset configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        dataset_config_file_path : str
            Path to the dataset configuration YAML file. This is not needed if
            ``dataset_config`` is provided.

            .. admonition:: What's in the dataset config file?

                The dataset configuration YAML depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        force : bool
            If :obj:`add_dataset` is called when there is already a dataset of the same
            type in the staging area, when ``force=True``, the existing staged dataset
            will be overwritten by the new one. When ``force=False``, the user will
            be prompted to confirm the overwrite first.

        Notes
        -----
        **Your dataset is in csv file?** You can use the
        :obj:`add_dataset` method instead.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        First, instantiate the client:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')

        Create a project if you don't have one:

        >>> from openlayer.tasks import TaskType
        >>>
        >>> project = client.create_project(
        ...     name="Churn Prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first project!",
        ... )

        If you already have a project created on the platform:

        >>> project = client.load_project(name="Your project name")

        Let's say you have a tabular classification project and your dataset looks like
        the following:

        >>> df
                    CreditScore  Geography    Balance  Churned
        0               618       France       321.92     1
        1               714      Germany      102001.22   0
        2               604       Spain       12333.15    0

        Prepare the dataset config:

        >>> dataset_config = {
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'label': 'training',  # or 'validation'
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }

        .. admonition:: What's in the dataset config?

                The dataset configuration depends on the project's :obj:`tasks.TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        You can now add this dataset to your project with:

        >>> project.add_dataset(
        ...     dataset_df=df,
        ...     dataset_config=dataset_config,
        ... )

        After adding the dataset to the project, it is staged, waiting to
        be committed and pushed to the platform.

        You can check what's on your staging area with :obj:`status`. If you want to
        push the dataset right away with a commit message, you can use the
        :obj:`commit` and :obj:`push` methods:

        >>> project.commit("Initial dataset commit.")
        >>> project.push()
        """
        return self.client.add_dataframe(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def commit(self, *args, **kwargs):
        """Adds a commit message to staged resources.

        Parameters
        ----------
        message : str
            The commit message, between 1 and 140 characters.
        force : bool
            If :obj:`commit` is called when there is already a commit message for the
            staging area, when ``force=True``, the existing message
            will be overwritten by the new one. When ``force=False``, the user will
            be prompted to confirm the overwrite first.

        Notes
        -----
        - To use this method, you must first add a model and/or dataset to the staging
            area using one of the ``add_*`` methods (e.g., :obj:`add_model`, :obj:`add_dataset`, :obj:`add_dataframe`).

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        A commit message is associated with a project version. The commit message is
        supposed to be a short description of the changes made from one version to
        the next.

        Let's say you have a project with a model and a dataset staged. You can confirm
        these resources are indeed in the staging area using the :obj:`status` method:

        >>> project.status()

        Now, you can add a commit message to the staged resources.

        >>> project.commit("Initial commit.")

        After adding the commit message, the resources are ready to be pushed to the
        platform. Use the :obj:`push` method to do so:

        >>> project.push()
        """
        return self.client.commit(*args, project_id=self.id, **kwargs)

    def push(self, *args, **kwargs):
        """Pushes the commited resources to the platform.

        Returns
        -------
        :obj:`ProjectVersion`
            An object that is used to check for upload progress and test statuses.
            Also contains other useful information about a project version.

        Notes
        -----
        - To use this method, you must first have committed your changes with the :obj:`commit` method.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        Let's say you have a project with a model and a dataset staged and committed.
        You can confirm these resources are indeed in the staging area using the
        :obj:`status` method:

        >>> project.status()

        You should see the staged resources as well as the commit message associated
        with them.

        Now, you can push the resources to the platform with:

        >>> project.push()
        """
        return self.client.push(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def export(self, *args, **kwargs):
        """Exports the commit bundle as a tarfile to the location specified
        by ``destination_dir``.

        Parameters
        ----------
        destination_dir : str
            Directory path to where the project's staging area should be exported.

        Notes
        -----
        - To use this method, you must first have committed your changes with the :obj:`commit` method.

        Examples
        --------
        Let's say you have a project with a model and a dataset staged and committed.
        You can confirm these resources are indeed in the staging area using the
        :obj:`status` method:

        >>> project.status()

        You should see the staged resources as well as the commit message associated
        with them.

        Now, you can export the resources to a speficied location with:

        >>> project.export(destination_dir="/path/to/destination")
        """
        return self.client.export(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def status(self, *args, **kwargs):
        """Shows the state of the staging area.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

        You can use the :obj:`status` method to check the state of the staging area.

        >>> project.status()

        The staging area can be in one of three states.

        You can have a clean staging area, which is the initial state as well as the
        state after you have pushed your changes to the platform
        (with the :obj:`push` method).

        You can have a staging area with different resources staged (e.g., models and
        datasets added with the :obj:`add_model`, :obj:`add_dataset`, and
        :obj:`add_dataframe` mehtods).

        Finally, you can have a staging area with resources staged and committed
        (with the :obj:`commit` method).
        """
        return self.client.status(*args, project_id=self.id, **kwargs)

    def restore(self, *args, **kwargs):
        """Removes the resources specified from the staging area.

        Parameters
        ----------
        *resource_names : str
            The names of the resources to restore, separated by comma. Valid resource
            names are ``"model"``, ``"training"``, and ``"validation"``.

            .. important::
                To see the names of the resources staged, use the :obj:`status` method.

        Examples
        --------
        **Related guide**: `How to upload datasets and models for development <https://docs.openlayer.com/docs/how-to-guides/upload-datasets-and-models>`_.

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
        return self.client.restore(*args, project_id=self.id, **kwargs)

    def create_inference_pipeline(self, *args, **kwargs):
        """Creates an inference pipeline in an Openlayer project.

        An inference pipeline represents a model that has been deployed in production.

        Parameters
        ----------
        name : str
            Name of your inference pipeline. If not specified, the name will be
            set to ``"production"``.

            .. important::
                The inference pipeline name must be unique within a project.

        description : str, optional
            Inference pipeline description. If not specified, the description will be
            set to ``"Monitoring production data."``.
        reference_df : pd.DataFrame, optional
            Dataframe containing your reference dataset. It is optional to provide the
            reference dataframe during the creation of the inference pipeline. If you
            wish, you can add it later with the
            :obj:`InferencePipeline.upload_reference_dataframe` or
            :obj:`InferencePipeline.upload_reference_dataset` methods. Not needed if
            ``reference_dataset_file_path`` is provided.
        reference_dataset_file_path : str, optional
            Path to the reference dataset CSV file. It is optional to provide the
            reference dataset file path during the creation of the inference pipeline.
            If you wish, you can add it later with the
            :obj:`InferencePipeline.upload_reference_dataframe`
            or :obj:`InferencePipeline.upload_reference_dataset` methods.
            Not needed if ``reference_df`` is provided.
        reference_dataset_config : Dict[str, any], optional
            Dictionary containing the reference dataset configuration. This is not
            needed if ``reference_dataset_config_file_path`` is provided.
        reference_dataset_config_file_path : str, optional
            Path to the reference dataset configuration YAML file. This is not needed
            if ``reference_dataset_config`` is provided.

        Returns
        -------
        InferencePipeline
            An object that is used to interact with an inference pipeline on the
            Openlayer platform.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        Instantiate the client and retrieve an existing project:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(
        ...     name="Churn prediction"
        ... )

        With the Project object retrieved, you are able to create an inference pipeline:

        >>> inference_pipeline = project.create_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ...     description="Online model deployed to SageMaker endpoint.",
        ... )


        With the InferencePipeline object created, you are able to upload a reference
        dataset (used to measure drift) and to publish production data to the Openlayer
        platform. Refer to :obj:`InferencePipeline.upload_reference_dataset` and
        :obj:`InferencePipeline.publish_batch_data` for detailed examples."""
        return self.client.create_inference_pipeline(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )

    def load_inference_pipeline(self, *args, **kwargs):
        """Loads an existing inference pipeline from an Openlayer project.

        Parameters
        ----------
        name : str, optional
            Name of the inference pipeline to be loaded.
            The name of the inference piepline is the one displayed on the
            Openlayer platform. If not specified, will try to load the
            inference pipeline named ``"production"``.

            .. note::
                If you haven't created the inference pipeline yet, you should use the
                :obj:`create_inference_pipeline` method.

        Returns
        -------
        InferencePipeline
            An object that is used to interact with an inference pipeline on the
            Openlayer platform.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        Instantiate the client and load a project:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")

        With the Project object retrieved, you are able to load the inference pipeline:

        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        With the InferencePipeline object created, you are able to upload a reference
        dataset (used to measure drift) and to publish production data to the Openlayer
        platform. Refer to :obj:`InferencePipeline.upload_reference_dataset` and
        :obj:`InferencePipeline.publish_batch_data` for detailed examples.
        """
        return self.client.load_inference_pipeline(
            *args, project_id=self.id, task_type=tasks.TaskType(self.taskType), **kwargs
        )
