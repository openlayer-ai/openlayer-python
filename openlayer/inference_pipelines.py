"""Module for the InferencePipeline class.
"""


class InferencePipeline:
    """An object containing information about an inference pipeline
    on the Openlayer platform."""

    def __init__(self, json, upload, client, task_type):
        self._json = json
        self.id = json["id"]
        self.project_id = json["projectId"]
        self.upload = upload
        self.client = client
        # pylint: disable=invalid-name
        self.taskType = task_type

    def __getattr__(self, name):
        if name in self._json:
            return self._json[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name}")

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"InferencePipeline(id={self.id})"

    def __repr__(self):
        return f"InferencePipeline({self._json})"

    def to_dict(self):
        """Returns object properties as a dict.

        Returns
        -------
        Dict with object properties.
        """
        return self._json

    def upload_reference_dataset(
        self,
        *args,
        **kwargs,
    ):
        r"""Uploads a reference dataset saved as a csv file to an inference pipeline.

        The reference dataset is used to measure drift in the inference pipeline.
        The different types of drift are measured by comparing the production data
        published to the platform with the reference dataset.

        Ideally, the reference dataset should be a representative sample of the
        training set used to train the deployed model.

        Parameters
        ----------
        file_path : str
            Path to the csv file containing the reference dataset.
        dataset_config : Dict[str, any], optional
            Dictionary containing the dataset configuration. This is not needed if
            ``dataset_config_file_path`` is provided.

            .. admonition:: What's in the dataset config?

                The dataset configuration depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        dataset_config_file_path : str
            Path to the dataset configuration YAML file. This is not needed if
            ``dataset_config`` is provided.

            .. admonition:: What's in the dataset config file?

                The dataset configuration YAML depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        Notes
        -----
        **Your dataset is in a pandas dataframe?** You can use the
        :obj:`upload_reference_dataframe` method instead.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        First, instantiate the client and retrieve an existing inference pipeline:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")
        >>>
        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        With the ``InferencePipeline`` object retrieved, you are able to upload a reference
        dataset.

        For example, if your project's task type is tabular classification and
        your dataset looks like the following:

        .. csv-table::
            :header: CreditScore, Geography, Balance, Churned

            618, France, 321.92, 1
            714, Germany, 102001.22, 0
            604, Spain, 12333.15, 0

        .. important::
            The labels in your csv **must** be integers that correctly index into the
            ``class_names`` array that you define (as shown below).
            E.g. 0 => 'Retained', 1 => 'Churned'

        Prepare the dataset config:

        >>> dataset_config = {
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }

        You can now upload this reference dataset to your project with:

        >>> inference_pipeline.upload_reference_dataset(
        ...     file_path='/path/to/dataset.csv',
        ...     dataset_config=dataset_config,
        ... )
        """
        return self.client.upload_reference_dataset(
            *args,
            inference_pipeline_id=self.id,
            task_type=self.taskType,
            **kwargs,
        )

    def upload_reference_dataframe(
        self,
        *args,
        **kwargs,
    ):
        r"""Uploads a reference dataset (a pandas dataframe) to an inference pipeline.

        The reference dataset is used to measure drift in the inference pipeline.
        The different types of drift are measured by comparing the production data
        published to the platform with the reference dataset.

        Ideally, the reference dataset should be a representative sample of the
        training set used to train the deployed model.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Dataframe containing the reference dataset.
        dataset_config : Dict[str, any], optional
            Dictionary containing the dataset configuration. This is not needed if
            ``dataset_config_file_path`` is provided.

            .. admonition:: What's in the dataset config?

                The dataset configuration depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        dataset_config_file_path : str
            Path to the dataset configuration YAML file. This is not needed if
            ``dataset_config`` is provided.

            .. admonition:: What's in the dataset config file?

                The dataset configuration YAML depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details.

        Notes
        -----
        **Your dataset is in csv file?** You can use the
        :obj:`upload_reference_dataset` method instead.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        First, instantiate the client and retrieve an existing inference pipeline:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")
        >>>
        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        With the ``InferencePipeline`` object retrieved, you are able to upload a reference
        dataset.

        For example, if your project's task type is tabular classification, your
        dataset looks like the following (stored in a pandas dataframe
        called ``df``):

        >>> df
                    CreditScore  Geography    Balance  Churned
        0               618       France       321.92     1
        1               714      Germany      102001.22   0
        2               604       Spain       12333.15    0

        .. important::
            The labels in your csv **must** be integers that correctly index into the
            ``class_names`` array that you define (as shown below).
            E.g. 0 => 'Retained', 1 => 'Churned'


        Prepare the dataset config:

        >>> dataset_config = {
        ...     'classNames': ['Retained', 'Churned'],
        ...     'labelColumnName': 'Churned',
        ...     'featureNames': ['CreditScore', 'Geography', 'Balance'],
        ...     'categoricalFeatureNames': ['Geography'],
        ... }

        You can now upload this reference dataset to your project with:

        >>> inference_pipeline.upload_reference_dataframe(
        ...     dataset_df=df,
        ...     dataset_config_file_path=dataset_config,
        ... )
        """
        return self.client.upload_reference_dataframe(
            *args,
            inference_pipeline_id=self.id,
            task_type=self.taskType,
            **kwargs,
        )

    def stream_data(self, *args, **kwargs):
        """Streams production data to the Openlayer platform.

        Parameters
        ----------
        stream_data: Union[Dict[str, any], List[Dict[str, any]]]
            Dictionary or list of dictionaries containing the production data. E.g.,
            ``{'CreditScore': 618, 'Geography': 'France', 'Balance': 321.92}``.
        stream_config : Dict[str, any], optional
            Dictionary containing the stream configuration. This is not needed if
            ``stream_config_file_path`` is provided.

            .. admonition:: What's in the config?

                The configuration for a stream of data depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/tabular-classification-dataset-config>`_
                for details. These configurations are
                the same for development and production data.

        stream_config_file_path : str
            Path to the configuration YAML file. This is not needed if
            ``stream_config`` is provided.

            .. admonition:: What's in the config file?

                The configuration for a stream of data depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/tabular-classification-dataset-config>`_
                for details. These configurations are
                the same for development and production data.

        Notes
        -----
        Production data usually contains the inference timestamps. This
        column is specified in the ``timestampsColumnName`` of the stream config file,
        and it should contain timestamps in the **UNIX format in seconds**.

        Production data also usually contains the prediction IDs. This
        column is specified in the ``inferenceIdColumnName`` of the stream config file.
        This column is particularly important when the ground truths are not available
        during inference time, and they are updated later.

        If the above are not provided, **Openlayer will generate inference IDs and use
        the current time as the inference timestamp**.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/set-up-monitoring>`_.

        First, instantiate the client and retrieve an existing inference pipeline:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")
        >>>
        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        With the ``InferencePipeline`` object retrieved, you can stream
        production data -- in this example, stored in a dictionary called
        ``stream_data`` -- with:

        >>> inference_pipeline.stream_data(
        ...     stream_data=stream_data,
        ...     stream_config=config,
        ... )
        """
        return self.client.stream_data(
            *args,
            inference_pipeline_id=self.id,
            task_type=self.taskType,
            **kwargs,
        )

    def publish_batch_data(self, *args, **kwargs):
        """Publishes a batch of production data to the Openlayer platform.

        Parameters
        ----------
        batch_df : pd.DataFrame
            Dataframe containing the batch of production data.
        batch_config : Dict[str, any], optional
            Dictionary containing the batch configuration. This is not needed if
            ``batch_config_file_path`` is provided.

            .. admonition:: What's in the config?

                The configuration for a batch of data depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details. These configurations are
                the same for development and batches of production data.

        batch_config_file_path : str
            Path to the configuration YAML file. This is not needed if
            ``batch_config`` is provided.

            .. admonition:: What's in the config file?

                The configuration for a batch of data depends on the :obj:`TaskType`.
                Refer to the `How to write dataset configs guides <https://docs.openlayer.com/docs/how-to-guides/write-dataset-configs>`_
                for details. These configurations are
                the same for development and batches of production data.

        Notes
        -----
        Production data usually has a column with the inference timestamps. This
        column is specified in the ``timestampsColumnName`` of the batch config file,
        and it should contain timestamps in the **UNIX format in seconds**.

        Production data also usually has a column with the prediction IDs. This
        column is specified in the ``inferenceIdColumnName`` of the batch config file.
        This column is particularly important when the ground truths are not available
        during inference time, and they are updated later.

        If the above are not provided, **Openlayer will generate inference IDs and use
        the current time as the inference timestamp**.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        First, instantiate the client and retrieve an existing inference pipeline:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")
        >>>
        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        With the ``InferencePipeline`` object retrieved, you can publish a batch
        of production data -- in this example, stored in a pandas dataframe
        called ``df`` -- with:

        >>> inference_pipeline.publish_batch_data(
        ...     batch_df=df,
        ...     batch_config=config,
        ... )
        """
        return self.client.publish_batch_data(
            *args,
            inference_pipeline_id=self.id,
            task_type=self.taskType,
            **kwargs,
        )

    def publish_ground_truths(self, *args, **kwargs):
        """
        (Deprecated since version 0.1.0a21.)

        .. deprecated:: 0.1.0a21

            Use :obj:`update_data` instead.
        """
        return self.client.publish_ground_truths(
            *args,
            inference_pipeline_id=self.id,
            **kwargs,
        )

    def update_data(self, *args, **kwargs):
        """Updates values for data already on the Openlayer platform.

        This method is frequently used to upload the ground truths of production data
        that was already published without them. This is useful when the ground truths are not
        available during inference time, but they shall be update later to enable
        performance metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing ground truths.

            The df must contain a column with the inference IDs, and another column
            with the ground truths.

        ground_truth_column_name : Optional[str]
            Name of the column containing the ground truths. Optional, defaults to
            ``None``.

        inference_id_column_name : str
            Name of the column containing the inference IDs. The inference IDs are
            used to match the ground truths with the production data already published.

        Examples
        --------
        **Related guide**: `How to set up monitoring <https://docs.openlayer.com/docs/how-to-guides/set-up-monitoring>`_.

        Let's say you have a batch of production data already published to the
        Openlayer platform (with the method :obj:`publish_batch_data`). Now, you want
        to update the ground truths of this batch.

        First, instantiate the client and retrieve an existing inference pipeline:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")
        >>>
        >>> inference_pipeline = project.load_inference_pipeline(
        ...     name="XGBoost model inference pipeline",
        ... )

        If your ``df`` with the ground truths looks like the following:

        >>> df
                    inference_id  label
        0             d56d2b2c      0
        1             3b0b2521      1
        2             8c294a3a      0

        You can publish the ground truths with:

        >>> inference_pipeline.update_data(
        ...     df=df,
        ...     inference_id_column_name='inference_id',
        ...     ground_truth_column_name='label',
        ... )
        """
        return self.client.update_data(
            *args,
            inference_pipeline_id=self.id,
            **kwargs,
        )
