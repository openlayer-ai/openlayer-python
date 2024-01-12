"""
Openlayer Python SDK.

Defines the core OpenlayerClient class that users can use to interact
with the Openlayer platform.

Typical usage example:

    import openlayer

    client = openlayer.OpenlayerClient("YOUR_API_KEY")
    project = client.create_project("My Project")
    project.add_dataframe(
        dataset_df=training_set,
        dataset_config_file_path="training_dataset_config.yaml",
    )
    project.add_dataframe(
        dataset_df=validation_set,
        dataset_config_file_path="validation_dataset_config.yaml",
    )
    project.status()
    project.push()
"""
import os
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from . import api, constants, exceptions, utils
from .inference_pipelines import InferencePipeline
from .project_versions import ProjectVersion
from .projects import Project
from .schemas import dataset_schemas, model_schemas
from .tasks import TaskType
from .validators import (
    baseline_model_validators,
    commit_validators,
    dataset_validators,
    inference_pipeline_validators,
    model_validators,
    project_validators,
)
from .version import __version__  # noqa: F401


class OpenlayerClient(object):
    """Client class that interacts with the Openlayer Platform.

    Parameters
    ----------
    api_key : str
        Your API key. You can find your workspace API key in your
        `account settings <https://docs.openlayer.com/docs/how-to-guides/find-your-api-key>`_
        settings page.
    verbose : bool, default True
        Whether to print out success messages to the console. E.g., when data is
        successfully uploaded, a resource is staged, etc.

    Examples
    --------
    **Relevant guide**: `How to find your API keys <https://docs.openlayer.com/docs/how-to-guides/find-your-api-key>`_.

    Instantiate a client with your api key:

    >>> import openlayer
    >>>
    >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
    """

    def __init__(self, api_key: str = None, verbose: bool = True):
        self.api = api.Api(api_key)
        self.verbose = verbose

        if not os.path.exists(constants.OPENLAYER_DIR):
            os.makedirs(constants.OPENLAYER_DIR)

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
            Type of ML task for the project. E.g. :obj:`TaskType.TabularClassification`
            or :obj:`TaskType.TextClassification`.

        description : str, optional
            Project description.

        Returns
        -------
        Project
            An object that is used to interact with projects on the Openlayer platform.



        Examples
        --------
        **Related guide**: `How to create and load projects <https://docs.openlayer.com/docs/how-to-guides/creating-and-loading-projects>`_.

        Instantiate the client and create the project:

        >>> import openlayer
        >>> from openlayer.tasks import TaskType
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.create_project(
        ...     name="Churn prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first error analysis playground",
        ... )

        With the Project object, you are able to start adding models and
        datasets to it. Refer to :obj:`Project.add_model` and :obj:`Project.add_dataset` or
        :obj:`Project.add_dataframe` for detailed examples.
        """
        try:
            project = self.load_project(name)
            warnings.warn(
                f"Found an existing project with name '{name}'. Loading it instead."
            )
        except exceptions.OpenlayerResourceNotFound:
            # Validate project
            project_config = {
                "name": name,
                "description": description,
                "task_type": task_type,
            }
            project_validator = project_validators.ProjectValidator(
                project_config=project_config
            )
            failed_validations = project_validator.validate()

            if failed_validations:
                raise exceptions.OpenlayerValidationError(
                    "There are issues with the project. \n"
                    "Make sure to fix all of the issues listed above before creating it.",
                ) from None

            endpoint = "projects"
            payload = {
                "name": name,
                "description": description,
                "taskType": task_type.value,
            }
            project_data = self.api.post_request(endpoint, body=payload)

            project = Project(project_data, self.api.upload, self)

            # Check if the staging area exists
            project_dir = os.path.join(constants.OPENLAYER_DIR, f"{project.id}/staging")
            os.makedirs(project_dir)

            if self.verbose:
                print(
                    f"Created your project. Navigate to {project.links['app']} to see it."
                )
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
            An object that is used to interact with projects on the Openlayer platform.

        Examples
        --------
        **Related guide**: `How to create and load projects <https://docs.openlayer.com/docs/how-to-guides/creating-and-loading-projects>`_.

        Instantiate the client and load the project:

        >>> import openlayer
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.load_project(name="Churn prediction")

        With the Project object loaded, you are able to add models and datasets to
        the it. Refer to :obj:`Project.add_model` and :obj:`Project.add_dataset` or
        :obj:`Project.add_dataframe` for detailed examples.
        """
        endpoint = f"projects?name={name}"
        project_data = self.api.get_request(endpoint)
        if len(project_data["items"]) == 0:
            raise exceptions.OpenlayerResourceNotFound(
                f"Project with name {name} not found."
            )
        project = Project(project_data["items"][0], self.api.upload, self)

        # Create the project staging area, if it doesn't yet exist
        project_dir = os.path.join(constants.OPENLAYER_DIR, f"{project.id}/staging")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        if self.verbose:
            print(f"Found your project. Navigate to {project.links['app']} to see it.")
        return project

    def create_or_load_project(
        self, name: str, task_type: TaskType, description: Optional[str] = None
    ) -> Project:
        """Convenience function that either creates or loads a project.

        If a project with the ``name`` specified already exists, it will be loaded.
        Otherwise, a new project will be created.

        Parameters
        ----------
        name : str
            Name of your project.

            .. important::
                The project name must be unique in a user's collection of projects.

        task_type : :obj:`TaskType`
            Type of ML task for the project. E.g. :obj:`TaskType.TabularClassification`
            or :obj:`TaskType.TextClassification`.

        description : str, optional
            Project description.

        Returns
        -------
        Project
            An object that is used to interact with projects on the Openlayer platform.

        Examples
        --------
        **Related guide**: `How to create and load projects <https://docs.openlayer.com/docs/how-to-guides/creating-and-loading-projects>`_.

        Instantiate the client and create or load the project:

        >>> import openlayer
        >>> from openlayer.tasks import TaskType
        >>>
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> project = client.create_or_load_project(
        ...     name="Churn prediction",
        ...     task_type=TaskType.TabularClassification,
        ...     description="My first error analysis playground",
        ... )

        With the Project object, you are able to start adding models and
        datasets to it. Refer to :obj:`Project.add_model` and :obj:`Project.add_dataset`
        or :obj:`Project.add_dataframe` for detailed examples.
        """
        try:
            return self.load_project(name)
        except exceptions.OpenlayerResourceNotFound:
            return self.create_project(
                name=name, task_type=task_type, description=description
            )

    def add_model(
        self,
        task_type: TaskType,
        model_config: Optional[Dict[str, any]] = None,
        model_config_file_path: Optional[str] = None,
        model_package_dir: Optional[str] = None,
        sample_data: Optional[pd.DataFrame] = None,
        force: bool = False,
        project_id: str = None,
    ):
        """Adds a model to a project's staging area."""
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
        if model_config is None and model_config_file_path is None:
            raise ValueError(
                "Either `model_config` or `model_config_file_path` must be provided."
            )

        # Validate model package
        model_validator = model_validators.get_validator(
            task_type=task_type,
            model_config=model_config,
            model_package_dir=model_package_dir,
            model_config_file_path=model_config_file_path,
            sample_data=sample_data,
        )
        failed_validations = model_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the model package. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        # Load model config and augment with defaults
        if model_config_file_path is not None:
            model_config = utils.read_yaml(model_config_file_path)
        model_data = model_schemas.ModelSchema().load(
            {"task_type": task_type.value, **model_config}
        )

        # Copy relevant resources to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            if model_package_dir:
                shutil.copytree(model_package_dir, temp_dir, dirs_exist_ok=True)
                utils.write_python_version(temp_dir)
                model_type = model_data.get("modelType", "full")
                model_data["modelType"] = model_type
            else:
                model_type = model_data.get("modelType", "shell")
                model_data["modelType"] = model_type

            utils.write_yaml(model_data, f"{temp_dir}/model_config.yaml")

            self._stage_resource(
                resource_name="model",
                resource_dir=temp_dir,
                project_id=project_id,
                force=force,
            )

    def add_baseline_model(
        self,
        project_id: str,
        task_type: TaskType,
        model_config: Optional[Dict[str, any]] = None,
        model_config_file_path: Optional[str] = None,
        force: bool = False,
    ):
        """
        **Coming soon...**

        Adds a baseline model to the project.

        Baseline models should be added together with training and validation
        sets. A model will then be trained on the platform using AutoML, using
        the parameters provided in the model config file.

        .. important::
            This feature is experimental and currently under development. Only
            tabular classification tasks are supported for now.

        Parameters
        ----------
        model_config : Dict[str, any], optional
            Dictionary containing the model configuration. This is not needed if
            ``model_config_file_path`` is provided. If none of these are provided,
            the default model config will be used.

            .. admonition:: What's on the model config file?

                For baseline models, the config should contain:

                - ``metadata`` : Dict[str, any], default {}
                    Dictionary containing metadata about the model. This is the
                    metadata that will be displayed on the Openlayer platform.

        model_config_file_path : str, optional
            Path to the model configuration YAML file. This is not needed if
            ``model_config`` is provided. If none of these are provided,
            the default model config will be used.

            .. admonition:: What's on the model config file?

                For baseline models, the content of the YAML file should contain:

                - ``metadata`` : Dict[str, any], default {}
                    Dictionary containing metadata about the model. This is the
                    metadata that will be displayed on the Openlayer platform.
        force : bool, optional
            Whether to force the addition of the baseline model to the project.
            If set to True, any existing staged baseline model will be overwritten.
        """
        if task_type is not TaskType.TabularClassification:
            raise exceptions.OpenlayerException(
                "Only tabular classification is supported for model baseline for now."
            )

        # Validate the baseline model
        baseline_model_validator = baseline_model_validators.get_validator(
            task_type=task_type,
            model_config=model_config,
            model_config_file_path=model_config_file_path,
        )
        failed_validations = baseline_model_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the baseline model. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        # Load model config and augment with defaults
        model_config = {} or model_config
        if model_config_file_path is not None:
            model_config = utils.read_yaml(model_config_file_path)
        model_config["modelType"] = "baseline"
        model_data = model_schemas.BaselineModelSchema().load(
            {"task_type": task_type.value, **model_config}
        )

        # Copy relevant resources to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
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
        task_type: TaskType,
        dataset_config: Optional[Dict[str, any]] = None,
        dataset_config_file_path: Optional[str] = None,
        project_id: str = None,
        force: bool = False,
    ):
        r"""Adds a dataset to a project's staging area (from a csv)."""
        if dataset_config is None and dataset_config_file_path is None:
            raise ValueError(
                "Either `dataset_config` or `dataset_config_file_path` must be"
                " provided."
            )

        # Validate dataset
        dataset_validator = dataset_validators.get_validator(
            task_type=task_type,
            dataset_config=dataset_config,
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
        if dataset_config_file_path is not None:
            dataset_config = utils.read_yaml(dataset_config_file_path)
        dataset_data = dataset_schemas.DatasetSchema().load(
            {"task_type": task_type.value, **dataset_config}
        )
        if dataset_data.get("columnNames") is None:
            dataset_data["columnNames"] = utils.get_column_names(file_path)

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
        dataset_df: pd.DataFrame,
        task_type: TaskType,
        dataset_config: Optional[Dict[str, any]] = None,
        dataset_config_file_path: Optional[str] = None,
        project_id: str = None,
        force: bool = False,
    ):
        r"""Adds a dataset to a project's staging area (from a pandas DataFrame)."""
        # --------------------------- Resource validations --------------------------- #
        if not isinstance(dataset_df, pd.DataFrame):
            raise exceptions.OpenlayerValidationError(
                f"- `dataset_df` is a `{type(dataset_df)}`, but it must be of type"
                " `pd.DataFrame`. \n"
            ) from None
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, str(uuid.uuid1()))
            dataset_df.to_csv(file_path, index=False)
            return self.add_dataset(
                file_path=file_path,
                project_id=project_id,
                dataset_config_file_path=dataset_config_file_path,
                dataset_config=dataset_config,
                force=force,
                task_type=task_type,
            )

    def commit(self, message: str, project_id: str, force: bool = False):
        """Adds a commit message to staged resources."""
        # Validate commit
        commit_validator = commit_validators.CommitValidator(commit_message=message)
        failed_validations = commit_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the commit message specified. \n"
                "Make sure to fix all of the issues listed above before committing.",
            ) from None

        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

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
                with open(
                    f"{project_dir}/commit.yaml", "r", encoding="UTF-8"
                ) as commit_file:
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

        llm_and_no_outputs = self._check_llm_and_no_outputs(project_dir=project_dir)
        if llm_and_no_outputs:
            warnings.warn(
                "You are committing an LLM without validation outputs computed "
                "in the validation set. This means that the platform will try to "
                "compute the validation outputs for you. This may take a while and "
                "there are costs associated with it."
            )
        commit = {
            "message": message,
            "date": time.ctime(),
            "computeOutputs": llm_and_no_outputs,
        }
        with open(f"{project_dir}/commit.yaml", "w", encoding="UTF-8") as commit_file:
            yaml.dump(commit, commit_file)

        if self.verbose:
            print("Committed!")

    def _check_llm_and_no_outputs(self, project_dir: str) -> bool:
        """Checks if the project's staging area contains an LLM and no outputs."""
        # Check if validation set has outputs
        validation_has_no_outputs = False
        if os.path.exists(f"{project_dir}/validation"):
            validation_dataset_config = utils.load_dataset_config_from_bundle(
                bundle_path=project_dir, label="validation"
            )
            output_column_name = validation_dataset_config.get("outputColumnName")
            validation_has_no_outputs = output_column_name is None

        # Check if the model is an LLM
        model_is_llm = False
        if os.path.exists(f"{project_dir}/model"):
            model_config = utils.read_yaml(f"{project_dir}/model/model_config.yaml")
            architecture_type = model_config.get("architectureType")
            model_type = model_config.get("modelType")

            if architecture_type == "llm" and model_type != "shell":
                model_is_llm = True

        return validation_has_no_outputs and model_is_llm

    def push(self, project_id: str, task_type: TaskType) -> Optional[ProjectVersion]:
        """Pushes the commited resources to the platform."""
        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

        if self._ready_for_push(project_dir=project_dir, task_type=task_type):
            with open(
                f"{project_dir}/commit.yaml", "r", encoding="UTF-8"
            ) as commit_file:
                commit = yaml.safe_load(commit_file)

            # Tar the project's staging area
            with tempfile.TemporaryDirectory() as tmp_dir:
                tar_file_path = os.path.join(tmp_dir, "tarfile")
                with tarfile.open(tar_file_path, mode="w:gz") as tar:
                    tar.add(project_dir, arcname=os.path.basename(project_dir))

                # Upload the tar file
                print(
                    "Pushing changes to the platform with the commit message: \n"
                    f"\t - Message: {commit['message']} \n"
                    f"\t - Date: {commit['date']}"
                )
                payload = {"commit": {"message": commit["message"]}}
                response_body = self.api.upload(
                    endpoint=f"projects/{project_id}/versions",
                    file_path=tar_file_path,
                    object_name="tarfile",
                    body=payload,
                )
                project_version = ProjectVersion(json=response_body, client=self)

            self._post_push_cleanup(project_dir=project_dir)

            if self.verbose:
                print("Pushed!")

            return project_version

    def _ready_for_push(self, project_dir: str, task_type: TaskType) -> bool:
        """Checks if the project's staging area is ready to be pushed to the platform.

        Parameters
        ----------
        project_dir : str
            Directory path to the project's staging area.

        Returns
        -------
        bool
            Indicates whether the project's staging area is ready to be pushed to the platform.
        """
        if not os.listdir(project_dir):
            print(
                "The staging area is clean and there is nothing committed to push. "
                "Please add model and/or datasets first, and then commit before pushing."
            )
            return False

        if not os.path.exists(f"{project_dir}/commit.yaml"):
            print(
                "There are resources staged, but you haven't committed them yet. "
                "Please commit before pushing"
            )
            return False

        # Validate bundle resources
        commit_bundle_validator = commit_validators.get_validator(
            task_type=task_type,
            bundle_path=project_dir,
            skip_dataset_validation=True,
            skip_model_validation=False,  # Don't skip because the sample data is different
        )
        failed_validations = commit_bundle_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the staged resources. \n"
                "Make sure to fix all of the issues listed above before pushing.",
            ) from None

        return True

    def _post_push_cleanup(self, project_dir: str) -> None:
        """Cleans up and re-creates the project's staging area after a push."""
        shutil.rmtree(project_dir)
        os.makedirs(project_dir, exist_ok=True)

    def export(self, destination_dir: str, project_id: str, task_type: TaskType):
        """Exports the commited resources as a tarfile to the location specified
        by ``destination_dir``.
        """
        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

        if self._ready_for_push(project_dir=project_dir, task_type=task_type):
            # Tar the project's staging area
            with tempfile.TemporaryDirectory() as tmp_dir:
                tar_file_path = os.path.join(tmp_dir, "tarfile")
                with tarfile.open(tar_file_path, mode="w:gz") as tar:
                    tar.add(project_dir, arcname=os.path.basename(project_dir))

                print(f"Exporting staging area to {destination_dir}.")
                shutil.copy(tar_file_path, os.path.expanduser(destination_dir))

            self._post_push_cleanup(project_dir=project_dir)
            print("Exported tarfile!")

    def status(self, project_id: str):
        """Shows the state of the staging area."""
        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

        if not os.listdir(project_dir):
            print(
                "The staging area is clean. You can stage models and/or datasets by"
                " using the corresponding `add` methods."
            )
            return

        if not os.path.exists(f"{project_dir}/commit.yaml"):
            print("The following resources are staged, waiting to be committed:")
            for file in os.listdir(project_dir):
                if file in constants.VALID_RESOURCE_NAMES:
                    print(f"\t - {file}")
            print("Use the `commit` method to add a commit message to your changes.")
            return

        with open(f"{project_dir}/commit.yaml", "r", encoding="UTF-8") as commit_file:
            commit = yaml.safe_load(commit_file)
        print("The following resources are committed, waiting to be pushed:")
        for file in os.listdir(project_dir):
            if file in constants.VALID_RESOURCE_NAMES:
                print(f"\t - {file}")
        print(f"Commit message from {commit['date']}:")
        print(f"\t {commit['message']}")
        print("Use the `push` method to push your changes to the platform.")

    def restore(self, *resource_names: str, project_id: str):
        """Removes the resource specified by ``resource_name`` from the staging area."""
        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

        for resource_name in resource_names:
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
        self, resource_name: str, resource_dir: str, project_id: str, force: bool
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
        if resource_name not in constants.VALID_RESOURCE_NAMES:
            raise ValueError(
                "Resource name must be one of 'model', 'training',"
                f" 'validation', or 'fine-tuning' but got '{resource_name}'."
            )

        project_dir = f"{constants.OPENLAYER_DIR}/{project_id}/staging"

        resources_staged = utils.list_resources_in_bundle(project_dir)

        if resource_name in resources_staged:
            print(f"Found an existing `{resource_name}` resource staged.")

            overwrite = "n"
            if not force:
                overwrite = input("Do you want to overwrite it? [y/n] ")
            if overwrite.lower() == "y" or force:
                print(f"Overwriting previously staged `{resource_name}` resource...")
                shutil.rmtree(project_dir + "/" + resource_name)
            else:
                print(f"Keeping the existing `{resource_name}` resource staged.")
                return

        shutil.copytree(resource_dir, project_dir + "/" + resource_name)

        if self.verbose:
            print(f"Staged the `{resource_name}` resource!")

    def load_project_version(self, version_id: str) -> Project:
        """Loads an existing project version from the Openlayer platform. Can be used
        to check the status of the project version and the number of passing, failing
        and skipped tests.

        Parameters
        ----------
        id : str
            UUID of the project to be loaded. You can find the UUID of a project by
            navigating to the project's page on the Openlayer platform.

            .. note::
                When you run :obj:`push`, it will return the project version object,
                which you can use to check your test statuses.

        Returns
        -------
        :obj:`project_versions.ProjectVersion`
            An object that is used to check for upload progress and test statuses.
            Also contains other useful information about a project version.

        Examples
        --------
        Instantiate the client and load the project version:

        >>> import openlayer
        >>> client = openlayer.OpenlayerClient('YOUR_API_KEY_HERE')
        >>>
        >>> version = client.load_project_version(id='YOUR_PROJECT_ID_HERE')
        >>> version.wait_for_completion()
        >>> version.print_test_report()

        With the :obj:`project_versions.ProjectVersion` object loaded, you are able to
        check progress and test statuses.
        """
        endpoint = f"versions/{version_id}"
        version_data = self.api.get_request(endpoint)
        version = ProjectVersion(version_data, self)
        return version

    def create_inference_pipeline(
        self,
        project_id: str,
        task_type: TaskType,
        name: str = "production",
        description: Optional[str] = None,
        reference_df: Optional[pd.DataFrame] = None,
        reference_dataset_file_path: Optional[str] = None,
        reference_dataset_config: Optional[Dict[str, any]] = None,
        reference_dataset_config_file_path: Optional[str] = None,
    ) -> InferencePipeline:
        """Creates an inference pipeline in an Openlayer project."""
        if (reference_df is None) ^ (reference_dataset_config_file_path is None) or (
            reference_dataset_file_path is None
        ) ^ (reference_dataset_config_file_path is None):
            raise ValueError(
                "You must specify both a reference dataset and"
                " its config or none of them."
            )
        if reference_df is not None and reference_dataset_file_path is not None:
            raise ValueError(
                "Please specify either a reference dataset or a reference dataset"
                " file path."
            )

        try:
            inference_pipeline = self.load_inference_pipeline(
                name=name, project_id=project_id, task_type=task_type
            )
            warnings.warn(
                f"Found an existing inference pipeline with name '{name}'. "
                "Loading it instead."
            )
        except exceptions.OpenlayerResourceNotFound:
            # Validate inference pipeline
            inference_pipeline_config = {
                "name": name or "production",
                "description": description or "Monitoring production data.",
                "storageType": api.STORAGE.value,
            }
            inference_pipeline_validator = (
                inference_pipeline_validators.InferencePipelineValidator(
                    inference_pipeline_config=inference_pipeline_config
                )
            )
            failed_validations = inference_pipeline_validator.validate()
            if failed_validations:
                raise exceptions.OpenlayerValidationError(
                    "There are issues with the inference pipeline. \n"
                    "Make sure to fix all of the issues listed above before"
                    " creating it.",
                ) from None

            # Load dataset config
            if reference_dataset_config_file_path is not None:
                reference_dataset_config = utils.read_yaml(
                    reference_dataset_config_file_path
                )

            if reference_dataset_config is not None:
                # Validate reference dataset and augment config
                dataset_validator = dataset_validators.get_validator(
                    task_type=task_type,
                    dataset_config=reference_dataset_config,
                    dataset_df=reference_df,
                )
                failed_validations = dataset_validator.validate()

                if failed_validations:
                    raise exceptions.OpenlayerValidationError(
                        "There are issues with the reference dataset and its config. \n"
                        "Make sure to fix all of the issues listed above before the"
                        " upload.",
                    ) from None

                reference_dataset_data = dataset_schemas.ReferenceDatasetSchema().load(
                    {"task_type": task_type.value, **reference_dataset_config}
                )

                # Copy relevant files to tmp dir if reference dataset is provided
                with tempfile.TemporaryDirectory() as tmp_dir:
                    utils.write_yaml(
                        reference_dataset_data, f"{tmp_dir}/dataset_config.yaml"
                    )
                    if reference_df is not None:
                        reference_df.to_csv(f"{tmp_dir}/dataset.csv", index=False)
                    else:
                        shutil.copy(
                            reference_dataset_file_path, f"{tmp_dir}/dataset.csv"
                        )

                    tar_file_path = os.path.join(tmp_dir, "tarfile")
                    with tarfile.open(tar_file_path, mode="w:gz") as tar:
                        tar.add(tmp_dir, arcname=os.path.basename("reference_dataset"))

                    endpoint = f"projects/{project_id}/inference-pipelines"
                    inference_pipeline_data = self.api.upload(
                        endpoint=endpoint,
                        file_path=tar_file_path,
                        object_name="tarfile",
                        body=inference_pipeline_config,
                        storage_uri_key="referenceDatasetUri",
                        method="POST",
                    )
            else:
                endpoint = f"projects/{project_id}/inference-pipelines"
                inference_pipeline_data = self.api.post_request(
                    endpoint=endpoint, body=inference_pipeline_config
                )
            inference_pipeline = InferencePipeline(
                inference_pipeline_data, self.api.upload, self, task_type
            )

            if self.verbose:
                print(
                    "Created your inference pipeline. Navigate to"
                    f" {inference_pipeline.links['app']} to see it."
                )
        return inference_pipeline

    def load_inference_pipeline(
        self,
        project_id: str,
        task_type: TaskType,
        name: Optional[str] = None,
    ) -> InferencePipeline:
        """Loads an existing inference pipeline from an Openlayer project."""
        name = name or "production"
        endpoint = f"projects/{project_id}/inference-pipelines?name={name}"
        inference_pipeline_data = self.api.get_request(endpoint)
        if len(inference_pipeline_data["items"]) == 0:
            raise exceptions.OpenlayerResourceNotFound(
                f"Inference pipeline with name {name} not found."
            )

        inference_pipeline = InferencePipeline(
            inference_pipeline_data["items"][0], self.api.upload, self, task_type
        )

        if self.verbose:
            print(
                "Found your inference pipeline."
                f" Navigate to {inference_pipeline.links['app']} to see it."
            )
        return inference_pipeline

    def upload_reference_dataset(
        self,
        inference_pipeline_id: str,
        task_type: TaskType,
        file_path: str,
        dataset_config: Optional[Dict[str, any]] = None,
        dataset_config_file_path: Optional[str] = None,
    ) -> None:
        """Uploads a reference dataset saved as a csv file to an inference pipeline."""
        if dataset_config is None and dataset_config_file_path is None:
            raise ValueError(
                "Either `dataset_config` or `dataset_config_file_path` must be"
                " provided."
            )
        if dataset_config_file_path is not None:
            dataset_config = utils.read_yaml(dataset_config_file_path)
        dataset_config["label"] = "reference"

        # Validate dataset
        dataset_validator = dataset_validators.get_validator(
            task_type=task_type,
            dataset_config=dataset_config,
            dataset_file_path=file_path,
        )
        failed_validations = dataset_validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the reference dataset and its config. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        # Load dataset config and augment with defaults
        dataset_data = dataset_schemas.ReferenceDatasetSchema().load(
            {"task_type": task_type.value, **dataset_config}
        )

        # Add default columns if not present
        if dataset_data.get("columnNames") is None:
            dataset_data["columnNames"] = utils.get_column_names(file_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy relevant files to tmp dir
            folder_path = os.path.join(tmp_dir, "reference")
            os.mkdir(folder_path)
            utils.write_yaml(dataset_data, f"{folder_path}/dataset_config.yaml")
            shutil.copy(file_path, folder_path)

            tar_file_path = os.path.join(tmp_dir, "tarfile")
            with tarfile.open(tar_file_path, mode="w:gz") as tar:
                tar.add(tmp_dir, arcname=os.path.basename("reference_dataset"))

            self.api.upload(
                endpoint=f"inference-pipelines/{inference_pipeline_id}",
                file_path=tar_file_path,
                object_name="tarfile",
                body={},
                storage_uri_key="referenceDatasetUri",
                method="PUT",
            )
        if self.verbose:
            print("Reference dataset uploaded!")

    def upload_reference_dataframe(
        self,
        inference_pipeline_id: str,
        task_type: TaskType,
        dataset_df: pd.DataFrame,
        dataset_config: Optional[Dict[str, any]] = None,
        dataset_config_file_path: Optional[str] = None,
    ) -> None:
        """Uploads a reference dataset (a pandas dataframe) to an inference pipeline."""
        # --------------------------- Resource validations --------------------------- #
        if not isinstance(dataset_df, pd.DataFrame):
            raise exceptions.OpenlayerValidationError(
                f"- `dataset_df` is a `{type(dataset_df)}`, but it must be of type"
                " `pd.DataFrame`. \n"
            ) from None
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "dataset.csv")
            dataset_df.to_csv(file_path, index=False)
            return self.upload_reference_dataset(
                file_path=file_path,
                inference_pipeline_id=inference_pipeline_id,
                dataset_config=dataset_config,
                dataset_config_file_path=dataset_config_file_path,
                task_type=task_type,
            )

    def stream_data(
        self,
        inference_pipeline_id: str,
        task_type: TaskType,
        stream_data: Union[Dict[str, any], List[Dict[str, any]]],
        stream_config: Optional[Dict[str, any]] = None,
        stream_config_file_path: Optional[str] = None,
    ) -> None:
        """Streams production data to the Openlayer platform."""
        if not isinstance(stream_data, (dict, list)):
            raise ValueError(
                "stream_data must be a dictionary or a list of dictionaries."
            )
        if isinstance(stream_data, dict):
            stream_data = [stream_data]

        stream_df = pd.DataFrame(stream_data)
        stream_config = self._validate_production_data_and_load_config(
            task_type=task_type,
            config=stream_config,
            config_file_path=stream_config_file_path,
            df=stream_df,
        )
        stream_config, stream_df = self._add_default_columns(
            config=stream_config, df=stream_df
        )

        # Remove the `label` for the upload
        stream_config.pop("label", None)

        body = {
            "config": stream_config,
            "rows": stream_df.to_dict(orient="records"),
        }
        self.api.post_request(
            endpoint=f"inference-pipelines/{inference_pipeline_id}/data-stream",
            body=body,
            include_metadata=False,
        )
        if self.verbose:
            print("Stream published!")

    def publish_batch_data(
        self,
        inference_pipeline_id: str,
        task_type: TaskType,
        batch_df: pd.DataFrame,
        batch_config: Optional[Dict[str, any]] = None,
        batch_config_file_path: Optional[str] = None,
    ) -> None:
        """Publishes a batch of production data to the Openlayer platform."""
        batch_config = self._validate_production_data_and_load_config(
            task_type=task_type,
            config=batch_config,
            config_file_path=batch_config_file_path,
            df=batch_df,
        )
        batch_config, batch_df = self._add_default_columns(
            config=batch_config, df=batch_df
        )

        # Add column names if missing
        if batch_config.get("columnNames") is None:
            batch_config["columnNames"] = list(batch_df.columns)

        # Get min and max timestamps
        earliest_timestamp = batch_df[batch_config["timestampColumnName"]].min()
        latest_timestamp = batch_df[batch_config["timestampColumnName"]].max()
        batch_row_count = len(batch_df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy save files to tmp dir
            batch_df.to_csv(f"{tmp_dir}/dataset.csv", index=False)
            utils.write_yaml(batch_config, f"{tmp_dir}/dataset_config.yaml")

            tar_file_path = os.path.join(tmp_dir, "tarfile")
            with tarfile.open(tar_file_path, mode="w:gz") as tar:
                tar.add(tmp_dir, arcname=os.path.basename("batch_data"))

            payload = {
                "performDataMerge": False,
                "earliestTimestamp": int(earliest_timestamp),
                "latestTimestamp": int(latest_timestamp),
                "rowCount": batch_row_count,
            }

            presigned_url_query_params_dict = {
                "earliestTimestamp": int(earliest_timestamp),
                "latestTimestamp": int(latest_timestamp),
                "storageInterface": api.STORAGE.value,
                "dataType": "data",
            }

            presigned_url_query_params = urllib.parse.urlencode(
                presigned_url_query_params_dict
            )

            self.api.upload(
                endpoint=f"inference-pipelines/{inference_pipeline_id}/data",
                file_path=tar_file_path,
                object_name="tarfile",
                body=payload,
                storage_uri_key="storageUri",
                method="POST",
                presigned_url_endpoint=(
                    f"inference-pipelines/{inference_pipeline_id}/presigned-url"
                ),
                presigned_url_query_params=presigned_url_query_params,
            )
        if self.verbose:
            print("Data published!")

    def _validate_production_data_and_load_config(
        self,
        task_type: TaskType,
        df: pd.DataFrame,
        config: Optional[Dict[str, any]] = None,
        config_file_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """Validates the production data and its config and returns a valid config
        populated with the default values."""
        if config is None and config_file_path is None:
            raise ValueError(
                "Either the config or the config file path must be provided."
            )

        if config_file_path is not None:
            if not os.path.exists(config_file_path):
                raise exceptions.OpenlayerValidationError(
                    f"The file specified by the config file path {config_file_path} does"
                    " not exist."
                ) from None
            config = utils.read_yaml(config_file_path)

        # Force label to be production
        config["label"] = "production"

        # Validate batch of data
        validator = dataset_validators.get_validator(
            task_type=task_type,
            dataset_config=config,
            dataset_df=df,
        )
        failed_validations = validator.validate()

        if failed_validations:
            raise exceptions.OpenlayerValidationError(
                "There are issues with the data and its config. \n"
                "Make sure to fix all of the issues listed above before the upload.",
            ) from None

        config = dataset_schemas.ProductionDataSchema().load(
            {"task_type": task_type.value, **config}
        )

        return config

    def _add_default_columns(
        self, config: Dict[str, any], df: pd.DataFrame
    ) -> Tuple[Dict[str, any], pd.DataFrame]:
        """Adds the default columns if not present and returns the updated config and
        dataframe."""
        columns_to_add = {"timestampColumnName", "inferenceIdColumnName"}
        for column in columns_to_add:
            if config.get(column) is None:
                config, df = self._add_default_column(
                    config=config, df=df, column_name=column
                )
        return config, df

    def _add_default_column(
        self, config: Dict[str, any], df: pd.DataFrame, column_name: str
    ) -> Tuple[Dict[str, any], pd.DataFrame]:
        """Adds the default column specified by ``column_name`` to the dataset config
        and dataframe."""
        df = df.copy()
        if column_name == "timestampColumnName":
            timestamp_column_name = f"timestamp_{str(uuid.uuid1())[:8]}"
            config["timestampColumnName"] = timestamp_column_name
            df.loc[:, timestamp_column_name] = int(time.time())
        elif column_name == "inferenceIdColumnName":
            inference_id_column_name = f"inference_id_{str(uuid.uuid1())[:8]}"
            config["inferenceIdColumnName"] = inference_id_column_name
            df.loc[:, inference_id_column_name] = [
                str(uuid.uuid1()) for _ in range(len(df))
            ]
        return config, df

    def publish_ground_truths(
        self,
        inference_pipeline_id: str,
        df: pd.DataFrame,
        inference_id_column_name: str,
        ground_truth_column_name: str,
    ):
        """Publishes ground truths to the Openlayer platform."""
        raise DeprecationWarning(
            "The `publish_ground_truths` method is deprecated.\n"
            "Please use `update_data` instead."
        )

    def update_data(
        self,
        inference_pipeline_id: str,
        df: pd.DataFrame,
        inference_id_column_name: str,
        ground_truth_column_name: Optional[str] = None,
    ) -> None:
        """Updates data already on the Openlayer platform."""
        # -------------------------------- Validations ------------------------------- #
        if not isinstance(df, pd.DataFrame):
            raise exceptions.OpenlayerValidationError(
                f"- `df` is a `{type(df)}`, but it must a" " `pd.DataFrame`. \n"
            ) from None
        if ground_truth_column_name is not None:
            if ground_truth_column_name not in df.columns:
                raise exceptions.OpenlayerValidationError(
                    f"- `df` does not contain the ground truth column name"
                    f" `{ground_truth_column_name}`. \n"
                ) from None
        if inference_id_column_name not in df.columns:
            raise exceptions.OpenlayerValidationError(
                f"- `df` does not contain the inference ID column name"
                f" `{inference_id_column_name}`. \n"
            ) from None

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy save files to tmp dir
            df.to_csv(f"{tmp_dir}/dataset.csv", index=False)

            payload = {
                "performDataMerge": True,
                "groundTruthColumnName": ground_truth_column_name,
                "inferenceIdColumnName": inference_id_column_name,
            }

            presigned_url_query_params_dict = {
                "storageInterface": api.STORAGE.value,
                "dataType": "groundTruths",
            }

            presigned_url_query_params = urllib.parse.urlencode(
                presigned_url_query_params_dict
            )

            self.api.upload(
                endpoint=f"inference-pipelines/{inference_pipeline_id}/data",
                file_path=f"{tmp_dir}/dataset.csv",
                object_name="dataset.csv",
                body=payload,
                storage_uri_key="storageUri",
                method="POST",
                presigned_url_endpoint=f"inference-pipelines/{inference_pipeline_id}/presigned-url",
                presigned_url_query_params=presigned_url_query_params,
            )
        if self.verbose:
            print("Uploaded data to be updated!")
