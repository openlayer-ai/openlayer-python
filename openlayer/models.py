import os
import subprocess
from enum import Enum
from typing import List, Set
import tempfile
import pandas as pd


class ModelType(Enum):
    """A selection of machine learning modeling frameworks supported by Openlayer.

    .. note::
        Our `sample notebooks <https://github.com/openlayer-ai/openlayer-python/tree/main/examples>`_
        show you how to use each one of these model types with Openlayer.
    """

    #: For custom built models.
    custom = "custom"
    #: For models built with `fastText <https://fasttext.cc/>`_.
    fasttext = "fasttext"
    #: For models built with `Keras <https://keras.io/>`_.
    keras = "keras"
    #: For models built with `PyTorch <https://pytorch.org/>`_.
    pytorch = "pytorch"
    #: For models built with `rasa <https://rasa.com/>`_.
    rasa = "rasa"
    #: For models built with `scikit-learn <https://scikit-learn.org/>`_.
    sklearn = "sklearn"
    #: For models built with `TensorFlow <https://www.tensorflow.org/>`_.
    tensorflow = "tensorflow"
    #: For models built with `Hugging Face transformers <https://huggingface.co/docs/transformers/index>`_.
    transformers = "transformers"
    #: For models built with `XGBoost <https://xgboost.readthedocs.io>`_.
    xgboost = "xgboost"


class Model:
    """An object containing information about a model on the Openlayer platform."""

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


class CondaEnvironment:
    """Conda environment manager abstraction.

    Parameters
    ----------
    env_name : str
        Name of the conda environment.
    requirements_file_path : str
        Path to the requirements file.
    python_version_file_path : str
        Path to the python version file.
    logs_file_path : str
        Path to the logs file.
    """

    def __init__(
        self,
        env_name: str,
        requirements_file_path: str,
        python_version_file_path: str,
        logs_file_path: str,
    ):
        if not self._conda_available():
            raise Exception("Conda is not available on this machine.")

        self.env_name = env_name
        self.requirements_file_path = requirements_file_path
        self.python_version_file_path = python_version_file_path
        self._conda_prefix = self._get_conda_prefix()
        self._logs_file_path = logs_file_path
        self._logs_file = subprocess.PIPE

    def __enter__(self):
        self._logs_file = open(self._logs_file_path, "wb")
        existing_envs = self.get_existing_envs()
        if self.env_name in existing_envs:
            print(f"Found existing conda environment '{self.env_name}'.")
        else:
            self.create()
            self.install_requirements()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deactivate()
        self._logs_file.close()

    def _conda_available(self) -> bool:
        """Checks if conda is available on the machine."""
        if os.environ.get("CONDA_EXE") is None:
            return False
        return True

    def _get_conda_prefix(self) -> str:
        """Gets the conda base environment prefix.

        E.g., '~/miniconda3' or '~/anaconda3'
        """
        prefix = subprocess.check_output(["conda", "info", "--base"])
        return prefix.decode("UTF-8").strip()

    def create(self):
        """Creates a conda environment with the specified name and python version."""
        print(f"Creating a new conda environment '{self.env_name}'...")

        with open(
            self.python_version_file_path, "r", encoding="UTF-8"
        ) as python_version_file:
            python_version = python_version_file.read().split(".")[:2]
            python_version = ".".join(python_version)

        try:
            subprocess.check_call(
                [
                    "conda",
                    "create",
                    "-n",
                    f"{self.env_name}",
                    f"python={python_version}",
                    "--yes",
                ],
                stdout=self._logs_file,
                stderr=self._logs_file,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to create conda environment '{self.env_name}' with python "
                f"version {python_version}."
                f"Error {err.returncode}: {err.output}"
            ) from None

    def delete(self):
        """Deletes the conda environment with the specified name."""
        print(f"Deleting conda environment '{self.env_name}'...")

        try:
            subprocess.check_call(
                ["conda", "env", "remove", "-n", f"{self.env_name}", "--yes"],
                stdout=self._logs_file,
                stderr=self._logs_file,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to delete conda environment '{self.env_name}'."
                f"Error {err.returncode}: {err.output}"
            ) from None

    def get_existing_envs(self) -> Set[str]:
        """Gets the names of all existing conda environments."""
        print("Checking existing conda environments...")
        list_envs_command = """
        conda env list | awk '{print $1}'
        """
        try:
            envs = subprocess.check_output(
                list_envs_command,
                shell=True,
                stderr=self._logs_file,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to list conda environments."
                f"Error {err.returncode}: {err.output}"
            ) from None
        envs = set(envs.decode("UTF-8").split("\n"))
        return envs

    def activate(self):
        """Activates the conda environment with the specified name."""
        print(f"Activating conda environment '{self.env_name}'...")

        activation_command = f"""
        eval $(conda shell.bash hook)
        source {self._conda_prefix}/etc/profile.d/conda.sh
        conda activate {self.env_name}"""

        try:
            subprocess.check_call(
                activation_command,
                stdout=self._logs_file,
                stderr=self._logs_file,
                shell=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to activate conda environment '{self.env_name}'."
                f"Error {err.returncode}: {err.output}"
            ) from None

    def deactivate(self):
        """Deactivates the conda environment with the specified name."""
        print(f"Deactivating conda environment '{self.env_name}'...")

        deactivation_command = f"""
        eval $(conda shell.bash hook)
        source {self._conda_prefix}/etc/profile.d/conda.sh
        conda deactivate"""

        try:
            subprocess.check_call(
                deactivation_command,
                shell=True,
                stdout=self._logs_file,
                stderr=self._logs_file,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to deactivate conda environment '{self.env_name}'."
                f"Error {err.returncode}: {err.output}"
            ) from None

    def install_requirements(self):
        """Installs the requirements from the specified requirements file."""
        print(f"Installing requirements in conda environment '{self.env_name}'...")

        try:
            self.run_commands(
                ["pip", "install", "-r", self.requirements_file_path],
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to install requirements from {self.requirements_file_path}."
                f"Error {err.returncode}: {err.output}"
            ) from None

    def run_commands(self, commands: List[str]):
        """Runs the specified commands inside the conda environment.

        Parameters
        ----------
        commands : List[str]
            List of commands to run.
        """
        full_command = f"""
        eval $(conda shell.bash hook)
        source {self._conda_prefix}/etc/profile.d/conda.sh
        conda activate {self.env_name}
        {" ".join(commands)}
        """
        subprocess.check_call(
            full_command,
            shell=True,
            stdout=self._logs_file,
            stderr=self._logs_file,
        )


class ModelRunner:
    """Wraps the model package and provides a uniform run method."""

    def __init__(self, model_package: str):
        self.model_package = model_package
        # TODO: change env name to the model id
        self._conda_environment = CondaEnvironment(
            env_name="new-openlayer",
            requirements_file_path=f"{model_package}/requirements.txt",
            python_version_file_path=f"{model_package}/python_version",
            logs_file_path=f"{model_package}/logs.txt",
        )

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in the conda
        environment.

        Parameters
        ----------
        input_data : pd.DataFrame
            Input data to run the model on.

        Returns
        -------
        pd.DataFrame
            Output from the model. The output is a dataframe with a single
            column named 'prediction' and lists of class probabilities as values.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input data to a csv file
            input_data.to_csv(f"{temp_dir}/input_data.csv", index=False)

            # Run the model in the conda environment
            with self._conda_environment as env:
                try:
                    env.run_commands(
                        [
                            "python",
                            f"{self.model_package}/prediction_job.py",
                            "--input",
                            f"{temp_dir}/input_data.csv",
                            "--output",
                            f"{temp_dir}/output_data.csv",
                        ]
                    )
                except subprocess.CalledProcessError as err:
                    raise Exception(
                        f"Failed to run the model in conda environment '{env.env_name}'."
                        f"Error {err.returncode}: {err.output}"
                    ) from None

            # Read the output data from the csv file
            output_data = pd.read_csv(f"{temp_dir}/output_data.csv")

        return output_data
