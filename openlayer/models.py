import logging
import os
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import List, Optional, Set

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
    logs_file_path : str, optional
        Where to log the output of the conda commands.
        If None, the output is shown in stdout.
    """

    def __init__(
        self,
        env_name: str,
        requirements_file_path: str,
        python_version_file_path: str,
        logs_file_path: Optional[str] = None,
    ):
        if not self._conda_available():
            raise Exception("Conda is not available on this machine.")

        self.env_name = env_name
        self.requirements_file_path = requirements_file_path
        self.python_version_file_path = python_version_file_path
        self._conda_prefix = self._get_conda_prefix()
        self._logs_file_path = logs_file_path
        self._logs = None

    def __enter__(self):
        self._logs = open(self._logs_file_path, "w")
        existing_envs = self.get_existing_envs()
        if self.env_name in existing_envs:
            logging.info("Found existing conda environment '%s'.", self.env_name)
        else:
            self.create()
            self.install_requirements()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deactivate()
        self._logs.close()

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
        logging.info("Creating a new conda environment '%s'...", self.env_name)

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
                stdout=self._logs,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to create conda environment '{self.env_name}' with python "
                f"version {python_version}."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def delete(self):
        """Deletes the conda environment with the specified name."""
        logging.info("Deleting conda environment '%s'...", self.env_name)

        try:
            subprocess.check_call(
                ["conda", "env", "remove", "-n", f"{self.env_name}", "--yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to delete conda environment '{self.env_name}'."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def get_existing_envs(self) -> Set[str]:
        """Gets the names of all existing conda environments."""
        logging.info("Checking existing conda environments...")

        list_envs_command = """
        conda env list | awk '{print $1}'
        """

        try:
            envs = subprocess.check_output(
                list_envs_command,
                shell=True,
                stderr=self._logs,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to list conda environments."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None
        envs = set(envs.decode("UTF-8").split("\n"))
        return envs

    def activate(self):
        """Activates the conda environment with the specified name."""
        logging.info("Activating conda environment '%s'...", self.env_name)

        activation_command = f"""
        eval $(conda shell.bash hook)
        source {self._conda_prefix}/etc/profile.d/conda.sh
        conda activate {self.env_name}"""

        try:
            subprocess.check_call(
                activation_command,
                stdout=self._logs,
                stderr=subprocess.STDOUT,
                shell=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to activate conda environment '{self.env_name}'."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def deactivate(self):
        """Deactivates the conda environment with the specified name."""
        logging.info("Deactivating conda environment '%s'...", self.env_name)

        deactivation_command = f"""
        eval $(conda shell.bash hook)
        source {self._conda_prefix}/etc/profile.d/conda.sh
        conda deactivate"""

        try:
            subprocess.check_call(
                deactivation_command,
                shell=True,
                stdout=self._logs,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to deactivate conda environment '{self.env_name}'."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def install_requirements(self):
        """Installs the requirements from the specified requirements file."""
        logging.info(
            "Installing requirements in conda environment '%s'...", self.env_name
        )

        try:
            self.run_commands(
                ["pip", "install", "-r", self.requirements_file_path],
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to install the depencies specified in the requirements.txt file."
                " Please check the model logs for details. \n"
                f"- Error code returned {err.returncode}: {err.output}"
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
            full_command, shell=True, stdout=self._logs, stderr=subprocess.STDOUT
        )


class ModelRunner:
    """Wraps the model package and provides a uniform run method."""

    def __init__(self, model_package: str, logs: Optional[str] = None):
        self.model_package = model_package

        # Save log to the model package if logs is not specified
        if logs is None:
            logs_file_path = f"{model_package}/model_run_logs.txt"

            logging.basicConfig(
                filename=logs_file_path,
                format="[%(asctime)s] %(levelname)s - %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # TODO: change env name to the model id
        self._conda_environment = CondaEnvironment(
            env_name="new-openlayer",
            requirements_file_path=f"{model_package}/requirements.txt",
            python_version_file_path=f"{model_package}/python_version",
            logs_file_path=logs_file_path,
        )

    def __del__(self):
        self._conda_environment.delete()

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
        # Copy the prediction job script to the model package
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copy(
            f"{current_file_dir}/prediction_job.py",
            f"{self.model_package}/prediction_job.py",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input data to a csv file
            input_data.to_csv(f"{temp_dir}/input_data.csv", index=False)

            # Run the model in the conda environment
            with self._conda_environment as env:
                logging.info("Running %s rows through the model...", len(input_data))
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
                    logging.error(
                        "Failed to run the model. Check the stacktrace above for details."
                    )
                    raise Exception(
                        "Failed to run the model in the conda environment."
                        " Please check the model logs for details. \n"
                        f" Error {err.returncode}: {err.output}"
                    ) from None

            logging.info("Successfully ran data through the model!")
            # Read the output data from the csv file
            output_data = pd.read_csv(f"{temp_dir}/output_data.csv")

        return output_data
