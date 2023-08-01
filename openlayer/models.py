# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module that contains structures relevant to interfacing models with Openlayer.

The ModelType enum chooses between different machine learning modeling frameworks.
The Model object contains information about a model on the Openlayer platform.
The CondaEnv object contains conda environment metadata relevant to the
Openlayer platform.
The ModelRunner runs input data through the model in a consistent Conda environment.

Typical usage example:

    validate=validate.OneOf(
        [model_framework.value for model_framework in ModelType],
        error=f"`model_type` must be one of the supported frameworks.",
    )

"""
import ast
import datetime
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Set

import openai
import pandas as pd
import pybars

from . import tasks, utils


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
    #: For large language models (LLMs), such as GPT
    llm = "llm"
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
        logger: Optional[logging.Logger] = None,
    ):
        self._conda_exe = self._get_executable()
        self._conda_prefix = self._get_conda_prefix()
        self._bash = self._get_bash()
        self.env_name = env_name
        self.requirements_file_path = requirements_file_path
        self.python_version_file_path = python_version_file_path
        self.logger = logger or logging.getLogger("validators")

    def __enter__(self):
        existing_envs = self.get_existing_envs()
        if self.env_name in existing_envs:
            self.logger.info("Found existing conda environment '%s'.", self.env_name)
        else:
            self.create()
            self.install_requirements()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.deactivate()

    def _get_executable(self) -> str:
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe is None:
            raise Exception("Conda is not available on this machine.")
        return conda_exe

    def _get_bash(self) -> str:
        """Gets the bash executable."""
        shell_path = shutil.which("bash")
        if shell_path is None:
            raise Exception("Bash is not available on this machine.")
        return shell_path

    def _get_conda_prefix(self) -> str:
        """Gets the conda base environment prefix.

        E.g., '~/miniconda3' or '~/anaconda3'
        """
        prefix = subprocess.check_output([self._conda_exe, "info", "--base"])
        return prefix.decode("UTF-8").strip()

    def create(self):
        """Creates a conda environment with the specified name and python version."""
        self.logger.info("Creating a new conda environment '%s'... \n", self.env_name)

        with open(
            self.python_version_file_path, "r", encoding="UTF-8"
        ) as python_version_file:
            python_version = python_version_file.read().split(".")[:2]
            python_version = ".".join(python_version)

        process = subprocess.Popen(
            [
                self._conda_exe,
                "create",
                "-n",
                f"{self.env_name}",
                f"python={python_version}",
                "--yes",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()

        if exitcode != 0:
            raise Exception(
                f"Failed to create conda environment '{self.env_name}' with python "
                f"version {python_version}."
            )

    def delete(self):
        """Deletes the conda environment with the specified name."""
        self.logger.info("Deleting conda environment '%s'...", self.env_name)

        process = subprocess.Popen(
            [
                self._conda_exe,
                "env",
                "remove",
                "-n",
                f"{self.env_name}",
                "--yes",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()

        if exitcode != 0:
            raise Exception(f"Failed to delete conda environment '{self.env_name}'.")

    def get_existing_envs(self) -> Set[str]:
        """Gets the names of all existing conda environments."""
        self.logger.info("Checking existing conda environments...")

        awk_command = "awk '{print $1}"
        list_envs_command = f"""
        {self._conda_exe} env list | {awk_command}'
        """

        try:
            envs = subprocess.check_output(
                list_envs_command,
                shell=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to list conda environments."
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None
        envs = set(envs.decode("UTF-8").split("\n"))
        return envs

    def activate(self):
        """Activates the conda environment with the specified name."""
        self.logger.info("Activating conda environment '%s'...", self.env_name)

        activation_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda activate {self.env_name}
        """

        try:
            subprocess.check_call(
                activation_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                shell=True,
            )
        except subprocess.CalledProcessError as err:
            raise Exception(
                f"Failed to activate conda environment '{self.env_name}'."
                f"- Error code returned {err.returncode}: {err.output}"
            ) from None

    def deactivate(self):
        """Deactivates the conda environment with the specified name."""
        self.logger.info("Deactivating conda environment '%s'...", self.env_name)

        deactivation_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda deactivate
        """

        try:
            subprocess.check_call(
                deactivation_command,
                shell=True,
                executable=self._bash,
                stdout=subprocess.DEVNULL,
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
        self.logger.info(
            "Installing requirements in conda environment '%s'...", self.env_name
        )

        exitcode = self.run_commands(
            ["pip", "install", "-r", self.requirements_file_path],
        )
        if exitcode != 0:
            raise Exception(
                "Failed to install the depencies specified in the requirements.txt file."
            )

    def run_commands(self, commands: List[str]):
        """Runs the specified commands inside the conda environment.

        Parameters
        ----------
        commands : List[str]
            List of commands to run.
        """
        full_command = f"""
        source {self._conda_prefix}/etc/profile.d/conda.sh
        eval $(conda shell.bash hook)
        conda activate {self.env_name}
        {" ".join(commands)}
        """
        process = subprocess.Popen(
            full_command,
            shell=True,
            executable=self._bash,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        with process.stdout:
            utils.log_subprocess_output(self.logger, process.stdout)
        exitcode = process.wait()
        return exitcode


class BaseModelRunner(ABC):
    """Wraps the model package and provides a uniform run method."""

    def __init__(self, model_package: str, logger: Optional[logging.Logger] = None):
        self.model_package = model_package

        # Use validators logger if no logger is provided
        self.logger = logger or logging.getLogger("validators")

        # TODO: change env name to the model id
        self._conda_environment = CondaEnvironment(
            env_name=f"model-runner-env-{datetime.datetime.now().strftime('%m-%d-%H-%M-%S-%f')}",
            requirements_file_path=f"{model_package}/requirements.txt",
            python_version_file_path=f"{model_package}/python_version",
            logger=self.logger,
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

        self._copy_prediction_job_script(current_file_dir)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input data to a csv file
            input_data.to_csv(f"{temp_dir}/input_data.csv", index=False)

            # Run the model in the conda environment
            with self._conda_environment as env:
                self.logger.info(
                    "Running %s rows through the model...", len(input_data)
                )
                exitcode = env.run_commands(
                    [
                        "python",
                        f"{self.model_package}/prediction_job.py",
                        "--input",
                        f"{temp_dir}/input_data.csv",
                        "--output",
                        f"{temp_dir}/output_data.csv",
                    ]
                )
                if exitcode != 0:
                    self.logger.error(
                        "Failed to run the model. Check the stacktrace above for details."
                    )
                    raise Exception(
                        "Failed to run the model in the conda environment."
                    ) from None

            self.logger.info("Successfully ran data through the model!")
            # Read the output data from the csv file
            output_data = pd.read_csv(f"{temp_dir}/output_data.csv")

            output_data = self._post_process_output(output_data)

        return output_data

    @abstractmethod
    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the correct prediction job script to the model package."""
        pass

    @abstractmethod
    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Performs any post-processing on the output data."""
        pass


class LLModelRunner(ABC):
    """Base LLM model runner.

    The run method gets the LLM's predictions one by one. Child classes
    should implement the _initialize_llm method to initialize the LLM (i.e.
    handle API keys, etc.) and the _get_llm_prediction method to get the
    LLM's prediction for a single input row.
    """

    def __init__(self, model_package: str, logger: Optional[logging.Logger] = None):
        self.model_config = utils.read_yaml(f"{model_package}/model_config.yaml")

        # Use validators logger if no logger is provided
        self.logger = logger or logging.getLogger("validators")

    @abstractmethod
    def _initialize_llm(self):
        """Initializes the LLM. E.g. sets API keys, loads the model, etc."""
        pass

    def run(self, input_data_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in the conda
        environment.
        """
        model_outputs = []

        for input_data_row in input_data_df.iterrows():
            input_variables_dict = input_data_row[1][
                self.model_config["inputVariableNames"]
            ].to_dict()
            input_text = self._inject_prompt_template(
                input_variables_dict=input_variables_dict
            )

            try:
                model_outputs.append(self._get_llm_output(input_text=input_text))
            except Exception as exc:
                model_outputs.append(
                    f"[Error] Could not get predictions for row: {exc}"
                )

        self.logger.info("Successfully ran data through the model!")
        return pd.DataFrame({"predictions": model_outputs})

    def _inject_prompt_template(self, input_variables_dict: dict) -> str:
        """Injects the input variables into the prompt template.

        The prompt template must contain handlebar expressions.

        Parameters
        ----------
        input_variables_dict : dict
            Dictionary of input variables to be injected into the prompt template.
            E.g. {"input_variable_1": "value_1", "input_variable_2": "value_2"}
        """
        self.logger.info("Injecting input variables into the prompt template...")
        compiler = pybars.Compiler()
        formatter = compiler.compile(self.model_config["promptTemplate"].strip())
        return formatter(input_variables_dict)

    @abstractmethod
    def _get_llm_output(self, input_text: str) -> str:
        """Implements the logic to get the output from the language model for
        a given input text."""
        pass


class OpenAIChatCompletionRunner(LLModelRunner):
    """Wraps OpenAI's chat completion model."""

    def __init__(
        self,
        model_package: str,
        logger: Optional[logging.Logger] = None,
        openai_api_key: str = None,
    ):
        super().__init__(model_package, logger)
        if openai_api_key is None:
            raise ValueError(
                "OpenAI API key must be provided. Please pass it as a parameter "
                "named 'openai_api_key'"
            )

        self.openai_api_key = openai_api_key
        self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the OpenAI chat completion model."""
        openai.api_key = (
            self.openai_api_key  # "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"
        )

    def _get_llm_output(self, input_text: str) -> str:
        """Gets the output from the OpenAI's chat completion model
        for a given input text."""
        return openai.ChatCompletion.create(
            model=self.model_config["model"],
            messages=[{"role": "user", "content": input_text}],
            **self.model_config.get("modelParameters", {}),
        )["choices"][0]["message"]["content"]


class ClassificationModelRunner(BaseModelRunner):
    """Wraps classification models."""

    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the classification prediction job script to the model package."""
        shutil.copy(
            f"{current_file_dir}/prediction_jobs/classification_prediction_job.py",
            f"{self.model_package}/prediction_job.py",
        )

    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the output data."""
        processed_output_data = output_data.copy()

        # Make the items list of floats (and not strings)
        processed_output_data["predictions"] = processed_output_data[
            "predictions"
        ].apply(ast.literal_eval)

        return processed_output_data


class RegressionModelRunner(BaseModelRunner):
    """Wraps regression models."""

    def _copy_prediction_job_script(self, current_file_dir: str):
        """Copies the regression prediction job script to the model package."""
        shutil.copy(
            f"{current_file_dir}/prediction_jobs/regression_prediction_job.py",
            f"{self.model_package}/prediction_job.py",
        )

    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the output data."""
        return output_data


# ----------------------------- Factory function ----------------------------- #
def get_model_runner(
    task_type: tasks.TaskType,
    model_package: str,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> BaseModelRunner:
    """Factory function to get the correct model runner for the specified task type.

    Parameters
    ----------
    task_type : tasks.TaskType
        Task type of the model.
    model_package : str
        Path to the model package.
    logger : Optional[logging.Logger], optional
        Logger to use, by default None
    """
    if task_type in [
        tasks.TaskType.TabularClassification,
        tasks.TaskType.TextClassification,
    ]:
        return ClassificationModelRunner(model_package, logger)
    elif task_type == tasks.TaskType.TabularRegression:
        return RegressionModelRunner(model_package, logger)
    elif task_type in [
        tasks.TaskType.LLM,
        tasks.TaskType.LLMNER,
        tasks.TaskType.LLMQuestionAnswering,
        tasks.TaskType.LLMSummarization,
        tasks.TaskType.LLMTranslation,
    ]:
        model_provider = utils.read_yaml(f"{model_package}/model_config.yaml").get(
            "modelProvider"
        )
        if model_provider == "OpenAI":
            return OpenAIChatCompletionRunner(
                model_package, logger, kwargs.get("openai_api_key")
            )
        else:
            raise ValueError(f"Model provider `{model_provider}` is not supported.")
    else:
        raise ValueError(f"Task type `{task_type}` is not supported.")
