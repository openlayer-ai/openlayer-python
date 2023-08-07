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
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Set

import anthropic
import cohere
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


# -------------------------- Abstract model runners -------------------------- #
class ModelRunnerInterface(ABC):
    """Interface for model runners."""

    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        self.logger = logger

        model_package = kwargs.get("model_package")
        if model_package is not None:
            self.init_from_model_package(model_package)
        else:
            self.init_from_kwargs(**kwargs)

        self.validate_minimum_viable_config()

    def init_from_model_package(self, model_package: str) -> None:
        """Initializes the model runner from the model package.

        I.e., using the model_config.yaml file located in the model package
        directory.
        """
        self.model_package = model_package

        # Model config is originally a dict with camelCase keys
        self.model_config = utils.camel_to_snake_dict(
            utils.read_yaml(f"{model_package}/model_config.yaml")
        )

        self._conda_environment = None
        self.in_memory = True
        python_version_file_path = f"{model_package}/python_version"
        requirements_file_path = f"{model_package}/requirements.txt"
        if os.path.isfile(python_version_file_path) and os.path.isfile(
            requirements_file_path
        ):
            self.in_memory = False
            self._conda_environment = CondaEnvironment(
                env_name=f"model-runner-env-{datetime.datetime.now().strftime('%m-%d-%H-%M-%S-%f')}",
                requirements_file_path=python_version_file_path,
                python_version_file_path=requirements_file_path,
                logger=self.logger,
            )

    def init_from_kwargs(self, **kwargs) -> None:
        """Initializes the model runner from the kwargs."""
        self.model_package = None
        self._conda_environment = None
        self.in_memory = True
        self.model_config = kwargs

    @abstractmethod
    def validate_minimum_viable_config(self) -> None:
        """Superficial validation of the minimum viable config needed to use
        the model runner.

        Each concrete model runner must implement this method.
        """
        pass

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model."""
        if self.in_memory:
            return self._run_in_memory(input_data)
        else:
            return self._run_in_conda(input_data)

    @abstractmethod
    def _run_in_memory(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the model in memory."""
        pass

    @abstractmethod
    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the model in a conda environment."""
        pass

    def __del__(self):
        if self._conda_environment is not None:
            self._conda_environment.delete()


class TraditionalMLModelRunner(ModelRunnerInterface):
    """Model runner for traditional ML models."""

    @abstractmethod
    def validate_minimum_viable_config(self) -> None:
        pass

    def _run_in_memory(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in memory."""
        raise NotImplementedError(
            "Running traditional ML in memory is not implemented yet. "
            "Please use the runner in a conda environment."
        )

    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in the conda
        environment.
        """
        self.logger.info("Running traditional ML model in conda environment...")

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
                        "Failed to run the model. Check the stack trace above for details."
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
        """Copies the correct prediction job script to the model package.

        Needed if the model is intended to be run in a conda environment."""
        pass

    @abstractmethod
    def _post_process_output(self, output_data: pd.DataFrame) -> pd.DataFrame:
        """Performs any post-processing on the output data.

        Needed if the model is intended to be run in a conda environment."""
        pass


class LLModelRunner(ModelRunnerInterface):
    """Model runner for LLMs."""

    @abstractmethod
    def _initialize_llm(self):
        """Initializes the LLM. E.g. sets API keys, loads the model, etc."""
        pass

    def validate_minimum_viable_config(self) -> None:
        """Validates the minimum viable config needed to use the LLM model
        runner.
        """
        if (
            self.model_config.get("input_variable_names") is None
            or self.model_config.get("prompt_template") is None
        ):
            raise ValueError(
                "Input variable names and prompt template must be provided."
            )

    def _run_in_memory(self, input_data_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in memory."""
        self.logger.info("Running LLM in memory...")
        model_outputs = []

        for input_data_row in input_data_df.iterrows():
            input_variables_dict = input_data_row[1][
                self.model_config["input_variable_names"]
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
        formatter = compiler.compile(self.model_config["prompt_template"].strip())
        return formatter(input_variables_dict)

    @abstractmethod
    def _get_llm_output(self, input_text: str) -> str:
        """Implements the logic to get the output from the language model for
        a given input text."""
        pass

    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs LLM prediction job in a conda environment."""
        raise NotImplementedError(
            "Running LLM in conda environment is not implemented yet. "
            "Please use the in-memory runner."
        )


# -------------------------- Concrete model runners -------------------------- #
class ClassificationModelRunner(TraditionalMLModelRunner):
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


class RegressionModelRunner(TraditionalMLModelRunner):
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


class OpenAIChatCompletionRunner(LLModelRunner):
    """Wraps OpenAI's chat completion model."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        if kwargs.get("openai_api_key") is None:
            raise ValueError(
                "OpenAI API key must be provided. Please pass it as the "
                "keyword argument 'openai_api_key'"
            )

        self.openai_api_key = kwargs["openai_api_key"]
        self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the OpenAI chat completion model."""
        openai.api_key = self.openai_api_key
        if self.model_config.get("model") is None:
            warnings.warn("No model specified. Defaulting to model 'gpt-3.5-turbo'.")
        if self.model_config.get("model_parameters") is None:
            warnings.warn("No model parameters specified. Using default parameters.")

    def _get_llm_output(self, input_text: str) -> str:
        """Gets the output from the OpenAI's chat completion model
        for a given input text."""
        return openai.ChatCompletion.create(
            model=self.model_config.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": input_text}],
            **self.model_config.get("model_parameters", {}),
        )["choices"][0]["message"]["content"]


class CohereGenerateModelRunner(LLModelRunner):
    """Wraps Cohere's Generate model."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        if kwargs.get("cohere_api_key") is None:
            raise ValueError(
                "Cohere API key must be provided. Please pass it as the "
                "keyword argument 'cohere_api_key'"
            )

        self.cohere_api_key = kwargs["cohere_api_key"]
        self._initialize_llm()

    def _initialize_llm(self):
        """Initializes Cohere's Generate model."""
        self.cohere_client = cohere.Client(self.cohere_api_key)
        if self.model_config.get("model") is None:
            warnings.warn("No model specified. Defaulting to model 'command'.")
        if self.model_config.get("model_parameters") is None:
            warnings.warn("No model parameters specified. Using default parameters.")

    def _get_llm_output(self, input_text: str) -> str:
        """Gets the output from Cohere's generate model
        for a given input text."""
        return self.cohere_client.generate(
            model=self.model_config.get("model", "command"),
            prompt=input_text,
            **self.model_config.get("model_parameters", {}),
        )[0].text


class AnthropicModelRunner(LLModelRunner):
    """Wraps Anthropic's models."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        if kwargs.get("anthropic_api_key") is None:
            raise ValueError(
                "Anthropic API key must be provided. Please pass it as the "
                "keyword argument 'anthropic_api_key'"
            )

        self.anthropic_api_key = kwargs["anthropic_api_key"]
        self._initialize_llm()

    def _initialize_llm(self):
        """Initializes Cohere's Generate model."""
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.anthropic_api_key,
        )
        if self.model_config.get("model") is None:
            warnings.warn("No model specified. Defaulting to model 'claude-2'.")
        if self.model_config.get("model_parameters") is None:
            warnings.warn("No model parameters specified. Using default parameters.")
            self.model_config["model_parameters"]["max_tokens_to_sample"] = 200
        elif "max_tokens_to_sample" not in self.model_config.get("model_parameters"):
            warnings.warn(
                "max_tokens_to_sample not specified. Using default max_tokens_to_sample of 200.",
            )
            self.model_config["model_parameters"]["max_tokens_to_sample"] = 200

    def _get_llm_output(self, input_text: str) -> str:
        """Gets the output from Cohere's generate model
        for a given input text."""
        return self.anthropic_client.completions.create(
            model=self.model_config.get("model", "claude-2"),
            prompt=f"{anthropic.HUMAN_PROMPT} {input_text} {anthropic.AI_PROMPT}",
            **self.model_config.get("model_parameters", {}),
        )["completion"]


# ----------------------------- Factory function ----------------------------- #
def get_model_runner(
    **kwargs,
) -> ModelRunnerInterface:
    """Factory function to get the correct model runner for the specified task type."""
    task_type = kwargs.get("task_type")
    model_package = kwargs.get("model_package")
    logger = kwargs.get("logger") or logging.getLogger("validators")

    # Try to infer task type if not provided
    if task_type is None and model_package is None:
        raise ValueError(
            "Task type could not be inferred. "
            "You must provide either the task type (task_type) as a keyword argument "
            "or the model package (model_package) with a model_config.yaml file that "
            "contains the task type written (taskType)."
        )
    elif task_type is None and model_package is not None:
        # Model config keys are originally camelCase, but we want to use snake_case
        model_config = utils.camel_to_snake_dict(
            utils.read_yaml(f"{model_package}/model_config.yaml")
        )
        task_type = tasks.TaskType(model_config.get("task_type"))
    elif task_type is not None and model_package is None:
        model_config = kwargs
    else:
        model_config = utils.camel_to_snake_dict(
            utils.read_yaml(f"{model_package}/model_config.yaml")
        )

    if task_type is None:
        raise ValueError(
            "Task type could not be inferred. "
            "You must provide either the task type (task_type) as a keyword argument "
            "or the model package (model_package) with a model_config.yaml file that "
            "contains the task type written (taskType)."
        )

    if task_type in [
        tasks.TaskType.TabularClassification,
        tasks.TaskType.TextClassification,
    ]:
        return ClassificationModelRunner(logger=logger, **kwargs)
    elif task_type == tasks.TaskType.TabularRegression:
        return RegressionModelRunner(logger=logger, **kwargs)
    elif task_type in [
        tasks.TaskType.LLM,
        tasks.TaskType.LLMNER,
        tasks.TaskType.LLMQuestionAnswering,
        tasks.TaskType.LLMSummarization,
        tasks.TaskType.LLMTranslation,
    ]:
        if model_package is not None:
            model_provider = model_config.get("model_provider")
        else:
            model_provider = kwargs.get("model_provider")

        if model_provider == "OpenAI":
            return OpenAIChatCompletionRunner(logger=logger, **kwargs)
        elif model_provider == "Cohere":
            return CohereGenerateModelRunner(logger=logger, **kwargs)
        elif model_provider == "Anthropic":
            return AnthropicModelRunner(logger=logger, **kwargs)
        else:
            raise ValueError(f"Model provider `{model_provider}` is not supported.")
    else:
        raise ValueError(f"Task type `{task_type}` is not supported.")
