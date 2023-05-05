"""Series of helper functions and classes that are used throughout the
OpenLayer Python client.
"""
import io
import logging
import os
import sys
import traceback
import warnings
from typing import Any, Dict

import pandas as pd
import yaml


# -------------------------- Helper context managers ------------------------- #
class LogStdout:
    """Helper class that suppresses the prints and writes them to the `log_file_path` file."""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(self.log_file_path, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HidePrints:
    """Helper class that suppresses the prints and warnings to stdout and Jupyter's stdout.

    Used as a context manager to hide the print / warning statements that can be inside the user's
    function while we test it.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("default")


# ----------------------------- Helper functions ----------------------------- #
def log_subprocess_output(logger: logging.Logger, pipe: io.BufferedReader):
    """Logs the output of a subprocess."""
    for line in iter(pipe.readline, b""):  # b'\n'-separated lines
        line = line.decode("UTF-8").strip()
        logger.info("%s", line)


def write_python_version(directory: str):
    """Writes the python version to the file `python_version` in the specified
    directory (`directory`).

    This is used to register the Python version of the user's environment in the
    when they are uploading a model package.

    Args:
        directory (str): the directory to write the file to.
    """
    with open(f"{directory}/python_version", "w", encoding="UTF-8") as file:
        file.write(
            str(sys.version_info.major)
            + "."
            + str(sys.version_info.minor)
            + "."
            + str(sys.version_info.micro)
        )


def remove_python_version(directory: str):
    """Removes the file `python_version` from the specified directory
    (`directory`).

    Args:
        directory (str): the directory to remove the file from.
    """
    os.remove(f"{directory}/python_version")


def read_yaml(filename: str) -> dict:
    """Reads a YAML file and returns it as a dictionary.

    Args:
        filename (str): the path to the YAML file.

    Returns:
        dict: the dictionary representation of the YAML file.
    """
    with open(filename, "r", encoding="UTF-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(dictionary: dict, filename: str):
    """Writes the dictionary to a YAML file in the specified directory (`dir`).

    Args:
        dictionary (dict): the dictionary to write to a YAML file.
        dir (str): the directory to write the file to.
    """
    with open(filename, "w", encoding="UTF-8") as stream:
        yaml.dump(dictionary, stream)


def get_exception_stacktrace(err: Exception):
    """Returns the stacktrace of the most recent exception.

    Returns:
        str: the stacktrace of the most recent exception.
    """
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))


def list_resources_in_bundle(bundle_path: str) -> list:
    """Lists the resources in the bundle.

    Args:
        bundle_path (str): the path to the bundle.

    Returns:
        list: the list of resources in the bundle.
    """
    # TODO: factor out list of valid resources
    # pylint: disable=invalid-name
    VALID_RESOURCES = {"baseline-model", "model", "training", "validation"}

    resources = []

    for resource in os.listdir(bundle_path):
        if resource in VALID_RESOURCES:
            resources.append(resource)
    return resources


def load_dataset_from_bundle(bundle_path: str, label: str) -> pd.DataFrame:
    """Loads a dataset from a commit bundle.

    Parameters
    ----------
    label : str
        The type of the dataset. Can be either "training" or "validation".

    Returns
    -------
    pd.DataFrame
        The dataset.
    """
    dataset_file_path = f"{bundle_path}/{label}/dataset.csv"

    dataset_df = pd.read_csv(dataset_file_path)

    return dataset_df


def load_dataset_config_from_bundle(bundle_path: str, label: str) -> Dict[str, Any]:
    """Loads a dataset config from a commit bundle.

    Parameters
    ----------
    label : str
        The type of the dataset. Can be either "training" or "validation".

    Returns
    -------
    Dict[str, Any]
        The dataset config.
    """
    dataset_config_file_path = f"{bundle_path}/{label}/dataset_config.yaml"

    with open(dataset_config_file_path, "r", encoding="UTF-8") as stream:
        dataset_config = yaml.safe_load(stream)

    return dataset_config


def load_model_config_from_bundle(bundle_path: str) -> Dict[str, Any]:
    """Loads a model config from a commit bundle.

    Returns
    -------
    Dict[str, Any]
        The model config.
    """
    model_config_file_path = f"{bundle_path}/model/model_config.yaml"

    with open(model_config_file_path, "r", encoding="UTF-8") as stream:
        model_config = yaml.safe_load(stream)

    return model_config
