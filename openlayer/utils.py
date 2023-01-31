import os
import sys
import traceback
import warnings

import yaml


# -------------------------- Helper context managers ------------------------- #
class HidePrints:
    """Helper class that suppresses the prints and warnings to stdout and Jupyter's stdout.

    Used as a context manager to hide the print / warning statements that can be inside the user's
    function while we test it.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys._jupyter_stdout = sys.stdout
        warnings.filterwarnings("default")


# ----------------------------- Helper functions ----------------------------- #
def write_python_version(directory: str):
    """Writes the python version to the file `python_version` in the specified
    directory (`directory`).

    This is used to register the Python version of the user's environment in the
    when they are uploading a model package.

    Args:
        directory (str): the directory to write the file to.
    """
    with open(f"{directory}/python_version", "w") as file:
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
    with open(filename, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(dictionary: dict, filename: str):
    """Writes the dictionary to a YAML file in the specified directory (`dir`).

    Args:
        dictionary (dict): the dictionary to write to a YAML file.
        dir (str): the directory to write the file to.
    """
    with open(filename, "w") as stream:
        yaml.dump(dictionary, stream)


def get_exception_stacktrace(err: Exception):
    """Returns the stacktrace of the most recent exception.

    Returns:
        str: the stacktrace of the most recent exception.
    """
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))
