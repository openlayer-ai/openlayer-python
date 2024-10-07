"""Series of helper functions and classes that are used throughout the
Openlayer SDK.
"""

import json
import os
from typing import Optional

import yaml


# ----------------------------- Helper functions ----------------------------- #
def get_env_variable(name: str) -> Optional[str]:
    """Returns the value of the specified environment variable.

    Args:
        name (str): the name of the environment variable.

    Returns:
        str: the value of the specified environment variable.
    """
    try:
        return os.environ[name]
    except KeyError:
        return None


def write_yaml(dictionary: dict, filename: str):
    """Writes the dictionary to a YAML file in the specified directory (`dir`).

    Args:
        dictionary (dict): the dictionary to write to a YAML file.
        dir (str): the directory to write the file to.
    """
    with open(filename, "w", encoding="UTF-8") as stream:
        yaml.dump(dictionary, stream)


def json_serialize(data):
    """
    Recursively attempts to convert data into JSON-serializable formats.
    """
    if isinstance(data, (str, int, float, bool, type(None))):
        return data  # Already JSON-serializable
    elif isinstance(data, dict):
        return {k: json_serialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_serialize(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(json_serialize(item) for item in data)
    else:
        # Fallback: Convert to string if not serializable
        try:
            json.dumps(data)
            return data  # Data was serializable
        except TypeError:
            return str(data)  # Not serializable, convert to string
