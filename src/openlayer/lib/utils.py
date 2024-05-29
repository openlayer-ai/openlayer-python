"""Series of helper functions and classes that are used throughout the
Openlayer SDK.
"""

import os
import json
from typing import Optional


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
