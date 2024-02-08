"""Module for storing constants used throughout the OpenLayer Python Client.
"""
import os

import marshmallow as ma

# ---------------------------- Commit/staging flow --------------------------- #
VALID_RESOURCE_NAMES = {"model", "training", "validation", "fine-tuning"}
OPENLAYER_DIR = os.path.join(os.path.expanduser("~"), ".openlayer")

# -------------------------------- Size limits ------------------------------- #
MAXIMUM_CHARACTER_LIMIT = 50000
MAXIMUM_TAR_FILE_SIZE = 25  # MB

# ----------------------------------- APIs ----------------------------------- #
REQUESTS_TIMEOUT = 60 * 60 * 3  # 3 hours

# ---------------------------- Validation patterns --------------------------- #
COLUMN_NAME_REGEX = validate = ma.validate.Regexp(
    r"^(?!openlayer)[a-zA-Z0-9_-]+$",
    error="strings that are not alphanumeric with underscores or hyphens."
    + " Spaces and special characters are not allowed."
    + " The string cannot start with `openlayer`.",
)
LANGUAGE_CODE_REGEX = ma.validate.Regexp(
    r"^[a-z]{2}(-[A-Z]{2})?$",
    error="`language` of the dataset is not in the ISO 639-1 (alpha-2 code) format.",
)

COLUMN_NAME_VALIDATION_LIST = [
    ma.validate.Length(
        min=1,
        max=60,
    ),
    COLUMN_NAME_REGEX,
]
# --------------------------- LLM usage costs table -------------------------- #
# Last update: 2024-02-05
OPENAI_COST_PER_TOKEN = {
    "babbage-002": {
        "input": 0.0004e-3,
        "output": 0.0004e-3,
    },
    "davinci-002": {
        "input": 0.002e-3,
        "output": 0.002e-3,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005e-3,
        "output": 0.0015e-3,
    },
    "gpt-3.5-turbo-0125": {
        "input": 0.0005e-3,
        "output": 0.0015e-3,
    },
    "gpt-3.5-turbo-0301": {
        "input": 0.0015e-3,
        "output": 0.002e-3,
    },
    "gpt-3.5-turbo-0613": {
        "input": 0.0015e-3,
        "output": 0.002e-3,
    },
    "gpt-3.5-turbo-1106": {
        "input": 0.001e-3,
        "output": 0.002e-3,
    },
    "gpt-3.5-turbo-16k-0613": {
        "input": 0.003e-3,
        "output": 0.004e-3,
    },
    "gpt-3.5-turbo-instruct": {
        "input": 0.0015e-3,
        "output": 0.002e-3,
    },
    "gpt-4": {
        "input": 0.03e-3,
        "output": 0.06e-3,
    },
    "gpt-4-0125-preview": {
        "input": 0.01e-3,
        "output": 0.03e-3,
    },
    "gpt-4-1106-preview": {
        "input": 0.01e-3,
        "output": 0.03e-3,
    },
    "gpt-4-0314": {
        "input": 0.03e-3,
        "output": 0.06e-3,
    },
    "gpt-4-1106-vision-preview": {
        "input": 0.01e-3,
        "output": 0.03e-3,
    },
    "gpt-4-32k": {
        "input": 0.06e-3,
        "output": 0.12e-3,
    },
    "gpt-4-32k-0314": {
        "input": 0.06e-3,
        "output": 0.12e-3,
    },
}
