"""Module for storing constants used throughout the OpenLayer Python Client.
"""
import os

import marshmallow as ma

# ---------------------------- Commit/staging flow --------------------------- #
VALID_RESOURCE_NAMES = {"model", "training", "validation", "fine-tuning"}
OPENLAYER_DIR = os.path.join(os.path.expanduser("~"), ".openlayer")

# -------------------------------- Size limits ------------------------------- #
MAXIMUM_CHARACTER_LIMIT = 10000
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
