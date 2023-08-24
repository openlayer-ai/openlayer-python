"""Module for storing constants used throughout the OpenLayer Python Client.
"""
import os

# ---------------------------- Commit/staging flow --------------------------- #
VALID_RESOURCE_NAMES = {"model", "training", "validation", "fine-tuning"}
OPENLAYER_DIR = os.path.join(os.path.expanduser("~"), ".openlayer")

# -------------------------------- Size limits ------------------------------- #
MAXIMUM_CHARACTER_LIMIT = 10000
MAXIMUM_TAR_FILE_SIZE = 25  # MB

# ----------------------------------- APIs ----------------------------------- #
REQUESTS_TIMEOUT = 60 * 60 * 3  # 3 hours
