"""Module for storing constants used throughout the OpenLayer SDK.
"""

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
    "gpt-4-turbo-preview": {
        "input": 0.01e-3,
        "output": 0.03e-3,
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
# Last update: 2024-03-26
AZURE_OPENAI_COST_PER_TOKEN = {
    "babbage-002": {
        "input": 0.0004e-3,
        "output": 0.0004e-3,
    },
    "davinci-002": {
        "input": 0.002e-3,
        "output": 0.002e-3,
    },
    "gpt-35-turbo": {"input": 0.0005e-3, "output": 0.0015e-3},
    "gpt-35-turbo-0125": {"input": 0.0005e-3, "output": 0.0015e-3},
    "gpt-35-turbo-instruct": {"input": 0.0015e-3, "output": 0.002e-3},
    "gpt-4-turbo": {"input": 0.01e-3, "output": 0.03e-3},
    "gpt-4-turbo-vision": {"input": 0.01e-3, "output": 0.03e-3},
    "gpt-4-8k": {"input": 0.03e-3, "output": 0.06e-3},
    "gpt-4-32k": {"input": 0.06e-3, "output": 0.12e-3},
}
