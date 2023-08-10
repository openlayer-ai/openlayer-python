"""Tests LLM runners.

Typical usage example:

    pytest test_llm_runners.py
"""
from typing import Dict

import anthropic
import pandas as pd

# pylint: disable=line-too-long
import pytest

from openlayer.model_runners import ll_model_runners

# --------------------------------- Test data -------------------------------- #
PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": """You will be provided with a product description and seed words, and your task is to generate a list
of product names and provide a short description of the target customer for such product. The output
must be a valid JSON with attributes `names` and `target_custommer`.""",
    },
    {"role": "assistant", "content": "Let's get started!"},
    {
        "role": "user",
        "content": "Product description: \n description: A home milkshake maker \n seed words: fast, healthy, compact",
    },
    {
        "role": "assistant",
        "content": """{
    "names": ["QuickBlend", "FitShake", "MiniMix"]
    "target_custommer": "College students that are into fitness and healthy living"
}""",
    },
    {
        "role": "user",
        "content": """description: {{ description }} \n
seed words: {{ seed_words }}""",
    },
]
INPUT_VARIABLES = ["description", "seed_words"]

DATA = pd.DataFrame(
    {
        "description": [
            "A smartwatch with fitness tracking capabilities",
            "An eco-friendly reusable water bottle",
        ],
        "seed_words": ["smart, fitness, health", "eco-friendly, reusable, water"],
    }
)

# ----------------------------- Expected results ----------------------------- #
OPENAI_PROMPT = [
    *PROMPT[:-1],
    {
        "role": "user",
        "content": """description: A smartwatch with fitness tracking capabilities \n\nseed words: smart, fitness, health""",
    },
]
COHERE_PROMPT = """S: You are a helpful assistant. 
U: You will be provided with a product description and seed words, and your task is to generate a list
of product names and provide a short description of the target customer for such product. The output
must be a valid JSON with attributes `names` and `target_custommer`. 
A: Let\'s get started! 
U: Product description: \n description: A home milkshake maker \n seed words: fast, healthy, compact 
A: {\n    "names": ["QuickBlend", "FitShake", "MiniMix"]\n    "target_custommer": "College students that are into fitness and healthy living"\n} 
U: description: A smartwatch with fitness tracking capabilities \n\nseed words: smart, fitness, health 
A:"""

ANTHROPIC_PROMPT = f"""{anthropic.HUMAN_PROMPT} You are a helpful assistant. {anthropic.HUMAN_PROMPT} You will be provided with a product description and seed words, and your task is to generate a list
of product names and provide a short description of the target customer for such product. The output
must be a valid JSON with attributes `names` and `target_custommer`. {anthropic.AI_PROMPT} Let\'s get started! {anthropic.HUMAN_PROMPT} Product description: 
 description: A home milkshake maker \n seed words: fast, healthy, compact {anthropic.AI_PROMPT} {{\n    "names": ["QuickBlend", "FitShake", "MiniMix"]\n    "target_custommer": "College students that are into fitness and healthy living"\n}} {anthropic.HUMAN_PROMPT} description: A smartwatch with fitness tracking capabilities \n\nseed words: smart, fitness, health {anthropic.AI_PROMPT}"""


# --------------------------------- Fixtures --------------------------------- #
@pytest.fixture
def anthropic_model_runner():
    """Returns an instance of Anthropic's runner."""
    return ll_model_runners.AnthropicModelRunner(
        prompt=PROMPT,
        input_variable_names=INPUT_VARIABLES,
        model="claude-2",
        model_parameters={},
        anthropic_api_key="try-to-guess",
    )


@pytest.fixture
def openai_chat_completion_runner():
    """Returns an instance of the OpenAI chat completion runner."""
    return ll_model_runners.OpenAIChatCompletionRunner(
        prompt=PROMPT,
        input_variable_names=INPUT_VARIABLES,
        model="gpt-3.5-turbo",
        model_parameters={},
        openai_api_key="try-to-guess",
    )


@pytest.fixture
def cohere_generate_runner():
    """Returns an instance of Cohere's generate runner."""
    return ll_model_runners.CohereGenerateModelRunner(
        prompt=PROMPT,
        input_variable_names=INPUT_VARIABLES,
        model="command",
        model_parameters={},
        cohere_api_key="try-to-guess",
    )


@pytest.fixture
def input_data_dict():
    """Returns a dictionary of input data."""
    return {
        "description": "A smartwatch with fitness tracking capabilities",
        "seed_words": "smart, fitness, health",
    }


# ----------------------------- Test functions ------------------------------ #
def test_prompt_injection(
    input_data_dict: Dict[str, str],
    openai_chat_completion_runner: ll_model_runners.OpenAIChatCompletionRunner,
):
    """Tests the prompt injection method."""
    injected_prompt = openai_chat_completion_runner._inject_prompt(input_data_dict)
    assert injected_prompt == OPENAI_PROMPT


def test_openai_chat_completion_input(
    openai_chat_completion_runner: ll_model_runners.OpenAIChatCompletionRunner,
):
    """Tests the input for the OpenAI chat completion runner."""
    input_data = openai_chat_completion_runner._get_llm_input(OPENAI_PROMPT)
    assert input_data == OPENAI_PROMPT


def test_cohere_generate_input(
    input_data_dict: Dict[str, str],
    cohere_generate_runner: ll_model_runners.CohereGenerateModelRunner,
):
    """Tests the input for the Cohere generate runner."""
    injected_prompt = cohere_generate_runner._inject_prompt(input_data_dict)
    input_data = cohere_generate_runner._get_llm_input(injected_prompt)
    assert input_data == COHERE_PROMPT


def test_anthropic_input(
    input_data_dict: Dict[str, str],
    anthropic_model_runner: ll_model_runners.AnthropicModelRunner,
):
    """Tests the input for the Cohere generate runner."""
    injected_prompt = anthropic_model_runner._inject_prompt(input_data_dict)
    input_data = anthropic_model_runner._get_llm_input(injected_prompt)
    assert input_data == ANTHROPIC_PROMPT
