# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module with the concrete LLM runners.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import anthropic
import cohere
import openai
import pandas as pd
import pybars

from . import base_model_runner


class LLModelRunner(base_model_runner.ModelRunnerInterface, ABC):
    """Extends the base model runner for LLMs."""

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
            or self.model_config.get("prompt") is None
        ):
            raise ValueError("Input variable names and prompt must be provided.")

        for message in self.model_config["prompt"]:
            if message.get("role") is None or message.get("content") is None:
                raise ValueError(
                    "Every item in the 'prompt' list must contain "
                    "'role' and 'content' keys."
                )
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    "The 'role' key in the 'prompt' list must be one of "
                    "'system', 'user', or 'assistant'."
                )

    def _run_in_memory(self, input_data_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the input data through the model in memory."""
        self.logger.info("Running LLM in memory...")
        model_outputs = []

        for input_data_row in input_data_df.iterrows():
            input_variables_dict = input_data_row[1][
                self.model_config["input_variable_names"]
            ].to_dict()
            injected_prompt = self._inject_prompt(
                input_variables_dict=input_variables_dict
            )
            llm_input = self._get_llm_input(injected_prompt)

            try:
                model_outputs.append(self._get_llm_output(llm_input))
            except Exception as exc:
                model_outputs.append(
                    f"[Error] Could not get predictions for row: {exc}"
                )

        self.logger.info("Successfully ran data through the model!")
        return pd.DataFrame({"predictions": model_outputs})

    def _inject_prompt(self, input_variables_dict: dict) -> List[Dict[str, str]]:
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

        injected_prompt = []
        for message in self.model_config["prompt"]:
            formatter = compiler.compile(message["content"].strip())
            injected_prompt.append(
                {"role": message["role"], "content": formatter(input_variables_dict)}
            )
        return injected_prompt

    @abstractmethod
    def _get_llm_input(self, injected_prompt: List[Dict[str, str]]) -> Union[List, str]:
        """Implements the logic to prepare the input for the language model."""
        pass

    @abstractmethod
    def _get_llm_output(self, llm_input: Union[List, str]) -> str:
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

    def _get_llm_input(self, injected_prompt: List[Dict[str, str]]) -> str:
        """Prepares the input for Anthropic's generate model."""
        llm_input = ""
        for message in injected_prompt:
            if message["role"] == "assistant":
                llm_input += f"{anthropic.AI_PROMPT} {message['content']} "
            elif message["role"] == "user" or message["role"] == "system":
                llm_input += f"{anthropic.HUMAN_PROMPT} {message['content']} "
            else:
                raise ValueError(
                    "Message role must be either 'assistant', 'user', or 'system' for Anthropic LLMs. "
                    f"Got: {message['role']}"
                )
        llm_input += f"{anthropic.AI_PROMPT}"
        return llm_input

    def _get_llm_output(self, llm_input: str) -> str:
        """Gets the output from Cohere's generate model
        for a given input text."""
        return self.anthropic_client.completions.create(
            model=self.model_config.get("model", "claude-2"),
            prompt=llm_input,
            **self.model_config.get("model_parameters", {}),
        )["completion"]


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

    def _get_llm_input(self, injected_prompt: List[Dict[str, str]]) -> str:
        """Prepares the input for Cohere's generate model."""
        llm_input = ""
        for message in injected_prompt:
            if message["role"] == "system":
                llm_input += f"S: {message['content']} \n"
            elif message["role"] == "assistant":
                llm_input += f"A: {message['content']} \n"
            elif message["role"] == "user":
                llm_input += f"U: {message['content']} \n"
            else:
                raise ValueError(
                    "Message role must be either 'system', 'assistant' or 'user'. "
                    f"Got: {message['role']}"
                )
        llm_input += "A:"
        return llm_input

    def _get_llm_output(self, llm_input: str) -> str:
        """Gets the output from Cohere's generate model
        for a given input text."""
        return self.cohere_client.generate(
            model=self.model_config.get("model", "command"),
            prompt=llm_input,
            **self.model_config.get("model_parameters", {}),
        )[0].text


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

    def _get_llm_input(
        self, injected_prompt: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepares the input for OpenAI's chat completion model."""
        return injected_prompt

    def _get_llm_output(self, llm_input: List[Dict[str, str]]) -> str:
        """Gets the output from the OpenAI's chat completion model
        for a given input text."""
        return openai.ChatCompletion.create(
            model=self.model_config.get("model", "gpt-3.5-turbo"),
            messages=llm_input,
            **self.model_config.get("model_parameters", {}),
        )["choices"][0]["message"]["content"]
