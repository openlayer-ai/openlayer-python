# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module with the concrete LLM runners.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import anthropic
import cohere
import openai
import pandas as pd
import pybars

from . import base_model_runner


class LLModelRunner(base_model_runner.ModelRunnerInterface, ABC):
    """Extends the base model runner for LLMs."""

    cost_estimates: List[float] = []

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

        run_cost = 0
        for input_data_row in input_data_df.iterrows():
            input_variables_dict = input_data_row[1][
                self.model_config["input_variable_names"]
            ].to_dict()
            injected_prompt = self._inject_prompt(
                input_variables_dict=input_variables_dict
            )
            llm_input = self._get_llm_input(injected_prompt)

            try:
                result = self._get_llm_output(llm_input)
                model_outputs.append(result["output"])
                run_cost += result["cost"]
            except Exception as exc:
                model_outputs.append(
                    f"[Error] Could not get predictions for row: {exc}"
                )

        self.logger.info("Successfully ran data through the model!")
        self.cost_estimates.append(run_cost)
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

    def _get_llm_output(
        self, llm_input: Union[List, str]
    ) -> Dict[str, Union[float, str]]:
        """Implements the logic to get the output from the language model for
        a given input text."""
        response = self._make_request(llm_input)
        return self._parse_response(response)

    @abstractmethod
    def _make_request(self, llm_input: Union[List, str]) -> Dict[str, Any]:
        """Makes a request to the language model."""
        pass

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """Parses the response from the LLM, extracting the cost and the output."""
        output = self._get_output(response)
        cost = self._get_cost_estimate(response)
        return {
            "output": output,
            "cost": cost,
        }

    @abstractmethod
    def _get_output(self, response: Dict[str, Any]) -> str:
        """Extracts the output from the response."""
        pass

    @abstractmethod
    def _get_cost_estimate(self, response: Dict[str, Any]) -> float:
        """Extracts the cost from the response."""
        pass

    def _run_in_conda(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Runs LLM prediction job in a conda environment."""
        raise NotImplementedError(
            "Running LLM in conda environment is not implemented yet. "
            "Please use the in-memory runner."
        )

    def get_cost_estimate(self, num_of_runs: Optional[int] = None) -> float:
        """Returns the cost estimate of the last num_of_runs."""
        if len(self.cost_estimates) == 0:
            return 0
        if num_of_runs is not None:
            if num_of_runs > len(self.cost):
                warnings.warn(
                    f"Number of runs ({num_of_runs}) is greater than the number of "
                    f"runs that have been executed with this runner ({len(self.cost_estimates)}). "
                    "Returning the cost of all runs so far."
                )
                return sum(self.cost_estimates)
            else:
                return sum(self.cost_estimates[-num_of_runs:])
        return self.cost_estimates[-1]


# -------------------------- Concrete model runners -------------------------- #
class AnthropicModelRunner(LLModelRunner):
    """Wraps Anthropic's models."""

    # Last update: 2023-08-15
    COST_PER_TOKEN = {
        "claude-2": {
            "input": 11.02e-6,
            "output": 32.68e-6,
        },
        "claude-instant": {
            "input": 1.63e-6,
            "output": 5.51e-6,
        },
    }

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

    def _make_request(self, llm_input: str) -> Dict[str, Any]:
        """Make the request to Anthropic's model
        for a given input."""
        return self.anthropic_client.completions.create(
            model=self.model_config.get("model", "claude-2"),
            prompt=llm_input,
            **self.model_config.get("model_parameters", {}),
        )

    def _get_output(self, response: Dict[str, Any]) -> str:
        """Gets the output from the response."""
        return response["completion"]

    def _get_cost_estimate(self, response: Dict[str, Any]) -> float:
        """Estimates the cost from the response."""
        return -1


class CohereGenerateModelRunner(LLModelRunner):
    """Wraps Cohere's Generate model."""

    # Last update: 2023-08-15
    COST_PER_TOKEN = 0.000015

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
        # Check if API key is valid -- Cohere's validation seems to be very shallow
        try:
            self.cohere_client = cohere.Client(
                api_key=self.cohere_api_key, check_api_key=True
            )
        except Exception as e:
            raise ValueError(
                "Cohere API key is invalid. Please pass a valid API key as the "
                f"keyword argument 'cohere_api_key' \n Error message: {e}"
            )
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

    def _make_request(self, llm_input: str) -> Dict[str, Any]:
        """Make the request to Cohere's Generate model
        for a given input."""
        return self.cohere_client.generate(
            model=self.model_config.get("model", "command"),
            prompt=llm_input,
            **self.model_config.get("model_parameters", {}),
        )

    def _get_output(self, response: Dict[str, Any]) -> str:
        """Gets the output from the response."""
        return response[0].text

    def _get_cost_estimate(self, response: Dict[str, Any]) -> float:
        """Estimates the cost from the response."""
        return -1


class OpenAIChatCompletionRunner(LLModelRunner):
    """Wraps OpenAI's chat completion model."""

    # Last update: 2023-08-15
    COST_PER_TOKEN = {
        "gpt-3.5-turbo": {
            "input": 0.0015e-3,
            "output": 0.002e-3,
        },
        "gpt-4": {
            "input": 0.03e-3,
            "output": 0.06e-3,
        },
    }

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

        self.cost: List[float] = []

    def _initialize_llm(self):
        """Initializes the OpenAI chat completion model."""
        openai.api_key = self.openai_api_key

        # Check if API key is valid
        try:
            openai.Model.list()
        except Exception as e:
            raise ValueError(
                "OpenAI API key is invalid. Please pass a valid API key as the "
                f"keyword argument 'openai_api_key' \n Error message: {e}"
            )
        if self.model_config.get("model") is None:
            warnings.warn("No model specified. Defaulting to model 'gpt-3.5-turbo'.")
        if self.model_config.get("model_parameters") is None:
            warnings.warn("No model parameters specified. Using default parameters.")

    def _get_llm_input(
        self, injected_prompt: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepares the input for OpenAI's chat completion model."""
        return injected_prompt

    def _make_request(self, llm_input: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make the request to OpenAI's chat completion model
        for a given input."""
        return openai.ChatCompletion.create(
            model=self.model_config.get("model", "gpt-3.5-turbo"),
            messages=llm_input,
            **self.model_config.get("model_parameters", {}),
        )

    def _get_output(self, response: Dict[str, Any]) -> str:
        """Gets the output from the response."""
        return response["choices"][0]["message"]["content"]

    def _get_cost_estimate(self, response: Dict[str, Any]) -> None:
        """Estimates the cost from the response."""
        model = self.model_config.get("model", "gpt-3.5-turbo")
        if model not in self.COST_PER_TOKEN:
            return -1
        else:
            num_input_tokens = response["usage"]["prompt_tokens"]
            num_output_tokens = response["usage"]["completion_tokens"]
            return (
                num_input_tokens * self.COST_PER_TOKEN[model]["input"]
                + num_output_tokens * self.COST_PER_TOKEN[model]["output"]
            )
