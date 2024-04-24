# pylint: disable=invalid-name,broad-exception-raised, consider-using-with
"""
Module with the concrete LLM runners.
"""

import datetime
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import openai
import pandas as pd
import pybars
from tqdm import tqdm

from .. import constants
from .. import exceptions as openlayer_exceptions
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

    def run(
        self, input_data: pd.DataFrame, output_column_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Runs the input data through the model."""
        if self.in_memory:
            return self._run_in_memory(
                input_data=input_data,
                output_column_name=output_column_name,
            )
        else:
            return self._run_in_conda(
                input_data=input_data, output_column_name=output_column_name
            )

    def _run_in_memory(
        self,
        input_data: pd.DataFrame,
        output_column_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Runs the input data through the model in memory and returns a pandas
        dataframe."""
        for output_df, _ in tqdm(
            self._run_in_memory_and_yield_progress(input_data, output_column_name),
            total=len(input_data),
            colour="BLUE",
        ):
            pass
        # pylint: disable=undefined-loop-variable
        return output_df

    def _run_in_memory_and_yield_progress(
        self,
        input_data: pd.DataFrame,
        output_column_name: Optional[str] = None,
    ) -> Generator[Tuple[pd.DataFrame, float], None, None]:
        """Runs the input data through the model in memory and yields the results
        and the progress."""
        self.logger.info("Running LLM in memory...")

        model_outputs = []
        timestamps = []
        run_exceptions = []
        run_cost = 0
        total_rows = len(input_data)
        current_row = 0

        for _, input_data_row in input_data.iterrows():
            # Check if output column already has a value to avoid re-running
            if output_column_name and output_column_name in input_data_row:
                output_value = input_data_row[output_column_name]
                if output_value is not None:
                    model_outputs.append(output_value)
                    if "output_time_utc" in input_data_row:
                        timestamps.append(input_data_row["output_time_utc"])
                    else:
                        timestamps.append(datetime.datetime.utcnow().isoformat())
                    current_row += 1
                    yield pd.DataFrame(
                        {"output": model_outputs, "output_time_utc": timestamps}
                    ), current_row / total_rows
                    continue

            output, cost, exceptions = self._run_single_input(input_data_row)

            model_outputs.append(output)
            run_cost += cost
            run_exceptions.append(exceptions)
            timestamps.append(datetime.datetime.utcnow().isoformat())
            current_row += 1

            yield pd.DataFrame(
                {
                    "output": model_outputs,
                    "output_time_utc": timestamps,
                    "exceptions": run_exceptions,
                }
            ), current_row / total_rows

        if (
            len(run_exceptions) > 0
            and None not in run_exceptions
            and len(set(run_exceptions)) == 1
        ):
            raise openlayer_exceptions.OpenlayerLlmException(
                f"Calculating all outputs failed with: {run_exceptions[0]}"
            )

        self.logger.info("Successfully ran data through the model!")

        self._report_exceptions(set(run_exceptions))
        self.cost_estimates.append(run_cost)

        yield pd.DataFrame(
            {
                "output": model_outputs,
                "output_time_utc": timestamps,
                "exceptions": run_exceptions,
            }
        ), 1.0

    def _run_single_input(
        self, input_data_row: pd.Series
    ) -> Tuple[str, float, Optional[Exception]]:
        """Runs the LLM on a single row of input data.

        Returns a tuple of the output, cost, and exceptions encountered.
        """
        input_variables_dict = input_data_row[
            self.model_config["input_variable_names"]
        ].to_dict()
        injected_prompt = self._inject_prompt(input_variables_dict=input_variables_dict)
        llm_input = self._get_llm_input(injected_prompt)

        try:
            outputs = self._get_llm_output(llm_input)
            return outputs["output"], outputs["cost"], None
        # pylint: disable=broad-except
        except Exception as exc:
            return None, 0, exc

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

    def _report_exceptions(self, exceptions: set) -> None:
        if len(exceptions) == 1 and None in exceptions:
            return
        warnings.warn(
            f"We couldn't get the outputs for all rows.\n"
            "Encountered the following exceptions while running the model: \n"
            f"{exceptions}\n"
            "After you fix the issues, you can call the `run` method again and provide "
            "the `output_column_name` argument to avoid re-running the model on rows "
            "that already have an output value."
        )

    def _run_in_conda(
        self, input_data: pd.DataFrame, output_column_name: Optional[str] = None
    ) -> pd.DataFrame:
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

    def run_and_yield_progress(
        self, input_data: pd.DataFrame, output_column_name: Optional[str] = None
    ) -> Generator[Tuple[pd.DataFrame, float], None, None]:
        """Runs the input data through the model and yields progress."""
        if self.in_memory:
            yield from self._run_in_memory_and_yield_progress(
                input_data=input_data,
                output_column_name=output_column_name,
            )
        else:
            raise NotImplementedError(
                "Running LLM in conda environment is not implemented yet. "
                "Please use the in-memory runner."
            )


# -------------------------- Concrete model runners -------------------------- #


class OpenAIChatCompletionRunner(LLModelRunner):
    """Wraps OpenAI's chat completion model."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        if kwargs.get("openai_api_key") is None:
            raise openlayer_exceptions.OpenlayerMissingLlmApiKey(
                "Please pass your OpenAI API key as the "
                "keyword argument 'openai_api_key'"
            )

        self.openai_client = openai.OpenAI(api_key=kwargs["openai_api_key"])
        self._initialize_llm()

        self.cost: List[float] = []

    def _initialize_llm(self):
        """Initializes the OpenAI chat completion model."""
        # Check if API key is valid
        try:
            self.openai_client.models.list()
        except Exception as e:
            raise openlayer_exceptions.OpenlayerInvalidLlmApiKey(
                "Please pass a valid OpenAI API key as the "
                f"keyword argument 'openai_api_key' \n Error message: {e}"
            ) from e
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
        response = self.openai_client.chat.completions.create(
            model=self.model_config.get("model", "gpt-3.5-turbo"),
            messages=llm_input,
            **self.model_config.get("model_parameters", {}),
        )
        return response

    def _get_output(self, response: Dict[str, Any]) -> str:
        """Gets the output from the response."""
        return response.choices[0].message.content

    def _get_cost_estimate(self, response: Dict[str, Any]) -> None:
        """Estimates the cost from the response."""
        model = self.model_config.get("model", "gpt-3.5-turbo")
        if model not in constants.OPENAI_COST_PER_TOKEN:
            return -1
        else:
            num_input_tokens = response.usage.prompt_tokens
            num_output_tokens = response.usage.completion_tokens
            return (
                num_input_tokens * constants.OPENAI_COST_PER_TOKEN[model]["input"]
                + num_output_tokens * constants.OPENAI_COST_PER_TOKEN[model]["output"]
            )
