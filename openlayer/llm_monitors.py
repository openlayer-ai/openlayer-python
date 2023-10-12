"""Module with classes for monitoring calls to LLMs."""

import logging
import time
from typing import Dict, List, Optional

import openai
import pandas as pd

import openlayer

from . import utils

logger = logging.getLogger(__name__)


class OpenAIMonitor:
    """Class for monitoring calls to OpenAI LLMs."""

    def __init__(
        self,
        publish: bool = False,
        openlayer_api_key: Optional[str] = None,
        openlayer_project_name: Optional[str] = None,
        openlayer_inference_pipeline_name: Optional[str] = None,
    ) -> None:
        # ------------------------------ Openlayer setup ----------------------------- #
        if openlayer_api_key is None:
            openlayer_api_key = utils.get_env_variable("OPENLAYER_API_KEY")
        if openlayer_project_name is None:
            openlayer_project_name = utils.get_env_variable("OPENLAYER_PROJECT_NAME")
        if openlayer_inference_pipeline_name is None:
            openlayer_inference_pipeline_name = utils.get_env_variable(
                "OPENLAYER_INFERENCE_PIPELINE_NAME"
            )
        if publish and (openlayer_api_key is None or openlayer_project_name is None):
            raise ValueError(
                "To publish data to Openlayer, you must provide an API key and "
                "a project name. This can be done by setting the environment "
                "variables `OPENLAYER_API_KEY` and `OPENLAYER_PROJECT_NAME`, or by "
                "passing them as arguments to the OpenAIMonitor constructor "
                "(`openlayer_api_key` and `openlayer_project_name`, respectively)."
            )
        self.openlayer_api_key = openlayer_api_key
        self.openlayer_project_name = openlayer_project_name
        self.openlayer_inference_pipeline_name = openlayer_inference_pipeline_name
        self.publish = publish
        self.inference_pipeline = self._load_inference_pipeline()

        # ------------------------------- OpenAI setup ------------------------------- #
        self.create_chat_completion = openai.ChatCompletion.create
        self.create_completion = openai.Completion.create
        self.modified_create_chat_completion = (
            self._get_modified_create_chat_completion()
        )
        self.modified_create_completion = self._get_modified_create_completion()

        self.df = pd.DataFrame(columns=["input", "output", "tokens", "latency"])
        self.monitoring_on = False

    def _load_inference_pipeline(self) -> Optional[openlayer.InferencePipeline]:
        """Load inference pipeline from the Openlayer platform.

        If no platform/project information is provided, returns None.
        """
        inference_pipeline = None
        if self.openlayer_api_key and self.openlayer_project_name:
            with utils.HidePrints():
                client = openlayer.OpenlayerClient(
                    api_key=self.openlayer_api_key,
                )
                project = client.load_project(name=self.openlayer_project_name)
                inference_pipeline = project.load_inference_pipeline(
                    name=self.openlayer_inference_pipeline_name
                )
        return inference_pipeline

    def _get_modified_create_chat_completion(self) -> callable:
        """Returns a modified version of the create method for openai.ChatCompletion."""

        def modified_create_chat_completion(*args, **kwargs) -> str:
            start_time = time.time()
            response = self.create_chat_completion(*args, **kwargs)
            latency = (time.time() - start_time) * 1000

            try:
                input_data = self.format_user_messages(kwargs["messages"])
                output_data = response.choices[0].message.content.strip()
                num_of_tokens = response.usage["total_tokens"]

                self._append_row_to_df(
                    input_data=input_data,
                    output_data=output_data,
                    num_of_tokens=num_of_tokens,
                    latency=latency,
                )

                self._handle_data_publishing()
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Failed to track chat request. %s", e)

            return response

        return modified_create_chat_completion

    def _get_modified_create_completion(self) -> callable:
        """Returns a modified version of the create method for openai.Completion"""

        def modified_create_completion(*args, **kwargs):
            start_time = time.time()
            response = self.create_completion(*args, **kwargs)
            latency = (time.time() - start_time) * 1000

            try:
                prompts = kwargs.get("prompt", [])
                prompts = [prompts] if isinstance(prompts, str) else prompts
                choices_splits = self.split_list(response.choices, len(prompts))

                for input_data, choices in zip(prompts, choices_splits):
                    output_data = choices[0].text.strip()
                    num_of_tokens = int(response.usage["total_tokens"] / len(prompts))

                    self._append_row_to_df(
                        input_data=input_data,
                        output_data=output_data,
                        num_of_tokens=num_of_tokens,
                        latency=latency,
                    )

                    self._handle_data_publishing()
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Failed to track completion request. %s", e)

            return response

        return modified_create_completion

    @staticmethod
    def format_user_messages(conversation_list: List[Dict[str, str]]) -> str:
        """Extracts the 'user' messages from the conversation list and returns them
        as a single string."""
        return "\n".join(
            item["content"] for item in conversation_list if item["role"] == "user"
        ).strip()

    @staticmethod
    def split_list(lst: List, n_parts: int) -> List[List]:
        """Split a list into n_parts."""
        # Calculate the base size and the number of larger parts
        base_size, extra = divmod(len(lst), n_parts)

        start = 0
        end = 0
        result = []
        for i in range(n_parts):
            # Calculate the size for this part
            part_size = base_size + 1 if i < extra else base_size

            # Update the end index for slicing
            end += part_size

            result.append(lst[start:end])

            # Update the start index for the next iteration
            start = end
        return result

    def _append_row_to_df(
        self, input_data: str, output_data: str, num_of_tokens: int, latency: float
    ) -> None:
        """Appends a row with input/output, number of tokens, and latency to the
        df."""
        row = pd.DataFrame(
            [
                {
                    "input": input_data,
                    "output": output_data,
                    "tokens": num_of_tokens,
                    "latency": latency,
                }
            ]
        )
        self.df = pd.concat([self.df, row], ignore_index=True)
        self.df = self.df.astype(
            {"input": object, "output": object, "tokens": int, "latency": float}
        )

    def _handle_data_publishing(self) -> None:
        """Handle data publishing.

        If `publish` is set to True, publish the latest row to Openlayer.
        """
        if self.publish:
            self.inference_pipeline.publish_batch_data(
                batch_df=self.df.tail(1), batch_config=self.data_config
            )

    def start_monitoring(self) -> None:
        """Setup monitoring for every call to OpenAI LLMs."""
        if self.monitoring_on:
            print("Monitoring is already on!\nTo stop it, call `stop_monitoring`.")
            return
        self.monitoring_on = True
        self._overwrite_completion_methods()
        print("All the calls to OpenAI models are now being monitored!")
        if self.publish:
            print(
                "Furthermore, since `publish` was set to True, the data is being"
                f" published to your '{self.openlayer_project_name}' Openlayer project."
            )
        print("The `data` attribute contails all the data collected.")
        print("To stop monitoring, call the `stop_monitoring` method.")

    def _overwrite_completion_methods(self) -> None:
        """Overwrites OpenAI's completion methods with the modified versions."""
        openai.ChatCompletion.create = self.modified_create_chat_completion
        openai.Completion.create = self.modified_create_completion

    def stop_monitoring(self):
        """Stop monitoring calls to OpenAI LLMs."""
        self._restore_completion_methods()
        self.monitoring_on = False
        print("Monitoring stopped.")
        if not self.publish:
            print(
                "To publish the data collected so far to your Openlayer project, "
                "call the `publish_batch_data` method."
            )

    def _restore_completion_methods(self) -> None:
        """Restores OpenAI's completion methods to their original versions."""
        openai.ChatCompletion.create = self.create_chat_completion
        openai.Completion.create = self.create_completion

    def publish_batch_data(self):
        """Publish the accumulated batch of data to Openlayer."""
        if self.inference_pipeline is None:
            raise ValueError(
                "To publish data to Openlayer, you must provide an API key and "
                "a project name to the OpenAIMonitor."
            )
        if self.publish:
            print(
                "You have set `publish` to True, so every request you've made so far"
                " was already published to Openlayer."
            )
            answer = input(
                "Do you want to publish the "
                "accumulated data again to Openlayer? [y/n]:"
            )
            if answer.lower() != "y":
                print(
                    "Canceled data publishing attempt.\nIf you want to use the "
                    "`publish_data` method manually, instantiate the OpenAIMonitor "
                    "with `publish=False`."
                )
                return
        self.inference_pipeline.publish_batch_data(
            batch_df=self.df, batch_config=self.data_config
        )

    @property
    def data_config(self) -> Dict[str, any]:
        """Data config for the df. Used for publishing data to Openlayer."""
        return {
            "inputVariableNames": ["input"],
            "label": "production",
            "outputColumnName": "output",
            "numOfTokenColumnName": "tokens",
            "latencyColumnName": "latency",
        }

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe with the input/output, number of tokens, and latency for each
        request after the `monitor` method was called."""
        return self.df
