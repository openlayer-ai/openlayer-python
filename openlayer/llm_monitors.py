"""Module with classes for monitoring calls to LLMs."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd

import openlayer

from . import tasks, utils

logger = logging.getLogger(__name__)


class OpenAIMonitor:
    """Monitor class used to keep track of OpenAI LLMs inferences.

    Parameters
    ----------
    publish : bool, optional
        Whether to publish the data to Openlayer as soon as it is available. If True,
        the Openlayer credentials must be provided (either as keyword arguments or as
        environment variables).
    accumulate_data : bool, False
        Whether to accumulate the data in a dataframe. If False (default), only the
        latest request is stored. If True, all the requests are stored in a dataframe,
        accessed through the `data` attribute.
    client : openai.api_client.Client, optional
        The OpenAI client. It is required if you are using openai>=1.0.0.
    openlayer_api_key : str, optional
        The Openlayer API key. If not provided, it is read from the environment
        variable ``OPENLAYER_API_KEY``. This is required if `publish` is set to True.
    openlayer_project_name : str, optional
        The Openlayer project name. If not provided, it is read from the environment
        variable ``OPENLAYER_PROJECT_NAME``. This is required if `publish` is set to True.
    openlayer_inference_pipeline_name : str, optional
        The Openlayer inference pipeline name. If not provided, it is read from the
        environment variable ``OPENLAYER_INFERENCE_PIPELINE_NAME``. This is required if
        `publish` is set to True and you gave your inference pipeline a name different
        than the default.

    Examples
    --------

    Let's say that you have a GPT model you want to monitor. You can turn on monitoring
    with Openlayer by simply doing:

    1. Set the environment variables:

    .. code-block:: bash

        export OPENAI_API_KEY=<your-openai-api-key>

        export OPENLAYER_API_KEY=<your-openlayer-api-key>
        export OPENLAYER_PROJECT_NAME=<your-project-name>

    2. Instantiate the monitor:

    >>> from opemlayer import llm_monitors
    >>> from openai import OpenAI
    >>>
    >>> openai_client = OpenAI()
    >>> monitor = llm_monitors.OpenAIMonitor(publish=True, client=openai_client)

    3. Start monitoring:

    >>> monitor.start_monitoring()

    From this point onwards, you can continue making requests to your model normally:

    >>> openai_client.chat.completions.create(
    >>>     model="gpt-3.5-turbo",
    >>>     messages=[
    >>>         {"role": "system", "content": "You are a helpful assistant."},
    >>>         {"role": "user", "content": "How are you doing today?"}
    >>>     ],
    >>> )

    Your data is automatically being published to your Openlayer project!

    If you no longer want to monitor your model, you can stop monitoring by calling:

    >>> monitor.stop_monitoring()

    """

    # Last update: 2024-01-05
    COST_PER_TOKEN = {
        "babbage-002": {
            "input": 0.0004e-3,
            "output": 0.0004e-3,
        },
        "davinci-002": {
            "input": 0.002e-3,
            "output": 0.002e-3,
        },
        "gpt-3.5-turbo": {
            "input": 0.003e-3,
            "output": 0.006e-3,
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
        "gpt-4-0314": {
            "input": 0.03e-3,
            "output": 0.06e-3,
        },
        "gpt-4-1106-preview": {
            "input": 0.01e-3,
            "output": 0.03e-3,
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

    def __init__(
        self,
        publish: bool = False,
        client=None,
        accumulate_data: bool = False,
        openlayer_api_key: Optional[str] = None,
        openlayer_project_name: Optional[str] = None,
        openlayer_inference_pipeline_name: Optional[str] = None,
    ) -> None:
        # Openlayer setup
        self.openlayer_api_key: str = None
        self.openlayer_project_name: str = None
        self.openlayer_inference_pipeline_name: str = None
        self.inference_pipeline: openlayer.InferencePipeline = None
        self._initialize_openlayer(
            publish=publish,
            api_key=openlayer_api_key,
            project_name=openlayer_project_name,
            inference_pipeline_name=openlayer_inference_pipeline_name,
        )
        self._load_inference_pipeline()

        # OpenAI setup
        self.openai_version = openai.__version__
        if self.openai_version.split(".", maxsplit=1)[0] == "1" and client is None:
            raise ValueError(
                "You must provide the OpenAI client for as the kwarg `client` for"
                " openai>=1.0.0."
            )
        self.openai_client = client
        self.create_chat_completion: callable = None
        self.create_completion: callable = None
        self.modified_create_chat_completion: callable = None
        self.modified_create_completion: callable = None
        self._initialize_openai()

        self.df = pd.DataFrame(columns=["input", "output", "tokens", "latency"])
        self.publish = publish
        self.accumulate_data = accumulate_data
        self.monitoring_on = False

    def _initialize_openlayer(
        self,
        publish: bool = False,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        inference_pipeline_name: Optional[str] = None,
    ) -> None:
        """Initializes the Openlayer attributes, if credentials are provided."""
        # Get credentials from environment variables if not provided
        if api_key is None:
            api_key = utils.get_env_variable("OPENLAYER_API_KEY")
        if project_name is None:
            project_name = utils.get_env_variable("OPENLAYER_PROJECT_NAME")
        if inference_pipeline_name is None:
            inference_pipeline_name = utils.get_env_variable(
                "OPENLAYER_INFERENCE_PIPELINE_NAME"
            )
        if publish and (api_key is None or project_name is None):
            raise ValueError(
                "To publish data to Openlayer, you must provide an API key and "
                "a project name. This can be done by setting the environment "
                "variables `OPENLAYER_API_KEY` and `OPENLAYER_PROJECT_NAME`, or by "
                "passing them as arguments to the OpenAIMonitor constructor "
                "(`openlayer_api_key` and `openlayer_project_name`, respectively)."
            )

        self.openlayer_api_key = api_key
        self.openlayer_project_name = project_name
        self.openlayer_inference_pipeline_name = inference_pipeline_name

    def _load_inference_pipeline(self) -> None:
        """Load inference pipeline from the Openlayer platform.

        If no platform/project information is provided, it is set to None.
        """
        inference_pipeline = None
        if self.openlayer_api_key and self.openlayer_project_name:
            with utils.HidePrints():
                client = openlayer.OpenlayerClient(
                    api_key=self.openlayer_api_key, verbose=False
                )
                project = client.create_project(
                    name=self.openlayer_project_name, task_type=tasks.TaskType.LLM
                )
                if self.openlayer_inference_pipeline_name:
                    inference_pipeline = project.load_inference_pipeline(
                        name=self.openlayer_inference_pipeline_name
                    )
                else:
                    inference_pipeline = project.create_inference_pipeline()

        self.inference_pipeline = inference_pipeline

    def _initialize_openai(self) -> None:
        """Initializes the OpenAI attributes."""
        if self.openai_version.split(".", maxsplit=1)[0] == "0":
            openai_api_key = utils.get_env_variable("OPENAI_API_KEY")
            openai.api_key = openai_api_key
            self.create_chat_completion = openai.ChatCompletion.create
            self.create_completion = openai.Completion.create
        else:
            self.create_chat_completion = self.openai_client.chat.completions.create
            self.create_completion = self.openai_client.completions.create
        self.modified_create_chat_completion = (
            self._get_modified_create_chat_completion()
        )
        self.modified_create_completion = self._get_modified_create_completion()

    def _get_modified_create_chat_completion(self) -> callable:
        """Returns a modified version of the create method for openai.ChatCompletion."""

        def modified_create_chat_completion(*args, **kwargs) -> str:
            start_time = time.time()
            response = self.create_chat_completion(*args, **kwargs)
            latency = (time.time() - start_time) * 1000

            try:
                prompt, input_data = self.format_input(kwargs["messages"])
                output_data = response.choices[0].message.content.strip()
                num_of_tokens = response.usage.total_tokens
                cost = self.get_cost_estimate(
                    model=kwargs.get("model"),
                    num_input_tokens=response.usage.prompt_tokens,
                    num_output_tokens=response.usage.completion_tokens,
                )

                config = self.data_config.copy()
                config["prompt"] = prompt
                config.update({"inputVariableNames": list(input_data.keys())})
                config["costColumnName"] = "cost"

                self._append_row_to_df(
                    input_data=input_data,
                    output_data=output_data,
                    num_of_tokens=num_of_tokens,
                    latency=latency,
                    cost=cost,
                )

                self._handle_data_publishing(config=config)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Failed to monitor chat request. %s", e)

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
                choices_splits = self._split_list(response.choices, len(prompts))

                for input_data, choices in zip(prompts, choices_splits):
                    output_data = choices[0].text.strip()
                    num_of_tokens = int(response.usage.total_tokens / len(prompts))
                    cost = self.get_cost_estimate(
                        model=kwargs.get("model"),
                        num_input_tokens=response.usage.prompt_tokens,
                        num_output_tokens=response.usage.completion_tokens,
                    )

                    self._append_row_to_df(
                        input_data={"message": input_data},
                        output_data=output_data,
                        num_of_tokens=num_of_tokens,
                        latency=latency,
                        cost=cost,
                    )

                    config = self.data_config.copy()
                    config["costColumnName"] = "cost"

                    self._handle_data_publishing(config=config)
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Failed to monitor completion request. %s", e)

            return response

        return modified_create_completion

    @staticmethod
    def format_input(
        messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Formats the input messages.

        Returns messages (prompt) replacing the user messages with input variables
        in brackets (e.g., ``{{ message_0 }}``) and a dictionary mapping the input variable
        names to the original user messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of messages that were sent to the chat completion model. Each message
            is a dictionary with the following keys:

            - ``role``: The role of the message. Can be either ``"user"`` or ``"system"``.
            - ``content``: The content of the message.

        Returns
        -------
        Tuple(List[Dict[str, str]], Dict[str, str])
            The formatted messages and the mapping from input variable names to the
            original user messages.
        """
        input_messages = []
        input_variables = {}
        for i, message in enumerate(messages):
            if message["role"] == "user":
                input_variable_name = f"message_{i}"
                input_messages.append(
                    {
                        "role": message["role"],
                        "content": f"{{{{ {input_variable_name} }}}}",
                    }
                )
                input_variables[input_variable_name] = message["content"]
            else:
                input_messages.append(message)
        return input_messages, input_variables

    @staticmethod
    def _split_list(lst: List, n_parts: int) -> List[List]:
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

    def get_cost_estimate(
        self, num_input_tokens: int, num_output_tokens: int, model: str
    ) -> float:
        """Returns the cost estimate for a given model and number of tokens."""
        if model not in self.COST_PER_TOKEN:
            return None
        cost_per_token = self.COST_PER_TOKEN[model]
        return (
            cost_per_token["input"] * num_input_tokens
            + cost_per_token["output"] * num_output_tokens
        )

    def _append_row_to_df(
        self,
        input_data: Dict[str, str],
        output_data: str,
        num_of_tokens: int,
        latency: float,
        cost: float,
    ) -> None:
        """Appends a row with input/output, number of tokens, and latency to the
        df."""
        row = pd.DataFrame(
            [
                {
                    **input_data,
                    **{
                        "output": output_data,
                        "tokens": num_of_tokens,
                        "latency": latency,
                        "cost": cost,
                    },
                }
            ]
        )
        if self.accumulate_data:
            self.df = pd.concat([self.df, row], ignore_index=True)
        else:
            self.df = row

        # Perform casting
        input_columns = [col for col in self.df.columns if col.startswith("message")]
        casting_dict = {col: object for col in input_columns}
        casting_dict.update(
            {"output": object, "tokens": int, "latency": float, "cost": float}
        )
        self.df = self.df.astype(casting_dict)

    def _handle_data_publishing(self, config: Optional[Dict[str, any]] = None) -> None:
        """Handle data publishing.

        If `publish` is set to True, publish the latest row to Openlayer.
        """
        if self.publish:
            self.inference_pipeline.stream_data(
                stream_data=self.df.tail(1).to_dict(orient="records"),
                stream_config=config or self.data_config,
            )

    def start_monitoring(self) -> None:
        """Switches monitoring for OpenAI LLMs on.

        After calling this method, all the calls to OpenAI's `Completion` and
        `ChatCompletion` APIs will be monitored.

        Refer to the `OpenAIMonitor` class docstring for an example.
        """
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
        print("To stop monitoring, call the `stop_monitoring` method.")

    def _overwrite_completion_methods(self) -> None:
        """Overwrites OpenAI's completion methods with the modified versions."""
        if self.openai_version.split(".", maxsplit=1)[0] == "0":
            openai.ChatCompletion.create = self.modified_create_chat_completion
            openai.Completion.create = self.modified_create_completion
        else:
            self.openai_client.chat.completions.create = (
                self.modified_create_chat_completion
            )
            self.openai_client.completions.create = self.modified_create_completion

    def stop_monitoring(self):
        """Switches monitoring for OpenAI LLMs off.

        After calling this method, all the calls to OpenAI's `Completion` and
        `ChatCompletion` APIs will stop being monitored.

        Refer to the `OpenAIMonitor` class docstring for an example.
        """
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
        if self.openai_version.split(".", maxsplit=1)[0] == "0":
            openai.ChatCompletion.create = self.create_chat_completion
            openai.Completion.create = self.create_completion
        else:
            self.openai_client.chat.completions.create = self.create_chat_completion
            self.openai_client.completions.create = self.create_completion

    def publish_batch_data(self):
        """Manually publish the accumulated data to Openlayer when automatic publishing
        is disabled (i.e., ``publish=False``)."""
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
            "inputVariableNames": ["message"],
            "label": "production",
            "outputColumnName": "output",
            "numOfTokenColumnName": "tokens",
            "latencyColumnName": "latency",
        }

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe accumulated after monitoring was switched on."""
        return self.df
