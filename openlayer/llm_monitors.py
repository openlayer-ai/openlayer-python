"""Module with classes for monitoring calls to LLMs."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd

from . import constants, utils
from .services import data_streamer

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
    monitor_output_only : bool, False
        Whether to monitor only the output of the model. If True, only the output of
        the model is logged.
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
    openlayer_inference_pipeline_id : str, optional
        The Openlayer inference pipeline id. If not provided, it is read from the
        environment variable ``OPENLAYER_INFERENCE_PIPELINE_ID``.
        This is only needed if you do not want to specify an inference pipeline name and
        project name, and you want to load the inference pipeline directly from its id.

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

    You can also use the ``monitor`` as a context manager:

    >>> monitor = llm_monitors.OpenAIMonitor(publish=True, client=openai_client)
    >>>
    >>> with monitor:
    >>>     openai_client.chat.completions.create(
    >>>         model="gpt-3.5-turbo",
    >>>         messages=[
    >>>             {"role": "system", "content": "You are a helpful assistant."},
    >>>             {"role": "user", "content": "How are you doing today?"}
    >>>         ],
    >>>     )

    This will automatically start and stop the monitoring for you.

    """

    def __init__(
        self,
        publish: bool = False,
        client=None,
        monitor_output_only: bool = False,
        accumulate_data: bool = False,
        openlayer_api_key: Optional[str] = None,
        openlayer_project_name: Optional[str] = None,
        openlayer_inference_pipeline_name: Optional[str] = None,
        openlayer_inference_pipeline_id: Optional[str] = None,
    ) -> None:
        self._initialize_openai(client)

        self.accumulate_data = accumulate_data
        self.monitor_output_only = monitor_output_only
        self.monitoring_on = False
        self.df = pd.DataFrame(columns=["input", "output", "tokens", "latency"])

        self.data_streamer = data_streamer.DataStreamer(
            openlayer_api_key=openlayer_api_key,
            openlayer_project_name=openlayer_project_name,
            openlayer_inference_pipeline_name=openlayer_inference_pipeline_name,
            openlayer_inference_pipeline_id=openlayer_inference_pipeline_id,
            publish=publish,
        )

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitoring()

    def _initialize_openai(self, client) -> None:
        """Initializes the OpenAI attributes."""
        self._validate_and_set_openai_client(client)
        self._set_create_methods()

    def _validate_and_set_openai_client(self, client) -> None:
        """Validate and set the OpenAI client."""
        self.openai_version = openai.__version__
        if self.openai_version.split(".", maxsplit=1)[0] == "1" and client is None:
            raise ValueError(
                "You must provide the OpenAI client for as the kwarg `client` for"
                " openai>=1.0.0."
            )
        self.openai_client = client

    def _set_create_methods(self) -> None:
        """Sets up the create methods for OpenAI's Completion and ChatCompletion."""
        # Original versions of the create methods
        if self.openai_version.startswith("0"):
            openai.api_key = utils.get_env_variable("OPENAI_API_KEY")
            self.create_chat_completion = openai.ChatCompletion.create
            self.create_completion = openai.Completion.create
        else:
            self.create_chat_completion = self.openai_client.chat.completions.create
            self.create_completion = self.openai_client.completions.create

        # Modified versions of the create methods
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
                # Extract data
                prompt, input_data = self.format_input(kwargs["messages"])
                output_data = response.choices[0].message.content.strip()
                num_of_tokens = response.usage.total_tokens
                cost = self.get_cost_estimate(
                    model=kwargs.get("model"),
                    num_input_tokens=response.usage.prompt_tokens,
                    num_output_tokens=response.usage.completion_tokens,
                )

                # Prepare config
                config = self.data_config.copy()
                config["prompt"] = prompt
                if not self.monitor_output_only:
                    config.update({"inputVariableNames": list(input_data.keys())})

                self._append_row_to_df(
                    input_data=input_data,
                    output_data=output_data,
                    num_of_tokens=num_of_tokens,
                    latency=latency,
                    cost=cost,
                )

                self.data_streamer.stream_data(
                    data=self.df.tail(1).to_dict(orient="records"),
                    config=config,
                )
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
                    # Extract data
                    output_data = choices[0].text.strip()
                    num_of_tokens = int(response.usage.total_tokens / len(prompts))
                    cost = self.get_cost_estimate(
                        model=kwargs.get("model"),
                        num_input_tokens=response.usage.prompt_tokens,
                        num_output_tokens=response.usage.completion_tokens,
                    )

                    # Prepare config
                    config = self.data_config.copy()
                    if not self.monitor_output_only:
                        config["prompt"] = [{"role": "user", "content": input_data}]
                        config["inputVariableNames"] = ["message"]

                    self._append_row_to_df(
                        input_data={"message": input_data},
                        output_data=output_data,
                        num_of_tokens=num_of_tokens,
                        latency=latency,
                        cost=cost,
                    )

                    self.data_streamer.stream_data(
                        data=self.df.tail(1).to_dict(orient="records"),
                        config=config,
                    )
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
        in brackets (e.g., ``{{ message_0 }}``) and a dictionary mapping the input
        variable names to the original user messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            List of messages that were sent to the chat completion model. Each message
            is a dictionary with the following keys:

            - ``role``: The role of the message. Can be either ``"user"`` or
            ``"system"``.
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

    @staticmethod
    def get_cost_estimate(
        num_input_tokens: int, num_output_tokens: int, model: str
    ) -> float:
        """Returns the cost estimate for a given model and number of tokens."""
        if model not in constants.OPENAI_COST_PER_TOKEN:
            return None
        cost_per_token = constants.OPENAI_COST_PER_TOKEN[model]
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
        if self.monitor_output_only:
            input_data = {}

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
        if self.data_streamer.publish:
            print(
                "Furthermore, since `publish` was set to True, the data is being"
                f" published to your '{self.data_streamer.openlayer_project_name}' Openlayer project."
            )
        print("To stop monitoring, call the `stop_monitoring` method.")

    def _overwrite_completion_methods(self) -> None:
        """Overwrites OpenAI's completion methods with the modified versions."""
        if self.openai_version.startswith("0"):
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
        if not self.data_streamer.publish:
            print(
                "To publish the data collected so far to your Openlayer project, "
                "call the `publish_batch_data` method."
            )

    def _restore_completion_methods(self) -> None:
        """Restores OpenAI's completion methods to their original versions."""
        if self.openai_version.startswith("0"):
            openai.ChatCompletion.create = self.create_chat_completion
            openai.Completion.create = self.create_completion
        else:
            self.openai_client.chat.completions.create = self.create_chat_completion
            self.openai_client.completions.create = self.create_completion

    def publish_batch_data(self):
        """Manually publish the accumulated data to Openlayer when automatic publishing
        is disabled (i.e., ``publish=False``)."""
        if self.data_streamer.publish:
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
        self.data_streamer.publish_batch_data(df=self.df, config=self.data_config)

    @property
    def data_config(self) -> Dict[str, any]:
        """Data config for the df. Used for publishing data to Openlayer."""
        return {
            "costColumnName": "cost",
            "inputVariableNames": [],
            "label": "production",
            "latencyColumnName": "latency",
            "numOfTokenColumnName": "tokens",
            "outputColumnName": "output",
        }

    @property
    def data(self) -> pd.DataFrame:
        """Dataframe accumulated after monitoring was switched on."""
        return self.df

    def monitor_thread_run(self, run: openai.types.beta.threads.run.Run) -> None:
        """Monitor a run from an OpenAI assistant.

        Once the run is completed, the thread data is published to Openlayer,
        along with the latency, cost, and number of tokens used."""
        self._type_check_run(run)

        # Do nothing if the run is not completed
        if run.status != "completed":
            return

        try:
            # Extract vars
            run_vars = self._extract_run_vars(run)

            # Convert thread to prompt
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=run_vars["openai_thread_id"], order="asc"
            )
            populated_prompt = self.thread_messages_to_prompt(messages)
            prompt, input_variables = self.format_input(populated_prompt)

            # Data
            input_data = {
                **input_variables,
                **{
                    "output": prompt[-1]["content"],
                    "tokens": run_vars["total_num_tokens"],
                    "latency": run_vars["latency"],
                    "cost": run_vars["cost"],
                    "openai_thread_id": run_vars["openai_thread_id"],
                    "openai_assistant_id": run_vars["openai_assistant_id"],
                    "timestamp": run_vars["timestamp"],
                },
            }

            # Config
            config = self.data_config.copy()
            config["inputVariableNames"] = input_variables.keys()
            config["prompt"] = prompt[:-1]  # Remove the last message (the output)
            config["timestampColumnName"] = "timestamp"

            self.data_streamer.stream_data(data=input_data, config=config)
            print("Data published to Openlayer.")
        # pylint: disable=broad-except
        except Exception as e:
            print(f"Failed to monitor run. {e}")

    def _type_check_run(self, run: openai.types.beta.threads.run.Run) -> None:
        """Validate the run object."""
        if not isinstance(run, openai.types.beta.threads.run.Run):
            raise ValueError(f"Expected a Run object, but got {type(run)}.")

    def _extract_run_vars(
        self, run: openai.types.beta.threads.run.Run
    ) -> Dict[str, any]:
        """Extract the variables from the run object."""
        return {
            "openai_thread_id": run.thread_id,
            "openai_assistant_id": run.assistant_id,
            "latency": (run.completed_at - run.created_at) * 1000,  # Convert to ms
            "timestamp": run.created_at,  # Convert to ms
            "num_input_tokens": run.usage["prompt_tokens"],
            "num_output_tokens": run.usage["completion_tokens"],
            "total_num_tokens": run.usage["total_tokens"],
            "cost": self.get_cost_estimate(
                model=run.model,
                num_input_tokens=run.usage["prompt_tokens"],
                num_output_tokens=run.usage["completion_tokens"],
            ),
        }

    @staticmethod
    def thread_messages_to_prompt(
        messages: List[openai.types.beta.threads.thread_message.ThreadMessage],
    ) -> List[Dict[str, str]]:
        """Given list of ThreadMessage, return its contents in the `prompt` format,
        i.e., a list of dicts with 'role' and 'content' keys."""
        prompt = []
        for message in list(messages):
            role = message.role
            contents = message.content

            for content in contents:
                content_type = content.type
                if content_type == "text":
                    text_content = content.text.value
                if content_type == "image_file":
                    text_content = content.image_file.file_id

                prompt.append(
                    {
                        "role": role,
                        "content": text_content,
                    }
                )
        return prompt
