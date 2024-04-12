"""Module with classes for monitoring calls to LLMs."""

import json
import logging
import time
import warnings
from typing import Dict, List, Optional

import openai

from . import constants, utils
from .tracing import tracer

logger = logging.getLogger(__name__)


class OpenAIMonitor:
    """Monitor inferences from OpenAI LLMs and upload traces to Openlayer.

    Parameters
    ----------
    client : openai.api_client.Client
        The OpenAI client. It is required if you are using openai>=1.0.0.

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
    >>> monitor = llm_monitors.OpenAIMonitor(client=openai_client)

    3. Use the OpenAI model as you normally would:

    From this point onwards, you can continue making requests to your model normally:

    >>> openai_client.chat.completions.create(
    >>>     model="gpt-3.5-turbo",
    >>>     messages=[
    >>>         {"role": "system", "content": "You are a helpful assistant."},
    >>>         {"role": "user", "content": "How are you doing today?"}
    >>>     ],
    >>> )

    The trace of this inference request is automatically uploaded to your Openlayer
    project.
    """

    def __init__(
        self,
        client=None,
        publish: Optional[bool] = None,
    ) -> None:
        self._initialize_openai(client)
        if publish is not None:
            warnings.warn(
                "The `publish` parameter is deprecated and will be removed in a future"
                " version. All traces are now automatically published to Openlayer.",
                DeprecationWarning,
                stacklevel=2,
            )

    def start_monitoring(self) -> None:
        """(Deprecated) Start monitoring the OpenAI assistant."""
        warnings.warn(
            "The `start_monitoring` method is deprecated and will be removed in a future"
            " version. Monitoring is now automatically enabled once the OpenAIMonitor"
            " is instantiated.",
            DeprecationWarning,
            stacklevel=2,
        )

    def stop_monitoring(self) -> None:
        """(Deprecated) Stop monitoring the OpenAI assistant."""
        warnings.warn(
            "The `stop_monitoring` method is deprecated and will be removed in a future"
            " version. Monitoring is now automatically enabled once the OpenAIMonitor"
            " is instantiated.",
            DeprecationWarning,
            stacklevel=2,
        )

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

        # Overwrite the original methods with the modified ones
        self._overwrite_completion_methods()

    def _get_modified_create_chat_completion(self) -> callable:
        """Returns a modified version of the create method for openai.ChatCompletion."""

        def modified_create_chat_completion(*args, **kwargs) -> str:
            stream = kwargs.get("stream", False)

            # Pop the reserved Openlayer kwargs
            inference_id = kwargs.pop("inference_id", None)

            if not stream:
                start_time = time.time()
                response = self.create_chat_completion(*args, **kwargs)
                end_time = time.time()

                # Try to add step to the trace
                try:
                    output_content = response.choices[0].message.content
                    output_function_call = response.choices[0].message.function_call
                    output_tool_calls = response.choices[0].message.tool_calls
                    if output_content:
                        output_data = output_content.strip()
                    elif output_function_call or output_tool_calls:
                        if output_function_call:
                            function_call = {
                                "name": output_function_call.name,
                                "arguments": json.loads(output_function_call.arguments),
                            }
                        else:
                            function_call = {
                                "name": output_tool_calls[0].function.name,
                                "arguments": json.loads(
                                    output_tool_calls[0].function.arguments
                                ),
                            }
                        output_data = function_call
                    else:
                        output_data = None
                    cost = self.get_cost_estimate(
                        model=response.model,
                        num_input_tokens=response.usage.prompt_tokens,
                        num_output_tokens=response.usage.completion_tokens,
                    )
                    trace_args = {
                        "end_time": end_time,
                        "inputs": {
                            "prompt": kwargs["messages"],
                        },
                        "output": output_data,
                        "latency": (end_time - start_time) * 1000,
                        "tokens": response.usage.total_tokens,
                        "cost": cost,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "model": response.model,
                        "model_parameters": kwargs.get("model_parameters"),
                        "raw_output": response.model_dump(),
                    }
                    if inference_id:
                        trace_args["id"] = str(inference_id)

                    self._add_to_trace(
                        **trace_args,
                    )
                # pylint: disable=broad-except
                except Exception as e:
                    logger.error("Failed to monitor chat request. %s", e)

                return response
            else:
                chunks = self.create_chat_completion(*args, **kwargs)

                def stream_chunks():
                    collected_output_data = []
                    collected_function_call = {
                        "name": "",
                        "arguments": "",
                    }
                    raw_outputs = []
                    start_time = time.time()
                    end_time = None
                    first_token_time = None
                    num_of_completion_tokens = None
                    latency = None
                    try:
                        i = 0
                        for i, chunk in enumerate(chunks):
                            raw_outputs.append(chunk.model_dump())
                            if i == 0:
                                first_token_time = time.time()
                            if i > 0:
                                num_of_completion_tokens = i + 1

                            delta = chunk.choices[0].delta

                            if delta.content:
                                collected_output_data.append(delta.content)
                            elif delta.function_call:
                                if delta.function_call.name:
                                    collected_function_call[
                                        "name"
                                    ] += delta.function_call.name
                                if delta.function_call.arguments:
                                    collected_function_call[
                                        "arguments"
                                    ] += delta.function_call.arguments
                            elif delta.tool_calls:
                                if delta.tool_calls[0].function.name:
                                    collected_function_call["name"] += delta.tool_calls[
                                        0
                                    ].function.name
                                if delta.tool_calls[0].function.arguments:
                                    collected_function_call[
                                        "arguments"
                                    ] += delta.tool_calls[0].function.arguments

                            yield chunk
                        end_time = time.time()
                        latency = (end_time - start_time) * 1000
                    # pylint: disable=broad-except
                    except Exception as e:
                        logger.error("Failed yield chunk. %s", e)
                    finally:
                        # Try to add step to the trace
                        try:
                            collected_output_data = [
                                message
                                for message in collected_output_data
                                if message is not None
                            ]
                            if collected_output_data:
                                output_data = "".join(collected_output_data)
                            else:
                                collected_function_call["arguments"] = json.loads(
                                    collected_function_call["arguments"]
                                )
                                output_data = collected_function_call
                            completion_cost = self.get_cost_estimate(
                                model=kwargs.get("model"),
                                num_input_tokens=0,
                                num_output_tokens=(
                                    num_of_completion_tokens
                                    if num_of_completion_tokens
                                    else 0
                                ),
                            )
                            trace_args = {
                                "end_time": end_time,
                                "inputs": {
                                    "prompt": kwargs["messages"],
                                },
                                "output": output_data,
                                "latency": latency,
                                "tokens": num_of_completion_tokens,
                                "cost": completion_cost,
                                "prompt_tokens": None,
                                "completion_tokens": num_of_completion_tokens,
                                "model": kwargs.get("model"),
                                "model_parameters": kwargs.get("model_parameters"),
                                "raw_output": raw_outputs,
                                "metadata": {
                                    "timeToFirstToken": (
                                        (first_token_time - start_time) * 1000
                                        if first_token_time
                                        else None
                                    )
                                },
                            }
                            if inference_id:
                                trace_args["id"] = str(inference_id)

                            self._add_to_trace(
                                **trace_args,
                            )
                        # pylint: disable=broad-except
                        except Exception as e:
                            logger.error("Failed to monitor chat request. %s", e)

                return stream_chunks()

        return modified_create_chat_completion

    def _get_modified_create_completion(self) -> callable:
        """Returns a modified version of the create method for openai.Completion"""

        def modified_create_completion(*args, **kwargs):
            start_time = time.time()
            response = self.create_completion(*args, **kwargs)
            end_time = time.time()

            try:
                prompts = kwargs.get("prompt", [])
                prompts = [prompts] if isinstance(prompts, str) else prompts
                choices_splits = self._split_list(response.choices, len(prompts))

                for input_data, choices in zip(prompts, choices_splits):
                    # Extract data
                    output_data = choices[0].text.strip()
                    num_of_tokens = int(response.usage.total_tokens / len(prompts))
                    cost = self.get_cost_estimate(
                        model=response.model,
                        num_input_tokens=response.usage.prompt_tokens,
                        num_output_tokens=response.usage.completion_tokens,
                    )

                    self._add_to_trace(
                        end_time=end_time,
                        inputs={
                            "prompt": [{"role": "user", "content": input_data}],
                        },
                        output=output_data,
                        tokens=num_of_tokens,
                        latency=(end_time - start_time) * 1000,
                        cost=cost,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        model=response.model,
                        model_parameters=kwargs.get("model_parameters"),
                        raw_output=response.model_dump(),
                    )
            # pylint: disable=broad-except
            except Exception as e:
                logger.error("Failed to monitor completion request. %s", e)

            return response

        return modified_create_completion

    def _add_to_trace(self, **kwargs) -> None:
        """Add a step to the trace."""
        tracer.add_openai_chat_completion_step_to_trace(
            **kwargs,
            provider="OpenAI",
        )

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

    def monitor_thread_run(self, run: "openai.types.beta.threads.run.Run") -> None:
        """Monitor a run from an OpenAI assistant.

        Once the run is completed, the thread data is published to Openlayer,
        along with the latency, cost, and number of tokens used."""
        self._type_check_run(run)

        # Do nothing if the run is not completed
        if run.status != "completed":
            return

        try:
            # Extract vars
            run_step_vars = self._extract_run_vars(run)
            metadata = self._extract_run_metadata(run)

            # Convert thread to prompt
            messages = self.openai_client.beta.threads.messages.list(
                thread_id=run.thread_id, order="asc"
            )
            prompt = self._thread_messages_to_prompt(messages)

            # Add step to the trace
            tracer.add_openai_chat_completion_step_to_trace(
                inputs={"prompt": prompt[:-1]},  # Remove the last message (the output)
                output=prompt[-1]["content"],
                **run_step_vars,
                metadata=metadata,
                provider="OpenAI",
            )

        # pylint: disable=broad-except
        except Exception as e:
            print(f"Failed to monitor run. {e}")

    def _type_check_run(self, run: "openai.types.beta.threads.run.Run") -> None:
        """Validate the run object."""
        if not isinstance(run, openai.types.beta.threads.run.Run):
            raise ValueError(f"Expected a Run object, but got {type(run)}.")

    def _extract_run_vars(
        self, run: "openai.types.beta.threads.run.Run"
    ) -> Dict[str, any]:
        """Extract the variables from the run object."""
        return {
            "start_time": run.created_at,
            "end_time": run.completed_at,
            "latency": (run.completed_at - run.created_at) * 1000,  # Convert to ms
            "prompt_tokens": run.usage.prompt_tokens,
            "completion_tokens": run.usage.completion_tokens,
            "tokens": run.usage.total_tokens,
            "model": run.model,
            "cost": self.get_cost_estimate(
                model=run.model,
                num_input_tokens=run.usage.prompt_tokens,
                num_output_tokens=run.usage.completion_tokens,
            ),
        }

    def _extract_run_metadata(
        self, run: "openai.types.beta.threads.run.Run"
    ) -> Dict[str, any]:
        """Extract the metadata from the run object."""
        return {
            "openaiThreadId": run.thread_id,
            "openaiAssistantId": run.assistant_id,
        }

    @staticmethod
    def _thread_messages_to_prompt(
        messages: List["openai.types.beta.threads.thread_message.ThreadMessage"],
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


class AzureOpenAIMonitor(OpenAIMonitor):
    """Monitor inferences from Azure OpenAI LLMs and upload traces to Openlayer.

    Parameters
    ----------
    client : openai.AzureOpenAI
        The AzureOpenAI client.

    Examples
    --------

    Let's say that you have a GPT model you want to monitor. You can turn on monitoring
    with Openlayer by simply doing:

    1. Set the environment variables:

    .. code-block:: bash

        export AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
        export AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
        export AZURE_OPENAI_DEPLOYMENT_NAME=<your-azure-openai-deployment-name>

        export OPENLAYER_API_KEY=<your-openlayer-api-key>
        export OPENLAYER_PROJECT_NAME=<your-project-name>

    2. Instantiate the monitor:

    >>> from opemlayer import llm_monitors
    >>> from openai import AzureOpenAI
    >>>
    >>> azure_client = AzureOpenAI(
    >>>     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    >>>     api_version="2024-02-01",
    >>>     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    >>> )
    >>> monitor = llm_monitors.OpenAIMonitor(client=azure_client)

    3. Use the Azure OpenAI model as you normally would:

    From this point onwards, you can continue making requests to your model normally:

    >>> completion = azure_client.chat.completions.create(
    >>>     model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    >>>     messages=[
    >>>         {"role": "system", "content": "You are a helpful assistant."},
    >>>         {"role": "user", "content": "How are you doing today?"},
    >>>     ]
    >>> )

    The trace of this inference request is automatically uploaded to your Openlayer
    project.
    """

    def __init__(
        self,
        client=None,
    ) -> None:
        super().__init__(client)

    @staticmethod
    def get_cost_estimate(
        num_input_tokens: int, num_output_tokens: int, model: str
    ) -> float:
        """Returns the cost estimate for a given model and number of tokens."""
        if model not in constants.AZURE_OPENAI_COST_PER_TOKEN:
            return None
        cost_per_token = constants.AZURE_OPENAI_COST_PER_TOKEN[model]
        return (
            cost_per_token["input"] * num_input_tokens
            + cost_per_token["output"] * num_output_tokens
        )

    def _add_to_trace(self, **kwargs) -> None:
        """Add a step to the trace."""
        tracer.add_openai_chat_completion_step_to_trace(
            **kwargs,
            name="Azure OpenAI Chat Completion",
            provider="Azure OpenAI",
        )
