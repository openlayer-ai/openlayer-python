"""Module with the logic to create and manage traces and steps."""

import contextvars
import inspect
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Tuple

from .. import utils
from ..services import data_streamer
from . import enums, steps, traces

logger = logging.getLogger(__name__)

_publish = utils.get_env_variable("OPENLAYER_DISABLE_PUBLISH") != "true"
_streamer = None
if _publish:
    _streamer = data_streamer.DataStreamer()

_current_step = contextvars.ContextVar("current_step")
_current_trace = contextvars.ContextVar("current_trace")


def get_current_trace() -> Optional[traces.Trace]:
    """Returns the current trace."""
    return _current_trace.get(None)


def get_current_step() -> Optional[steps.Step]:
    """Returns the current step."""
    return _current_step.get(None)


@contextmanager
def create_step(
    name: str,
    step_type: enums.StepType = enums.StepType.USER_CALL,
    inputs: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[steps.Step, None, None]:
    """Starts a trace and yields a Step object."""
    new_step: steps.Step = steps.step_factory(
        step_type=step_type, name=name, inputs=inputs, output=output, metadata=metadata
    )
    new_step.start_time = time.time()

    parent_step: Optional[steps.Step] = get_current_step()
    is_root_step: bool = parent_step is None

    if parent_step is None:
        logger.debug("Starting a new trace...")
        current_trace = traces.Trace()
        _current_trace.set(current_trace)  # Set the current trace in context
        current_trace.add_step(new_step)
    else:
        logger.debug("Adding step %s to parent step %s", name, parent_step.name)
        current_trace = get_current_trace()
        parent_step.add_nested_step(new_step)

    token = _current_step.set(new_step)
    try:
        yield new_step
    finally:
        if new_step.end_time is None:
            new_step.end_time = time.time()
        if new_step.latency is None:
            latency = (new_step.end_time - new_step.start_time) * 1000  # in ms
            new_step.latency = latency

        _current_step.reset(token)
        if is_root_step:
            logger.debug("Ending the trace...")
            trace_data, input_variable_names = process_trace_for_upload(current_trace)
            config = {
                "outputColumnName": "output",
                "inputVariableNames": input_variable_names,
                "label": "production",
                "groundTruthColumnName": "groundTruth",
                "latencyColumnName": "latency",
                "costColumnName": "cost",
                "numOfTokenColumnName": "tokens",
                "timestampColumnName": "inferenceTimestamp",
                "inferenceIdColumnName": "inferenceId",
            }
            if isinstance(new_step, steps.ChatCompletionStep):
                config.update(
                    {
                        "prompt": new_step.inputs.get("prompt"),
                    }
                )
            if _publish:
                try:
                    _streamer.stream_data(data=trace_data, config=config)
                except Exception:  # pylint: disable=broad-except
                    logger.error("Could not stream data to Openlayer")
        else:
            logger.debug("Ending step %s", name)


def add_openai_chat_completion_step_to_trace(**kwargs) -> None:
    """Adds an OpenAI chat completion step to the trace."""
    with create_step(
        step_type=enums.StepType.CHAT_COMPLETION,
        name=kwargs.get("name", "OpenAI Chat Completion"),
    ) as step:
        step.log(**kwargs)


# ----------------------------- Tracing decorator ---------------------------- #
def trace(*step_args, **step_kwargs):
    """Decorator to trace a function.

    Examples
    --------

    To trace a function, simply decorate it with the ``@trace()`` decorator. By doing so,
    the functions inputs, outputs, and metadata will be automatically logged to your
    Openlayer project.

    >>> import os
    >>> from openlayer.tracing import tracer
    >>>
    >>> # Set the environment variables
    >>> os.environ["OPENLAYER_API_KEY"] = "YOUR_OPENLAYER_API_KEY_HERE"
    >>> os.environ["OPENLAYER_PROJECT_NAME"] = "YOUR_OPENLAYER_PROJECT_NAME_HERE"
    >>>
    >>> # Decorate all the functions you want to trace
    >>> @tracer.trace()
    >>> def main(user_query: str) -> str:
    >>>     context = retrieve_context(user_query)
    >>>     answer = generate_answer(user_query, context)
    >>>     return answer
    >>>
    >>> @tracer.trace()
    >>> def retrieve_context(user_query: str) -> str:
    >>>     return "Some context"
    >>>
    >>> @tracer.trace()
    >>> def generate_answer(user_query: str, context: str) -> str:
    >>>     return "Some answer"
    >>>
    >>> # Every time the main function is called, the data is automatically
    >>> # streamed to your Openlayer project. E.g.:
    >>> main("What is the meaning of life?")
    """

    def decorator(func):
        func_signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if step_kwargs.get("name") is None:
                step_kwargs["name"] = func.__name__
            with create_step(*step_args, **step_kwargs) as step:
                output = func(*func_args, **func_kwargs)
                end_time = time.time()
                latency = (end_time - step.start_time) * 1000  # in ms

                bound = func_signature.bind(*func_args, **func_kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)
                inputs.pop("self", None)
                inputs.pop("cls", None)

                step.log(
                    inputs=inputs,
                    output=output,
                    end_time=end_time,
                    latency=latency,
                )
            return output

        return wrapper

    return decorator


# --------------------- Helper post-processing functions --------------------- #
def process_trace_for_upload(
    trace_obj: traces.Trace,
) -> Tuple[Dict[str, Any], List[str]]:
    """Post processing of the trace data before uploading to Openlayer.

    This is done to ensure backward compatibility with data on Openlayer.
    """
    root_step = trace_obj.steps[0]

    input_variables = root_step.inputs
    input_variable_names = list(input_variables.keys())

    processed_steps = bubble_up_costs_and_tokens(trace_obj.to_dict())

    trace_data = {
        **input_variables,
        "inferenceTimestamp": root_step.start_time,
        "inferenceId": str(root_step.id),
        "output": root_step.output,
        "groundTruth": root_step.ground_truth,
        "latency": root_step.latency,
        "cost": processed_steps[0].get("cost", 0),
        "tokens": processed_steps[0].get("tokens", 0),
        "steps": processed_steps,
    }

    return trace_data, input_variable_names


def bubble_up_costs_and_tokens(
    trace_dict: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Adds the cost and number of tokens of nested steps to their parent steps."""

    def add_step_costs_and_tokens(step: Dict[str, Any]) -> Tuple[float, int]:
        step_cost = step_tokens = 0

        if "cost" in step and step["cost"] is not None:
            step_cost += step["cost"]
        if "tokens" in step and step["tokens"] is not None:
            step_tokens += step["tokens"]

        # Recursively add costs and tokens from nested steps
        for nested_step in step.get("steps", []):
            nested_cost, nested_tokens = add_step_costs_and_tokens(nested_step)
            step_cost += nested_cost
            step_tokens += nested_tokens

        if "steps" in step:
            if step_cost > 0 and "cost" not in step:
                step["cost"] = step_cost
            if step_tokens > 0 and "tokens" not in step:
                step["tokens"] = step_tokens

        return step_cost, step_tokens

    for root_step_dict in trace_dict:
        add_step_costs_and_tokens(root_step_dict)

    return trace_dict
