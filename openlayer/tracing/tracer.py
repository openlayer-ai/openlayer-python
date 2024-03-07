"""Module with the logic to create and manage traces and steps."""

import contextvars
import inspect
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..services import data_streamer
from . import steps, traces

logger = logging.getLogger(__name__)

_streamer = None
try:
    _streamer = data_streamer.DataStreamer(publish=True)
except Exception as exc:
    logger.error(
        "You have not provided enough information to upload traces to Openlayer."
        "\n%s \n"
        "To upload the traces, please provide the missing information and try again.",
        exc,
    )

_current_step = contextvars.ContextVar("current_step")
_current_trace = contextvars.ContextVar("current_trace")


@contextmanager
def create_step(
    name: str,
    step_type: str = "user_call",
    inputs: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Dict[str, any] = {},
) -> Generator[steps.Step, None, None]:
    """Starts a trace and yields a Step object."""
    new_step: steps.Step = steps.step_factory(
        step_type=step_type, name=name, inputs=inputs, output=output, metadata=metadata
    )

    parent_step: Optional[steps.Step] = _current_step.get(None)
    is_root_step: bool = parent_step is None

    if parent_step is None:
        logger.debug("Starting a new trace...")
        current_trace = traces.Trace()
        _current_trace.set(current_trace)  # Set the current trace in context
        current_trace.add_step(new_step)
    else:
        logger.debug(f"Adding step {name} to parent step {parent_step.name}")
        current_trace = _current_trace.get()
        parent_step.add_nested_step(new_step)

    token = _current_step.set(new_step)
    try:
        yield new_step
    finally:
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
            }
            if isinstance(new_step, steps.OpenAIChatCompletionStep):
                config.update(
                    {
                        "costColumnName": "cost",
                        "numOfTokenColumnName": "tokens",
                        "prompt": new_step.inputs.get("prompt"),
                    }
                )
            if _streamer:
                _streamer.stream_data(data=trace_data, config=config)
            else:
                logger.warning(
                    "Trace computed but not uploaded to Openlayer. "
                    "You have not provided enough information to upload traces to"
                    " Openlayer."
                )
        else:
            logger.debug(f"Ending step {name}")


def process_trace_for_upload(trace: traces.Trace) -> Tuple[Dict[str, Any], List[str]]:
    """Post processing of the trace data before uploading to Openlayer.

    This is done to ensure backward compatibility with data on Openlayer.
    """
    root_step = trace.steps[0]

    input_variables = root_step.inputs
    input_variable_names = list(input_variables.keys())

    trace_data = {
        **input_variables,
        "output": root_step.output,
        "groundTruth": root_step.ground_truth,
        "latency": root_step.latency,
        "steps": trace.to_dict(),
    }
    # Extra fields for openai_chat_completion step
    if isinstance(root_step, steps.OpenAIChatCompletionStep):
        trace_data.update(
            {
                "cost": root_step.cost,
                "tokens": root_step.prompt_tokens + root_step.completion_tokens,
            }
        )

    return trace_data, input_variable_names


def trace(*step_args, **step_kwargs):
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
