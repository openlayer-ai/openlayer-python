"""Module with the logic to create and manage traces and steps."""

import inspect
from typing import Any, Dict, Optional, Generator
from contextlib import contextmanager
import contextvars
from functools import wraps

from . import steps
from . import traces
import time

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
    new_step = steps.step_factory(
        step_type=step_type, name=name, inputs=inputs, output=output, metadata=metadata
    )
    parent_step = _current_step.get(None)
    is_root_step = parent_step is None

    if parent_step is None:
        print("Starting a new trace...")
        current_trace = traces.Trace()
        _current_trace.set(current_trace)  # Set the current trace in context
        current_trace.add_step(new_step)
    else:
        print(f"Adding step {name} to parent step {parent_step.name}")
        current_trace = _current_trace.get()
        parent_step.add_nested_step(new_step)

    token = _current_step.set(new_step)

    try:
        yield new_step
    finally:
        _current_step.reset(token)
        if is_root_step:
            print("Ending the trace...")
            print("-" * 80)
            print(current_trace.to_dict())
            print("-" * 80)
        else:
            # TODO: stream to Openlayer
            print(f"Ending step {name}")


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

                step.update_data(
                    inputs=inputs,
                    output=output,
                    end_time=end_time,
                    latency=latency,
                )
            return output

        return wrapper

    return decorator
