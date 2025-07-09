"""Module with the logic to create and manage traces and steps."""

import time
import asyncio
import inspect
import logging
import contextvars
from typing import Any, Dict, List, Tuple, Optional, Awaitable, Generator
from functools import wraps
from contextlib import contextmanager

from . import enums, steps, traces
from .. import utils
from ..._client import Openlayer
from ..._base_client import DefaultHttpxClient
from ...types.inference_pipelines.data_stream_params import ConfigLlmData

logger = logging.getLogger(__name__)

# ----------------------------- Module setup and globals ----------------------------- #

TRUE_LIST = ["true", "on", "1"]

_publish = utils.get_env_variable("OPENLAYER_DISABLE_PUBLISH") not in TRUE_LIST
_verify_ssl = (
    utils.get_env_variable("OPENLAYER_VERIFY_SSL") or "true"
).lower() in TRUE_LIST
_client = None


def _get_client() -> Optional[Openlayer]:
    """Get or create the Openlayer client with lazy initialization."""
    global _client
    if not _publish:
        return None

    if _client is None:
        # Lazy initialization - create client when first needed
        if _verify_ssl:
            _client = Openlayer()
        else:
            _client = Openlayer(
                http_client=DefaultHttpxClient(
                    verify=False,
                ),
            )
    return _client


_current_step = contextvars.ContextVar("current_step")
_current_trace = contextvars.ContextVar("current_trace")
_rag_context = contextvars.ContextVar("rag_context")

# ----------------------------- Public API functions ----------------------------- #


def get_current_trace() -> Optional[traces.Trace]:
    """Returns the current trace."""
    return _current_trace.get(None)


def get_current_step() -> Optional[steps.Step]:
    """Returns the current step."""
    return _current_step.get(None)


def get_rag_context() -> Optional[Dict[str, Any]]:
    """Returns the current context."""
    return _rag_context.get(None)


@contextmanager
def create_step(
    name: str,
    step_type: enums.StepType = enums.StepType.USER_CALL,
    inputs: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    inference_pipeline_id: Optional[str] = None,
) -> Generator[steps.Step, None, None]:
    """Starts a trace and yields a Step object."""
    new_step, is_root_step, token = _create_and_initialize_step(
        step_name=name,
        step_type=step_type,
        inputs=inputs,
        output=output,
        metadata=metadata,
    )
    try:
        yield new_step
    finally:
        if new_step.end_time is None:
            new_step.end_time = time.time()
        if new_step.latency is None:
            latency = (new_step.end_time - new_step.start_time) * 1000  # in ms
            new_step.latency = latency

        _current_step.reset(token)
        _handle_trace_completion(
            is_root_step=is_root_step,
            step_name=name,
            inference_pipeline_id=inference_pipeline_id,
        )


def add_chat_completion_step_to_trace(**kwargs) -> None:
    """Adds a chat completion step to the trace."""
    with create_step(
        step_type=enums.StepType.CHAT_COMPLETION,
        name=kwargs.get("name", "Chat Completion"),
    ) as step:
        step.log(**kwargs)


def trace(
    *step_args,
    inference_pipeline_id: Optional[str] = None,
    context_kwarg: Optional[str] = None,
    **step_kwargs,
):
    """Decorator to trace a function.

    Examples
    --------

    To trace a function, simply decorate it with the ``@trace()`` decorator. By doing
    so, the functions inputs, outputs, and metadata will be automatically logged to your
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

            with create_step(
                *step_args, inference_pipeline_id=inference_pipeline_id, **step_kwargs
            ) as step:
                output = exception = None
                try:
                    output = func(*func_args, **func_kwargs)
                except Exception as exc:
                    _log_step_exception(step, exc)
                    exception = exc

                # Extract inputs and finalize logging using optimized helper
                _process_wrapper_inputs_and_outputs(
                    step=step,
                    func_signature=func_signature,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                    context_kwarg=context_kwarg,
                    output=output,
                )

                if exception is not None:
                    raise exception
            return output

        return wrapper

    return decorator


def trace_async(
    *step_args,
    inference_pipeline_id: Optional[str] = None,
    context_kwarg: Optional[str] = None,
    **step_kwargs,
):
    """Decorator to trace async functions and async generators.

    This decorator automatically detects whether the function is a regular async function
    or an async generator and handles both cases appropriately.

    Examples
    --------

    To trace a regular async function:

    >>> @tracer.trace_async()
    >>> async def main(user_query: str) -> str:
    >>>     context = retrieve_context(user_query)
    >>>     answer = generate_answer(user_query, context)
    >>>     return answer

    To trace an async generator function:

    >>> @tracer.trace_async()
    >>> async def stream_response(query: str):
    >>>     async for chunk in openai_client.chat.completions.create(...):
    >>>         yield chunk.choices[0].delta.content
    """

    def decorator(func):
        func_signature = inspect.signature(func)

        if step_kwargs.get("name") is None:
            step_kwargs["name"] = func.__name__
        step_name = step_kwargs["name"]

        if asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            # Check if it's specifically an async generator function
            if inspect.isasyncgenfunction(func):
                # For async generators, use class-based approach to delay trace creation
                # until actual iteration begins (not when generator object is created)
                @wraps(func)
                def async_generator_wrapper(*func_args, **func_kwargs):
                    class TracedAsyncGenerator:
                        def __init__(self):
                            self._original_gen = None
                            self._step = None
                            self._is_root_step = False
                            self._token = None
                            self._output_chunks = []
                            self._trace_initialized = False

                        def __aiter__(self):
                            return self

                        async def __anext__(self):
                            # Initialize tracing on first iteration only
                            if not self._trace_initialized:
                                self._original_gen = func(*func_args, **func_kwargs)
                                self._step, self._is_root_step, self._token = _create_and_initialize_step(
                                    step_name=step_name,
                                    step_type=enums.StepType.USER_CALL,
                                    inputs=None,
                                    output=None,
                                    metadata=None,
                                )
                                self._inputs = _extract_function_inputs(
                                    func_signature=func_signature,
                                    func_args=func_args,
                                    func_kwargs=func_kwargs,
                                    context_kwarg=context_kwarg,
                                )
                                self._trace_initialized = True

                            try:
                                chunk = await self._original_gen.__anext__()
                                self._output_chunks.append(chunk)
                                return chunk
                            except StopAsyncIteration:
                                # Finalize trace when generator is exhausted
                                output = _join_output_chunks(self._output_chunks)
                                _finalize_async_generator_step(
                                    step=self._step,
                                    token=self._token,
                                    is_root_step=self._is_root_step,
                                    step_name=step_name,
                                    inputs=self._inputs,
                                    output=output,
                                    inference_pipeline_id=inference_pipeline_id,
                                )
                                raise
                            except Exception as exc:
                                # Handle exceptions
                                if self._step:
                                    _log_step_exception(self._step, exc)
                                    output = _join_output_chunks(self._output_chunks)
                                    _finalize_async_generator_step(
                                        step=self._step,
                                        token=self._token,
                                        is_root_step=self._is_root_step,
                                        step_name=step_name,
                                        inputs=self._inputs,
                                        output=output,
                                        inference_pipeline_id=inference_pipeline_id,
                                    )
                                raise

                    return TracedAsyncGenerator()

                return async_generator_wrapper
            else:
                # Create wrapper for regular async functions
                @wraps(func)
                async def async_function_wrapper(*func_args, **func_kwargs):
                    with create_step(
                        *step_args,
                        inference_pipeline_id=inference_pipeline_id,
                        **step_kwargs,
                    ) as step:
                        output = exception = None

                        try:
                            output = await func(*func_args, **func_kwargs)
                        except Exception as exc:
                            _log_step_exception(step, exc)
                            exception = exc
                            raise

                        # Extract inputs and finalize logging
                        _process_wrapper_inputs_and_outputs(
                            step=step,
                            func_signature=func_signature,
                            func_args=func_args,
                            func_kwargs=func_kwargs,
                            context_kwarg=context_kwarg,
                            output=output,
                        )

                        return output

                return async_function_wrapper
        else:
            # For sync functions, use the existing logic with optimizations
            @wraps(func)
            def sync_wrapper(*func_args, **func_kwargs):
                with create_step(
                    *step_args,
                    inference_pipeline_id=inference_pipeline_id,
                    **step_kwargs,
                ) as step:
                    output = exception = None
                    try:
                        output = func(*func_args, **func_kwargs)
                    except Exception as exc:
                        _log_step_exception(step, exc)
                        exception = exc

                    # Extract inputs and finalize logging
                    _process_wrapper_inputs_and_outputs(
                        step=step,
                        func_signature=func_signature,
                        func_args=func_args,
                        func_kwargs=func_kwargs,
                        context_kwarg=context_kwarg,
                        output=output,
                    )

                    if exception is not None:
                        raise exception
                return output

            return sync_wrapper

    return decorator


def log_context(context: List[str]) -> None:
    """Logs context information to the current step of the trace.

    The `context` parameter should be a list of strings representing the
    context chunks retrieved by the context retriever."""
    current_step = get_current_step()
    if current_step:
        _rag_context.set(context)
        current_step.log(metadata={"context": context})
    else:
        logger.warning("No current step found to log context.")


def run_async_func(coroutine: Awaitable[Any]) -> Any:
    """Runs an async function while preserving the context. This is needed
    for tracing async functions.
    """
    context, result = asyncio.run(_invoke_with_context(coroutine))
    for key, value in context.items():
        key.set(value)
    return result


# ----------------------------- Helper functions for create_step ----------------------------- #


def _create_and_initialize_step(
    step_name: str,
    step_type: enums.StepType = enums.StepType.USER_CALL,
    inputs: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[steps.Step, bool, Any]:
    """Create a new step and initialize trace/parent relationships.

    Returns:
        Tuple of (step, is_root_step, token)
    """
    new_step = steps.step_factory(
        step_type=step_type,
        name=step_name,
        inputs=inputs,
        output=output,
        metadata=metadata,
    )
    new_step.start_time = time.time()

    parent_step = get_current_step()
    is_root_step = parent_step is None

    if parent_step is None:
        logger.debug("Starting a new trace...")
        current_trace = traces.Trace()
        _current_trace.set(current_trace)
        _rag_context.set(None)
        current_trace.add_step(new_step)
    else:
        logger.debug("Adding step %s to parent step %s", step_name, parent_step.name)
        current_trace = get_current_trace()
        parent_step.add_nested_step(new_step)

    token = _current_step.set(new_step)
    return new_step, is_root_step, token


def _handle_trace_completion(
    is_root_step: bool, step_name: str, inference_pipeline_id: Optional[str] = None
) -> None:
    """Handle trace completion and data streaming."""
    if is_root_step:
        logger.debug("Ending the trace...")
        current_trace = get_current_trace()
        trace_data, input_variable_names = post_process_trace(current_trace)

        config = dict(
            ConfigLlmData(
                output_column_name="output",
                input_variable_names=input_variable_names,
                latency_column_name="latency",
                cost_column_name="cost",
                timestamp_column_name="inferenceTimestamp",
                inference_id_column_name="inferenceId",
                num_of_token_column_name="tokens",
            )
        )
        if "groundTruth" in trace_data:
            config.update({"ground_truth_column_name": "groundTruth"})
        if "context" in trace_data:
            config.update({"context_column_name": "context"})

        if isinstance(get_current_step(), steps.ChatCompletionStep):
            config.update(
                {
                    "prompt": get_current_step().inputs.get("prompt"),
                }
            )
        if _publish:
            try:
                client = _get_client()
                if client:
                    client.inference_pipelines.data.stream(
                        inference_pipeline_id=inference_pipeline_id
                        or utils.get_env_variable("OPENLAYER_INFERENCE_PIPELINE_ID"),
                        rows=[trace_data],
                        config=config,
                    )
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Could not stream data to Openlayer %s", err)
    else:
        logger.debug("Ending step %s", step_name)


# ----------------------------- Helper functions for trace decorators ----------------------------- #


def _log_step_exception(step: steps.Step, exception: Exception) -> None:
    """Log exception metadata to a step."""
    step.log(metadata={"Exceptions": str(exception)})


def _process_wrapper_inputs_and_outputs(
    step: steps.Step,
    func_signature: inspect.Signature,
    func_args: tuple,
    func_kwargs: dict,
    context_kwarg: Optional[str],
    output: Any,
) -> None:
    """Extract function inputs and finalize step logging - common pattern across wrappers."""
    inputs = _extract_function_inputs(
        func_signature=func_signature,
        func_args=func_args,
        func_kwargs=func_kwargs,
        context_kwarg=context_kwarg,
    )
    _finalize_step_logging(
        step=step, inputs=inputs, output=output, start_time=step.start_time
    )


def _extract_function_inputs(
    func_signature: inspect.Signature,
    func_args: tuple,
    func_kwargs: dict,
    context_kwarg: Optional[str] = None,
) -> dict:
    """Extract and clean function inputs for logging."""
    bound = func_signature.bind(*func_args, **func_kwargs)
    bound.apply_defaults()
    inputs = dict(bound.arguments)
    inputs.pop("self", None)
    inputs.pop("cls", None)

    # Handle context kwarg if specified
    if context_kwarg:
        if context_kwarg in inputs:
            log_context(inputs.get(context_kwarg))
        else:
            logger.warning(
                "Context kwarg `%s` not found in inputs of the current function.",
                context_kwarg,
            )

    return inputs


def _finalize_step_logging(
    step: steps.Step,
    inputs: dict,
    output: Any,
    start_time: float,
) -> None:
    """Finalize step timing and logging."""
    if step.end_time is None:
        step.end_time = time.time()
    if step.latency is None:
        step.latency = (step.end_time - start_time) * 1000  # in ms

    step.log(
        inputs=inputs,
        output=output,
        end_time=step.end_time,
        latency=step.latency,
    )


# ----------------------------- Async generator specific functions ----------------------------- #



def _finalize_async_generator_step(
    step: steps.Step,
    token: Any,
    is_root_step: bool,
    step_name: str,
    inputs: dict,
    output: Any,
    inference_pipeline_id: Optional[str] = None,
) -> None:
    """Finalize async generator step - called when generator is consumed."""
    _current_step.reset(token)
    _finalize_step_logging(
        step=step, inputs=inputs, output=output, start_time=step.start_time
    )
    _handle_trace_completion(
        is_root_step=is_root_step,
        step_name=step_name,
        inference_pipeline_id=inference_pipeline_id,
    )


def _join_output_chunks(output_chunks: List[Any]) -> str:
    """Join output chunks into a single string, filtering out None values."""
    return "".join(str(chunk) for chunk in output_chunks if chunk is not None)


# ----------------------------- Utility functions ----------------------------- #


async def _invoke_with_context(
    coroutine: Awaitable[Any],
) -> Tuple[contextvars.Context, Any]:
    """Runs a coroutine and preserves the context variables set within it."""
    result = await coroutine
    context = contextvars.copy_context()
    return context, result


def post_process_trace(
    trace_obj: traces.Trace,
) -> Tuple[Dict[str, Any], List[str]]:
    """Post processing of the trace data before uploading to Openlayer.

    This is done to ensure backward compatibility with data on Openlayer.
    """
    root_step = trace_obj.steps[0]

    input_variables = root_step.inputs
    if input_variables:
        input_variable_names = list(input_variables.keys())
    else:
        input_variable_names = []

    processed_steps = trace_obj.to_dict()

    trace_data = {
        "inferenceTimestamp": root_step.start_time,
        "inferenceId": str(root_step.id),
        "output": root_step.output,
        "latency": root_step.latency,
        "cost": processed_steps[0].get("cost", 0),
        "tokens": processed_steps[0].get("tokens", 0),
        "steps": processed_steps,
        **root_step.metadata,
    }
    if root_step.ground_truth:
        trace_data["groundTruth"] = root_step.ground_truth
    if input_variables:
        trace_data.update(input_variables)

    context = get_rag_context()
    if context:
        trace_data["context"] = context

    return trace_data, input_variable_names
