"""Module with the logic to create and manage traces and steps."""

import asyncio
import contextvars
import inspect
import json
import logging
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional, Tuple, Union

from ..._base_client import DefaultHttpxClient
from ..._client import Openlayer
from ...types.inference_pipelines.data_stream_params import ConfigLlmData
from .. import utils
from . import enums, steps, traces
from ..guardrails.base import GuardrailResult, GuardrailAction
from .context import UserSessionContext

# Type aliases for callback functions
OnFlushFailureCallback = Callable[[Dict[str, Any], Dict[str, Any], Exception], None]
OnReplaySuccessCallback = Callable[[Dict[str, Any], Dict[str, Any]], None]
OnReplayFailureCallback = Callable[[Dict[str, Any], Dict[str, Any], Exception], None]

logger = logging.getLogger(__name__)

# ----------------------------- Module setup and globals ----------------------------- #

TRUE_LIST = ["true", "on", "1"]

_publish = utils.get_env_variable("OPENLAYER_DISABLE_PUBLISH") not in TRUE_LIST
_verify_ssl = (utils.get_env_variable("OPENLAYER_VERIFY_SSL") or "true").lower() in TRUE_LIST
_client = None

# Configuration variables for programmatic setup
_configured_api_key: Optional[str] = None
_configured_pipeline_id: Optional[str] = None
_configured_base_url: Optional[str] = None
_configured_timeout: Optional[Union[int, float]] = None
_configured_max_retries: Optional[int] = None

# Offline buffering and callback configuration
_configured_on_flush_failure: Optional[OnFlushFailureCallback] = None
_configured_offline_buffer_enabled: bool = False
_configured_offline_buffer_path: Optional[str] = None
_configured_max_buffer_size: Optional[int] = None


def configure(
    api_key: Optional[str] = None,
    inference_pipeline_id: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[Union[int, float]] = None,
    max_retries: Optional[int] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
    offline_buffer_enabled: bool = False,
    offline_buffer_path: Optional[str] = None,
    max_buffer_size: Optional[int] = None,
) -> None:
    """Configure the Openlayer tracer with custom settings.

    This function allows you to programmatically set the API key, inference pipeline ID,
    base URL, timeout, retry settings, and offline buffering for the Openlayer client,
    instead of relying on environment variables.

    Args:
        api_key: The Openlayer API key. If not provided, falls back to OPENLAYER_API_KEY environment variable.
        inference_pipeline_id: The default inference pipeline ID to use for tracing.
            If not provided, falls back to OPENLAYER_INFERENCE_PIPELINE_ID environment variable.
        base_url: The base URL for the Openlayer API. If not provided, falls back to
            OPENLAYER_BASE_URL environment variable or the default.
        timeout: The timeout for the Openlayer API in seconds (int or float). Defaults to 60 seconds.
        max_retries: The maximum number of retries for failed API requests. Defaults to 2.
        on_flush_failure: Optional callback function called when trace data fails to send to Openlayer.
            Should accept (trace_data, config, error) as arguments.
        offline_buffer_enabled: Enable offline buffering of failed traces. Defaults to False.
        offline_buffer_path: Directory path for storing buffered traces. Defaults to ~/.openlayer/buffer.
        max_buffer_size: Maximum number of trace files to store in buffer. Defaults to 1000.

    Examples:
        >>> import openlayer.lib.tracing.tracer as tracer
        >>> # Configure with API key and pipeline ID
        >>> tracer.configure(api_key="your_api_key_here", inference_pipeline_id="your_pipeline_id_here")
        >>> # Configure with failure callback and offline buffering
        >>> def on_failure(trace_data, config, error):
        ...     print(f"Failed to send trace: {error}")
        ...     # Could also log to monitoring system, send alerts, etc.
        >>> tracer.configure(
        ...     api_key="your_api_key_here",
        ...     inference_pipeline_id="your_pipeline_id_here",
        ...     on_flush_failure=on_failure,
        ...     offline_buffer_enabled=True,
        ...     offline_buffer_path="/tmp/openlayer_buffer",
        ...     max_buffer_size=500,
        ... )
        >>> # Now use the decorators normally
        >>> @tracer.trace()
        >>> def my_function():
        ...     return "result"
    """
    global \
        _configured_api_key, \
        _configured_pipeline_id, \
        _configured_base_url, \
        _configured_timeout, \
        _configured_max_retries, \
        _client
    global \
        _configured_on_flush_failure, \
        _configured_offline_buffer_enabled, \
        _configured_offline_buffer_path, \
        _configured_max_buffer_size, \
        _offline_buffer

    _configured_api_key = api_key
    _configured_pipeline_id = inference_pipeline_id
    _configured_base_url = base_url
    _configured_timeout = timeout
    _configured_max_retries = max_retries
    _configured_on_flush_failure = on_flush_failure
    _configured_offline_buffer_enabled = offline_buffer_enabled
    _configured_offline_buffer_path = offline_buffer_path
    _configured_max_buffer_size = max_buffer_size

    # Reset the client and buffer so they get recreated with new configuration
    _client = None
    _offline_buffer = None


def _get_client() -> Optional[Openlayer]:
    """Get or create the Openlayer client with lazy initialization."""
    global _client
    if not _publish:
        return None

    if _client is None:
        # Lazy initialization - create client when first needed
        client_kwargs = {}

        # Use configured API key if available, otherwise fall back to environment variable
        if _configured_api_key is not None:
            client_kwargs["api_key"] = _configured_api_key

        # Use configured base URL if available, otherwise fall back to environment variable
        if _configured_base_url is not None:
            client_kwargs["base_url"] = _configured_base_url

        if _configured_timeout is not None:
            client_kwargs["timeout"] = _configured_timeout

        if _configured_max_retries is not None:
            client_kwargs["max_retries"] = _configured_max_retries

        if _verify_ssl:
            _client = Openlayer(**client_kwargs)
        else:
            _client = Openlayer(
                http_client=DefaultHttpxClient(
                    verify=False,
                ),
                **client_kwargs,
            )
    return _client


_current_step = contextvars.ContextVar("current_step")
_current_trace = contextvars.ContextVar("current_trace")
_rag_context = contextvars.ContextVar("rag_context")

# ----------------------------- Offline Buffer Implementation ----------------------------- #


class OfflineBuffer:
    """Handles offline buffering of trace data when platform communication fails."""

    def __init__(
        self,
        buffer_path: Optional[str] = None,
        max_buffer_size: Optional[int] = None,
    ):
        """Initialize the offline buffer.

        Args:
            buffer_path: Directory path for storing buffered traces.
                        Defaults to ~/.openlayer/buffer.
            max_buffer_size: Maximum number of trace files to store.
                           Defaults to 1000.
        """
        self.buffer_path = Path(buffer_path or os.path.expanduser("~/.openlayer/buffer"))
        self.max_buffer_size = max_buffer_size or 1000
        self._lock = threading.RLock()

        # Create buffer directory if it doesn't exist
        self.buffer_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Initialized offline buffer at %s", self.buffer_path)

    def store_trace(self, trace_data: Dict[str, Any], config: Dict[str, Any], inference_pipeline_id: str) -> bool:
        """Store a failed trace to the offline buffer.

        Args:
            trace_data: The trace data that failed to send
            config: The configuration used for streaming
            inference_pipeline_id: The pipeline ID used

        Returns:
            True if successfully stored, False otherwise
        """
        try:
            with self._lock:
                # Check buffer size limit
                existing_files = list(self.buffer_path.glob("trace_*.json"))
                if len(existing_files) >= self.max_buffer_size:
                    # Remove oldest file to make room
                    oldest_file = min(existing_files, key=lambda f: f.stat().st_mtime)
                    oldest_file.unlink()
                    logger.debug("Removed oldest buffered trace: %s", oldest_file)

                # Create filename with timestamp and unique suffix
                timestamp = int(time.time() * 1000)  # milliseconds
                unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
                filename = f"trace_{timestamp}_{os.getpid()}_{unique_id}.json"
                file_path = self.buffer_path / filename

                # Prepare the complete data payload
                buffered_payload = {
                    "trace_data": trace_data,
                    "config": config,
                    "inference_pipeline_id": inference_pipeline_id,
                    "timestamp": timestamp,
                    "metadata": {
                        "buffer_version": "1.0",
                        "created_by": "openlayer-python-sdk",
                        "process_id": os.getpid(),
                    },
                }

                # Write to file atomically
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(buffered_payload, f, ensure_ascii=False, indent=2)

                logger.info("Stored trace to offline buffer: %s", file_path)
                return True

        except Exception as e:
            logger.error("Failed to store trace to offline buffer: %s", e)
            return False

    def get_buffered_traces(self) -> List[Dict[str, Any]]:
        """Get all buffered traces from disk.

        Returns:
            List of buffered trace payloads
        """
        traces = []

        try:
            with self._lock:
                trace_files = sorted(self.buffer_path.glob("trace_*.json"), key=lambda f: f.stat().st_mtime)

                for file_path in trace_files:
                    try:
                        with file_path.open("r", encoding="utf-8") as f:
                            payload = json.load(f)
                            payload["_file_path"] = str(file_path)
                            traces.append(payload)
                    except Exception as e:
                        logger.error("Failed to read buffered trace %s: %s", file_path, e)

        except Exception as e:
            logger.error("Failed to get buffered traces: %s", e)

        return traces

    def remove_trace(self, file_path: str) -> bool:
        """Remove a successfully replayed trace from the buffer.

        Args:
            file_path: Path to the trace file to remove

        Returns:
            True if successfully removed, False otherwise
        """
        try:
            with self._lock:
                Path(file_path).unlink(missing_ok=True)
                logger.debug("Removed successfully replayed trace: %s", file_path)
                return True
        except Exception as e:
            logger.error("Failed to remove buffered trace %s: %s", file_path, e)
            return False

    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status.

        Returns:
            Dictionary with buffer statistics
        """
        try:
            with self._lock:
                trace_files = list(self.buffer_path.glob("trace_*.json"))
                total_size = sum(f.stat().st_size for f in trace_files)

                return {
                    "buffer_path": str(self.buffer_path),
                    "total_traces": len(trace_files),
                    "max_buffer_size": self.max_buffer_size,
                    "total_size_bytes": total_size,
                    "oldest_trace": (min(trace_files, key=lambda f: f.stat().st_mtime).name if trace_files else None),
                    "newest_trace": (max(trace_files, key=lambda f: f.stat().st_mtime).name if trace_files else None),
                }
        except Exception as e:
            logger.error("Failed to get buffer status: %s", e)
            return {"error": str(e)}

    def clear_buffer(self) -> int:
        """Clear all buffered traces.

        Returns:
            Number of traces removed
        """
        try:
            with self._lock:
                trace_files = list(self.buffer_path.glob("trace_*.json"))
                count = len(trace_files)

                for file_path in trace_files:
                    file_path.unlink(missing_ok=True)

                logger.info("Cleared %d traces from offline buffer", count)
                return count
        except Exception as e:
            logger.error("Failed to clear buffer: %s", e)
            return 0


# Global offline buffer instance
_offline_buffer: Optional[OfflineBuffer] = None


def _get_offline_buffer() -> Optional[OfflineBuffer]:
    """Get or create the offline buffer instance."""
    global _offline_buffer

    if _configured_offline_buffer_enabled and _offline_buffer is None:
        _offline_buffer = OfflineBuffer(
            buffer_path=_configured_offline_buffer_path,
            max_buffer_size=_configured_max_buffer_size,
        )

    return _offline_buffer if _configured_offline_buffer_enabled else None


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
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
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
            on_flush_failure=on_flush_failure,
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
    guardrails: Optional[List[Any]] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
    **step_kwargs,
):
    """Decorator to trace a function with optional guardrails.

    Examples
    --------

    To trace a function, simply decorate it with the ``@trace()`` decorator. By doing
    so, the functions inputs, outputs, and metadata will be automatically logged to your
    Openlayer project.

    >>> import os
    >>> from openlayer.tracing import tracer
    >>> from openlayer.lib.guardrails import PIIGuardrail
    >>>
    >>> # Set the environment variables
    >>> os.environ["OPENLAYER_API_KEY"] = "YOUR_OPENLAYER_API_KEY_HERE"
    >>> os.environ["OPENLAYER_PROJECT_NAME"] = "YOUR_OPENLAYER_PROJECT_NAME_HERE"
    >>>
    >>> # Create guardrail instance
    >>> pii_guardrail = PIIGuardrail(name="PII Protection")
    >>>
    >>> # Decorate functions with tracing and guardrails
    >>> @tracer.trace(guardrails=[pii_guardrail])
    >>> def main(user_query: str) -> str:
    >>>     context = retrieve_context(user_query)
    >>>     answer = generate_answer(user_query, context)
    >>>     return answer
    >>>
    >>> @tracer.trace()
    >>> def retrieve_context(user_query: str) -> str:
    >>>     return "Some context"
    >>>
    >>> @tracer.trace(guardrails=[pii_guardrail])
    >>> def generate_answer(user_query: str, context: str) -> str:
    >>>     return "Some answer"
    >>>
    >>> # Every time the main function is called, the data is automatically
    >>> # streamed to your Openlayer project. E.g.:
    >>> main("What is the meaning of life?")
    """

    def decorator(func):
        func_signature = inspect.signature(func)

        if step_kwargs.get("name") is None:
            step_kwargs["name"] = func.__name__
        step_name = step_kwargs["name"]

        # Check if it's a generator function
        if inspect.isgeneratorfunction(func):
            # For sync generators, use class-based approach to delay trace creation
            # until actual iteration begins (not when generator object is created)
            @wraps(func)
            def sync_generator_wrapper(*func_args, **func_kwargs):
                class TracedSyncGenerator:
                    def __init__(self):
                        self._original_gen = None
                        self._step = None
                        self._is_root_step = False
                        self._token = None
                        self._output_chunks = []
                        self._trace_initialized = False

                    def __iter__(self):
                        return self

                    def __next__(self):
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
                            chunk = next(self._original_gen)
                            self._output_chunks.append(chunk)
                            return chunk
                        except StopIteration:
                            # Finalize trace when generator is exhausted
                            output = _join_output_chunks(self._output_chunks)
                            _finalize_sync_generator_step(
                                step=self._step,
                                token=self._token,
                                is_root_step=self._is_root_step,
                                step_name=step_name,
                                inputs=self._inputs,
                                output=output,
                                inference_pipeline_id=inference_pipeline_id,
                                on_flush_failure=on_flush_failure,
                            )
                            raise
                        except Exception as exc:
                            # Handle exceptions
                            if self._step:
                                _log_step_exception(self._step, exc)
                                output = _join_output_chunks(self._output_chunks)
                                _finalize_sync_generator_step(
                                    step=self._step,
                                    token=self._token,
                                    is_root_step=self._is_root_step,
                                    step_name=step_name,
                                    inputs=self._inputs,
                                    output=output,
                                    inference_pipeline_id=inference_pipeline_id,
                                    on_flush_failure=on_flush_failure,
                                )
                            raise

                return TracedSyncGenerator()

            return sync_generator_wrapper
        else:
            # Handle regular functions with guardrail support
            @wraps(func)
            def wrapper(*func_args, **func_kwargs):
                if step_kwargs.get("name") is None:
                    step_kwargs["name"] = func.__name__

                with create_step(
                    *step_args,
                    inference_pipeline_id=inference_pipeline_id,
                    **step_kwargs,
                ) as step:
                    output = exception = None
                    original_inputs = None
                    modified_inputs = None
                    guardrail_metadata = {}

                    try:
                        # Extract original inputs for guardrail processing
                        original_inputs = _extract_function_inputs(
                            func_signature=func_signature,
                            func_args=func_args,
                            func_kwargs=func_kwargs,
                            context_kwarg=context_kwarg,
                        )

                        # Apply input guardrails
                        modified_inputs, input_guardrail_metadata = _apply_input_guardrails(
                            guardrails or [],
                            original_inputs,
                        )
                        guardrail_metadata.update(input_guardrail_metadata)

                        # Check if function execution should be skipped
                        if (
                            hasattr(modified_inputs, "__class__")
                            and modified_inputs.__class__.__name__ == "SkipFunctionExecution"
                        ):
                            # Function execution was blocked with SKIP_FUNCTION strategy
                            output = None
                            logger.debug(
                                "Function %s execution skipped by guardrail",
                                func.__name__,
                            )
                        else:
                            # Execute function with potentially modified inputs
                            if modified_inputs != original_inputs:
                                # Reconstruct function arguments from modified inputs
                                bound = func_signature.bind(*func_args, **func_kwargs)
                                bound.apply_defaults()

                                # Update bound arguments with modified values
                                for (
                                    param_name,
                                    modified_value,
                                ) in modified_inputs.items():
                                    if param_name in bound.arguments:
                                        bound.arguments[param_name] = modified_value

                                output = func(*bound.args, **bound.kwargs)
                            else:
                                output = func(*func_args, **func_kwargs)

                        # Apply output guardrails (skip if function was skipped)
                        if (
                            hasattr(modified_inputs, "__class__")
                            and modified_inputs.__class__.__name__ == "SkipFunctionExecution"
                        ):
                            final_output, output_guardrail_metadata = output, {}
                            # Use original inputs for logging since modified_inputs
                            # is a special marker
                            modified_inputs = original_inputs
                        else:
                            final_output, output_guardrail_metadata = _apply_output_guardrails(
                                guardrails or [],
                                output,
                                modified_inputs or original_inputs,
                            )
                        guardrail_metadata.update(output_guardrail_metadata)

                        if final_output != output:
                            output = final_output

                    except Exception as exc:
                        # Check if this is a guardrail exception
                        if hasattr(exc, "guardrail_name"):
                            guardrail_metadata[f"{exc.guardrail_name}_blocked"] = {
                                "action": "blocked",
                                "reason": exc.reason,
                                "metadata": getattr(exc, "metadata", {}),
                            }

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
                        guardrail_metadata=guardrail_metadata,
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
    guardrails: Optional[List[Any]] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
    **step_kwargs,
):
    """Decorator to trace async functions and async generators.

    This decorator automatically detects whether the function is a regular async
    function
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
                                    on_flush_failure=on_flush_failure,
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
                                        on_flush_failure=on_flush_failure,
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
                        guardrail_metadata = {}

                        try:
                            # Apply input guardrails if provided
                            if guardrails:
                                try:
                                    inputs = _extract_function_inputs(
                                        func_signature=func_signature,
                                        func_args=func_args,
                                        func_kwargs=func_kwargs,
                                        context_kwarg=context_kwarg,
                                    )

                                    # Process inputs through guardrails
                                    modified_inputs, input_metadata = _apply_input_guardrails(
                                        guardrails,
                                        inputs,
                                    )
                                    guardrail_metadata.update(input_metadata)

                                    # Execute function with potentially modified inputs
                                    if modified_inputs != inputs:
                                        # Reconstruct function arguments from modified inputs
                                        bound = func_signature.bind(*func_args, **func_kwargs)
                                        bound.apply_defaults()

                                        # Update bound arguments with modified values
                                        for (
                                            param_name,
                                            modified_value,
                                        ) in modified_inputs.items():
                                            if param_name in bound.arguments:
                                                bound.arguments[param_name] = modified_value

                                        output = await func(*bound.args, **bound.kwargs)
                                    else:
                                        output = await func(*func_args, **func_kwargs)
                                except Exception as e:
                                    # Log guardrail errors but don't fail function execution
                                    logger.error("Guardrail error: %s", e)
                                    output = await func(*func_args, **func_kwargs)
                            else:
                                output = await func(*func_args, **func_kwargs)

                        except Exception as exc:
                            _log_step_exception(step, exc)
                            raise exc

                        # Apply output guardrails if provided
                        if guardrails and output is not None:
                            try:
                                final_output, output_metadata = _apply_output_guardrails(
                                    guardrails,
                                    output,
                                    _extract_function_inputs(
                                        func_signature=func_signature,
                                        func_args=func_args,
                                        func_kwargs=func_kwargs,
                                        context_kwarg=context_kwarg,
                                    ),
                                )
                                guardrail_metadata.update(output_metadata)

                                if final_output != output:
                                    output = final_output
                            except Exception as e:
                                # Log guardrail errors but don't fail function execution
                                logger.error("Output guardrail error: %s", e)

                        # Extract inputs and finalize logging
                        _process_wrapper_inputs_and_outputs(
                            step=step,
                            func_signature=func_signature,
                            func_args=func_args,
                            func_kwargs=func_kwargs,
                            context_kwarg=context_kwarg,
                            output=output,
                            guardrail_metadata=guardrail_metadata,
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
                    guardrail_metadata = {}
                    try:
                        # Apply input guardrails if provided
                        if guardrails:
                            try:
                                inputs = _extract_function_inputs(
                                    func_signature=func_signature,
                                    func_args=func_args,
                                    func_kwargs=func_kwargs,
                                    context_kwarg=context_kwarg,
                                )

                                # Process inputs through guardrails
                                modified_inputs, input_metadata = _apply_input_guardrails(
                                    guardrails,
                                    inputs,
                                )
                                guardrail_metadata.update(input_metadata)

                                # Execute function with potentially modified inputs
                                if modified_inputs != inputs:
                                    # Reconstruct function arguments from modified inputs
                                    bound = func_signature.bind(*func_args, **func_kwargs)
                                    bound.apply_defaults()

                                    # Update bound arguments with modified values
                                    for (
                                        param_name,
                                        modified_value,
                                    ) in modified_inputs.items():
                                        if param_name in bound.arguments:
                                            bound.arguments[param_name] = modified_value

                                    output = func(*bound.args, **bound.kwargs)
                                else:
                                    output = func(*func_args, **func_kwargs)
                            except Exception as e:
                                # Log guardrail errors but don't fail function execution
                                logger.error("Guardrail error: %s", e)
                                output = func(*func_args, **func_kwargs)
                        else:
                            output = func(*func_args, **func_kwargs)

                    except Exception as exc:
                        _log_step_exception(step, exc)
                        exception = exc

                    # Apply output guardrails if provided
                    if guardrails and output is not None:
                        try:
                            final_output, output_metadata = _apply_output_guardrails(
                                guardrails,
                                output,
                                _extract_function_inputs(
                                    func_signature=func_signature,
                                    func_args=func_args,
                                    func_kwargs=func_kwargs,
                                    context_kwarg=context_kwarg,
                                ),
                            )
                            guardrail_metadata.update(output_metadata)

                            if final_output != output:
                                output = final_output
                        except Exception as e:
                            # Log guardrail errors but don't fail function execution
                            logger.error("Output guardrail error: %s", e)

                    # Extract inputs and finalize logging
                    _process_wrapper_inputs_and_outputs(
                        step=step,
                        func_signature=func_signature,
                        func_args=func_args,
                        func_kwargs=func_kwargs,
                        context_kwarg=context_kwarg,
                        output=output,
                        guardrail_metadata=guardrail_metadata,
                    )

                    if exception is not None:
                        raise exception
                return output

            return sync_wrapper

    return decorator


def log_output(output: Any) -> None:
    """Logs output information to the current step of the trace.

    This will overwrite the output of the currently active step instead of
    relying on the returned object from the traced function.

    Args:
        output: The output value to log to the current step.
    """
    current_step = get_current_step()
    if current_step:
        logger.debug("Logging output to current step: %s", output)
        current_step.log(output=output, metadata={"manual_output_logged": True})
    else:
        logger.warning("No current step found to log output.")


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


def update_current_trace(**kwargs) -> None:
    """Updates the current trace metadata with the provided values.

    This function allows users to set trace-level metadata dynamically
    during execution without having to pass it through function arguments.

    All provided key-value pairs will be stored in the trace metadata.

    Example:
        >>> from openlayer.lib import trace, update_current_trace
        >>>
        >>> @trace()
        >>> def my_function():
        >>> # Update trace with user context
        >>>     update_current_trace(
        >>>         inferenceId="custom_inference_id",
        >>>         user_id="user123",
        >>>         session_id="sess456",
        >>>         custom_field="any_value"
        >>>     )
        >>>     return "result"
    """
    current_trace = get_current_trace()
    if current_trace is None:
        logger.warning(
            "update_current_trace() called without an active trace. "
            "Make sure to call this function within a traced context "
            "(e.g., inside a function decorated with @trace)."
        )
        return

    current_trace.update_metadata(**kwargs)
    logger.debug("Updated current trace metadata")


def update_current_step(
    attributes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Updates the current step with the provided attributes.

    This function allows users to set step-level metadata dynamically
    during execution.

    Args:
        attributes: Optional dictionary of attributes to set on the step
        metadata: Optional dictionary of metadata to merge with existing metadata

    Example:
        >>> from openlayer.lib import trace, update_current_step
        >>>
        >>> @trace()
        >>> def my_function():
        >>> # Update current step with additional context
        >>>     update_current_step(
        >>>         metadata={"model_version": "v1.2.3"}
        >>>     )
        >>>     return "result"
    """
    current_step = get_current_step()
    if current_step is None:
        logger.warning(
            "update_current_step() called without an active step. "
            "Make sure to call this function within a traced context "
            "(e.g., inside a function decorated with @trace)."
        )
        return

    # Update step attributes using the existing log method
    update_data = {}
    if metadata is not None:
        # Merge with existing metadata
        existing_metadata = current_step.metadata or {}
        existing_metadata.update(metadata)
        update_data["metadata"] = existing_metadata

    # Handle generic attributes by setting them directly on the step
    if attributes is not None:
        for key, value in attributes.items():
            setattr(current_step, key, value)

    if update_data:
        current_step.log(**update_data)

    logger.debug("Updated current step metadata")


def run_async_func(coroutine: Awaitable[Any]) -> Any:
    """Runs an async function while preserving the context. This is needed
    for tracing async functions.
    """
    context, result = asyncio.run(_invoke_with_context(coroutine))
    for key, value in context.items():
        key.set(value)
    return result


def replay_buffered_traces(
    max_retries: int = 3,
    on_replay_success: Optional[OnReplaySuccessCallback] = None,
    on_replay_failure: Optional[OnReplayFailureCallback] = None,
) -> Dict[str, Any]:
    """Replay all buffered traces to the Openlayer platform.

    This function attempts to send all traces stored in the offline buffer
    to Openlayer. Successfully sent traces are removed from the buffer.

    Args:
        max_retries: Maximum number of retries per trace. Defaults to 3.
        on_replay_success: Optional callback called when a trace is successfully replayed.
            Should accept (trace_data, config) as arguments.
        on_replay_failure: Optional callback called when a trace fails to replay.
            Should accept (trace_data, config, error) as arguments.

    Returns:
        Dictionary with replay statistics including success/failure counts.

    Examples:
        >>> import openlayer.lib.tracing.tracer as tracer
        >>> def on_success(trace_data, config):
        ...     print(f"Successfully replayed trace: {trace_data['inferenceId']}")
        >>> def on_failure(trace_data, config, error):
        ...     print(f"Failed to replay trace: {error}")
        >>> result = tracer.replay_buffered_traces(
        ...     max_retries=3, on_replay_success=on_success, on_replay_failure=on_failure
        ... )
        >>> print(f"Replayed {result['success_count']} traces successfully")
    """
    offline_buffer = _get_offline_buffer()
    if not offline_buffer:
        logger.warning("Offline buffer not enabled - nothing to replay")
        return {
            "total_traces": 0,
            "success_count": 0,
            "failure_count": 0,
            "error": "Offline buffer not enabled",
        }

    client = _get_client()
    if not client:
        logger.error("No Openlayer client available for replay")
        return {
            "total_traces": 0,
            "success_count": 0,
            "failure_count": 0,
            "error": "No Openlayer client available",
        }

    buffered_traces = offline_buffer.get_buffered_traces()
    total_traces = len(buffered_traces)
    success_count = 0
    failure_count = 0
    failed_traces = []

    logger.info("Starting replay of %d buffered traces", total_traces)

    for trace_payload in buffered_traces:
        trace_data = trace_payload["trace_data"]
        config = trace_payload["config"]
        inference_pipeline_id = trace_payload["inference_pipeline_id"]
        file_path = trace_payload["_file_path"]

        success = False
        last_error = None

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = client.inference_pipelines.data.stream(
                    inference_pipeline_id=inference_pipeline_id,
                    rows=[trace_data],
                    config=config,
                )

                # Success - remove from buffer and count it
                offline_buffer.remove_trace(file_path)
                success_count += 1
                success = True

                logger.debug(
                    "Successfully replayed trace %s (attempt %d)",
                    trace_data.get("inferenceId", "unknown"),
                    attempt + 1,
                )

                # Call success callback if provided
                if on_replay_success:
                    try:
                        on_replay_success(trace_data, config)
                    except Exception as callback_err:
                        logger.error("Error in replay success callback: %s", callback_err)

                break

            except Exception as err:
                last_error = err
                logger.debug(
                    "Failed to replay trace %s (attempt %d): %s",
                    trace_data.get("inferenceId", "unknown"),
                    attempt + 1,
                    err,
                )

                # If this is the last attempt, mark as failed
                if attempt == max_retries - 1:
                    failure_count += 1
                    failed_traces.append(
                        {
                            "trace_id": trace_data.get("inferenceId", "unknown"),
                            "error": str(err),
                            "file_path": file_path,
                        }
                    )

                    # Call failure callback if provided
                    if on_replay_failure:
                        try:
                            on_replay_failure(trace_data, config, err)
                        except Exception as callback_err:
                            logger.error("Error in replay failure callback: %s", callback_err)

    result = {
        "total_traces": total_traces,
        "success_count": success_count,
        "failure_count": failure_count,
        "failed_traces": failed_traces,
    }

    logger.info(
        "Replay completed: %d/%d traces successfully sent",
        success_count,
        total_traces,
    )

    return result


def get_buffer_status() -> Dict[str, Any]:
    """Get the current status of the offline buffer.

    Returns:
        Dictionary with buffer statistics including trace count, size, and path.

    Examples:
        >>> import openlayer.lib.tracing.tracer as tracer
        >>> status = tracer.get_buffer_status()
        >>> print(f"Buffer contains {status['total_traces']} traces")
        >>> print(f"Buffer size: {status['total_size_bytes']} bytes")
    """
    offline_buffer = _get_offline_buffer()
    if not offline_buffer:
        return {
            "enabled": False,
            "error": "Offline buffer not enabled",
        }

    status = offline_buffer.get_buffer_status()
    status["enabled"] = True
    return status


def clear_offline_buffer() -> Dict[str, Any]:
    """Clear all traces from the offline buffer.

    This permanently removes all buffered traces from disk.
    Use with caution as this operation cannot be undone.

    Returns:
        Dictionary with the number of traces removed.

    Examples:
        >>> import openlayer.lib.tracing.tracer as tracer
        >>> result = tracer.clear_offline_buffer()
        >>> print(f"Removed {result['traces_removed']} buffered traces")
    """
    offline_buffer = _get_offline_buffer()
    if not offline_buffer:
        return {
            "traces_removed": 0,
            "error": "Offline buffer not enabled",
        }

    traces_removed = offline_buffer.clear_buffer()
    return {"traces_removed": traces_removed}


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
    is_root_step: bool,
    step_name: str,
    inference_pipeline_id: Optional[str] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
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

        # Add reserved column configurations for user context
        if "user_id" in trace_data:
            config.update({"user_id_column_name": "user_id"})
        if "session_id" in trace_data:
            config.update({"session_id_column_name": "session_id"})
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
            # Use provided pipeline_id, or fall back to configured default,
            # or finally to environment variable
            inference_pipeline_id = (
                inference_pipeline_id
                or _configured_pipeline_id
                or utils.get_env_variable("OPENLAYER_INFERENCE_PIPELINE_ID")
            )
            client = _get_client()

            if client:
                try:
                    response = client.inference_pipelines.data.stream(
                        inference_pipeline_id=inference_pipeline_id,
                        rows=[trace_data],
                        config=config,
                    )
                    print(
                        "Successfully streamed data to Openlayer. Response:",
                        response.to_json(),
                    )

                except Exception as err:  # pylint: disable=broad-except
                    logger.error(traceback.format_exc())
                    logger.error(
                        "Could not stream data to Openlayer (pipeline_id: %s, base_url: %s) Error: %s",
                        inference_pipeline_id,
                        client.base_url if client else "N/A",
                        err,
                    )

                    # Handle failure callback and offline buffering
                    _handle_streaming_failure(
                        trace_data=trace_data,
                        config=config,
                        inference_pipeline_id=inference_pipeline_id,
                        error=err,
                        on_flush_failure=on_flush_failure,
                    )
    else:
        logger.debug("Ending step %s", step_name)


def _handle_streaming_failure(
    trace_data: Dict[str, Any],
    config: Dict[str, Any],
    inference_pipeline_id: str,
    error: Exception,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
) -> None:
    """Handle streaming failure by calling callback and/or storing to buffer.

    Args:
        trace_data: The trace data that failed to send
        config: The configuration used for streaming
        inference_pipeline_id: The pipeline ID used
        error: The exception that occurred
    """
    try:
        # Call the failure callback if configured (per-trace takes priority over global)
        failure_callback = on_flush_failure or _configured_on_flush_failure
        if failure_callback:
            try:
                failure_callback(trace_data, config, error)
                logger.debug("Called on_flush_failure callback")
            except Exception as callback_err:
                logger.error("Error in on_flush_failure callback: %s", callback_err)

        # Store to offline buffer if enabled
        offline_buffer = _get_offline_buffer()
        if offline_buffer:
            success = offline_buffer.store_trace(
                trace_data=trace_data,
                config=config,
                inference_pipeline_id=inference_pipeline_id,
            )
            if success:
                logger.info("Stored failed trace to offline buffer for later replay")
            else:
                logger.error("Failed to store trace to offline buffer")

    except Exception as handler_err:
        logger.error("Error handling streaming failure: %s", handler_err)


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
    guardrail_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Extract function inputs and finalize step logging - common pattern across
    wrappers."""
    inputs = _extract_function_inputs(
        func_signature=func_signature,
        func_args=func_args,
        func_kwargs=func_kwargs,
        context_kwarg=context_kwarg,
    )
    _finalize_step_logging(
        step=step,
        inputs=inputs,
        output=output,
        start_time=step.start_time,
        guardrail_metadata=guardrail_metadata,
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
    guardrail_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Finalize step timing and logging."""
    if step.end_time is None:
        step.end_time = time.time()
    if step.latency is None:
        step.latency = (step.end_time - start_time) * 1000  # in ms

    # Check if manual output was logged
    if step.metadata.get("manual_output_logged"):
        logger.debug("Using manually logged output for step: %s", step.name)
    else:
        step.log(output=output)

    # Start with existing metadata instead of overwriting it
    step_metadata = step.metadata.copy() if step.metadata else {}

    # Add guardrail metadata to step metadata
    if guardrail_metadata:
        step_metadata["guardrails"] = guardrail_metadata

        # Add summary fields for easy filtering
        step_metadata["has_guardrails"] = True
        step_metadata["guardrail_actions"] = [metadata.get("action") for metadata in guardrail_metadata.values()]
        step_metadata["guardrail_names"] = [
            key.replace("input_", "").replace("output_", "") for key in guardrail_metadata.keys()
        ]

        # Add flags for specific actions for easy filtering
        actions = step_metadata["guardrail_actions"]
        step_metadata["guardrail_blocked"] = "blocked" in actions
        step_metadata["guardrail_modified"] = "redacted" in actions or "modified" in actions
        step_metadata["guardrail_allowed"] = "allow" in actions
    else:
        step_metadata["has_guardrails"] = False

    step.log(
        inputs=inputs,
        end_time=step.end_time,
        latency=step.latency,
        metadata=step_metadata,
    )


# ----------------------------- Generator specific functions ----------------------------- #


def _finalize_sync_generator_step(
    step: steps.Step,
    token: Any,
    is_root_step: bool,
    step_name: str,
    inputs: dict,
    output: Any,
    inference_pipeline_id: Optional[str] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
) -> None:
    """Finalize sync generator step - called when generator is consumed."""
    try:
        _current_step.reset(token)
    except ValueError:
        # Context variable was created in a different context (e.g., different thread)
        # This can happen in async/multi-threaded environments like FastAPI/OpenWebUI
        # We can safely ignore this as the step finalization will still complete
        logger.debug("Context variable reset failed - generator consumed in different context")

    _finalize_step_logging(step=step, inputs=inputs, output=output, start_time=step.start_time)

    _handle_trace_completion(
        is_root_step=is_root_step,
        step_name=step_name,
        inference_pipeline_id=inference_pipeline_id,
        on_flush_failure=on_flush_failure,
    )


def _finalize_async_generator_step(
    step: steps.Step,
    token: Any,
    is_root_step: bool,
    step_name: str,
    inputs: dict,
    output: Any,
    inference_pipeline_id: Optional[str] = None,
    on_flush_failure: Optional[OnFlushFailureCallback] = None,
) -> None:
    """Finalize async generator step - called when generator is consumed."""
    _current_step.reset(token)
    _finalize_step_logging(step=step, inputs=inputs, output=output, start_time=step.start_time)
    _handle_trace_completion(
        is_root_step=is_root_step,
        step_name=step_name,
        inference_pipeline_id=inference_pipeline_id,
        on_flush_failure=on_flush_failure,
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
        "inferenceId": trace_obj.inference_id or str(root_step.id),
        "output": root_step.output,
        "latency": root_step.latency,
        "cost": processed_steps[0].get("cost", 0),
        "tokens": processed_steps[0].get("tokens", 0),
        "steps": processed_steps,
        **root_step.metadata,
    }

    # Include trace-level metadata if set - extract keys to row/record level
    if trace_obj.metadata is not None:
        # Add each trace metadata key directly to the row/record level
        trace_data.update(trace_obj.metadata)

    # Add reserved columns for user and session context
    user_id = UserSessionContext.get_user_id()
    if user_id is not None:
        trace_data["user_id"] = user_id

    session_id = UserSessionContext.get_session_id()
    if session_id is not None:
        trace_data["session_id"] = session_id

    if root_step.ground_truth:
        trace_data["groundTruth"] = root_step.ground_truth
    if input_variables:
        trace_data.update(input_variables)

    context = get_rag_context()
    if context:
        trace_data["context"] = context

    return trace_data, input_variable_names


# ----------------------------- Guardrail helper functions ----------------------------- #


def _apply_input_guardrails(
    guardrails: List[Any],
    inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply guardrails to function inputs, creating guardrail steps.

    Args:
        guardrails: List of guardrail instances
        inputs: Extracted function inputs

    Returns:
        Tuple of (modified_inputs, guardrail_metadata)
    """
    if not guardrails:
        return inputs, {}

    modified_inputs = inputs.copy()
    overall_metadata = {}

    for i, guardrail in enumerate(guardrails):
        try:
            # Import here to avoid circular imports
            from ..guardrails.base import BaseGuardrail, GuardrailBlockedException

            if not isinstance(guardrail, BaseGuardrail):
                logger.warning("Skipping invalid guardrail: %s", guardrail)
                continue

            if not guardrail.is_enabled():
                continue

            # Create a guardrail step for this check
            with create_step(
                name=f"{guardrail.name} - Input",
                step_type=enums.StepType.GUARDRAIL,
            ) as guardrail_step:
                try:
                    # Apply the guardrail
                    result = guardrail.check_input(modified_inputs)

                    # Store guardrail metadata for main function step
                    guardrail_key = f"input_{guardrail.name.lower().replace(' ', '_')}"
                    overall_metadata[guardrail_key] = {
                        "action": result.action.value,
                        "reason": result.reason,
                        "metadata": result.metadata or {},
                    }

                    # Prepare step logging data
                    step_log_data = {
                        "action": result.action.value,
                        "reason": result.reason,
                        "data_type": "input",
                        "inputs": {"original_data": modified_inputs},
                    }

                    if result.action.value == "block":
                        # Handle the block according to strategy
                        final_inputs, block_metadata = _handle_guardrail_block(
                            guardrail=guardrail,
                            result=result,
                            modified_inputs=modified_inputs,
                            guardrail_metadata=overall_metadata,
                            guardrail_key=guardrail_key,
                            is_input=True,
                        )

                        # Add final output if different
                        if final_inputs != modified_inputs:
                            step_log_data["output"] = final_inputs

                        # Log once with all data
                        guardrail_step.log(**step_log_data)
                        return final_inputs, overall_metadata

                    elif result.action.value == "modify" and result.modified_data is not None:
                        step_log_data["output"] = result.modified_data
                        modified_inputs = result.modified_data
                        logger.debug("Guardrail %s modified inputs", guardrail.name)

                    else:  # allow
                        step_log_data["output"] = modified_inputs

                    # Single log call with all data
                    guardrail_step.log(**step_log_data)

                except Exception as e:
                    # Create error result for the guardrail step
                    error_result = GuardrailResult(
                        action=GuardrailAction.ALLOW,  # Default to allow on error
                        reason=f"Guardrail error: {str(e)}",
                        metadata={"error": str(e), "error_type": type(e).__name__},
                    )
                    guardrail_step.log(
                        inputs={"original_data": modified_inputs},
                        output=modified_inputs,
                    )

                    if hasattr(e, "guardrail_name"):
                        # Re-raise guardrail exceptions
                        raise
                    else:
                        # Log other exceptions but don't fail the trace
                        logger.error("Error applying input guardrail %s: %s", guardrail.name, e)
                        guardrail_key = f"input_{guardrail.name.lower().replace(' ', '_')}"
                        overall_metadata[guardrail_key] = {
                            "action": "error",
                            "reason": str(e),
                            "metadata": {"error_type": type(e).__name__},
                            "guardrail_name": guardrail.name,
                        }

        except Exception as e:
            # Handle exceptions that occur outside the guardrail step context
            if hasattr(e, "guardrail_name"):
                raise
            else:
                logger.error(
                    "Error setting up input guardrail %s: %s",
                    getattr(guardrail, "name", f"guardrail_{i}"),
                    e,
                )

    return modified_inputs, overall_metadata


def _apply_output_guardrails(guardrails: List[Any], output: Any, inputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Apply guardrails to function output, creating guardrail steps.

    Args:
        guardrails: List of guardrail instances
        output: Function output
        inputs: Function inputs for context

    Returns:
        Tuple of (modified_output, guardrail_metadata)
    """
    if not guardrails:
        return output, {}

    modified_output = output
    overall_metadata = {}

    for i, guardrail in enumerate(guardrails):
        try:
            # Import here to avoid circular imports
            from ..guardrails.base import BaseGuardrail, GuardrailBlockedException

            if not isinstance(guardrail, BaseGuardrail):
                logger.warning("Skipping invalid guardrail: %s", guardrail)
                continue

            if not guardrail.is_enabled():
                continue

            # Create a guardrail step for this check
            with create_step(
                name=f"{guardrail.name} - Output",
                step_type=enums.StepType.GUARDRAIL,
            ) as guardrail_step:
                try:
                    # Apply the guardrail
                    result = guardrail.check_output(modified_output, inputs)

                    # Store guardrail metadata for main function step
                    guardrail_key = f"output_{guardrail.name.lower().replace(' ', '_')}"
                    overall_metadata[guardrail_key] = {
                        "action": result.action.value,
                        "reason": result.reason,
                        "metadata": result.metadata or {},
                    }

                    # Prepare step logging data
                    step_log_data = {
                        "action": result.action.value,
                        "reason": result.reason,
                        "data_type": "output",
                        "inputs": {"original_data": modified_output},
                    }

                    if result.action.value == "block":
                        # Handle the block according to strategy
                        final_output, block_metadata = _handle_guardrail_block(
                            guardrail=guardrail,
                            result=result,
                            modified_output=modified_output,
                            guardrail_metadata=overall_metadata,
                            guardrail_key=guardrail_key,
                            is_input=False,
                        )

                        # Add final output if different
                        if final_output != modified_output:
                            step_log_data["output"] = final_output

                        # Log once with all data
                        guardrail_step.log(**step_log_data)
                        return final_output, overall_metadata

                    elif result.action.value == "modify" and result.modified_data is not None:
                        step_log_data["output"] = result.modified_data
                        modified_output = result.modified_data
                        logger.debug("Guardrail %s modified output", guardrail.name)

                    else:  # allow
                        step_log_data["output"] = modified_output

                    # Single log call with all data
                    guardrail_step.log(**step_log_data)

                except Exception as e:
                    # Create error result for the guardrail step
                    error_result = GuardrailResult(
                        action=GuardrailAction.ALLOW,  # Default to allow on error
                        reason=f"Guardrail error: {str(e)}",
                        metadata={"error": str(e), "error_type": type(e).__name__},
                    )
                    guardrail_step.log(
                        inputs={"original_data": modified_output},
                        output=modified_output,
                    )

                    if hasattr(e, "guardrail_name"):
                        # Re-raise guardrail exceptions
                        raise
                    else:
                        # Log other exceptions but don't fail the trace
                        logger.error("Error applying output guardrail %s: %s", guardrail.name, e)
                        guardrail_key = f"output_{guardrail.name.lower().replace(' ', '_')}"
                        overall_metadata[guardrail_key] = {
                            "action": "error",
                            "reason": str(e),
                            "metadata": {"error_type": type(e).__name__},
                        }
                        guardrail_step.log(**overall_metadata[guardrail_key])

        except Exception as e:
            # Handle exceptions that occur outside the guardrail step context
            if hasattr(e, "guardrail_name"):
                raise
            else:
                logger.error(
                    "Error setting up output guardrail %s: %s",
                    getattr(guardrail, "name", f"guardrail_{i}"),
                    e,
                )

    return modified_output, overall_metadata


def _handle_guardrail_block(
    guardrail: Any,
    result: Any,
    modified_inputs: Optional[Dict[str, Any]] = None,
    modified_output: Optional[Any] = None,
    guardrail_metadata: Optional[Dict[str, Any]] = None,
    guardrail_key: Optional[str] = None,
    is_input: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Handle different block strategies for guardrails.

    Args:
        guardrail: The guardrail instance
        result: The GuardrailResult with block action
        modified_inputs: Current inputs (for input guardrails)
        modified_output: Current output (for output guardrails)
        guardrail_metadata: Current guardrail metadata
        guardrail_key: Key for storing metadata
        is_input: True if this is an input guardrail, False for output

    Returns:
        Tuple of (data, metadata) or raises exception based on strategy
    """
    from ..guardrails.base import BlockStrategy, GuardrailBlockedException

    strategy = getattr(result, "block_strategy", None)
    if strategy is None:
        strategy = BlockStrategy.RAISE_EXCEPTION

    # Update metadata to reflect the blocking strategy used
    if guardrail_metadata is not None and guardrail_key is not None:
        guardrail_metadata[guardrail_key].update(
            {
                "block_strategy": strategy.value,
                "error_message": getattr(result, "error_message", None),
            }
        )

    if strategy == BlockStrategy.RAISE_EXCEPTION:
        # Original behavior - raise exception (breaks pipeline)
        raise GuardrailBlockedException(
            guardrail_name=guardrail.name,
            reason=result.reason or f"{'Input' if is_input else 'Output'} blocked by guardrail",
            metadata=result.metadata,
        )

    elif strategy == BlockStrategy.RETURN_EMPTY:
        # Return empty/None response (graceful)
        if is_input:
            # For input blocking, return empty inputs
            empty_inputs = {key: "" for key in (modified_inputs or {})}
            logger.info("Guardrail %s blocked input, returning empty inputs", guardrail.name)
            return empty_inputs, guardrail_metadata or {}
        else:
            # For output blocking, return None
            logger.info("Guardrail %s blocked output, returning None", guardrail.name)
            return None, guardrail_metadata or {}

    elif strategy == BlockStrategy.RETURN_ERROR_MESSAGE:
        # Return error message (graceful)
        error_msg = getattr(result, "error_message", "Request blocked due to policy violation")
        logger.info(
            "Guardrail %s blocked %s, returning error message",
            guardrail.name,
            "input" if is_input else "output",
        )

        if is_input:
            # For input blocking, replace inputs with error message
            error_inputs = {key: error_msg for key in (modified_inputs or {})}
            return error_inputs, guardrail_metadata or {}
        else:
            # For output blocking, return error message
            return error_msg, guardrail_metadata or {}

    elif strategy == BlockStrategy.SKIP_FUNCTION:
        # Skip function execution, return None (graceful)
        logger.info(
            "Guardrail %s blocked %s, skipping execution",
            guardrail.name,
            "input" if is_input else "output",
        )

        if is_input:
            # For input blocking, this will be handled by the main wrapper
            # We'll use a special marker to indicate function should be skipped
            class SkipFunctionExecution:
                pass

            return SkipFunctionExecution(), guardrail_metadata or {}
        else:
            # For output blocking, return None
            return None, guardrail_metadata or {}

    else:
        # Fallback to raising exception
        logger.warning("Unknown block strategy %s, falling back to raising exception", strategy)
        raise GuardrailBlockedException(
            guardrail_name=guardrail.name,
            reason=result.reason or f"{'Input' if is_input else 'Output'} blocked by guardrail",
            metadata=result.metadata,
        )
