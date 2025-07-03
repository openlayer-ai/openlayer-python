"""Module with the Openlayer tracing processor for OpenAI Agents SDK."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, Union
from uuid import uuid4

from ..tracing import tracer, steps, traces, enums
from .. import utils

try:
    from agents import tracing  # type: ignore[import]

    HAVE_AGENTS = True
except ImportError:
    HAVE_AGENTS = False

    class FileSpanExporter:
        """Write spans/traces to a JSONL file under `logs/`.

        Requires OpenAI Agents SDK: Make sure to install it with ``pip install agents``.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError("The `agents` package is not installed. Please install it with `pip install agents`.")

    class OpenAIAgentsTracingProcessor:
        """Tracing processor for the `OpenAI Agents SDK <https://openai.github.io/openai-agents-python/>`_.

        Traces all intermediate steps of your OpenAI Agent to Openlayer.

        Requirements: Make sure to install the OpenAI Agents SDK with ``pip install agents``.

        Args:
            **kwargs: Additional metadata to associate with all traces.

        Example:
            .. code-block:: python

                from agents import (
                    Agent,
                    FileSearchTool,
                    Runner,
                    WebSearchTool,
                    function_tool,
                    set_trace_processors,
                )

                from openlayer.lib.integrations.openai_agents import OpenAIAgentsTracingProcessor

                set_trace_processors([OpenAIAgentsTracingProcessor()])


                @function_tool
                def get_weather(city: str) -> str:
                    return f"The weather in {city} is sunny"


                haiku_agent = Agent(
                    name="Haiku agent",
                    instructions="Always respond in haiku form",
                    model="o3-mini",
                    tools=[get_weather],
                )
                agent = Agent(
                    name="Assistant",
                    tools=[WebSearchTool()],
                    instructions="speak in spanish. use Haiku agent if they ask for a haiku or for the weather",
                    handoffs=[haiku_agent],
                )

                result = await Runner.run(
                    agent,
                    "write a haiku about the weather today and tell me a recent news story about new york",
                )
                print(result.final_output)
        """  # noqa: E501

        def __init__(self, *args, **kwargs):
            raise ImportError("The `agents` package is not installed. Please install it with `pip install agents`.")


logger = logging.getLogger(__name__)


def repo_path(relative_path: Union[str, Path]) -> Path:
    """Get path relative to the current working directory."""
    return Path.cwd() / relative_path


if HAVE_AGENTS:

    class FileSpanExporter(tracing.TracingProcessor):
        """Write spans/traces to a JSONL file under `logs/`."""

        def __init__(self, logfile: Union[str, Path] = "logs/agent_traces.jsonl") -> None:
            path = repo_path(logfile)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.logfile = path

        def on_trace_start(self, trace: tracing.Trace) -> None:
            """Handle the start of a trace."""
            self._write_item({"event": "trace_start", "trace": trace})

        def on_trace_end(self, trace: tracing.Trace) -> None:
            """Handle the end of a trace."""
            self._write_item({"event": "trace_end", "trace": trace})

        def on_span_start(self, span: tracing.Span) -> None:
            """Handle the start of a span."""
            self._write_item({"event": "span_start", "span": span})

        def on_span_end(self, span: tracing.Span) -> None:
            """Handle the end of a span."""
            self._write_item({"event": "span_end", "span": span})

        def shutdown(self) -> None:
            """Shutdown the exporter."""
            pass

        def force_flush(self) -> None:
            """Force flush any pending data."""
            pass

        def _write_item(self, item: Dict[str, Any]) -> None:
            """Write an item to the log file."""
            with self.logfile.open("a", encoding="utf-8") as f:
                try:
                    # Extract the actual trace/span data for logging
                    if "trace" in item:
                        trace_data = item["trace"].export() if hasattr(item["trace"], "export") else str(item["trace"])
                        log_entry = {
                            "event": item["event"],
                            "type": "trace",
                            "data": trace_data,
                            "timestamp": time.time()
                        }
                    elif "span" in item:
                        span_data = {
                            "span_id": getattr(item["span"], "span_id", None),
                            "trace_id": getattr(item["span"], "trace_id", None),
                            "parent_id": getattr(item["span"], "parent_id", None),
                            "span_data": self._extract_span_data(item["span"]),
                            "started_at": getattr(item["span"], "started_at", None),
                            "ended_at": getattr(item["span"], "ended_at", None),
                            "error": getattr(item["span"], "error", None),
                        }
                        log_entry = {
                            "event": item["event"],
                            "type": "span",
                            "data": span_data,
                            "timestamp": time.time()
                        }
                    else:
                        log_entry = {"event": item["event"], "data": str(item), "timestamp": time.time()}
                    
                    f.write(json.dumps(log_entry, default=str) + "\n")
                except Exception as e:
                    f.write(json.dumps({"error": str(e), "raw_data": str(item), "timestamp": time.time()}) + "\n")

        def _extract_span_data(self, span: tracing.Span) -> Dict[str, Any]:
            """Extract data from a span for logging."""
            span_data = getattr(span, "span_data", None)
            if span_data:
                if hasattr(span_data, "dict") and callable(getattr(span_data, "dict")):
                    try:
                        return span_data.dict()
                    except Exception:
                        pass
                if hasattr(span_data, "__dict__"):
                    return vars(span_data)
            return {"raw_data": str(span_data)}

    class RunData(TypedDict):
        step: steps.Step
        trace_id: str
        start_time: float
        parent_step: Optional[steps.Step]

    class OpenAIAgentsTracingProcessor(tracing.TracingProcessor):  # type: ignore[no-redef]
        """Tracing processor for the `OpenAI Agents SDK <https://openai.github.io/openai-agents-python/>`_.

        Traces all intermediate steps of your OpenAI Agent to Openlayer.

        Requirements: Make sure to install the OpenAI Agents SDK with ``pip install agents``.

        Args:
            **kwargs: Additional metadata to associate with all traces.

        Example:
            .. code-block:: python

                from agents import (
                    Agent,
                    FileSearchTool,
                    Runner,
                    WebSearchTool,
                    function_tool,
                    set_trace_processors,
                )

                from openlayer.lib.integrations.openai_agents import OpenAIAgentsTracingProcessor

                set_trace_processors([OpenAIAgentsTracingProcessor()])


                @function_tool
                def get_weather(city: str) -> str:
                    return f"The weather in {city} is sunny"


                haiku_agent = Agent(
                    name="Haiku agent",
                    instructions="Always respond in haiku form",
                    model="o3-mini",
                    tools=[get_weather],
                )
                agent = Agent(
                    name="Assistant",
                    tools=[WebSearchTool()],
                    instructions="speak in spanish. use Haiku agent if they ask for a haiku or for the weather",
                    handoffs=[haiku_agent],
                )

                result = await Runner.run(
                    agent,
                    "write a haiku about the weather today and tell me a recent news story about new york",
                )
                print(result.final_output)
        """  # noqa: E501

        def __init__(self, **kwargs: Any) -> None:
            """Initialize the OpenAI Agents tracing processor.

            Args:
                **kwargs: Additional metadata to associate with all traces.
            """
            self.metadata: Dict[str, Any] = kwargs or {}
            self._runs: Dict[str, RunData] = {}
            self._root_traces: set[str] = set()  # Track root traces

        def on_trace_start(self, trace: tracing.Trace) -> None:
            """Handle the start of a trace (root agent workflow)."""
            if self._get_trace_name(trace):
                trace_name = self._get_trace_name(trace)
            elif trace.name:
                trace_name = trace.name
            else:
                trace_name = "Agent workflow"

            # Check if we're in an existing trace context
            current_step = tracer.get_current_step()
            current_trace = tracer.get_current_trace()

            if current_step is not None:
                # We're inside a @trace() decorated function - create as nested step
                step = steps.step_factory(
                    step_type=enums.StepType.USER_CALL,
                    name=trace_name,
                    inputs=self._extract_trace_inputs(trace),
                    metadata={**self.metadata, "trace_id": trace.trace_id},
                )
                step.start_time = time.time()
                current_step.add_nested_step(step)
                parent_step = current_step
            elif current_trace is not None:
                # There's an existing trace but no current step
                step = steps.step_factory(
                    step_type=enums.StepType.USER_CALL,
                    name=trace_name,
                    inputs=self._extract_trace_inputs(trace),
                    metadata={**self.metadata, "trace_id": trace.trace_id},
                )
                step.start_time = time.time()
                current_trace.add_step(step)
                parent_step = None
            else:
                # No existing trace - create new one (standalone mode)
                current_trace = traces.Trace()
                tracer._current_trace.set(current_trace)
                tracer._rag_context.set(None)

                step = steps.step_factory(
                    step_type=enums.StepType.USER_CALL,
                    name=trace_name,
                    inputs=self._extract_trace_inputs(trace),
                    metadata={**self.metadata, "trace_id": trace.trace_id},
                )
                step.start_time = time.time()
                current_trace.add_step(step)
                parent_step = None

                # Track root traces (those without existing context)
                self._root_traces.add(trace.trace_id)

            self._runs[trace.trace_id] = RunData(
                step=step,
                trace_id=trace.trace_id,
                start_time=step.start_time,
                parent_step=parent_step,
            )

        def on_trace_end(self, trace: tracing.Trace) -> None:
            """Handle the end of a trace (root agent workflow)."""
            run_data = self._runs.pop(trace.trace_id, None)
            if not run_data:
                return

            step = run_data["step"]
            is_root_trace = trace.trace_id in self._root_traces

            if is_root_trace:
                self._root_traces.remove(trace.trace_id)

            # Update step with final data
            if step.end_time is None:
                step.end_time = time.time()
            if step.latency is None:
                step.latency = (step.end_time - step.start_time) * 1000

            # Set output from trace
            step.output = utils.json_serialize(self._extract_trace_outputs(trace))

            # Add trace metadata
            trace_dict = trace.export() or {}
            step.metadata.update(
                {
                    "agent_trace_metadata": trace_dict.get("metadata", {}),
                    "group_id": trace_dict.get("group_id"),
                }
            )

            # Only upload trace if this was a root trace and we're not in a @trace() context
            if is_root_trace and tracer.get_current_step() is None:
                self._process_and_upload_trace(step)

        def on_span_start(self, span: tracing.Span) -> None:
            """Handle the start of a span (individual agent step)."""
            # Find parent - either from span.parent_id or trace-level parent
            parent_run = None
            parent_id = getattr(span, "parent_id", None)
            trace_id = getattr(span, "trace_id", None)
            span_id = getattr(span, "span_id", None)
            
            if parent_id and parent_id in self._runs:
                parent_run = self._runs[parent_id]
            elif trace_id in self._runs:
                parent_run = self._runs[trace_id]

            if parent_run is None:
                logger.warning(f"No trace info found for span, skipping: {span_id}")
                return

            # Determine step type and name based on span data
            step_type, step_name = self._get_step_info_from_span(span)

            # Extract inputs and metadata from span
            inputs = self._extract_span_inputs(span)
            metadata = self._extract_span_metadata(span)
            metadata.update(self.metadata)

            # Create step
            step = steps.step_factory(
                step_type=step_type,
                name=step_name,
                inputs=inputs,
                metadata=metadata,
            )

            # Set timing
            started_at = getattr(span, "started_at", None)
            step.start_time = datetime.fromisoformat(started_at).timestamp() if started_at else time.time()

            # Add to parent
            parent_step = parent_run["step"]
            parent_step.add_nested_step(step)

            # Store run data
            self._runs[span_id] = RunData(
                step=step,
                trace_id=parent_run["trace_id"],
                start_time=step.start_time,
                parent_step=parent_step,
            )

        def on_span_end(self, span: tracing.Span) -> None:
            """Handle the end of a span (individual agent step)."""
            span_id = getattr(span, "span_id", None)
            run_data = self._runs.pop(span_id, None) if span_id else None
            if not run_data:
                return

            step = run_data["step"]

            # Update timing
            if step.end_time is None:
                ended_at = getattr(span, "ended_at", None)
                step.end_time = datetime.fromisoformat(ended_at).timestamp() if ended_at else time.time()
            if step.latency is None:
                step.latency = (step.end_time - step.start_time) * 1000

            # Set outputs and additional metadata
            step.output = utils.json_serialize(self._extract_span_outputs(span))

            # Add span metadata
            step.metadata.update(
                {
                    "openai_parent_id": getattr(span, "parent_id", None),
                    "openai_trace_id": getattr(span, "trace_id", None),
                    "openai_span_id": span_id,
                }
            )

            # Handle errors
            error = getattr(span, "error", None)
            if error:
                step.metadata["error"] = str(error)

            # Extract token usage and model info for chat completion steps
            if isinstance(step, steps.ChatCompletionStep):
                self._update_llm_step_from_span(step, span)

        def shutdown(self) -> None:
            """Shutdown the processor and flush any remaining data."""
            # No additional cleanup needed for Openlayer integration
            pass

        def force_flush(self) -> None:
            """Force flush any pending data."""
            # No additional flushing needed for Openlayer integration
            pass

        def _get_trace_name(self, trace: tracing.Trace) -> Optional[str]:
            """Extract a meaningful name from the trace."""
            trace_dict = trace.export() or {}
            return trace_dict.get("name") or trace.name

        def _extract_trace_inputs(self, trace: tracing.Trace) -> Dict[str, Any]:
            """Extract inputs from trace data."""
            trace_dict = trace.export() or {}
            return {
                "trace_data": trace_dict,
                "trace_id": trace.trace_id,
            }

        def _extract_trace_outputs(self, trace: tracing.Trace) -> Dict[str, Any]:
            """Extract outputs from trace data."""
            trace_dict = trace.export() or {}
            return {
                "trace_result": trace_dict,
                "trace_id": trace.trace_id,
            }

        def _get_step_info_from_span(self, span: tracing.Span) -> tuple[enums.StepType, str]:
            """Determine step type and name from span data."""
            span_data = getattr(span, "span_data", None)
            span_type = getattr(span_data, "type", None) if span_data else None
            span_name = getattr(span, "name", None) or "Unknown"

            # Map OpenAI Agent span types to Openlayer step types
            if span_type == "completion" or "completion" in str(type(span_data)).lower():
                return enums.StepType.CHAT_COMPLETION, f"Agent Completion - {span_name}"
            elif span_type == "tool_call" or "tool" in str(type(span_data)).lower():
                return enums.StepType.USER_CALL, f"Agent Tool - {span_name}"
            elif span_type == "function" or "function" in str(type(span_data)).lower():
                return enums.StepType.USER_CALL, f"Agent Function - {span_name}"
            else:
                return enums.StepType.USER_CALL, f"Agent Step - {span_name}"

        def _extract_span_inputs(self, span: tracing.Span) -> Dict[str, Any]:
            """Extract inputs from span data."""
            span_id = getattr(span, "span_id", None)
            inputs = {"span_id": span_id}

            span_data = getattr(span, "span_data", None)
            if span_data:
                span_dict = self._span_data_to_dict(span_data)
                inputs.update(span_dict.get("inputs", {}))

                # For completion spans, extract prompt-like data
                if hasattr(span_data, "messages"):
                    inputs["prompt"] = getattr(span_data, "messages", [])
                elif hasattr(span_data, "input"):
                    inputs["input"] = getattr(span_data, "input", None)

            return inputs

        def _extract_span_outputs(self, span: tracing.Span) -> Dict[str, Any]:
            """Extract outputs from span data."""
            span_id = getattr(span, "span_id", None)
            outputs = {"span_id": span_id}

            span_data = getattr(span, "span_data", None)
            if span_data:
                span_dict = self._span_data_to_dict(span_data)
                outputs.update(span_dict.get("outputs", {}))

                # For completion spans, extract response data
                if hasattr(span_data, "response"):
                    outputs["response"] = getattr(span_data, "response", None)
                elif hasattr(span_data, "output"):
                    outputs["output"] = getattr(span_data, "output", None)

            return outputs

        def _extract_span_metadata(self, span: tracing.Span) -> Dict[str, Any]:
            """Extract metadata from span data."""
            metadata = {
                "span_name": getattr(span, "name", None),
                "started_at": getattr(span, "started_at", None),
                "ended_at": getattr(span, "ended_at", None),
            }

            span_data = getattr(span, "span_data", None)
            if span_data:
                span_dict = self._span_data_to_dict(span_data)
                metadata.update(span_dict.get("metadata", {}))

            return metadata

        def _update_llm_step_from_span(self, step: steps.ChatCompletionStep, span: tracing.Span) -> None:
            """Update LLM step with model information from span data."""
            span_data = getattr(span, "span_data", None)
            if not span_data:
                return

            span_dict = self._span_data_to_dict(span_data)

            # Extract model information
            if "model" in span_dict:
                step.model = span_dict["model"]
            if "provider" in span_dict:
                step.provider = span_dict["provider"]
            else:
                step.provider = "OpenAI"  # Default for OpenAI Agents

            # Extract token usage
            usage = span_dict.get("usage", {})
            if usage:
                step.prompt_tokens = usage.get("prompt_tokens", 0)
                step.completion_tokens = usage.get("completion_tokens", 0)
                step.tokens = usage.get("total_tokens", step.prompt_tokens + step.completion_tokens)

        def _span_data_to_dict(self, span_data: Any) -> Dict[str, Any]:
            """Convert span data to dictionary format."""
            if hasattr(span_data, "dict") and callable(getattr(span_data, "dict")):
                try:
                    return span_data.dict()
                except Exception:
                    pass

            if hasattr(span_data, "__dict__"):
                return vars(span_data)

            return {"raw_data": str(span_data)}

        def _process_and_upload_trace(self, root_step: steps.Step) -> None:
            """Process and upload the completed trace (only for standalone root traces)."""
            current_trace = tracer.get_current_trace()
            if not current_trace:
                return

            # Post-process the trace
            trace_data, input_variable_names = tracer.post_process_trace(current_trace)

            # Configure trace data for upload
            config = dict(
                tracer.ConfigLlmData(
                    output_column_name="output",
                    input_variable_names=input_variable_names,
                    latency_column_name="latency",
                    cost_column_name="cost",
                    timestamp_column_name="inferenceTimestamp",
                    inference_id_column_name="inferenceId",
                    num_of_token_column_name="tokens",
                )
            )

            # Add additional config based on trace data
            if "groundTruth" in trace_data:
                config.update({"ground_truth_column_name": "groundTruth"})
            if "context" in trace_data:
                config.update({"context_column_name": "context"})
            if isinstance(root_step, steps.ChatCompletionStep) and root_step.inputs and "prompt" in root_step.inputs:
                config.update({"prompt": root_step.inputs["prompt"]})

            # Upload trace data to Openlayer
            if tracer._publish:
                try:
                    tracer._client.inference_pipelines.data.stream(
                        inference_pipeline_id=utils.get_env_variable("OPENLAYER_INFERENCE_PIPELINE_ID"),
                        rows=[trace_data],
                        config=config,
                    )
                except Exception as err:  # pylint: disable=broad-except
                    logger.error("Could not stream data to Openlayer: %s", err)

            # Reset trace context only for standalone traces
            tracer._current_trace.set(None)
