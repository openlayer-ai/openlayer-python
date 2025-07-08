"""Module with the Openlayer tracing processor for OpenAI Agents SDK."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union, List

from ..tracing import tracer, steps, enums

try:
    from agents import tracing  # type: ignore[import]

    HAVE_AGENTS = True
except ImportError:
    HAVE_AGENTS = False

logger = logging.getLogger(__name__)


def repo_path(relative_path: Union[str, Path]) -> Path:
    """Get path relative to the current working directory."""
    return Path.cwd() / relative_path


class ParsedSpanData:
    """Parsed span data with meaningful input/output extracted."""

    def __init__(
        self,
        name: str,
        span_type: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.span_type = span_type
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata or {}
        self.model = model
        self.provider = provider
        self.usage = usage


def _extract_messages_from_input(input_data: Any) -> List[Dict[str, Any]]:
    """Extract and normalize messages from input data.

    This helper function eliminates duplicate message processing logic.
    """
    if not isinstance(input_data, (list, tuple)):
        return []

    prompt_messages = []
    for msg in input_data:
        if isinstance(msg, dict):
            prompt_messages.append(msg)
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            prompt_messages.append({"role": msg.role, "content": msg.content})
        elif hasattr(msg, "__dict__"):
            # Try to convert object to dict
            msg_dict = vars(msg)
            prompt_messages.append(msg_dict)

    return prompt_messages


def _extract_response_output(response_output: Any) -> Optional[Dict[str, Any]]:
    """Extract actual output content from response.output object.

    This helper function consolidates complex response extraction logic.
    """
    if not response_output:
        return None

    try:
        if isinstance(response_output, str):
            # Sometimes output might be a string directly
            return {"output": response_output}

        if isinstance(response_output, list) and response_output:
            first_item = response_output[0]

            # Check if this is a function call (common for handoffs)
            if (
                hasattr(first_item, "type")
                and first_item.type == "function_call"
                and hasattr(first_item, "name")
            ):
                # This is a function call response, create meaningful description
                func_name = first_item.name
                return {"output": f"Made function call: {func_name}"}

            # Check if this is a ResponseOutputMessage (actual LLM response)
            elif (
                hasattr(first_item, "type")
                and first_item.type == "message"
                and hasattr(first_item, "content")
                and first_item.content
            ):
                # This is the actual LLM response in ResponseOutputMessage format
                content_list = first_item.content
                if isinstance(content_list, list) and content_list:
                    # Look for ResponseOutputText in the content
                    for content_item in content_list:
                        if (
                            hasattr(content_item, "type")
                            and content_item.type == "output_text"
                            and hasattr(content_item, "text")
                            and content_item.text
                        ):
                            return {"output": content_item.text}
                    # No output_text found, try first content item
                    first_content = content_list[0]
                    if hasattr(first_content, "text"):
                        return {"output": first_content.text}
                    else:
                        return {"output": str(first_content)}
                else:
                    return {"output": str(content_list)}

            # Otherwise try to extract message content normally (legacy format)
            elif hasattr(first_item, "content") and first_item.content:
                # Extract text from content parts
                content_parts = first_item.content

                if isinstance(content_parts, list) and content_parts:
                    first_content = content_parts[0]

                    if hasattr(first_content, "text") and first_content.text:
                        return {"output": first_content.text}
                    elif hasattr(first_content, "content"):
                        # Sometimes the text might be in a 'content' field
                        return {"output": str(first_content.content)}
                    else:
                        # Fallback: try to convert the whole content to string
                        return {"output": str(first_content)}
                elif isinstance(content_parts, str):
                    # Sometimes content_parts might be a string directly
                    return {"output": content_parts}
                else:
                    # Fallback: convert whatever we have to string
                    return {"output": str(content_parts)}
            elif hasattr(first_item, "text"):
                # Sometimes the text might be directly on the message
                return {"output": first_item.text}
            else:
                # No text content found - indicate this was a non-text response
                return {"output": "Agent response (no text content)"}
        else:
            # Fallback for unknown response formats
            return {"output": "Agent response (unknown format)"}

    except Exception:
        return None


def parse_span_data(span_data: Any) -> ParsedSpanData:
    """Parse OpenAI Agents SDK span data to extract meaningful input/output."""
    try:
        # First try to use the official export() method
        content = {}
        if hasattr(span_data, "export") and callable(getattr(span_data, "export")):
            try:
                content = span_data.export()
            except Exception:
                pass

        # Get span type
        span_type = content.get("type") or getattr(span_data, "type", "unknown")

        # Initialize parsed data
        name = _get_span_name(span_data, span_type)
        input_data = None
        output_data = None
        metadata = content.copy()
        model = None
        provider = None
        usage = None

        # Parse based on span type
        if span_type == "function":
            input_data = getattr(span_data, "input", None)
            output_data = getattr(span_data, "output", None)

            # Try to extract function arguments from exported content
            function_args = content.get("input", {})
            function_name = content.get("name", "unknown_function")
            function_output = content.get("output", None)

            # Use content data if span attributes are empty
            if not input_data and function_args:
                input_data = function_args

            # Parse JSON string arguments into proper objects
            if input_data and isinstance(input_data, dict):
                # Check if we have a single 'input' key with a JSON string value
                if "input" in input_data and isinstance(input_data["input"], str):
                    try:
                        # Try to parse the JSON string
                        parsed_args = json.loads(input_data["input"])
                        input_data = parsed_args
                    except (json.JSONDecodeError, TypeError):
                        # Keep original string format if parsing fails
                        pass

            if not output_data and function_output is not None:
                output_data = function_output

            metadata.pop("input", None)
            metadata.pop("output", None)

        elif span_type == "generation":
            input_data = getattr(span_data, "input", None)
            output_data = getattr(span_data, "output", None)
            model = getattr(span_data, "model", None)
            provider = "OpenAI"

            # Extract usage information
            usage_obj = getattr(span_data, "usage", None)
            if usage_obj:
                usage = _extract_usage_dict(usage_obj)

            # Extract prompt information from input using helper function
            if input_data:
                prompt_messages = _extract_messages_from_input(input_data)
                if prompt_messages:
                    input_data = {
                        "messages": prompt_messages,
                        "prompt": prompt_messages,
                    }

            metadata.pop("input", None)
            metadata.pop("output", None)

        elif span_type == "response":
            return _parse_response_span_data(span_data)

        elif span_type == "agent":
            output_data = {"output_type": getattr(span_data, "output_type", None)}

        elif span_type == "handoff":
            # Extract handoff information from the span data
            input_data = {}
            from_agent = getattr(span_data, "from_agent", None)
            to_agent = getattr(span_data, "to_agent", None)

            # Try to extract from the exported content as well
            if from_agent is None and "from_agent" in content:
                from_agent = content["from_agent"]
            if to_agent is None and "to_agent" in content:
                to_agent = content["to_agent"]

            # If to_agent is still None, check for other fields that might indicate the
            #  target
            if to_agent is None:
                # Sometimes handoff data might be in other fields
                handoff_data = getattr(span_data, "data", {})
                if isinstance(handoff_data, dict):
                    to_agent = handoff_data.get("to_agent") or handoff_data.get(
                        "target_agent"
                    )

            input_data = {
                "from_agent": from_agent or "Unknown Agent",
                "to_agent": to_agent or "Unknown Target",
            }

        elif span_type == "custom":
            data = getattr(span_data, "data", {})
            input_data = data.get("input")
            output_data = data.get("output")
            metadata.pop("data", None)

        # Ensure input/output are dictionaries
        if input_data is not None and not isinstance(input_data, dict):
            input_data = {"input": input_data}

        if output_data is not None and not isinstance(output_data, dict):
            output_data = {"output": output_data}

        return ParsedSpanData(
            name=name,
            span_type=span_type,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            model=model,
            provider=provider,
            usage=usage,
        )

    except Exception as e:
        logger.error(f"Failed to parse span data: {e}")
        return ParsedSpanData(
            name="Unknown", span_type="unknown", metadata={"parse_error": str(e)}
        )


def _get_span_name(span_data: Any, span_type: str) -> str:
    """Get appropriate name for the span."""
    if hasattr(span_data, "name") and span_data.name:
        return span_data.name
    elif span_type == "generation":
        return "Generation"
    elif span_type == "response":
        return "Response"
    elif span_type == "handoff":
        return "Handoff"
    elif span_type == "agent":
        return "Agent"
    elif span_type == "function":
        return "Function"
    else:
        return span_type.title()


def _parse_response_span_data(span_data: Any) -> ParsedSpanData:
    """Parse response span data to extract meaningful conversation content."""
    response = getattr(span_data, "response", None)

    if response is None:
        return ParsedSpanData(
            name="Response", span_type="response", metadata={"no_response": True}
        )

    input_data = None
    output_data = None
    usage = None
    model = None
    metadata = {}

    try:
        # Extract input - this might be available in some cases
        if hasattr(span_data, "input") and span_data.input:
            input_data = {"input": span_data.input}

            # Try to extract prompt/messages from input using helper function
            prompt_messages = _extract_messages_from_input(span_data.input)
            if prompt_messages:
                input_data["messages"] = prompt_messages
                input_data["prompt"] = prompt_messages

        # Extract agent instructions and tools from the response object if available
        instructions = None
        tools_info = None

        if response and hasattr(response, "instructions") and response.instructions:
            instructions = response.instructions

        if response and hasattr(response, "tools") and response.tools:
            tools_info = []
            for tool in response.tools:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    tools_info.append(
                        {"name": tool.name, "description": tool.description}
                    )
                elif isinstance(tool, dict):
                    tools_info.append(
                        {
                            "name": tool.get("name", "unknown"),
                            "description": tool.get("description", ""),
                        }
                    )

        # Create comprehensive prompt with system instructions if we found them
        if instructions or tools_info:
            # Start with system instructions if available
            enhanced_messages = []
            if instructions:
                enhanced_messages.append({"role": "system", "content": instructions})

            # Add tool descriptions as system context if available
            if tools_info:
                tools_description = "Available tools:\n" + "\n".join(
                    [f"- {tool['name']}: {tool['description']}" for tool in tools_info]
                )
                enhanced_messages.append(
                    {"role": "system", "content": tools_description}
                )

            # Add the original user messages
            if input_data and "messages" in input_data:
                enhanced_messages.extend(input_data["messages"])
            elif (
                input_data
                and "input" in input_data
                and isinstance(input_data["input"], list)
            ):
                enhanced_messages.extend(input_data["input"])

            # Update input_data with enhanced prompt
            if not input_data:
                input_data = {}
            input_data["messages"] = enhanced_messages
            input_data["prompt"] = enhanced_messages
            input_data["instructions"] = instructions
            if tools_info:
                input_data["tools"] = tools_info

        # Extract output from response.output using helper function
        if hasattr(response, "output") and response.output:
            output_data = _extract_response_output(response.output)
            if not output_data:
                # Try fallback approaches
                try:
                    if hasattr(response, "text") and response.text:
                        output_data = {"output": response.text}
                    elif hasattr(response, "output"):
                        output_data = {"output": "Agent response (extraction failed)"}
                except Exception:
                    output_data = {"output": "Response content extraction failed"}

        # Extract model and usage
        if hasattr(response, "model"):
            model = response.model

        if hasattr(response, "usage") and response.usage:
            usage = _extract_usage_dict(response.usage)

        # Add response metadata
        if hasattr(response, "id"):
            metadata["response_id"] = response.id
        if hasattr(response, "object"):
            metadata["response_object"] = response.object
        if hasattr(response, "tools"):
            metadata["tools_count"] = len(response.tools) if response.tools else 0

    except Exception as e:
        logger.error(f"Failed to parse response span data: {e}")
        metadata["parse_error"] = str(e)

    return ParsedSpanData(
        name="Response",
        span_type="response",
        input_data=input_data,
        output_data=output_data,
        metadata=metadata,
        model=model,
        provider="OpenAI",
        usage=usage,
    )


def _extract_usage_dict(usage_obj: Any) -> Dict[str, Any]:
    """Extract usage information as a dictionary."""
    if not usage_obj:
        return {}

    try:
        # Try model_dump first (Pydantic models)
        if hasattr(usage_obj, "model_dump"):
            result = usage_obj.model_dump()
            return result

        # Try __dict__ next
        elif hasattr(usage_obj, "__dict__"):
            result = vars(usage_obj)
            return result

        # Manual extraction with multiple field name conventions
        else:
            # Try different field naming conventions
            usage_dict = {}

            # OpenAI Responses API typically uses these field names
            for input_field in ["input_tokens", "prompt_tokens"]:
                if hasattr(usage_obj, input_field):
                    usage_dict["input_tokens"] = getattr(usage_obj, input_field)
                    usage_dict["prompt_tokens"] = getattr(usage_obj, input_field)
                    break

            for output_field in ["output_tokens", "completion_tokens"]:
                if hasattr(usage_obj, output_field):
                    usage_dict["output_tokens"] = getattr(usage_obj, output_field)
                    usage_dict["completion_tokens"] = getattr(usage_obj, output_field)
                    break

            for total_field in ["total_tokens"]:
                if hasattr(usage_obj, total_field):
                    usage_dict["total_tokens"] = getattr(usage_obj, total_field)
                    break

            # If we couldn't find specific fields, try to get all attributes
            if not usage_dict:
                for attr in dir(usage_obj):
                    if not attr.startswith("_") and not callable(
                        getattr(usage_obj, attr)
                    ):
                        value = getattr(usage_obj, attr)
                        usage_dict[attr] = value

            return usage_dict
    except Exception:
        return {"usage_extraction_error": "Failed to extract usage"}


# Global reference to the active OpenlayerTracerProcessor
_active_openlayer_processor: Optional["OpenlayerTracerProcessor"] = None


def capture_user_input(trace_id: str, user_input: str) -> None:
    """Capture user input at the application level.

    This is a convenience function that forwards to the active OpenlayerTracerProcessor.

    Args:
        trace_id: The trace ID to associate the input with
        user_input: The user's input message
    """
    if _active_openlayer_processor:
        _active_openlayer_processor.capture_user_input(trace_id, user_input)


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID if available.

    Returns:
        The current trace ID or None if not available
    """
    # This would need to be implemented based on the OpenAI Agents SDK
    # For now, we'll need to pass the trace_id explicitly
    return None


def _extract_span_attributes(span: Any) -> Dict[str, Any]:
    """Extract common span attributes to eliminate duplicate getattr calls.

    This helper function consolidates span attribute extraction patterns.
    """
    return {
        "span_id": getattr(span, "span_id", None),
        "trace_id": getattr(span, "trace_id", None),
        "parent_id": getattr(span, "parent_id", None),
    }


def _extract_token_counts(usage: Dict[str, Any]) -> Dict[str, int]:
    """Extract token counts from usage data with field name variations.

    This helper function eliminates duplicate token extraction logic.
    """
    if not usage or not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Support multiple field name conventions
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _configure_chat_completion_step(
    step: steps.ChatCompletionStep,
    start_time: float,
    model: str,
    provider: str,
    usage: Dict[str, Any],
    model_parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Configure ChatCompletionStep attributes to eliminate duplicate setup code.

    This helper function consolidates ChatCompletionStep attribute setting.
    """
    token_counts = _extract_token_counts(usage)

    step.start_time = start_time
    step.model = model
    step.provider = provider
    step.prompt_tokens = token_counts["prompt_tokens"]
    step.completion_tokens = token_counts["completion_tokens"]
    step.tokens = token_counts["total_tokens"]
    step.model_parameters = model_parameters or {}


class OpenlayerTracerProcessor(tracing.TracingProcessor):  # type: ignore[no-redef]
    """Tracing processor for the `OpenAI Agents SDK
    <https://openai.github.io/openai-agents-python/>`_.

    Traces all intermediate steps of your OpenAI Agent to Openlayer using the official
    span data models and export() methods for standardized data extraction.

    Requirements: Make sure to install the OpenAI Agents SDK with
    ``pip install openai-agents``.



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

            from openlayer.lib.integrations.openai_agents import (
                OpenlayerTracerProcessor,
            )

            set_trace_processors([OpenlayerTracerProcessor()])


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
                instructions="speak in spanish. use Haiku agent if they ask for a haiku"
                "or for the weather",
                handoffs=[haiku_agent],
            )

            result = await Runner.run(
                agent,
                "write a haiku about the weather today and tell me a recent news story"
                "about new york",
            )
            print(result.final_output)
    """  # noqa: E501

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the OpenAI Agents tracing processor.

        Args:
            **kwargs: Additional metadata to associate with all traces.
        """
        self.metadata: Dict[str, Any] = kwargs or {}
        self._active_traces: Dict[str, Dict[str, Any]] = {}
        self._active_steps: Dict[str, steps.Step] = {}
        self._current_user_inputs: Dict[str, List[str]] = (
            {}
        )  # Track user inputs by trace_id

        self._trace_first_meaningful_input: Dict[str, Dict[str, Any]] = {}
        self._trace_last_meaningful_output: Dict[str, Dict[str, Any]] = {}

        # Track step hierarchy using span_id -> step mapping and parent relationships
        self._span_to_step: Dict[str, steps.Step] = {}  # span_id -> step
        self._step_parents: Dict[str, str] = {}  # span_id -> parent_span_id
        self._step_children: Dict[str, List[str]] = (
            {}
        )  # span_id -> list of child span_ids
        self._children_already_added: Dict[str, set] = (
            {}
        )  # parent_span_id -> set of added child_span_ids

        # Collect root-level steps for each trace (steps without parents)
        self._trace_root_steps: Dict[str, List[steps.Step]] = {}

        # Register this processor as the active one for user input capture
        global _active_openlayer_processor
        _active_openlayer_processor = self

    def on_trace_start(self, trace: tracing.Trace) -> None:
        """Handle the start of a trace (root agent workflow)."""
        try:
            # Get trace information
            trace_export = trace.export() if hasattr(trace, "export") else {}
            trace_name = trace_export.get("workflow_name", "Agent Workflow")

            # Initialize trace data collection
            self._active_traces[trace.trace_id] = {
                "trace_name": trace_name,
                "trace_export": trace_export,
                "start_time": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to handle trace start: {e}")

    def on_trace_end(self, trace: tracing.Trace) -> None:
        """Handle the end of a trace (root agent workflow)."""
        try:
            trace_data = self._active_traces.pop(trace.trace_id, None)
            if not trace_data:
                return

            # Calculate total duration
            end_time = time.time()
            duration = end_time - trace_data["start_time"]

            # Get all collected root steps for this trace
            steps_list = self._trace_root_steps.pop(trace.trace_id, [])

            # Remove duplicates based on step ID (keep the most recent one)
            unique_steps = {}
            for step in steps_list:
                step_id = getattr(step, "id", None)
                if step_id:
                    unique_steps[step_id] = step
                else:
                    # If no ID, add anyway (shouldn't happen normally)
                    unique_steps[id(step)] = step

            steps_list = list(unique_steps.values())

            if steps_list:
                # Create a root step that encompasses all collected steps
                trace_name = trace_data.get("trace_name", "Agent Workflow")

                # Get meaningful input/output if available
                first_input = self._trace_first_meaningful_input.get(trace.trace_id)
                last_output = self._trace_last_meaningful_output.get(trace.trace_id)

                # Create inputs from first meaningful input or from user input
                inputs = first_input or {}
                if trace.trace_id in self._current_user_inputs:
                    user_inputs = self._current_user_inputs[trace.trace_id]
                    if user_inputs:
                        inputs["user_query"] = user_inputs[
                            -1
                        ]  # Use the last user input

                # Create output from last meaningful output
                output = "Agent workflow completed"
                if last_output:
                    if isinstance(last_output, dict) and "output" in last_output:
                        output = last_output["output"]
                    else:
                        output = str(last_output)

                # Create consolidated trace using the standard tracer API
                with tracer.create_step(
                    name=trace_name,
                    step_type=enums.StepType.USER_CALL,
                    inputs=inputs,
                    output=output,
                    metadata={**self.metadata, "trace_id": trace.trace_id},
                ) as root_step:
                    # Add all collected root steps as nested steps
                    # The nested steps will automatically include their own nested steps
                    for step in steps_list:
                        root_step.add_nested_step(step)

                    # Set the end time to match the trace end time
                    root_step.end_time = end_time
                    root_step.latency = duration * 1000  # Convert to ms

            # Clean up trace-specific data
            self._current_user_inputs.pop(trace.trace_id, None)
            self._trace_first_meaningful_input.pop(trace.trace_id, None)
            self._trace_last_meaningful_output.pop(trace.trace_id, None)

            # Clean up span hierarchy tracking for this trace
            # We need to find all spans that belong to this trace and remove them
            spans_to_remove = []
            for span_id, step in list(self._span_to_step.items()):
                # Check if this span belongs to the ended trace
                if (
                    hasattr(step, "metadata")
                    and step.metadata.get("trace_id") == trace.trace_id
                ):
                    spans_to_remove.append(span_id)

            # Remove span mappings for this trace
            for span_id in spans_to_remove:
                self._span_to_step.pop(span_id, None)
                self._step_parents.pop(span_id, None)
                self._step_children.pop(span_id, None)

        except Exception as e:
            logger.error(f"Failed to handle trace end: {e}")

    def on_span_start(self, span: tracing.Span) -> None:
        """Handle the start of a span (individual agent step)."""
        try:
            # Extract span attributes using helper function
            span_attrs = _extract_span_attributes(span)
            span_id = span_attrs["span_id"]
            trace_id = span_attrs["trace_id"]
            parent_id = span_attrs["parent_id"]

            if not span_id or not trace_id:
                return

            if trace_id not in self._active_traces:
                return

            # Extract span data
            span_data = getattr(span, "span_data", None)
            if not span_data:
                return

            # Create the appropriate Openlayer step based on span type
            step = self._create_step_for_span(span, span_data)
            if step:
                # Store the step mapping
                self._active_steps[span_id] = step
                self._span_to_step[span_id] = step

                # Track parent-child relationships
                if parent_id:
                    self._step_parents[span_id] = parent_id

                    # Add to parent's children list
                    if parent_id not in self._step_children:
                        self._step_children[parent_id] = []
                    self._step_children[parent_id].append(span_id)

                    # Track that this child has been added to prevent duplicates
                    if parent_id not in self._children_already_added:
                        self._children_already_added[parent_id] = set()

                    # Add this step as a nested step to its parent (if parent exists)
                    parent_step = self._span_to_step.get(parent_id)
                    if parent_step:
                        parent_step.add_nested_step(step)
                        self._children_already_added[parent_id].add(span_id)
                else:
                    # This is a root-level step (no parent)
                    if trace_id not in self._trace_root_steps:
                        self._trace_root_steps[trace_id] = []
                    self._trace_root_steps[trace_id].append(step)

        except Exception as e:
            logger.error(f"Failed to handle span start: {e}")

    def on_span_end(self, span: tracing.Span) -> None:
        """Handle the end of a span (individual agent step)."""
        try:
            # Extract span attributes using helper function
            span_attrs = _extract_span_attributes(span)
            span_id = span_attrs["span_id"]
            trace_id = span_attrs["trace_id"]

            if not span_id:
                return

            step = self._active_steps.pop(span_id, None)
            if not step:
                return

            # Update step with final span data
            span_data = getattr(span, "span_data", None)
            if span_data:
                self._update_step_with_span_data(step, span, span_data)

                if trace_id and span_data:
                    parsed_data = parse_span_data(span_data)

                    # Track meaningful span types (response, generation, custom)
                    if parsed_data.span_type in ["response", "generation", "custom"]:
                        # Track first meaningful input
                        if (
                            parsed_data.input_data
                            and trace_id not in self._trace_first_meaningful_input
                        ):
                            self._trace_first_meaningful_input[trace_id] = (
                                parsed_data.input_data
                            )

                        # Track last meaningful output
                        if parsed_data.output_data:
                            self._trace_last_meaningful_output[trace_id] = (
                                parsed_data.output_data
                            )

            # Handle any orphaned children (children that were created before their
            # parent)
            # BUT only add children that haven't already been added
            if span_id in self._step_children:
                already_added = self._children_already_added.get(span_id, set())
                for child_span_id in self._step_children[span_id]:
                    if child_span_id not in already_added:
                        child_step = self._span_to_step.get(child_span_id)
                        if child_step:
                            step.add_nested_step(child_step)
                            already_added.add(child_span_id)

            # Set end time
            ended_at = getattr(span, "ended_at", None)
            if ended_at:
                try:
                    step.end_time = datetime.fromisoformat(
                        ended_at.replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, AttributeError):
                    step.end_time = time.time()
            else:
                step.end_time = time.time()

            # Calculate latency
            if hasattr(step, "start_time") and step.start_time:
                step.latency = (step.end_time - step.start_time) * 1000  # Convert to ms

        except Exception as e:
            logger.error(f"Failed to handle span end: {e}")

    def _create_step_for_span(
        self, span: tracing.Span, span_data: Any
    ) -> Optional[steps.Step]:
        """Create the appropriate Openlayer step for a span."""
        try:
            # Parse the span data using our new parsing approach
            parsed_data = parse_span_data(span_data)

            # Get basic span info using helper function
            span_attrs = _extract_span_attributes(span)
            started_at = getattr(span, "started_at", None)
            start_time = time.time()
            if started_at:
                try:
                    start_time = datetime.fromisoformat(
                        started_at.replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, AttributeError):
                    pass

            metadata = {
                **self.metadata,
                **span_attrs,  # Use extracted attributes
                "span_type": parsed_data.span_type,
                "started_at": started_at,
                **parsed_data.metadata,
            }

            # Create step based on span type
            if parsed_data.span_type == "generation":
                return self._create_generation_step(parsed_data, start_time, metadata)
            elif parsed_data.span_type == "function":
                return self._create_function_step(parsed_data, start_time, metadata)
            elif parsed_data.span_type == "agent":
                return self._create_agent_step(parsed_data, start_time, metadata)
            elif parsed_data.span_type == "handoff":
                return self._create_handoff_step(parsed_data, start_time, metadata)
            elif parsed_data.span_type == "response":
                return self._create_response_step(parsed_data, start_time, metadata)
            else:
                return self._create_generic_step(parsed_data, start_time, metadata)

        except Exception as e:
            logger.error(f"Failed to create step for span: {e}")
            return None

    def _create_generation_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create a generation step from GenerationSpanData."""
        # Extract inputs and outputs from parsed data
        inputs = parsed_data.input_data or {}
        output = self._extract_output_from_parsed_data(parsed_data, "LLM response")

        # Extract model and usage info from parsed data
        model = parsed_data.model or "unknown"
        model_config = parsed_data.metadata.get("model_config", {})

        # Create step without immediately sending to Openlayer
        step = steps.ChatCompletionStep(
            name=f"LLM Generation ({model})",
            inputs=inputs,
            output=output,
            metadata=metadata,
        )

        # Use helper function to configure ChatCompletionStep attributes
        _configure_chat_completion_step(
            step=step,
            start_time=start_time,
            model=model,
            provider=parsed_data.provider or "OpenAI",
            usage=parsed_data.usage or {},
            model_parameters=model_config,
        )

        return step

    def _create_function_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create a function call step from FunctionSpanData."""
        function_name = parsed_data.name or "unknown_function"
        function_input = parsed_data.input_data or {}
        function_output = parsed_data.output_data or {}

        inputs = function_input if function_input else {}
        output = function_output if function_output else "Function completed"

        # Create step without immediately sending to Openlayer
        step = steps.UserCallStep(
            name=f"Tool Call: {function_name}",
            inputs=inputs,
            output=output,
            metadata=metadata,
        )
        step.start_time = start_time
        return step

    def _create_agent_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create an agent step from AgentSpanData."""
        agent_name = parsed_data.name or "Agent"
        tools = parsed_data.metadata.get("tools", [])
        handoffs = parsed_data.metadata.get("handoffs", [])
        output_type = parsed_data.metadata.get("output_type", "str")

        inputs = {
            "agent_name": agent_name,
            "available_tools": tools,
            "available_handoffs": handoffs,
            "output_type": output_type,
        }

        # Create more meaningful output for agent steps
        if handoffs and len(handoffs) > 0:
            handoff_list = ", ".join(handoffs)
            output = f"Agent {agent_name} initialized with handoffs to: {handoff_list}"
        elif tools and len(tools) > 0:
            tools_list = ", ".join(
                [tool if isinstance(tool, str) else str(tool) for tool in tools]
            )
            output = f"Agent {agent_name} initialized with tools: {tools_list}"
        else:
            output = f"Agent {agent_name} initialized and ready"

        # Create step without immediately sending to Openlayer
        step = steps.UserCallStep(
            name=f"Agent: {agent_name}", inputs=inputs, output=output, metadata=metadata
        )
        step.start_time = start_time

        return step

    def _create_handoff_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create a handoff step from HandoffSpanData."""
        from_agent = (
            parsed_data.input_data.get("from_agent", "unknown")
            if parsed_data.input_data
            else "unknown"
        )
        to_agent = (
            parsed_data.input_data.get("to_agent", "unknown")
            if parsed_data.input_data
            else "unknown"
        )

        inputs = {"from_agent": from_agent, "to_agent": to_agent}

        # Create step without immediately sending to Openlayer
        step = steps.UserCallStep(
            name=f"Handoff: {from_agent} → {to_agent}",
            inputs=inputs,
            output=f"Handed off from {from_agent} to {to_agent}",
            metadata=metadata,
        )
        step.start_time = start_time
        return step

    def _create_response_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create a response step from ResponseSpanData."""
        response_id = parsed_data.metadata.get("response_id", "unknown")

        # Start with proper input data from parsed_data
        inputs = {}

        # Use the parsed input data which contains the actual conversation messages
        if parsed_data.input_data:
            inputs.update(parsed_data.input_data)

            # If we have messages, format them properly for ChatCompletion
            if "messages" in parsed_data.input_data:
                messages = parsed_data.input_data["messages"]
                inputs["messages"] = messages
                inputs["prompt"] = messages  # Also add as prompt for compatibility

                # Create a readable prompt summary
                user_messages = [
                    msg.get("content", "")
                    for msg in messages
                    if msg.get("role") == "user"
                ]
                if user_messages:
                    inputs["user_query"] = user_messages[
                        -1
                    ]  # Use the last user message

            # If we have input field, use it as well
            if "input" in parsed_data.input_data:
                input_data = parsed_data.input_data["input"]
                if isinstance(input_data, list) and input_data:
                    # Extract user content from input list
                    user_content = next(
                        (
                            msg.get("content", "")
                            for msg in input_data
                            if msg.get("role") == "user"
                        ),
                        "",
                    )
                    if user_content:
                        inputs["user_query"] = user_content
                        if "messages" not in inputs:
                            inputs["messages"] = input_data
                            inputs["prompt"] = input_data

        # If we still don't have good input, try to get user input from
        # application-level capture
        if not inputs or ("user_query" not in inputs and "messages" not in inputs):
            trace_id = metadata.get("trace_id")
            if trace_id:
                user_input = self._get_user_input_for_trace(trace_id)
                if user_input:
                    inputs["user_query"] = user_input
                    inputs["messages"] = [{"role": "user", "content": user_input}]
                    inputs["prompt"] = [{"role": "user", "content": user_input}]

        # Fallback to response_id if we still have no good input
        if not inputs:
            inputs = {"response_id": response_id}

        # Use the parsed output data which contains the actual conversation content
        output = self._extract_output_from_parsed_data(
            parsed_data, "Response processed"
        )

        # Always create ChatCompletionStep for response spans - tokens will be updated
        #  in span end handler
        step = steps.ChatCompletionStep(
            name="Agent Response", inputs=inputs, output=output, metadata=metadata
        )

        # Use helper function to configure ChatCompletionStep attributes
        _configure_chat_completion_step(
            step=step,
            start_time=start_time,
            model=parsed_data.model or "unknown",
            provider=parsed_data.provider or "OpenAI",
            usage=parsed_data.usage or {},
        )

        return step

    def _extract_function_calls_from_messages(
        self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> None:
        """Extract function calls from conversation messages and create Tool Call steps.

        This ensures that handoff functions that are captured as handoff spans
        are also captured as Tool Call steps with their proper inputs and outputs.
        """
        try:
            trace_id = metadata.get("trace_id")
            if not trace_id:
                return

            # Check if this appears to be a cumulative conversation history vs.
            # incremental function calls
            # Cumulative histories contain multiple different function calls from the
            # entire conversation
            function_call_names = set()
            for message in messages:
                if isinstance(message, dict) and message.get("type") == "function_call":
                    function_call_names.add(message.get("name", ""))

            # If we have multiple different function types, this is likely cumulative
            #  conversation history
            # We should skip extracting function calls to avoid duplicates
            if len(function_call_names) > 1:
                return

            # Find function calls and their outputs in the messages
            function_calls = {}

            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue

                # Look for function calls
                if message.get("type") == "function_call":
                    call_id = message.get("call_id")
                    function_name = message.get("name", "unknown_function")
                    if call_id:
                        function_calls[call_id] = {
                            "name": function_name,
                            "arguments": message.get("arguments", "{}"),
                            "call_id": call_id,
                        }

                # Look for function call outputs
                elif message.get("type") == "function_call_output":
                    call_id = message.get("call_id")
                    output = message.get("output")
                    if call_id and call_id in function_calls:
                        function_calls[call_id]["output"] = output

            # Create Tool Call steps for function calls that don't have corresponding
            # function spans
            for call_id, func_data in function_calls.items():
                function_name = func_data["name"]

                # Skip if this function already has a dedicated function span
                # (this is for handoff functions that only get handoff spans)
                if self._should_create_tool_call_step(function_name, trace_id):
                    self._create_tool_call_step_from_message(func_data, metadata)

        except Exception as e:
            logger.error(f"Failed to extract function calls from messages: {e}")

    def _should_create_tool_call_step(self, function_name: str, trace_id: str) -> bool:
        """Check if we should create a Tool Call step for this function.

        We create Tool Call steps for regular tools that don't already have dedicated
        spans.
        We do NOT create Tool Call steps for handoff functions since they already get
        Handoff spans.
        """
        # Common handoff function patterns
        handoff_patterns = ["transfer_to_", "handoff_to_", "switch_to_"]

        # Check if this looks like a handoff function
        is_handoff_function = any(
            function_name.startswith(pattern) for pattern in handoff_patterns
        )

        # Do NOT create Tool Call steps for handoff functions since they already get
        # Handoff spans
        if is_handoff_function:
            return False

        # For non-handoff functions, we might want to create Tool Call steps
        # if they don't have their own function spans (but this case is rare)
        # For now, we'll be conservative and not create Tool Call steps from message
        # extraction
        # since regular tools already get proper function spans
        return False

    def _create_tool_call_step_from_message(
        self, func_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Create a Tool Call step from function call message data."""
        try:
            function_name = func_data["name"]
            arguments = func_data.get("arguments", "{}")
            output = func_data.get("output", "Function completed")

            # Parse JSON arguments
            inputs = {}
            if arguments:
                try:
                    inputs = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                except (json.JSONDecodeError, TypeError):
                    inputs = {"arguments": arguments}

            # Create the Tool Call step
            step = steps.UserCallStep(
                name=f"Tool Call: {function_name}",
                inputs=inputs,
                output=output,
                metadata=metadata,
            )
            step.start_time = time.time()
            step.end_time = time.time()
            step.latency = 0  # Minimal latency for extracted function calls

            # Add to the trace steps collection
            trace_id = metadata.get("trace_id")
            if trace_id:
                if trace_id not in self._trace_root_steps:
                    self._trace_root_steps[trace_id] = []
                self._trace_root_steps[trace_id].append(step)

        except Exception as e:
            logger.error(f"Failed to create Tool Call step from message: {e}")

    def _create_generic_step(
        self, parsed_data: ParsedSpanData, start_time: float, metadata: Dict[str, Any]
    ) -> steps.Step:
        """Create a generic step for unknown span types."""
        name = parsed_data.name or f"Unknown {parsed_data.span_type}"

        # Use parsed input/output data
        inputs = parsed_data.input_data or {}
        output = self._extract_output_from_parsed_data(parsed_data, "Completed")

        # Create step without immediately sending to Openlayer
        step = steps.UserCallStep(
            name=f"{parsed_data.span_type.title()}: {name}",
            inputs=inputs,
            output=output,
            metadata=metadata,
        )
        step.start_time = start_time
        return step

    def _extract_usage_from_response(self, response: Any, field: str = None) -> int:
        """Extract usage information from response object."""
        if not response:
            return 0

        usage = getattr(response, "usage", None)
        if not usage:
            return 0

        if field == "input_tokens":
            return getattr(usage, "input_tokens", 0)
        elif field == "output_tokens":
            return getattr(usage, "output_tokens", 0)
        elif field == "total_tokens":
            return getattr(usage, "total_tokens", 0)
        else:
            # Return usage dict for metadata
            return {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

    def _update_step_with_span_data(
        self, step: steps.Step, span: tracing.Span, span_data: Any
    ) -> None:
        """Update step with final span data."""
        try:
            # Parse the span data to get the latest information including usage/tokens
            parsed_data = parse_span_data(span_data)

            # Extract function calls from response spans when conversation data becomes
            #  available
            if (
                parsed_data.span_type == "response"
                and parsed_data.input_data
                and "input" in parsed_data.input_data
            ):
                input_data = parsed_data.input_data["input"]
                if isinstance(input_data, list) and input_data:
                    # Create metadata dictionary for function call extraction using
                    #  helper
                    span_attrs = _extract_span_attributes(span)
                    function_metadata = {
                        **span_attrs,
                        "span_type": parsed_data.span_type,
                    }
                    self._extract_function_calls_from_messages(
                        input_data, function_metadata
                    )

            # Update inputs with the latest parsed input data if available
            if parsed_data.input_data and isinstance(step, steps.ChatCompletionStep):
                # Check if the new input data is richer than what we currently have
                current_inputs = getattr(step, "inputs", {})
                new_input_data = parsed_data.input_data

                # Always update if we have no inputs or generic placeholder
                should_update = (
                    not current_inputs or current_inputs.get("response_id") == "unknown"
                )

                # Also update if the new data has significantly more information
                if not should_update and new_input_data:
                    # Count rich fields in current vs new input data
                    rich_fields = ["instructions", "tools", "messages", "prompt"]
                    current_rich_count = sum(
                        1 for field in rich_fields if field in current_inputs
                    )
                    new_rich_count = sum(
                        1 for field in rich_fields if field in new_input_data
                    )

                    # Update if new data has more rich fields
                    if new_rich_count > current_rich_count:
                        should_update = True

                    # Also update if new data has agent instructions and current doesn't
                    elif (
                        "instructions" in new_input_data
                        and "instructions" not in current_inputs
                    ):
                        should_update = True

                    # Also update if new data has tools and current doesn't
                    elif "tools" in new_input_data and "tools" not in current_inputs:
                        should_update = True

                if should_update:
                    # Update with better input data
                    step.inputs.update(new_input_data)

            # Update function steps with input arguments when they become available
            elif (
                parsed_data.input_data
                and hasattr(step, "inputs")
                and parsed_data.span_type == "function"
            ):
                current_inputs = getattr(step, "inputs", {})
                if not current_inputs or current_inputs == {}:
                    # Function inputs are now available, update the step
                    step.inputs = parsed_data.input_data

                    # Parse JSON string arguments into proper objects if needed
                    if (
                        isinstance(step.inputs, dict)
                        and "input" in step.inputs
                        and isinstance(step.inputs["input"], str)
                    ):
                        try:
                            # Try to parse the JSON string
                            parsed_args = json.loads(step.inputs["input"])
                            step.inputs = parsed_args
                        except (json.JSONDecodeError, TypeError):
                            # Keep original string format if parsing fails
                            pass

            # Update output if it's still generic
            if parsed_data.output_data:
                updated_output = self._extract_output_from_parsed_data(parsed_data, "")

                if (
                    updated_output and updated_output.strip()
                ):  # Check if we have meaningful content
                    # For agent spans, don't override meaningful output with generic
                    #  output_data
                    if (
                        parsed_data.span_type == "agent"
                        and step.output
                        and "initialized" in step.output
                        and updated_output == "{'output_type': 'str'}"
                    ):
                        pass  # Skip agent output override - keeping meaningful output
                    # For response spans, always update if we have better content
                    elif parsed_data.span_type == "response" and (
                        step.output == "Response processed"
                        or len(updated_output) > len(step.output)
                    ):
                        step.output = updated_output
                    # For other span types, update if it's different and not generic
                    elif (
                        updated_output != step.output
                        and updated_output != "Response processed"
                    ):
                        step.output = updated_output
                elif (
                    parsed_data.span_type == "response"
                    and step.output == "Response processed"
                ):
                    # For response spans, try harder to extract actual LLM output
                    actual_output = self._extract_actual_llm_output(span_data)
                    if actual_output and actual_output.strip():
                        step.output = actual_output
            elif (
                parsed_data.span_type == "response"
                and step.output == "Response processed"
            ):
                # Even if no parsed output_data, try to extract from raw span_data
                actual_output = self._extract_actual_llm_output(span_data)
                if actual_output and actual_output.strip():
                    step.output = actual_output

            # Special handling for handoff steps - update with corrected target agent
            if parsed_data.span_type == "handoff" and hasattr(step, "inputs"):
                current_inputs = getattr(step, "inputs", {})

                # Check if we have better handoff data now
                if parsed_data.input_data:
                    from_agent = parsed_data.input_data.get("from_agent")
                    to_agent = parsed_data.input_data.get("to_agent")

                    # Update if we now have a valid target agent
                    if (
                        to_agent
                        and to_agent != "Unknown Target"
                        and to_agent != current_inputs.get("to_agent")
                    ):
                        # Update the step inputs
                        step.inputs["to_agent"] = to_agent
                        if from_agent:
                            step.inputs["from_agent"] = from_agent

                        # Update the step name and output to reflect the correct handoff
                        step.name = f"Handoff: {from_agent} → {to_agent}"
                        step.output = f"Handed off from {from_agent} to {to_agent}"

            # For ChatCompletionStep, update token information using helper function
            if isinstance(step, steps.ChatCompletionStep) and parsed_data.usage:
                token_counts = _extract_token_counts(parsed_data.usage)

                if (
                    token_counts["prompt_tokens"] > 0
                    or token_counts["completion_tokens"] > 0
                ):
                    step.prompt_tokens = token_counts["prompt_tokens"]
                    step.completion_tokens = token_counts["completion_tokens"]
                    step.tokens = token_counts["total_tokens"]

                    # Also update model if available
                    if parsed_data.model:
                        step.model = parsed_data.model

        except Exception as e:
            logger.error(f"Failed to update step with span data: {e}")

    def shutdown(self) -> None:
        """Shutdown the processor and flush any remaining data."""
        try:
            # Clean up any remaining traces and steps
            self._cleanup_dict_with_warning(self._active_traces, "active traces")
            self._cleanup_dict_with_warning(self._active_steps, "active steps")
            self._cleanup_dict_with_warning(
                self._trace_root_steps, "collected trace steps"
            )
            self._cleanup_dict_with_warning(
                self._current_user_inputs, "captured user inputs"
            )
            self._cleanup_dict_with_warning(
                self._trace_first_meaningful_input, "meaningful inputs"
            )
            self._cleanup_dict_with_warning(
                self._trace_last_meaningful_output, "meaningful outputs"
            )

            # Clean up span hierarchy tracking
            self._cleanup_dict_with_warning(self._span_to_step, "span-to-step mappings")
            self._cleanup_dict_with_warning(self._step_parents, "parent relationships")
            self._cleanup_dict_with_warning(self._step_children, "child relationships")

            # Clear the global reference
            global _active_openlayer_processor
            if _active_openlayer_processor is self:
                _active_openlayer_processor = None
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def force_flush(self) -> None:
        """Force flush any pending data."""
        # No additional flushing needed for Openlayer integration
        pass

    def capture_user_input(self, trace_id: str, user_input: str) -> None:
        """Capture user input at the application level.

        Since the OpenAI Agents SDK doesn't echo back user input in spans,
        we need to capture it at the application level.

        Args:
            trace_id: The trace ID to associate the input with
            user_input: The user's input message
        """
        if trace_id not in self._current_user_inputs:
            self._current_user_inputs[trace_id] = []
        self._current_user_inputs[trace_id].append(user_input)

    def _get_user_input_for_trace(self, trace_id: str) -> Optional[str]:
        """Get the most recent user input for a trace."""
        inputs = self._current_user_inputs.get(trace_id, [])
        return inputs[-1] if inputs else None

    def _extract_output_from_parsed_data(
        self, parsed_data: ParsedSpanData, fallback: str = "Completed"
    ) -> str:
        """Extract output from parsed span data with consistent logic."""
        if parsed_data.output_data:
            if (
                isinstance(parsed_data.output_data, dict)
                and "output" in parsed_data.output_data
            ):
                return parsed_data.output_data["output"]
            else:
                return str(parsed_data.output_data)
        return fallback

    def _extract_actual_llm_output(self, span_data: Any) -> Optional[str]:
        """Attempt to extract the actual LLM output from the span_data."""
        try:
            # First, try using the export() method if available
            if hasattr(span_data, "export") and callable(getattr(span_data, "export")):
                try:
                    exported = span_data.export()
                    if isinstance(exported, dict) and "output" in exported:
                        output_val = exported["output"]
                        if output_val is not None:
                            return str(output_val)
                except Exception:
                    pass

            # Try to access response.output if it's a response span
            if hasattr(span_data, "response") and span_data.response:
                response = span_data.response

                # First check for response.text (most common for actual LLM text)
                if hasattr(response, "text") and response.text:
                    return response.text

                # Then check response.output for messages/function calls using helper
                #  function
                if hasattr(response, "output") and response.output:
                    extracted_output = _extract_response_output(response.output)
                    if extracted_output and "output" in extracted_output:
                        return extracted_output["output"]

                # Try other response attributes that might contain the text
                for attr in ["content", "message"]:
                    if hasattr(response, attr):
                        val = getattr(response, attr)
                        if val:
                            return str(val)

            # Try direct span_data attributes
            for attr in ["output", "text", "content", "message", "response_text"]:
                if hasattr(span_data, attr):
                    val = getattr(span_data, attr)
                    if val is not None:
                        return str(val)

            # If span_data is a dict, try common output keys
            if isinstance(span_data, dict):
                for key in [
                    "output",
                    "text",
                    "content",
                    "message",
                    "response",
                    "result",
                ]:
                    if key in span_data and span_data[key] is not None:
                        return str(span_data[key])

            return None

        except Exception:
            return None

    def _cleanup_dict_with_warning(self, dict_obj: Dict, name: str) -> None:
        """Helper to clean up dictionaries with warning logging."""
        if dict_obj:
            dict_obj.clear()
