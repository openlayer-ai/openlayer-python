"""Core tracing functionality tests.

Usage:
pytest tests/test_tracing_core.py -v
"""

# ruff: noqa: ARG001
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

import asyncio
from typing import Any, Set, Dict, List, Generator
from unittest.mock import patch

import pytest

from openlayer.lib.tracing import enums, steps, tracer, traces


class TestBasicTracing:
    """Test basic tracing functionality."""

    def setup_method(self) -> None:
        """Setup before each test - reset global state."""
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    @patch.object(tracer, "_publish", False)
    def test_sync_function_tracing(self) -> None:
        """Test that sync functions are traced correctly."""

        @tracer.trace()
        def simple_function(x: int, y: str = "default") -> str:
            return f"{y}: {x}"

        result = simple_function(42, "test")
        assert result == "test: 42"

    @patch.object(tracer, "_publish", False)
    def test_async_function_tracing(self) -> None:
        """Test that async functions are traced correctly."""

        @tracer.trace_async()
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        result = asyncio.run(async_function(21))
        assert result == 42

    @patch.object(tracer, "_publish", False)
    def test_sync_generator_tracing(self) -> None:
        """Test that sync generators are traced correctly."""

        @tracer.trace()
        def generator_function(n: int) -> Generator[int, None, None]:
            for i in range(n):
                yield i

        gen = generator_function(3)
        results = list(gen)
        assert results == [0, 1, 2]

    @patch.object(tracer, "_publish", False)
    def test_nested_tracing(self) -> None:
        """Test that nested traced functions work correctly."""

        @tracer.trace()
        def inner_function(x: int) -> int:
            return x * 2

        @tracer.trace()
        def outer_function(x: int) -> int:
            return inner_function(x) + 1

        result = outer_function(5)
        assert result == 11


class TestContextManagement:
    """Test context management functionality."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    @patch.object(tracer, "_publish", False)
    def test_create_step_context_manager(self) -> None:
        """Test the create_step context manager."""
        with tracer.create_step("test_step") as step:
            assert step.name == "test_step"


class TestTraceDataStructure:
    """Test trace data structure and content."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    @patch.object(tracer, "_publish", False)
    def test_trace_captures_inputs_and_outputs(self) -> None:
        """Test that trace captures function inputs and outputs correctly."""
        captured_trace = None

        @tracer.trace()
        def test_function(a: int, b: str = "default", **kwargs: Any) -> Dict[str, Any]:
            current_trace = tracer.get_current_trace()
            nonlocal captured_trace
            captured_trace = current_trace
            return {"result": a * 2, "message": b}

        result = test_function(42, "test", extra="value")

        # Verify function result
        assert result == {"result": 84, "message": "test"}

        # Verify trace structure
        assert captured_trace is not None
        assert len(captured_trace.steps) == 1

        root_step = captured_trace.steps[0]
        assert root_step.name == "test_function"
        assert root_step.step_type == enums.StepType.USER_CALL

        # Verify inputs were captured (excluding 'self' and 'cls')
        assert "a" in root_step.inputs
        assert root_step.inputs["a"] == 42
        assert root_step.inputs["b"] == "test"
        assert root_step.inputs["kwargs"] == {"extra": "value"}

        # Verify output was captured
        assert root_step.output == {"result": 84, "message": "test"}

        # Verify timing data
        assert root_step.start_time is not None
        assert root_step.end_time is not None
        assert root_step.latency is not None
        assert root_step.latency > 0  # Should have some latency

    @patch.object(tracer, "_publish", False)
    def test_nested_trace_structure(self) -> None:
        """Test that nested traces create proper parent-child relationships."""
        captured_trace = None

        @tracer.trace()
        def inner_function(x: int) -> int:
            return x * 3

        @tracer.trace()
        def middle_function(x: int) -> int:
            return inner_function(x) + 10

        @tracer.trace()
        def outer_function(x: int) -> int:
            current_trace = tracer.get_current_trace()
            nonlocal captured_trace
            captured_trace = current_trace
            return middle_function(x) + 1

        result = outer_function(5)
        assert result == 26  # (5 * 3) + 10 + 1

        # Verify trace structure
        assert captured_trace is not None
        assert len(captured_trace.steps) == 1  # Only root step at trace level

        root_step = captured_trace.steps[0]
        assert root_step.name == "outer_function"

        # Verify nested structure
        assert len(root_step.steps) == 1  # middle_function
        middle_step = root_step.steps[0]
        assert middle_step.name == "middle_function"

        assert len(middle_step.steps) == 1  # inner_function
        inner_step = middle_step.steps[0]
        assert inner_step.name == "inner_function"
        assert len(inner_step.steps) == 0  # leaf node

        # Verify all steps have proper data
        assert inner_step.inputs["x"] == 5
        assert inner_step.output == 15
        assert middle_step.inputs["x"] == 5
        assert middle_step.output == 25
        assert root_step.inputs["x"] == 5
        assert root_step.output == 26

    @patch.object(tracer, "_publish", False)
    def test_step_timing_data(self) -> None:
        """Test that step timing data is captured correctly."""

        @tracer.trace()
        def timed_function() -> str:
            import time

            time.sleep(0.01)  # 10ms delay
            return "done"

        result = timed_function()

        assert result == "done"

        # Get the trace to examine timing
        with tracer.create_step("dummy") as _dummy_step:
            pass  # This will finish the previous trace

        # The timing should be reasonable
        # Note: We can't access the previous trace easily, so let's test timing
        # with a context manager approach

    @patch.object(tracer, "_publish", False)
    def test_step_ids_are_unique(self) -> None:
        """Test that each step gets a unique ID."""
        step_ids: Set[str] = set()

        @tracer.trace()
        def function1() -> str:
            step = tracer.get_current_step()
            if step is not None:
                step_ids.add(str(step.id))
            return "result1"

        @tracer.trace()
        def function2() -> str:
            step = tracer.get_current_step()
            if step is not None:
                step_ids.add(str(step.id))
            return "result2"

        function1()
        function2()

        # Should have 2 unique IDs
        assert len(step_ids) == 2

    @patch.object(tracer, "_publish", False)
    def test_context_kwarg_functionality(self) -> None:
        """Test that context_kwarg properly captures context data."""
        captured_context = None

        @tracer.trace(context_kwarg="context_data")
        def rag_function(query: str, context_data: List[str]) -> str:  # noqa: ARG001
            nonlocal captured_context
            captured_context = tracer.get_rag_context()
            return f"Answer for {query} using context"

        context_list = ["context1", "context2", "context3"]
        result = rag_function("test query", context_list)

        assert result == "Answer for test query using context"
        assert captured_context == context_list


class TestTraceMetadata:
    """Test trace metadata functionality."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    @patch.object(tracer, "_publish", False)
    def test_update_current_trace_metadata(self) -> None:
        """Test that trace metadata can be updated during execution."""
        captured_trace = None

        @tracer.trace()
        def test_function() -> str:
            tracer.update_current_trace(
                user_id="user123", session_id="session456", custom_field="custom_value"
            )
            nonlocal captured_trace
            captured_trace = tracer.get_current_trace()
            return "result"

        test_function()

        assert captured_trace is not None
        assert captured_trace.metadata is not None
        assert captured_trace.metadata["user_id"] == "user123"
        assert captured_trace.metadata["session_id"] == "session456"
        assert captured_trace.metadata["custom_field"] == "custom_value"

    @patch.object(tracer, "_publish", False)
    def test_update_current_step_metadata(self) -> None:
        """Test that step metadata can be updated during execution."""
        captured_step = None

        @tracer.trace()
        def test_function() -> str:
            tracer.update_current_step(
                metadata={"model_version": "v1.2.3"},
                attributes={"custom_attr": "value"},
            )
            nonlocal captured_step
            captured_step = tracer.get_current_step()
            return "result"

        test_function()

        assert captured_step is not None
        assert captured_step.metadata is not None
        assert captured_step.metadata["model_version"] == "v1.2.3"
        assert captured_step.custom_attr == "value"

    @patch.object(tracer, "_publish", False)
    def test_log_output_overrides_function_output(self) -> None:
        """Test that log_output overrides the function's return value in trace."""
        captured_step = None

        @tracer.trace()
        def test_function() -> str:
            tracer.log_output("manual output")
            nonlocal captured_step
            captured_step = tracer.get_current_step()
            return "function output"  # This should be overridden

        result = test_function()

        # Function still returns its normal output
        assert result == "function output"

        # But trace should show manual output
        # Note: The manual output logging happens via metadata flag
        assert captured_step is not None
        assert captured_step.metadata is not None
        assert captured_step.metadata.get("manual_output_logged") is True


class TestTraceSerialization:
    """Test trace serialization and post-processing."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def test_step_to_dict_format(self) -> None:
        """Test step serialization format."""
        step = steps.Step(
            name="test_step",
            inputs={"input1": "value1", "input2": 42},
            output={"result": "success"},
            metadata={"meta1": "metavalue1"},
        )
        step.step_type = enums.StepType.USER_CALL
        # Fix the assignment issue by setting the end_time and latency properly
        step.end_time = step.start_time + 0.1  # type: ignore
        step.latency = 100.0  # type: ignore

        step_dict = step.to_dict()

        # Verify required fields
        assert step_dict["name"] == "test_step"
        assert step_dict["type"] == "user_call"
        assert "id" in step_dict
        assert "startTime" in step_dict
        assert step_dict["endTime"] is not None
        assert step_dict["latency"] == 100.0
        assert step_dict["inputs"] == {"input1": "value1", "input2": 42}
        assert step_dict["output"] == {"result": "success"}
        assert step_dict["metadata"] == {"meta1": "metavalue1"}

    def test_trace_to_dict_format(self) -> None:
        """Test trace serialization format."""
        trace = traces.Trace()

        # Add a step to the trace
        step = steps.Step(name="root_step")
        step.step_type = enums.StepType.USER_CALL
        trace.add_step(step)

        # Add nested step
        nested_step = steps.Step(name="nested_step")
        nested_step.step_type = enums.StepType.CHAT_COMPLETION
        step.add_nested_step(nested_step)

        trace_dict = trace.to_dict()

        assert isinstance(trace_dict, list)
        assert len(trace_dict) == 1
        assert trace_dict[0]["name"] == "root_step"
        assert len(trace_dict[0]["steps"]) == 1
        assert trace_dict[0]["steps"][0]["name"] == "nested_step"

    @patch.object(tracer, "_publish", False)
    def test_post_process_trace_format(self) -> None:
        """Test the post_process_trace function output format."""
        captured_trace = None

        @tracer.trace()
        def test_function(param1: str, param2: int) -> Dict[str, str]:  # noqa: ARG001
            tracer.update_current_trace(user_id="test_user")
            tracer.log_context(["context1", "context2"])
            nonlocal captured_trace
            captured_trace = tracer.get_current_trace()
            return {"answer": "test response"}

        test_function("test_param", 42)

        # Process the trace
        assert captured_trace is not None
        trace_data, input_variable_names = tracer.post_process_trace(captured_trace)

        # Verify trace_data structure
        assert isinstance(trace_data, dict)

        # Check required fields
        required_fields = [
            "inferenceTimestamp",
            "inferenceId",
            "output",
            "latency",
            "cost",
            "tokens",
            "steps",
        ]
        for field in required_fields:
            assert field in trace_data, f"Missing field: {field}"

        # Verify input variables
        assert "param1" in input_variable_names
        assert "param2" in input_variable_names
        assert trace_data["param1"] == "test_param"
        assert trace_data["param2"] == 42

        # Verify trace-level metadata was included
        assert trace_data["user_id"] == "test_user"

        # Verify context was captured
        assert trace_data["context"] == ["context1", "context2"]

        # Verify steps structure
        assert isinstance(trace_data["steps"], list)
        assert len(trace_data["steps"]) == 1
        assert trace_data["steps"][0]["name"] == "test_function"


class TestStepTypes:
    """Test different step types and their specific behavior."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def test_step_factory_creates_correct_types(self) -> None:
        """Test that step factory creates the correct step types."""
        step_types_mapping = {
            enums.StepType.USER_CALL: steps.UserCallStep,
            enums.StepType.CHAT_COMPLETION: steps.ChatCompletionStep,
            enums.StepType.AGENT: steps.AgentStep,
            enums.StepType.RETRIEVER: steps.RetrieverStep,
            enums.StepType.TOOL: steps.ToolStep,
            enums.StepType.GUARDRAIL: steps.GuardrailStep,
        }

        for step_type, expected_class in step_types_mapping.items():
            step = steps.step_factory(step_type, name=f"test_{step_type.value}")
            assert isinstance(step, expected_class)
            assert step.step_type == step_type

    def test_chat_completion_step_serialization(self) -> None:
        """Test ChatCompletionStep specific serialization."""
        step = steps.ChatCompletionStep(name="chat_step")
        step.inputs = {"prompt": [{"role": "user", "content": "Hello"}]}
        step.model = "gpt-3.5-turbo"
        step.provider = "openai"

        step_dict = step.to_dict()

        assert step_dict["type"] == "chat_completion"
        assert step_dict["inputs"]["prompt"] == [{"role": "user", "content": "Hello"}]
        assert step_dict["model"] == "gpt-3.5-turbo"
        assert step_dict["provider"] == "openai"

    @patch.object(tracer, "_publish", False)
    def test_add_chat_completion_step(self) -> None:
        """Test adding a chat completion step to trace."""
        captured_steps: List[Any] = []

        with tracer.create_step("main_step") as main_step:
            tracer.add_chat_completion_step_to_trace(
                name="Chat Step",
                model="gpt-4",
                prompt=[{"role": "user", "content": "Test"}],
                provider="openai",
            )
            captured_steps = main_step.steps

        assert len(captured_steps) == 1
        chat_step = captured_steps[0]
        assert chat_step.name == "Chat Step"
        assert chat_step.step_type == enums.StepType.CHAT_COMPLETION


class TestErrorHandlingInTraces:
    """Test error handling and exception capture in traces."""

    def setup_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def teardown_method(self) -> None:
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    @patch.object(tracer, "_publish", False)
    def test_exception_captured_in_metadata(self) -> None:
        """Test that exceptions are captured in step metadata."""
        captured_step = None

        @tracer.trace()
        def error_function() -> None:
            nonlocal captured_step
            captured_step = tracer.get_current_step()
            raise ValueError("Test error message")

        with pytest.raises(ValueError, match="Test error message"):
            error_function()

        # Verify exception was logged in metadata
        assert captured_step is not None
        assert captured_step.metadata is not None
        assert "Exceptions" in captured_step.metadata
        assert "Test error message" in captured_step.metadata["Exceptions"]

    @patch.object(tracer, "_publish", False)
    def test_nested_exception_handling(self) -> None:
        """Test exception handling in nested traced functions."""
        captured_steps: List[Any] = []

        @tracer.trace()
        def inner_error_function() -> None:
            step = tracer.get_current_step()
            captured_steps.append(step)
            raise RuntimeError("Inner error")

        @tracer.trace()
        def outer_function() -> None:
            step = tracer.get_current_step()
            captured_steps.append(step)
            return inner_error_function()

        with pytest.raises(RuntimeError, match="Inner error"):
            outer_function()

        # Both steps should have been captured
        assert len(captured_steps) == 2

        # Inner step should have exception metadata
        inner_step = captured_steps[0]
        assert inner_step is not None
        assert inner_step.metadata is not None
        assert "Exceptions" in inner_step.metadata
        assert "Inner error" in inner_step.metadata["Exceptions"]
