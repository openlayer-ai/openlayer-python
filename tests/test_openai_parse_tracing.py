"""Tests for OpenAI parse method tracing functionality.

Usage:
pytest tests/test_openai_parse_tracing.py -v
"""
# pyright: reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportArgumentType=false, reportMissingParameterType=false

import time
from typing import Any, Dict
from unittest.mock import Mock, MagicMock, patch

import pytest

# Mock openai imports for testing
openai_mock = MagicMock()
openai_mock.OpenAI = Mock
openai_mock.AzureOpenAI = Mock
openai_mock.AsyncOpenAI = Mock
openai_mock.AsyncAzureOpenAI = Mock


class MockParsedResponse:
    """Mock class representing a Pydantic model response."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        
    def model_dump(self) -> Dict[str, Any]:
        return self._data
    
    @property
    def name(self) -> str:
        return self._data.get("name", "")
    
    @property
    def age(self) -> int:
        return self._data.get("age", 0)


class MockParseResponse:
    """Mock response from OpenAI parse method."""
    def __init__(self, parsed_data: Dict[str, Any], **kwargs: Any):
        self.parsed = MockParsedResponse(parsed_data)
        self.usage = Mock()
        self.usage.total_tokens = kwargs.get("total_tokens", 100)
        self.usage.prompt_tokens = kwargs.get("prompt_tokens", 50)
        self.usage.completion_tokens = kwargs.get("completion_tokens", 50)
        self.model = kwargs.get("model", "gpt-4o-mini")
        
    def model_dump(self) -> Dict[str, Any]:
        return {
            "parsed": self.parsed._data,
            "usage": {
                "total_tokens": self.usage.total_tokens,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
            "model": self.model,
        }


class TestOpenAIParseTracing:
    """Test OpenAI parse method tracing functionality."""

    def setup_method(self) -> None:
        """Setup before each test."""
        # Mock the openai module
        self.openai_patcher = patch('sys.modules', {'openai': openai_mock})
        self.openai_patcher.start()
        
        # Import after mocking
        with patch('openlayer.lib.integrations.openai_tracer.HAVE_OPENAI', True):
            from openlayer.lib.integrations.openai_tracer import (  # type: ignore
                trace_openai,
                create_trace_args,
                handle_non_streaming_parse,
                parse_structured_output_data,
            )
            self.trace_openai = trace_openai  # type: ignore
            self.handle_non_streaming_parse = handle_non_streaming_parse  # type: ignore
            self.parse_structured_output_data = parse_structured_output_data  # type: ignore
            self.create_trace_args = create_trace_args  # type: ignore

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        self.openai_patcher.stop()

    @patch('openlayer.lib.integrations.openai_tracer.add_to_trace')
    def test_trace_openai_patches_parse_method(self, mock_add_to_trace) -> None:  # noqa: ARG002
        """Test that trace_openai patches the parse method when it exists."""
        # Create mock client with parse method
        mock_client = Mock()
        mock_client.chat.completions.create = Mock()
        mock_client.chat.completions.parse = Mock()
        
        # Configure isinstance to return True for OpenAI client
        with patch('isinstance', return_value=True):
            # Trace the client
            traced_client = self.trace_openai(mock_client)
            
            # Verify both create and parse methods are patched
            assert traced_client.chat.completions.create != mock_client.chat.completions.create
            assert traced_client.chat.completions.parse != mock_client.chat.completions.parse

    @patch('openlayer.lib.integrations.openai_tracer.add_to_trace')
    def test_trace_openai_handles_missing_parse_method(self, mock_add_to_trace) -> None:  # noqa: ARG002
        """Test that trace_openai handles clients without parse method gracefully."""
        # Create mock client without parse method
        mock_client = Mock()
        mock_client.chat.completions.create = Mock()
        # Don't add parse method
        
        with patch('isinstance', return_value=True):
            with patch('hasattr', return_value=False):
                # Should not raise an error
                traced_client = self.trace_openai(mock_client)
                
                # Only create method should be patched
                assert traced_client.chat.completions.create != mock_client.chat.completions.create

    @patch('openlayer.lib.integrations.openai_tracer.add_to_trace')
    def test_handle_non_streaming_parse_success(self, mock_add_to_trace) -> None:
        """Test successful non-streaming parse method handling."""
        # Mock parse function
        mock_parse_func = Mock()
        parsed_data = {"name": "John Doe", "age": 30, "occupation": "Engineer"}
        mock_response = MockParseResponse(parsed_data)
        mock_parse_func.return_value = mock_response
        
        # Test parameters
        kwargs = {
            "messages": [{"role": "user", "content": "Test message"}],
            "model": "gpt-4o-mini",
            "response_format": "Person",
        }
        
        # Call the handler
        result = self.handle_non_streaming_parse(
            mock_parse_func,
            inference_id="test-123",
            is_azure_openai=False,
            **kwargs
        )
        
        # Verify the result
        assert result == mock_response
        
        # Verify trace was added
        mock_add_to_trace.assert_called_once()
        call_args = mock_add_to_trace.call_args[1]
        
        assert call_args["is_azure_openai"] == False
        assert "inputs" in call_args
        assert "output" in call_args
        assert "latency" in call_args
        assert "tokens" in call_args
        assert call_args["metadata"]["method"] == "parse"

    def test_parse_structured_output_data_with_pydantic_model(self) -> None:
        """Test parsing structured output data from Pydantic model response."""
        parsed_data = {"name": "Alice Smith", "age": 25, "occupation": "Designer"}
        mock_response = MockParseResponse(parsed_data)
        
        result = self.parse_structured_output_data(mock_response)
        
        assert result == parsed_data

    def test_parse_structured_output_data_with_dict_object(self) -> None:
        """Test parsing structured output data from dict-like object."""
        # Mock response with dict-like parsed object
        mock_response = Mock()
        mock_response.parsed = Mock()
        mock_response.parsed.__dict__ = {"key": "value", "number": 42}
        
        # Mock hasattr to simulate no model_dump method
        def mock_hasattr(obj, attr):  # noqa: ARG001
            if attr == 'model_dump':
                return False
            elif attr == '__dict__':
                return True
            return False
        
        with patch('builtins.hasattr', side_effect=mock_hasattr):
            result = self.parse_structured_output_data(mock_response)
            
        assert result == {"key": "value", "number": 42}

    @patch('openlayer.lib.integrations.openai_tracer.parse_non_streaming_output_data')
    def test_parse_structured_output_data_fallback_to_regular_parsing(self, mock_parse_regular) -> None:
        """Test fallback to regular parsing when structured data is not available."""
        # Mock response without parsed attribute
        mock_response = Mock()
        del mock_response.parsed  # Remove parsed attribute
        
        mock_parse_regular.return_value = "fallback content"
        
        result = self.parse_structured_output_data(mock_response)
        
        assert result == "fallback content"
        mock_parse_regular.assert_called_once_with(mock_response)

    @patch('openlayer.lib.integrations.openai_tracer.parse_non_streaming_output_data')
    @patch('openlayer.lib.integrations.openai_tracer.logger')
    def test_parse_structured_output_data_error_handling(self, mock_logger, mock_parse_regular) -> None:
        """Test error handling in parse_structured_output_data."""
        # Mock response that will cause an exception
        mock_response = Mock()
        mock_response.parsed = Mock()
        mock_response.parsed.model_dump.side_effect = Exception("Test error")
        
        mock_parse_regular.return_value = "error fallback"
        
        result = self.parse_structured_output_data(mock_response)
        
        # Should fall back to regular parsing
        assert result == "error fallback"
        # Should log the error
        mock_logger.error.assert_called_once()
        mock_parse_regular.assert_called_once_with(mock_response)

    def test_create_trace_args_with_parse_metadata(self) -> None:
        """Test that create_trace_args properly handles parse-specific metadata."""
        metadata = {
            "method": "parse",
            "response_format": "Person",
            "custom_field": "value"
        }
        
        trace_args = self.create_trace_args(
            end_time=time.time(),
            inputs={"prompt": [{"role": "user", "content": "test"}]},
            output={"name": "test", "age": 25},
            latency=100.0,
            tokens=50,
            prompt_tokens=25,
            completion_tokens=25,
            model="gpt-4o-mini",
            metadata=metadata,
        )
        
        assert trace_args["metadata"]["method"] == "parse"
        assert trace_args["metadata"]["response_format"] == "Person"
        assert trace_args["metadata"]["custom_field"] == "value"

    @patch('openlayer.lib.integrations.openai_tracer.add_to_trace')
    @patch('openlayer.lib.integrations.openai_tracer.logger')
    def test_handle_non_streaming_parse_error_handling(self, mock_logger, mock_add_to_trace) -> None:  # noqa: ARG002
        """Test error handling in non-streaming parse method."""
        # Mock parse function that raises an error
        mock_parse_func = Mock()
        mock_parse_func.side_effect = Exception("API Error")
        
        kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "model": "gpt-4o-mini",
        }
        
        # Should raise the original exception
        with pytest.raises(Exception, match="API Error"):
            self.handle_non_streaming_parse(
                mock_parse_func,
                **kwargs
            )

    @patch('openlayer.lib.integrations.openai_tracer.add_to_trace')
    @patch('openlayer.lib.integrations.openai_tracer.logger')
    def test_handle_non_streaming_parse_tracing_error(self, mock_logger, mock_add_to_trace) -> None:
        """Test handling of tracing errors that shouldn't affect the main response."""
        # Mock parse function that succeeds
        mock_parse_func = Mock()
        parsed_data = {"name": "Test", "age": 30}
        mock_response = MockParseResponse(parsed_data)
        mock_parse_func.return_value = mock_response
        
        # Mock add_to_trace to raise an error
        mock_add_to_trace.side_effect = Exception("Tracing error")
        
        kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "model": "gpt-4o-mini",
        }
        
        # Should return the response despite tracing error
        result = self.handle_non_streaming_parse(
            mock_parse_func,
            **kwargs
        )
        
        assert result == mock_response
        # Should log the tracing error
        mock_logger.error.assert_called_once()


class TestAsyncOpenAIParseTracing:
    """Test async OpenAI parse method tracing functionality."""

    def setup_method(self) -> None:
        """Setup before each test."""
        # Mock the openai module
        self.openai_patcher = patch('sys.modules', {'openai': openai_mock})
        self.openai_patcher.start()

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        self.openai_patcher.stop()

    @patch('openlayer.lib.integrations.async_openai_tracer.add_to_trace')
    def test_trace_async_openai_patches_parse_method(self, mock_add_to_trace) -> None:  # noqa: ARG002
        """Test that trace_async_openai patches the parse method when it exists."""
        with patch('openlayer.lib.integrations.async_openai_tracer.HAVE_OPENAI', True):
            from openlayer.lib.integrations.async_openai_tracer import trace_async_openai
            
            # Create mock async client with parse method
            mock_client = Mock()
            mock_client.chat.completions.create = Mock()
            mock_client.chat.completions.parse = Mock()
            
            # Configure isinstance to return True for AsyncOpenAI client
            with patch('isinstance', return_value=True):
                # Trace the client
                traced_client = trace_async_openai(mock_client)
                
                # Verify both create and parse methods are patched
                assert traced_client.chat.completions.create != mock_client.chat.completions.create
                assert traced_client.chat.completions.parse != mock_client.chat.completions.parse

    @patch('openlayer.lib.integrations.async_openai_tracer.add_to_trace')
    async def test_handle_async_non_streaming_parse_success(self, mock_add_to_trace) -> None:
        """Test successful async non-streaming parse method handling."""
        with patch('openlayer.lib.integrations.async_openai_tracer.HAVE_OPENAI', True):
            from openlayer.lib.integrations.async_openai_tracer import handle_async_non_streaming_parse
            
            # Mock async parse function
            mock_parse_func = Mock()
            parsed_data = {"name": "Jane Doe", "age": 28, "occupation": "Developer"}
            mock_response = MockParseResponse(parsed_data)
            
            async def async_return():
                return mock_response
            
            mock_parse_func.return_value = async_return()
            
            # Test parameters
            kwargs = {
                "messages": [{"role": "user", "content": "Test message"}],
                "model": "gpt-4o-mini",
                "response_format": "Person",
            }
            
            # Call the async handler
            result = await handle_async_non_streaming_parse(
                mock_parse_func,
                inference_id="async-test-123",
                is_azure_openai=False,
                **kwargs
            )
            
            # Verify the result
            assert result == mock_response
            
            # Verify trace was added
            mock_add_to_trace.assert_called_once()
            call_args = mock_add_to_trace.call_args[1]
            
            assert call_args["is_azure_openai"] == False
            assert call_args["metadata"]["method"] == "parse"