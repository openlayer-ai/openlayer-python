"""Test LiteLLM integration."""

import builtins
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest  # type: ignore


class TestLiteLLMIntegration:
    """Test LiteLLM integration functionality."""

    def test_import_without_litellm(self):
        """Test that the module can be imported even when LiteLLM is not available."""
        # This should not raise an ImportError
        from openlayer.lib.integrations import litellm_tracer
        
        # The HAVE_LITELLM flag should be set correctly
        assert hasattr(litellm_tracer, 'HAVE_LITELLM')

    def test_trace_litellm_raises_import_error_without_dependency(self):
        """Test that trace_litellm raises ImportError when LiteLLM is not available."""
        with patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', False):
            from openlayer.lib.integrations.litellm_tracer import trace_litellm
            
            with pytest.raises(ImportError) as exc_info:  # type: ignore
                trace_litellm()
            
            assert "LiteLLM library is not installed" in str(exc_info.value)  # type: ignore
            assert "pip install litellm" in str(exc_info.value)  # type: ignore

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    @patch('openlayer.lib.integrations.litellm_tracer.litellm')
    def test_trace_litellm_patches_completion(self, mock_litellm: Mock) -> None:
        """Test that trace_litellm successfully patches litellm.completion."""
        from openlayer.lib.integrations.litellm_tracer import trace_litellm
        
        # Mock the original completion function
        original_completion = Mock()
        mock_litellm.completion = original_completion
        
        # Call trace_litellm
        trace_litellm()
        
        # Verify that litellm.completion was replaced
        assert mock_litellm.completion != original_completion
        assert callable(mock_litellm.completion)

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_detect_provider_from_model_name(self):
        """Test provider detection from model names."""
        from openlayer.lib.integrations.litellm_tracer import detect_provider_from_model_name
        
        test_cases = [
            ("gpt-4", "OpenAI"),
            ("gpt-3.5-turbo", "OpenAI"),
            ("claude-3-opus-20240229", "Anthropic"),
            ("claude-3-haiku-20240307", "Anthropic"),
            ("gemini-pro", "Google"),
            ("llama-2-70b", "Meta"),
            ("mistral-large-latest", "Mistral"),
            ("command-r-plus", "Cohere"),
            ("unknown-model", "unknown"),
        ]
        
        for model_name, expected_provider in test_cases:
            assert detect_provider_from_model_name(model_name) == expected_provider

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_get_model_parameters(self):
        """Test model parameters extraction."""
        from openlayer.lib.integrations.litellm_tracer import get_model_parameters
        
        kwargs = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 150,
            "stream": True,
            "custom_param": "ignored",
        }
        
        params = get_model_parameters(kwargs)
        
        expected_params = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 150,
            "n": 1,  # default value
            "stream": True,
            "stop": None,  # default value
            "presence_penalty": 0.0,  # default value
            "frequency_penalty": 0.0,  # default value
            "logit_bias": None,  # default value
            "logprobs": False,  # default value
            "top_logprobs": None,  # default value
            "parallel_tool_calls": True,  # default value
            "seed": None,  # default value
            "response_format": None,  # default value
            "timeout": None,  # default value
            "api_base": None,  # default value
            "api_version": None,  # default value
        }
        
        assert params == expected_params

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_extract_usage_from_response(self):
        """Test usage extraction from response."""
        from openlayer.lib.integrations.litellm_tracer import extract_usage_from_response
        
        # Mock response with usage
        mock_response = Mock()
        mock_usage = Mock()
        mock_usage.total_tokens = 100
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 50
        mock_response.usage = mock_usage
        
        usage = extract_usage_from_response(mock_response)
        
        expected_usage = {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
        }
        
        assert usage == expected_usage

        # Test response without usage
        mock_response_no_usage = Mock(spec=[])  # No usage attribute
        usage_no_data = extract_usage_from_response(mock_response_no_usage)
        
        expected_no_usage = {
            "total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        
        assert usage_no_data == expected_no_usage

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_parse_non_streaming_output_data(self):
        """Test parsing output data from non-streaming responses."""
        from openlayer.lib.integrations.litellm_tracer import parse_non_streaming_output_data
        
        # Mock response with content
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Hello, world!"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        output = parse_non_streaming_output_data(mock_response)
        assert output == "Hello, world!"
        
        # Mock response with function call
        mock_response_func = Mock()
        mock_choice_func = Mock()
        mock_message_func = Mock()
        mock_message_func.content = None
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.arguments = '{"location": "New York"}'
        mock_message_func.function_call = mock_function_call
        mock_choice_func.message = mock_message_func
        mock_response_func.choices = [mock_choice_func]
        
        output_func = parse_non_streaming_output_data(mock_response_func)
        expected_func_output = {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
        assert output_func == expected_func_output

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_create_trace_args(self):
        """Test trace arguments creation."""
        from openlayer.lib.integrations.litellm_tracer import create_trace_args
        
        args: Dict[str, Any] = create_trace_args(
            end_time=1234567890.0,
            inputs={"prompt": "test"},
            output="response",
            latency=1500.0,
            tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            model="gpt-4",
            id="test-id"
        )
        
        expected_args = {
            "end_time": 1234567890.0,
            "inputs": {"prompt": "test"},
            "output": "response",
            "latency": 1500.0,
            "tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "model": "gpt-4",
            "model_parameters": None,
            "raw_output": None,
            "metadata": {},
            "id": "test-id",
        }
        
        assert args == expected_args

    def test_lib_init_trace_litellm_function_exists(self):
        """Test that trace_litellm function is available in lib.__init__."""
        from openlayer.lib import trace_litellm
        
        assert callable(trace_litellm)

    def test_lib_init_trace_litellm_import_error(self):
        """Test that lib.trace_litellm raises ImportError when litellm is not available."""
        from openlayer.lib import trace_litellm
        
        # Mock import to fail for litellm specifically
        original_import = builtins.__import__
        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'litellm':
                raise ImportError("No module named 'litellm'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:  # type: ignore
                trace_litellm()
            
            assert "litellm is required for LiteLLM tracing" in str(exc_info.value)  # type: ignore
            assert "pip install litellm" in str(exc_info.value)  # type: ignore

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    def test_extract_litellm_metadata(self):
        """Test extraction of LiteLLM-specific metadata."""
        from openlayer.lib.integrations.litellm_tracer import extract_litellm_metadata
        
        # Mock response with hidden params
        mock_response = Mock()
        mock_hidden_params = {
            'response_cost': 0.002,
            'api_base': 'https://api.openai.com/v1',
            'api_version': 'v1',
            'model_info': {'provider': 'openai', 'mode': 'chat'},
            'custom_llm_provider': 'openai'
        }
        mock_response._hidden_params = mock_hidden_params
        mock_response.system_fingerprint = 'fp_12345'
        
        metadata = extract_litellm_metadata(mock_response, 'gpt-4')
        
        expected_metadata = {
            'cost': 0.002,
            'api_base': 'https://api.openai.com/v1',
            'api_version': 'v1',
            'model_info': {'provider': 'openai', 'mode': 'chat'},
            'system_fingerprint': 'fp_12345'
        }
        
        assert metadata == expected_metadata

    @patch('openlayer.lib.integrations.litellm_tracer.HAVE_LITELLM', True)
    @patch('openlayer.lib.integrations.litellm_tracer.litellm')
    def test_detect_provider_with_litellm_method(self, mock_litellm: Mock) -> None:
        """Test provider detection using LiteLLM's get_llm_provider method."""
        from openlayer.lib.integrations.litellm_tracer import detect_provider_from_response
        
        # Mock LiteLLM's get_llm_provider method
        mock_litellm.get_llm_provider.return_value = ('gpt-4', 'openai', None, None)
        
        mock_response = Mock(spec=[])  # No special attributes
        
        provider = detect_provider_from_response(mock_response, 'gpt-4')
        
        assert provider == 'openai'
        mock_litellm.get_llm_provider.assert_called_once_with('gpt-4')
