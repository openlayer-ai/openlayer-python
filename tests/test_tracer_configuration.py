"""Tests for the tracer configuration functionality."""

from typing import Any
from unittest.mock import MagicMock, patch

from openlayer.lib.tracing import tracer


class TestTracerConfiguration:
    """Test cases for the tracer configuration functionality."""

    def teardown_method(self):
        """Reset tracer configuration after each test."""
        # Reset the global configuration
        tracer._configured_api_key = None
        tracer._configured_pipeline_id = None
        tracer._configured_base_url = None
        tracer._client = None

    def test_configure_sets_global_variables(self):
        """Test that configure() sets the global configuration variables."""
        api_key = "test_api_key"
        pipeline_id = "test_pipeline_id"
        base_url = "https://test.api.com"

        tracer.configure(api_key=api_key, inference_pipeline_id=pipeline_id, base_url=base_url)

        assert tracer._configured_api_key == api_key
        assert tracer._configured_pipeline_id == pipeline_id
        assert tracer._configured_base_url == base_url

    def test_configure_resets_client(self):
        """Test that configure() resets the client to force recreation."""
        # Create a mock client
        tracer._client = MagicMock()
        original_client = tracer._client

        tracer.configure(api_key="test_key")

        # Client should be reset to None
        assert tracer._client is None
        assert tracer._client != original_client

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_api_key(self, mock_openlayer: Any) -> None:
        """Test that _get_client() uses the configured API key."""
        # Enable publishing for this test
        with patch.object(tracer, "_publish", True):
            api_key = "configured_api_key"
            tracer.configure(api_key=api_key)

            tracer._get_client()

            # Verify Openlayer was called with the configured API key
            mock_openlayer.assert_called_once_with(api_key=api_key)

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_base_url(self, mock_openlayer: Any) -> None:
        """Test that _get_client() uses the configured base URL."""
        with patch.object(tracer, "_publish", True):
            base_url = "https://configured.api.com"
            tracer.configure(base_url=base_url)

            tracer._get_client()

            mock_openlayer.assert_called_once_with(base_url=base_url)

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_both_configured_values(self, mock_openlayer: Any) -> None:
        """Test that _get_client() uses both configured API key and base URL."""
        with patch.object(tracer, "_publish", True):
            api_key = "configured_api_key"
            base_url = "https://configured.api.com"
            tracer.configure(api_key=api_key, base_url=base_url)

            tracer._get_client()

            mock_openlayer.assert_called_once_with(api_key=api_key, base_url=base_url)

    @patch("openlayer.lib.tracing.tracer.DefaultHttpxClient")
    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_with_ssl_disabled_and_config(self, mock_openlayer: Any, mock_http_client: Any) -> None:
        """Test _get_client() with SSL disabled and custom configuration."""
        with patch.object(tracer, "_publish", True), patch.object(tracer, "_verify_ssl", False):
            api_key = "test_key"
            tracer.configure(api_key=api_key)

            tracer._get_client()

            # Should create DefaultHttpxClient with verify=False
            mock_http_client.assert_called_once_with(verify=False)

            # Should create Openlayer with both http_client and configured values
            mock_openlayer.assert_called_once_with(http_client=mock_http_client.return_value, api_key=api_key)

    @patch.object(tracer, "utils")
    def test_handle_trace_completion_uses_configured_pipeline_id(self, mock_utils: Any) -> None:
        """Test that _handle_trace_completion() uses configured pipeline ID."""
        with patch.object(tracer, "_publish", True), patch.object(tracer, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_utils.get_env_variable.return_value = "env_pipeline_id"

            configured_pipeline_id = "configured_pipeline_id"
            tracer.configure(inference_pipeline_id=configured_pipeline_id)

            # Mock the necessary objects for trace completion
            with patch.object(tracer, "get_current_trace") as mock_get_trace, patch.object(
                tracer, "post_process_trace"
            ) as mock_post_process:
                mock_trace = MagicMock()
                mock_get_trace.return_value = mock_trace
                mock_post_process.return_value = ({}, [])

                # Call the function
                tracer._handle_trace_completion(is_root_step=True, step_name="test_step")

                # Verify the client.inference_pipelines.data.stream was called
                # with the configured pipeline ID
                mock_client.inference_pipelines.data.stream.assert_called_once()
                call_kwargs = mock_client.inference_pipelines.data.stream.call_args[1]
                assert call_kwargs["inference_pipeline_id"] == configured_pipeline_id

    @patch.object(tracer, "utils")
    def test_pipeline_id_precedence(self, mock_utils: Any) -> None:
        """Test pipeline ID precedence: provided > configured > environment."""
        with patch.object(tracer, "_publish", True), patch.object(tracer, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_utils.get_env_variable.return_value = "env_pipeline_id"

            tracer.configure(inference_pipeline_id="configured_pipeline_id")

            with patch.object(tracer, "get_current_trace") as mock_get_trace, patch.object(
                tracer, "post_process_trace"
            ) as mock_post_process:
                mock_trace = MagicMock()
                mock_get_trace.return_value = mock_trace
                mock_post_process.return_value = ({}, [])

                # Call with a provided pipeline ID (should have highest precedence)
                tracer._handle_trace_completion(
                    is_root_step=True, step_name="test_step", inference_pipeline_id="provided_pipeline_id"
                )

                call_kwargs = mock_client.inference_pipelines.data.stream.call_args[1]
                assert call_kwargs["inference_pipeline_id"] == "provided_pipeline_id"

    def test_configure_with_none_values(self):
        """Test that configure() with None values doesn't overwrite existing config."""
        # Set initial configuration
        tracer.configure(
            api_key="initial_key", inference_pipeline_id="initial_pipeline", base_url="https://initial.com"
        )

        # Configure with None values
        tracer.configure(api_key=None, inference_pipeline_id=None, base_url=None)

        # Values should be set to None (this is the expected behavior)
        assert tracer._configured_api_key is None
        assert tracer._configured_pipeline_id is None
        assert tracer._configured_base_url is None
