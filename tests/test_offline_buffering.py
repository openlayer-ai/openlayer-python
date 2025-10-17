"""Tests for offline buffering functionality in the tracer module."""

import json
import time
import tempfile
from typing import Any, Dict, List, Tuple
from pathlib import Path
from unittest.mock import Mock, patch

from openlayer.lib.tracing.tracer import (
    OfflineBuffer,
    configure,
    get_buffer_status,
    _get_offline_buffer,
    clear_offline_buffer,
    replay_buffered_traces,
    _handle_streaming_failure,
)


class TestOfflineBuffer:
    """Test cases for the OfflineBuffer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.buffer_path = Path(self.temp_dir) / "test_buffer"
        self.buffer = OfflineBuffer(buffer_path=str(self.buffer_path), max_buffer_size=3)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_buffer_initialization(self):
        """Test offline buffer initialization."""
        assert self.buffer.buffer_path == self.buffer_path
        assert self.buffer.max_buffer_size == 3
        assert self.buffer_path.exists()
        assert self.buffer_path.is_dir()

    def test_store_trace_success(self):
        """Test successful trace storage."""
        trace_data = {"inferenceId": "test-123", "output": "test output"}
        config = {"output_column_name": "output"}
        pipeline_id = "test-pipeline"

        result = self.buffer.store_trace(trace_data, config, pipeline_id)

        assert result is True

        # Check that file was created
        trace_files = list(self.buffer_path.glob("trace_*.json"))
        assert len(trace_files) == 1

        # Check file contents
        with trace_files[0].open("r") as f:
            stored_data = json.load(f)

        assert stored_data["trace_data"] == trace_data
        assert stored_data["config"] == config
        assert stored_data["inference_pipeline_id"] == pipeline_id
        assert "timestamp" in stored_data
        assert "metadata" in stored_data

    def test_store_trace_max_buffer_size(self):
        """Test that buffer respects max size limit."""
        trace_data = {"inferenceId": "test", "output": "output"}
        config = {"output_column_name": "output"}
        pipeline_id = "test-pipeline"

        # Store traces up to max size
        for i in range(5):  # More than max_buffer_size (3)
            time.sleep(0.01)  # Ensure different timestamps
            result = self.buffer.store_trace({**trace_data, "inferenceId": f"test-{i}"}, config, pipeline_id)
            assert result is True

        # Should only have max_buffer_size files
        trace_files = list(self.buffer_path.glob("trace_*.json"))
        assert len(trace_files) == 3

    def test_get_buffered_traces(self):
        """Test retrieving buffered traces."""
        # Store some traces
        for i in range(2):
            trace_data = {"inferenceId": f"test-{i}", "output": f"output-{i}"}
            config = {"output_column_name": "output"}
            self.buffer.store_trace(trace_data, config, "test-pipeline")
            time.sleep(0.01)

        traces = self.buffer.get_buffered_traces()

        assert len(traces) == 2
        assert all("_file_path" in trace for trace in traces)
        assert all("trace_data" in trace for trace in traces)
        assert all("config" in trace for trace in traces)

    def test_remove_trace(self):
        """Test removing a trace from buffer."""
        trace_data = {"inferenceId": "test-123", "output": "test output"}
        config = {"output_column_name": "output"}
        self.buffer.store_trace(trace_data, config, "test-pipeline")

        # Get the file path
        traces = self.buffer.get_buffered_traces()
        file_path = traces[0]["_file_path"]

        # Remove the trace
        result = self.buffer.remove_trace(file_path)
        assert result is True

        # Check it's gone
        traces = self.buffer.get_buffered_traces()
        assert len(traces) == 0

    def test_get_buffer_status(self):
        """Test getting buffer status."""
        # Store a trace
        trace_data = {"inferenceId": "test-123", "output": "test output"}
        config = {"output_column_name": "output"}
        self.buffer.store_trace(trace_data, config, "test-pipeline")

        status = self.buffer.get_buffer_status()

        assert status["total_traces"] == 1
        assert status["max_buffer_size"] == 3
        assert status["total_size_bytes"] > 0
        assert "oldest_trace" in status
        assert "newest_trace" in status

    def test_clear_buffer(self):
        """Test clearing all buffered traces."""
        # Store some traces
        for i in range(2):
            trace_data = {"inferenceId": f"test-{i}", "output": f"output-{i}"}
            config = {"output_column_name": "output"}
            self.buffer.store_trace(trace_data, config, "test-pipeline")

        # Clear buffer
        count = self.buffer.clear_buffer()

        assert count == 2
        traces = self.buffer.get_buffered_traces()
        assert len(traces) == 0


class TestTracerConfiguration:
    """Test cases for tracer configuration with offline buffering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Reset configuration
        configure()

    def test_configure_offline_buffering(self):
        """Test configuring offline buffering."""

        def failure_callback(trace_data: Dict[str, Any], config: Dict[str, Any], error: Exception) -> None:
            pass

        configure(
            on_flush_failure=failure_callback,
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
            max_buffer_size=100,
        )

        # Test that configuration was set
        from openlayer.lib.tracing.tracer import (
            _configured_max_buffer_size,
            _configured_on_flush_failure,
            _configured_offline_buffer_path,
            _configured_offline_buffer_enabled,
        )

        assert _configured_on_flush_failure == failure_callback
        assert _configured_offline_buffer_enabled is True
        assert _configured_offline_buffer_path == self.temp_dir
        assert _configured_max_buffer_size == 100

    def test_get_offline_buffer_disabled(self):
        """Test that offline buffer returns None when disabled."""
        configure(offline_buffer_enabled=False)
        buffer = _get_offline_buffer()
        assert buffer is None

    def test_get_offline_buffer_enabled(self):
        """Test that offline buffer is created when enabled."""
        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )
        buffer = _get_offline_buffer()
        assert buffer is not None
        assert isinstance(buffer, OfflineBuffer)


class TestStreamingFailureHandler:
    """Test cases for streaming failure handling."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.failure_callback_calls: List[Tuple[Dict[str, Any], Dict[str, Any], Exception]] = []

        def mock_failure_callback(trace_data: Dict[str, Any], config: Dict[str, Any], error: Exception) -> None:
            self.failure_callback_calls.append((trace_data, config, error))

        self.mock_failure_callback = mock_failure_callback

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        configure()

    def test_handle_streaming_failure_with_callback(self):
        """Test failure handling with callback only."""
        configure(on_flush_failure=self.mock_failure_callback)

        trace_data = {"inferenceId": "test-123"}
        config = {"output_column_name": "output"}
        error = Exception("Network error")

        _handle_streaming_failure(trace_data, config, "test-pipeline", error)

        assert len(self.failure_callback_calls) == 1
        assert self.failure_callback_calls[0][0] == trace_data
        assert self.failure_callback_calls[0][1] == config
        assert self.failure_callback_calls[0][2] == error

    def test_handle_streaming_failure_with_buffer(self):
        """Test failure handling with offline buffer."""
        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        trace_data = {"inferenceId": "test-123"}
        config = {"output_column_name": "output"}
        error = Exception("Network error")

        _handle_streaming_failure(trace_data, config, "test-pipeline", error)

        # Check that trace was stored
        buffer = _get_offline_buffer()
        assert buffer is not None
        traces = buffer.get_buffered_traces()
        assert len(traces) == 1
        assert traces[0]["trace_data"] == trace_data

    def test_handle_streaming_failure_callback_exception(self):
        """Test that callback exceptions don't break failure handling."""

        def failing_callback(_trace_data: Dict[str, Any], _config: Dict[str, Any], _error: Exception) -> None:
            raise Exception("Callback error")

        configure(
            on_flush_failure=failing_callback,
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        trace_data = {"inferenceId": "test-123"}
        config = {"output_column_name": "output"}
        error = Exception("Network error")

        # Should not raise exception
        _handle_streaming_failure(trace_data, config, "test-pipeline", error)

        # Buffer should still work
        buffer = _get_offline_buffer()
        assert buffer is not None
        traces = buffer.get_buffered_traces()
        assert len(traces) == 1


class TestBufferUtilityFunctions:
    """Test cases for buffer utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        configure()

    def test_get_buffer_status_disabled(self):
        """Test buffer status when disabled."""
        configure(offline_buffer_enabled=False)
        status = get_buffer_status()

        assert status["enabled"] is False
        assert "error" in status

    def test_get_buffer_status_enabled(self):
        """Test buffer status when enabled."""
        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        status = get_buffer_status()

        assert status["enabled"] is True
        assert "total_traces" in status
        assert "max_buffer_size" in status

    def test_clear_offline_buffer_disabled(self):
        """Test clearing buffer when disabled."""
        configure(offline_buffer_enabled=False)
        result = clear_offline_buffer()

        assert result["traces_removed"] == 0
        assert "error" in result

    def test_clear_offline_buffer_enabled(self):
        """Test clearing buffer when enabled."""
        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        # Store a trace first
        buffer = _get_offline_buffer()
        assert buffer is not None
        buffer.store_trace({"inferenceId": "test"}, {"output_column_name": "output"}, "test-pipeline")

        result = clear_offline_buffer()

        assert result["traces_removed"] == 1

    @patch("openlayer.lib.tracing.tracer._get_client")
    def test_replay_buffered_traces_success(self, mock_get_client: Mock) -> None:
        """Test successful replay of buffered traces."""
        # Setup
        mock_client = Mock()
        mock_response = Mock()
        mock_client.inference_pipelines.data.stream.return_value = mock_response
        mock_get_client.return_value = mock_client

        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        # Store some traces
        buffer = _get_offline_buffer()
        assert buffer is not None
        for i in range(2):
            buffer.store_trace(
                {"inferenceId": f"test-{i}", "output": f"output-{i}"}, {"output_column_name": "output"}, "test-pipeline"
            )

        success_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        def on_success(trace_data: Dict[str, Any], config: Dict[str, Any]) -> None:
            success_calls.append((trace_data, config))

        # Replay traces
        result = replay_buffered_traces(on_replay_success=on_success)

        assert result["total_traces"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert len(success_calls) == 2

        # Check that traces were removed from buffer
        traces = buffer.get_buffered_traces()
        assert len(traces) == 0

    @patch("openlayer.lib.tracing.tracer._get_client")
    def test_replay_buffered_traces_failure(self, mock_get_client: Mock) -> None:
        """Test replay failure handling."""
        # Setup
        mock_client = Mock()
        mock_client.inference_pipelines.data.stream.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        # Store a trace
        buffer = _get_offline_buffer()
        assert buffer is not None
        buffer.store_trace(
            {"inferenceId": "test-123", "output": "output"}, {"output_column_name": "output"}, "test-pipeline"
        )

        failure_calls: List[Tuple[Dict[str, Any], Dict[str, Any], Exception]] = []

        def on_failure(trace_data: Dict[str, Any], config: Dict[str, Any], error: Exception) -> None:
            failure_calls.append((trace_data, config, error))

        # Replay traces
        result = replay_buffered_traces(max_retries=2, on_replay_failure=on_failure)

        assert result["total_traces"] == 1
        assert result["success_count"] == 0
        assert result["failure_count"] == 1
        assert len(failure_calls) == 1

        # Check that trace is still in buffer
        traces = buffer.get_buffered_traces()
        assert len(traces) == 1

    def test_replay_buffered_traces_disabled(self):
        """Test replay when buffer is disabled."""
        configure(offline_buffer_enabled=False)
        result = replay_buffered_traces()

        assert result["total_traces"] == 0
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert "error" in result

    @patch("openlayer.lib.tracing.tracer._get_client")
    def test_replay_buffered_traces_no_client(self, mock_get_client: Mock) -> None:
        """Test replay when no client is available."""
        mock_get_client.return_value = None

        configure(
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        result = replay_buffered_traces()

        assert result["total_traces"] == 0
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert "error" in result


class TestEndToEndIntegration:
    """End-to-end integration tests for offline buffering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        configure()

    @patch("openlayer.lib.tracing.tracer._get_client")
    @patch("openlayer.lib.tracing.tracer._publish", True)
    def test_trace_with_offline_buffering(self, mock_get_client: Mock) -> None:
        """Test full tracing flow with offline buffering."""

        # Setup failing client
        mock_client = Mock()
        mock_client.inference_pipelines.data.stream.side_effect = Exception("Network error")
        mock_client.base_url = "https://api.openlayer.com"
        mock_get_client.return_value = mock_client

        failure_calls: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []

        def on_failure(trace_data: Dict[str, Any], config: Dict[str, Any], error: Exception) -> None:
            failure_calls.append((trace_data, config, str(error)))

        configure(
            api_key="test-key",
            inference_pipeline_id="test-pipeline",
            on_flush_failure=on_failure,
            offline_buffer_enabled=True,
            offline_buffer_path=self.temp_dir,
        )

        # Create test trace data
        trace_data: Dict[str, Any] = {
            "inferenceId": "test-123",
            "output": "test output",
            "steps": [],
        }
        config: Dict[str, Any] = {"output_column_name": "output"}

        # Test the failure handler directly
        _handle_streaming_failure(trace_data, config, "test-pipeline", Exception("Network error"))

        # Check that callback was called
        assert len(failure_calls) == 1

        # Check that trace was buffered
        buffer = _get_offline_buffer()
        assert buffer is not None
        traces = buffer.get_buffered_traces()
        assert len(traces) == 1
