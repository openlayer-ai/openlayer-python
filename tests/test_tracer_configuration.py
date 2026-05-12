"""Tests for the tracer configuration functionality."""

import os
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openlayer.lib.tracing import tracer


def _reset_tracer_state() -> None:
    """Clear tracer config and lazily-built state between tests."""
    tracer._tracer_config.clear()
    tracer._client = None
    tracer._client_init_logged = False
    tracer._configure_deprecation_logged = False


class TestTracerConfiguration:
    """Test cases for the tracer configuration functionality."""

    def teardown_method(self):
        """Reset tracer configuration after each test."""
        _reset_tracer_state()

    def test_init_sets_config_values(self):
        """init() stores explicitly-passed values in _tracer_config."""
        api_key = "test_api_key"
        pipeline_id = "test_pipeline_id"
        base_url = "https://test.api.com"
        timeout = 30.5
        max_retries = 5

        tracer.init(
            api_key=api_key,
            inference_pipeline_id=pipeline_id,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        assert tracer._tracer_config["api_key"] == api_key
        assert tracer._tracer_config["inference_pipeline_id"] == pipeline_id
        assert tracer._tracer_config["base_url"] == base_url
        assert tracer._tracer_config["timeout"] == timeout
        assert tracer._tracer_config["max_retries"] == max_retries

    def test_init_resets_client(self):
        """init() resets the lazily-built client so it gets recreated."""
        tracer._client = MagicMock()
        original_client = tracer._client

        tracer.init(api_key="test_key")

        assert tracer._client is None
        assert tracer._client != original_client

    def test_init_merges_partial_calls(self):
        """Repeated init() calls merge; omitted args preserve prior state."""
        tracer.init(api_key="A", inference_pipeline_id="P", base_url="https://x")
        tracer.init(api_key="B")  # only api_key passed

        assert tracer._tracer_config["api_key"] == "B"
        assert tracer._tracer_config["inference_pipeline_id"] == "P"
        assert tracer._tracer_config["base_url"] == "https://x"

    def test_init_explicit_none_clears(self):
        """Explicit None overrides env var; init(api_key=None) yields None from resolver."""
        tracer.init(api_key="A")
        with patch.dict(os.environ, {"OPENLAYER_API_KEY": "env-value"}):
            tracer.init(api_key=None)
            assert tracer.resolve_api_key() is None

    def test_init_with_no_args_is_noop(self):
        """init() with no args should not mutate _tracer_config (just resets client)."""
        tracer.init(api_key="A", inference_pipeline_id="P")
        snapshot = dict(tracer._tracer_config)
        tracer.init()
        assert tracer._tracer_config == snapshot

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_api_key(self, mock_openlayer: Any) -> None:
        """_get_client() passes the configured API key through to Openlayer()."""
        with patch.object(tracer, "_publish", True):
            tracer.init(api_key="configured_api_key")
            tracer._get_client()
            mock_openlayer.assert_called_once_with(api_key="configured_api_key")

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_base_url(self, mock_openlayer: Any) -> None:
        with patch.object(tracer, "_publish", True):
            tracer.init(base_url="https://configured.api.com")
            tracer._get_client()
            mock_openlayer.assert_called_once_with(base_url="https://configured.api.com")

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_both_configured_values(self, mock_openlayer: Any) -> None:
        with patch.object(tracer, "_publish", True):
            tracer.init(api_key="k", base_url="https://b")
            tracer._get_client()
            mock_openlayer.assert_called_once_with(api_key="k", base_url="https://b")

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_timeout(self, mock_openlayer: Any) -> None:
        with patch.object(tracer, "_publish", True):
            tracer.init(timeout=45.5)
            tracer._get_client()
            mock_openlayer.assert_called_once_with(timeout=45.5)

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_configured_max_retries(self, mock_openlayer: Any) -> None:
        with patch.object(tracer, "_publish", True):
            tracer.init(max_retries=10)
            tracer._get_client()
            mock_openlayer.assert_called_once_with(max_retries=10)

    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_uses_all_configured_values(self, mock_openlayer: Any) -> None:
        with patch.object(tracer, "_publish", True):
            tracer.init(
                api_key="k", base_url="https://b", timeout=25, max_retries=3
            )
            tracer._get_client()
            mock_openlayer.assert_called_once_with(
                api_key="k", base_url="https://b", timeout=25, max_retries=3
            )

    @patch("openlayer.lib.tracing.tracer.DefaultHttpxClient")
    @patch("openlayer.lib.tracing.tracer.Openlayer")
    def test_get_client_with_ssl_disabled_and_config(
        self, mock_openlayer: Any, mock_http_client: Any
    ) -> None:
        with patch.object(tracer, "_publish", True), patch.object(tracer, "_verify_ssl", False):
            tracer.init(api_key="test_key")
            tracer._get_client()
            mock_http_client.assert_called_once_with(verify=False)
            mock_openlayer.assert_called_once_with(
                http_client=mock_http_client.return_value, api_key="test_key"
            )

    def test_pipeline_id_precedence(self) -> None:
        """Resolver precedence: explicit arg > _tracer_config > env var."""
        with patch.dict(os.environ, {"OPENLAYER_INFERENCE_PIPELINE_ID": "env"}):
            # No config: env wins.
            assert tracer.resolve_pipeline_id() == "env"
            # _tracer_config set: _tracer_config wins.
            tracer.init(inference_pipeline_id="cfg")
            assert tracer.resolve_pipeline_id() == "cfg"
            # _tracer_config set to None: returns None (overrides env).
            tracer.init(inference_pipeline_id=None)
            assert tracer.resolve_pipeline_id() is None

    def test_init_preserves_none_values_explicit(self):
        """Explicit None values are stored in _tracer_config and clear env fallback."""
        tracer.init(
            api_key="initial_key",
            inference_pipeline_id="initial_pipeline",
            base_url="https://initial.com",
            timeout=60.0,
            max_retries=5,
        )

        # Explicit None on every knob — under merge semantics, these are stored.
        tracer.init(
            api_key=None,
            inference_pipeline_id=None,
            base_url=None,
            timeout=None,
            max_retries=None,
        )

        assert tracer._tracer_config["api_key"] is None
        assert tracer._tracer_config["inference_pipeline_id"] is None
        assert tracer._tracer_config["base_url"] is None
        assert tracer._tracer_config["timeout"] is None
        assert tracer._tracer_config["max_retries"] is None


class TestConfigureDeprecation:
    """Tests for the deprecated configure() alias."""

    def teardown_method(self):
        _reset_tracer_state()

    def test_configure_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracer.configure(api_key="x")
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_configure_delegates_to_init(self):
        """configure(**kwargs) should produce the same _tracer_config as init(**kwargs)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tracer.configure(api_key="cfg", inference_pipeline_id="pipe")
        assert tracer._tracer_config["api_key"] == "cfg"
        assert tracer._tracer_config["inference_pipeline_id"] == "pipe"


class TestResolvers:
    """Tests for the env-var resolver helpers."""

    def teardown_method(self):
        _reset_tracer_state()

    def test_resolver_falls_back_to_env(self):
        with patch.dict(
            os.environ,
            {
                "OPENLAYER_API_KEY": "env-key",
                "OPENLAYER_INFERENCE_PIPELINE_ID": "env-pipe",
                "OPENLAYER_BASE_URL": "https://env.example",
            },
        ):
            assert tracer.resolve_api_key() == "env-key"
            assert tracer.resolve_pipeline_id() == "env-pipe"
            assert tracer.resolve_base_url() == "https://env.example"

    def test_resolver_config_wins_over_env(self):
        with patch.dict(os.environ, {"OPENLAYER_API_KEY": "env-key"}):
            tracer.init(api_key="cfg-key")
            assert tracer.resolve_api_key() == "cfg-key"

    def test_get_config_redacts_api_key(self):
        tracer.init(api_key="super-secret", inference_pipeline_id="pipe")
        cfg = tracer.get_tracer_config()
        assert cfg["api_key"] == "***"
        assert cfg["inference_pipeline_id"] == "pipe"

    def test_get_config_returns_none_when_unset(self):
        cfg = tracer.get_tracer_config()
        assert cfg["api_key"] is None
        assert cfg["inference_pipeline_id"] is None


class TestLangchainCallbackResolver:
    """Regression test: the LangChain callback must route through the resolver,
    not read OPENLAYER_INFERENCE_PIPELINE_ID directly. Bug from the pre-resolver
    code where init(inference_pipeline_id=...) was silently ignored by LangChain."""

    def teardown_method(self):
        _reset_tracer_state()

    def test_langchain_callback_uses_configured_pipeline(self):
        try:
            from openlayer.lib.integrations import langchain_callback  # noqa: F401
        except ImportError:
            pytest.skip("langchain-core not installed")

        # No env var set; only init() sets the pipeline.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENLAYER_INFERENCE_PIPELINE_ID", None)
            tracer.init(inference_pipeline_id="from-init")
            assert tracer.resolve_pipeline_id() == "from-init"

        # The bug-fix is that the callback's stream call uses tracer.resolve_pipeline_id()
        # at langchain_callback.py:252 and :1310. Verifying source-level routing here
        # rather than mocking the full LangChain callback machinery.
        import inspect

        source = inspect.getsource(langchain_callback)
        assert "resolve_pipeline_id" in source
        assert 'get_env_variable("OPENLAYER_INFERENCE_PIPELINE_ID")' not in source
