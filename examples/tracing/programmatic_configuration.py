"""
Example: Configuring the Openlayer Tracer

Demonstrates the three ways to configure the tracer and how they compose:
  1. Environment variables (canonical for deployments)
  2. init() — programmatic, idempotent, merges on repeated calls
  3. Per-decorator override via @trace(inference_pipeline_id=...)

Precedence (highest first):
  decorator argument  >  init()  >  environment variable  >  default

Also shows the deprecated configure() alias, kept for backward compatibility.
"""

import os
import openai
from openlayer.lib import init, configure, get_tracer_config, trace, trace_openai


def example_environment_variables():
    """Canonical deployment path — env vars only, no code changes needed."""
    print("=== Environment Variables Approach ===")

    os.environ["OPENLAYER_API_KEY"] = "your_openlayer_api_key_here"
    os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your_pipeline_id_here"
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    @trace()
    def generate_response(query: str) -> str:
        client = trace_openai(openai.OpenAI())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
        )
        return response.choices[0].message.content

    print(f"Response: {generate_response('What is machine learning?')}")


def example_programmatic_init():
    """Programmatic configuration via init() — preferred for notebooks/apps."""
    print("\n=== Programmatic init() ===")

    init(
        api_key="your_openlayer_api_key_here",
        inference_pipeline_id="your_pipeline_id_here",
        # base_url="https://onprem.example.com",  # Optional, for on-prem deployments
    )

    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    @trace()
    def generate_response(query: str) -> str:
        client = trace_openai(openai.OpenAI())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
        )
        return response.choices[0].message.content

    print(f"Response: {generate_response('What is deep learning?')}")


def example_init_merges_on_repeat():
    """init() is idempotent and merges — safe to call multiple times."""
    print("\n=== init() Merge Semantics ===")

    init(api_key="key-A", inference_pipeline_id="pipeline-A")
    # Later in the program, override just one knob — the rest is preserved.
    init(inference_pipeline_id="pipeline-B")

    cfg = get_tracer_config()  # API key is redacted in the returned dict
    print(f"Resolved config: api_key={cfg['api_key']}, pipeline_id={cfg['inference_pipeline_id']}")
    # api_key=***, pipeline_id=pipeline-B


def example_per_decorator_override():
    """The @trace decorator can override the configured pipeline per-call."""
    print("\n=== Per-Decorator Override ===")

    init(api_key="your_openlayer_api_key_here", inference_pipeline_id="default_pipeline_id")

    @trace()
    def default_pipeline_function(query: str) -> str:
        return f"Response to: {query}"

    @trace(inference_pipeline_id="specific_pipeline_id")
    def specific_pipeline_function(query: str) -> str:
        return f"Specific response to: {query}"

    default_pipeline_function("Question 1")   # default_pipeline_id
    specific_pipeline_function("Question 2")  # specific_pipeline_id


def example_mixed_env_and_init():
    """Env var for API key, init() for pipeline — both honored via resolver."""
    print("\n=== Mixed (Env Var + init()) ===")

    os.environ["OPENLAYER_API_KEY"] = "your_openlayer_api_key_here"
    init(inference_pipeline_id="programmatic_pipeline_id")

    @trace()
    def mixed_config_function(query: str) -> str:
        return f"Mixed config response to: {query}"

    print(f"Response: {mixed_config_function('What is the best approach?')}")


def example_deprecated_configure():
    """configure() is kept for backward compatibility but is deprecated.

    It now merges (rather than replacing) state and emits a DeprecationWarning
    on each call. New code should use init() instead.
    """
    print("\n=== Deprecated configure() (still works) ===")

    configure(
        api_key="your_openlayer_api_key_here",
        inference_pipeline_id="your_pipeline_id_here",
    )


if __name__ == "__main__":
    print("Openlayer Tracing Configuration Examples")
    print("=" * 50)
    print("Note: Replace placeholder API keys and pipeline IDs with real values.\n")

    try:
        example_environment_variables()
        example_programmatic_init()
        example_init_merges_on_repeat()
        example_per_decorator_override()
        example_mixed_env_and_init()
        example_deprecated_configure()
    except Exception as e:
        print(f"Example failed (expected with placeholder keys): {e}")
        print("\nTo run this example successfully:")
        print("1. Replace placeholder API keys with real values")
        print("2. Replace pipeline IDs with real Openlayer pipeline IDs")
