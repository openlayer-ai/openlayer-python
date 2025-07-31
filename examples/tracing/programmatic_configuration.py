"""
Example: Programmatic Configuration for Openlayer Tracing

This example demonstrates how to configure Openlayer tracing programmatically
using the configure() function, instead of relying on environment variables.
"""

import os
import openai
from openlayer.lib import configure, trace, trace_openai


def example_environment_variables():
    """Traditional approach using environment variables."""
    print("=== Environment Variables Approach ===")

    # Set environment variables (traditional approach)
    os.environ["OPENLAYER_API_KEY"] = "your_openlayer_api_key_here"
    os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your_pipeline_id_here"
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    # Use the @trace decorator
    @trace()
    def generate_response(query: str) -> str:
        """Generate a response using OpenAI."""
        # Configure OpenAI client and trace it
        client = trace_openai(openai.OpenAI())

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
        )
        return response.choices[0].message.content

    # Test the function
    result = generate_response("What is machine learning?")
    print(f"Response: {result}")


def example_programmatic_configuration():
    """New approach using programmatic configuration."""
    print("\n=== Programmatic Configuration Approach ===")

    # Configure Openlayer programmatically
    configure(
        api_key="your_openlayer_api_key_here",
        inference_pipeline_id="your_pipeline_id_here",
        # base_url="https://api.openlayer.com/v1"  # Optional: custom base URL
    )

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    # Use the @trace decorator (no environment variables needed for Openlayer)
    @trace()
    def generate_response_programmatic(query: str) -> str:
        """Generate a response using OpenAI with programmatic configuration."""
        # Configure OpenAI client and trace it
        client = trace_openai(openai.OpenAI())

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            max_tokens=100,
        )
        return response.choices[0].message.content

    # Test the function
    result = generate_response_programmatic("What is deep learning?")
    print(f"Response: {result}")


def example_per_decorator_override():
    """Example showing how to override pipeline ID per decorator."""
    print("\n=== Per-Decorator Pipeline ID Override ===")

    # Configure default settings
    configure(
        api_key="your_openlayer_api_key_here",
        inference_pipeline_id="default_pipeline_id",
    )

    # Function using default pipeline ID
    @trace()
    def default_pipeline_function(query: str) -> str:
        return f"Response to: {query}"

    # Function using specific pipeline ID (overrides default)
    @trace(inference_pipeline_id="specific_pipeline_id")
    def specific_pipeline_function(query: str) -> str:
        return f"Specific response to: {query}"

    # Test both functions
    default_pipeline_function("Question 1")  # Uses default_pipeline_id
    specific_pipeline_function("Question 2")  # Uses specific_pipeline_id

    print("Both functions executed with different pipeline IDs")


def example_mixed_configuration():
    """Example showing mixed environment and programmatic configuration."""
    print("\n=== Mixed Configuration Approach ===")

    # Set API key via environment variable
    os.environ["OPENLAYER_API_KEY"] = "your_openlayer_api_key_here"

    # Set pipeline ID programmatically
    configure(inference_pipeline_id="programmatic_pipeline_id")

    @trace()
    def mixed_config_function(query: str) -> str:
        """Function using mixed configuration."""
        return f"Mixed config response to: {query}"

    # Test the function
    result = mixed_config_function("What is the best approach?")
    print(f"Response: {result}")


if __name__ == "__main__":
    print("Openlayer Tracing Configuration Examples")
    print("=" * 50)

    # Note: Replace the placeholder API keys and IDs with real values
    print("Note: Replace placeholder API keys and pipeline IDs with real values before running.")
    print()

    try:
        # Run examples (these will fail without real API keys)
        example_environment_variables()
        example_programmatic_configuration()
        example_per_decorator_override()
        example_mixed_configuration()

    except Exception as e:
        print(f"Example failed (expected with placeholder keys): {e}")
        print("\nTo run this example successfully:")
        print("1. Replace placeholder API keys with real values")
        print("2. Replace pipeline IDs with real Openlayer pipeline IDs")
        print("3. Ensure you have valid OpenAI and Openlayer accounts")
