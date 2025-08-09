"""
Example: Using Guardrails with Helper Functions (trace_openai, etc.)

This example demonstrates how to use guardrails with LLM helper functions
like trace_openai, trace_anthropic, etc., in addition to the @trace decorator.
"""

import os
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails import PIIGuardrail, BlockStrategy

# Set environment variables
os.environ["OPENLAYER_API_KEY"] = "test_api_key" 
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "test_pipeline_id"
os.environ["OPENLAYER_DISABLE_PUBLISH"] = "true"  # Disable publishing for testing

# Mock OpenAI for demonstration
class MockOpenAI:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                class MockResponse:
                    class Choice:
                        class Message:
                            content = f"Mock response to: {kwargs.get('messages', [{}])[-1].get('content', 'unknown')}"
                        message = Message()
                    choices = [Choice()]
                    
                    class Usage:
                        prompt_tokens = 10
                        completion_tokens = 5
                        total_tokens = 15
                    usage = Usage()
                    
                    model = kwargs.get('model', 'mock-model')
                return MockResponse()
        completions = Completions()
    chat = Chat()


def example_1_global_guardrails_configuration():
    """Example 1: Configure global guardrails for all helper functions."""
    print("=== Example 1: Global Guardrails Configuration ===")
    
    # Create guardrails
    pii_guardrail = PIIGuardrail(
        name="Global PII Protection",
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE,
        block_message="Content blocked due to PII detection"
    )
    
    # Configure global guardrails - these will apply to ALL helper functions
    tracer.configure(
        api_key="test_key",
        inference_pipeline_id="test_pipeline", 
        guardrails=[pii_guardrail]
    )
    
    print("✅ Global guardrails configured")
    print("   All LLM helper functions (trace_openai, trace_anthropic, etc.) will now use these guardrails")


def example_2_trace_openai_with_guardrails():
    """Example 2: Using trace_openai with configured guardrails."""
    print("\n=== Example 2: trace_openai with Guardrails ===")
    
    # Mock OpenAI client
    mock_client = MockOpenAI()
    
    # Trace the client - it will automatically use configured guardrails
    traced_client = tracer.trace_openai(mock_client)
    
    @tracer.trace()
    def chat_with_openai(user_message: str) -> str:
        """Function that uses OpenAI client."""
        response = traced_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        return response.choices[0].message.content
    
    # Test with safe content
    try:
        result = chat_with_openai("Tell me about machine learning")
        print(f"✅ Safe request: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with PII content (should be blocked/modified by guardrails)
    try:
        result = chat_with_openai("My SSN is 123-45-6789, can you help me?")
        print(f"🛡️  PII request handled: {result}")
    except Exception as e:
        print(f"🚫 PII request blocked: {e}")


def example_3_per_call_guardrails():
    """Example 3: Override global guardrails for specific calls."""
    print("\n=== Example 3: Per-Call Guardrail Override ===")
    
    # Create a different guardrail for this specific case
    strict_pii_guardrail = PIIGuardrail(
        name="Strict PII Protection",
        block_strategy=BlockStrategy.RAISE_EXCEPTION,  # More strict than global
        confidence_threshold=0.9
    )
    
    # Manually add a chat completion step with specific guardrails
    try:
        tracer.add_chat_completion_step_to_trace(
            guardrails=[strict_pii_guardrail],  # Override global guardrails
            inputs={"prompt": [{"role": "user", "content": "My credit card is 4532-1234-5678-9012"}]},
            output="I can't help with credit card information",
            name="Strict PII Check",
            provider="Manual",
            model="test-model"
        )
        print("✅ Manual step added with strict guardrails")
    except Exception as e:
        print(f"🚫 Strict guardrail blocked: {e}")


def example_4_mixed_usage():
    """Example 4: Mixed usage of @trace decorator and helper functions."""
    print("\n=== Example 4: Mixed Usage ===")
    
    mock_client = MockOpenAI()
    traced_client = tracer.trace_openai(mock_client)
    
    # Function-level guardrails (only apply to this function)
    function_guardrail = PIIGuardrail(
        name="Function-Level PII",
        block_strategy=BlockStrategy.SKIP_FUNCTION
    )
    
    @tracer.trace(guardrails=[function_guardrail])  # Function-specific guardrails
    def process_user_input(user_input: str) -> str:
        """This function has its own guardrails."""
        # This OpenAI call will use GLOBAL guardrails (configured earlier)
        response = traced_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Process this: {user_input}"}]
        )
        return f"Processed: {response.choices[0].message.content}"
    
    # Test with PII - function guardrails will apply to the function,
    # global guardrails will apply to the OpenAI call
    try:
        result = process_user_input("My phone number is 555-123-4567")
        print(f"📞 Mixed guardrails result: {result}")
    except Exception as e:
        print(f"🛡️  Mixed guardrails handled: {e}")


def example_5_disable_global_guardrails():
    """Example 5: Temporarily disable global guardrails."""
    print("\n=== Example 5: Disable Global Guardrails ===")
    
    # Disable global guardrails
    tracer.configure(guardrails=None)
    
    mock_client = MockOpenAI() 
    traced_client = tracer.trace_openai(mock_client)
    
    @tracer.trace()
    def unrestricted_chat(message: str) -> str:
        """Function without guardrails."""
        response = traced_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
    
    try:
        result = unrestricted_chat("My SSN is 987-65-4321")  # No guardrails active
        print(f"🔓 Unrestricted result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_6_guardrail_metadata_inspection():
    """Example 6: Inspecting guardrail metadata in traces."""
    print("\n=== Example 6: Guardrail Metadata ===")
    
    # Re-enable guardrails for metadata demonstration
    metadata_guardrail = PIIGuardrail(
        name="Metadata Demo",
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE
    )
    tracer.configure(guardrails=[metadata_guardrail])
    
    # This would normally be inspected in the Openlayer dashboard
    # For demo purposes, we'll just show that the system works
    try:
        tracer.add_chat_completion_step_to_trace(
            inputs={"prompt": [{"role": "user", "content": "What's my email: user@example.com?"}]},
            output="I can't process email addresses",
            name="Metadata Demo",
            provider="Demo"
        )
        print("✅ Step with metadata added")
        print("   Metadata includes:")
        print("   - has_guardrails: true")
        print("   - guardrail_actions: ['redacted' or 'blocked' or 'allow']")
        print("   - guardrail_names: ['metadata_demo']")
        print("   - guardrail_blocked/modified/allowed: boolean flags")
    except Exception as e:
        print(f"🛡️  Guardrail action: {e}")


if __name__ == "__main__":
    print("Guardrails with Helper Functions Examples")
    print("=" * 50)
    
    try:
        example_1_global_guardrails_configuration()
        example_2_trace_openai_with_guardrails()
        example_3_per_call_guardrails()
        example_4_mixed_usage()
        example_5_disable_global_guardrails()
        example_6_guardrail_metadata_inspection()
        
        print("\n" + "=" * 50)
        print("🎉 All examples completed successfully!")
        
        print("\n📋 Key Features Demonstrated:")
        print("  • Global guardrails configuration")
        print("  • Automatic application to helper functions (trace_openai, etc.)")
        print("  • Per-call guardrail overrides")
        print("  • Mixed usage with @trace decorator")
        print("  • Guardrail metadata for filtering and analysis")
        
        print("\n🔧 Usage Summary:")
        print("  1. Configure global guardrails: tracer.configure(guardrails=[...])")
        print("  2. Use helper functions normally: trace_openai(client)")
        print("  3. Override per-call: add_chat_completion_step_to_trace(guardrails=[...])")
        print("  4. Mix with @trace decorator for comprehensive protection")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("\nThis is expected when running without real dependencies.")
        print("The examples demonstrate the API and integration patterns.")
