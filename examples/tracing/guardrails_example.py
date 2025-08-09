"""
Example: Using Guardrails with Openlayer Tracing

This example demonstrates how to use guardrails to protect against PII leakage
and other security concerns in traced functions.
"""

import os
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails import PIIGuardrail, GuardrailBlockedException

# Set environment variables (replace with your actual values)
os.environ["OPENLAYER_API_KEY"] = "your_openlayer_api_key_here"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your_pipeline_id_here"


def example_1_no_pii():
    """Example 1: Normal query with no PII - should do nothing."""
    print("=== Example 1: No PII Detection ===")
    
    # Create PII guardrail
    pii_guardrail = PIIGuardrail(name="PII Protection")
    
    @tracer.trace(guardrails=[pii_guardrail])
    def process_query(user_query: str) -> str:
        """Process a user query and return a response."""
        return f"Here's information about {user_query}: Turtles are reptiles..."
    
    try:
        result = process_query("tell me about turtles")
        print(f"Result: {result}")
        print("✅ Query processed successfully - no PII detected")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_blocked_ssn():
    """Example 2: Query with SSN - should be blocked."""
    print("\n=== Example 2: SSN Detection (Blocked) ===")
    
    # Create PII guardrail with SSN in block list
    pii_guardrail = PIIGuardrail(
        name="PII Protection",
        block_entities={"US_SSN", "CREDIT_CARD"},  # High-risk PII
        redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS"}  # Medium-risk PII
    )
    
    @tracer.trace(guardrails=[pii_guardrail])
    def process_query(user_query: str) -> str:
        """Process a user query and return a response."""
        return f"Processing: {user_query}"
    
    try:
        result = process_query("here is my SSN: 123-45-6789")
        print(f"Result: {result}")
    except GuardrailBlockedException as e:
        print(f"🚫 Request blocked by guardrail: {e}")
        print("✅ Successfully blocked high-risk PII")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def example_3_redacted_phone():
    """Example 3: Query with phone number - should be redacted."""
    print("\n=== Example 3: Phone Number Detection (Redacted) ===")
    
    # Create PII guardrail
    pii_guardrail = PIIGuardrail(
        name="PII Protection",
        block_entities={"US_SSN", "CREDIT_CARD"},  # High-risk PII
        redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"}  # Medium-risk PII
    )
    
    @tracer.trace(guardrails=[pii_guardrail])
    def process_query(user_query: str) -> str:
        """Process a user query and return a response."""
        return f"I'll help you with that request: {user_query}"
    
    try:
        result = process_query("here is my phone number: 555-123-4567")
        print(f"Result: {result}")
        print("✅ Phone number successfully redacted")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_multiple_guardrails():
    """Example 4: Multiple guardrails with different configurations."""
    print("\n=== Example 4: Multiple Guardrails ===")
    
    # Create multiple guardrails with different settings
    strict_pii_guardrail = PIIGuardrail(
        name="Strict PII",
        block_entities={"US_SSN", "CREDIT_CARD", "US_PASSPORT"},
        redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS"},
        confidence_threshold=0.8  # Higher confidence required
    )
    
    lenient_pii_guardrail = PIIGuardrail(
        name="Lenient PII", 
        block_entities=set(),  # Don't block anything
        redact_entities={"PERSON", "LOCATION"},
        confidence_threshold=0.6  # Lower confidence threshold
    )
    
    @tracer.trace(guardrails=[strict_pii_guardrail, lenient_pii_guardrail])
    def process_user_data(user_input: str) -> str:
        """Process user data with multiple guardrail layers."""
        return f"Processed data: {user_input}"
    
    try:
        result = process_user_data("Hi, I'm John Smith from New York, call me at 555-0123")
        print(f"Result: {result}")
        print("✅ Multiple guardrails applied successfully")
    except GuardrailBlockedException as e:
        print(f"🚫 Request blocked: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_output_guardrails():
    """Example 5: Guardrails on function output."""
    print("\n=== Example 5: Output Guardrails ===")
    
    # Create PII guardrail that also checks outputs
    pii_guardrail = PIIGuardrail(
        name="Input/Output PII Protection",
        block_entities={"US_SSN"},
        redact_entities={"EMAIL_ADDRESS", "PHONE_NUMBER"}
    )
    
    @tracer.trace(guardrails=[pii_guardrail])
    def generate_response(query: str) -> str:
        """Generate a response that might contain PII."""
        # Simulate a function that might accidentally include PII in output
        if "contact" in query.lower():
            return "You can reach our support at support@company.com or call 555-HELP"
        return "How can I help you today?"
    
    try:
        result = generate_response("How do I contact support?")
        print(f"Result: {result}")
        print("✅ Output PII successfully redacted")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_6_custom_configuration():
    """Example 6: Custom guardrail configuration."""
    print("\n=== Example 6: Custom Configuration ===")
    
    # Create highly customized PII guardrail
    custom_pii_guardrail = PIIGuardrail(
        name="Custom PII Guardrail",
        block_entities={"CREDIT_CARD", "US_SSN", "US_BANK_NUMBER"},
        redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION", "DATE_TIME"},
        confidence_threshold=0.75,
        language="en"
    )
    
    @tracer.trace(guardrails=[custom_pii_guardrail])
    def process_customer_request(request: str) -> str:
        """Process a customer service request."""
        return f"Thank you for your request. We'll process: {request}"
    
    try:
        result = process_customer_request(
            "Hi, I'm Jane Doe from San Francisco. "
            "My account issue happened on January 15th, 2024. "
            "Please call me at (555) 987-6543 or email jane.doe@email.com"
        )
        print(f"Result: {result}")
        print("✅ Custom guardrail configuration applied")
    except GuardrailBlockedException as e:
        print(f"🚫 Request blocked: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_7_disabled_guardrail():
    """Example 7: Disabled guardrail."""
    print("\n=== Example 7: Disabled Guardrail ===")
    
    # Create disabled guardrail
    disabled_guardrail = PIIGuardrail(
        name="Disabled PII Guardrail",
        enabled=False,  # Guardrail is disabled
        block_entities={"US_SSN"}
    )
    
    @tracer.trace(guardrails=[disabled_guardrail])
    def process_data(data: str) -> str:
        """Process data with disabled guardrail."""
        return f"Processed: {data}"
    
    try:
        result = process_data("SSN: 123-45-6789")  # Would normally be blocked
        print(f"Result: {result}")
        print("✅ Disabled guardrail allowed request to pass through")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Openlayer Guardrails Examples")
    print("=" * 50)
    
    # Note: These examples require presidio to be installed
    print("Note: These examples require presidio. Install with:")
    print("pip install presidio-analyzer presidio-anonymizer")
    print()
    
    try:
        # Run all examples
        example_1_no_pii()
        example_2_blocked_ssn()
        example_3_redacted_phone()
        example_4_multiple_guardrails()
        example_5_output_guardrails()
        example_6_custom_configuration()
        example_7_disabled_guardrail()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        print("\nGuardrail metadata is automatically added to your Openlayer traces.")
        print("Check your Openlayer dashboard to see the guardrail actions and metadata.")
        
    except ImportError as e:
        if "presidio" in str(e):
            print(f"❌ Presidio not installed: {e}")
            print("Install presidio with: pip install presidio-analyzer presidio-anonymizer")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("\nTo run these examples successfully:")
        print("1. Install presidio: pip install presidio-analyzer presidio-anonymizer")
        print("2. Replace placeholder API keys with real values")
        print("3. Ensure you have a valid Openlayer account and pipeline ID")
