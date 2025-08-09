"""
Example: Using @trace() Decorator with Guardrails

This example demonstrates how to use guardrails with the @trace() decorator
for function-level protection and monitoring.
"""

import os
from typing import Dict, Any
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails import PIIGuardrail, BlockStrategy

# Set environment variables
os.environ["OPENLAYER_API_KEY"] = "your_api_key_here"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your_pipeline_id_here"

def main():
    """Main example demonstrating @trace() decorator with guardrails."""
    print("🛡️  @trace() Decorator with Guardrails Examples")
    print("=" * 60)
    
    # Example 1: Basic PII Protection
    print("\n📋 Example 1: Basic PII Protection")
    
    pii_guard = PIIGuardrail(
        name="Basic PII Protection",
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE,
        block_message="Request blocked due to sensitive information detected"
    )
    
    @tracer.trace(guardrails=[pii_guard])
    def process_user_query(user_input: str) -> str:
        """Process user input with PII protection."""
        # Simulate some processing
        processed = f"Processed query: {user_input}"
        return processed
    
    # Test with safe content
    try:
        result = process_user_query("Tell me about machine learning")
        print(f"✅ Safe content: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with PII content
    try:
        result = process_user_query("My SSN is 123-45-6789, can you help?")
        print(f"🛡️  PII handled: {result}")
    except Exception as e:
        print(f"🚫 PII blocked: {e}")
    
    # Example 2: Multiple Guardrails with Different Strategies
    print("\n📋 Example 2: Multiple Guardrails")
    
    # Strict guardrail for SSNs (blocks completely)
    ssn_guard = PIIGuardrail(
        name="SSN Protection",
        block_entities={"US_SSN"},
        redact_entities=set(),
        block_strategy=BlockStrategy.RAISE_EXCEPTION,
        confidence_threshold=0.8
    )
    
    # Lenient guardrail for phone numbers (redacts)
    phone_guard = PIIGuardrail(
        name="Phone Protection", 
        block_entities=set(),
        redact_entities={"PHONE_NUMBER"},
        confidence_threshold=0.7
    )
    
    @tracer.trace(guardrails=[ssn_guard, phone_guard])
    def handle_customer_data(customer_info: str) -> str:
        """Handle customer data with layered protection."""
        return f"Customer data processed: {customer_info}"
    
    # Test with phone number (should be redacted)
    try:
        result = handle_customer_data("Contact me at 555-123-4567")
        print(f"📞 Phone redacted: {result}")
    except Exception as e:
        print(f"🚫 Blocked: {e}")
    
    # Test with SSN (should be blocked completely)
    try:
        result = handle_customer_data("My SSN: 987-65-4321")
        print(f"❌ Should not reach here: {result}")
    except Exception as e:
        print(f"🛡️  SSN blocked: {e}")
    
    # Example 3: Custom Guardrail Logic
    print("\n📋 Example 3: Custom Guardrail")
    
    class CustomContentFilter(PIIGuardrail):
        """Custom guardrail that blocks specific keywords."""
        
        def __init__(self, **config):
            super().__init__(name="Custom Content Filter", **config)
            self.blocked_keywords = config.get("blocked_keywords", ["password", "secret"])
        
        def check_input(self, inputs: Dict[str, Any]):
            text = str(inputs).lower()
            for keyword in self.blocked_keywords:
                if keyword in text:
                    return self._create_block_result(f"Blocked keyword: {keyword}")
            return self._create_allow_result()
        
        def check_output(self, output: Any, inputs: Dict[str, Any]):
            text = str(output).lower()
            for keyword in self.blocked_keywords:
                if keyword in text:
                    return self._create_modify_result(
                        str(output).replace(keyword, "[REDACTED]"),
                        f"Redacted keyword: {keyword}"
                    )
            return self._create_allow_result()
    
    custom_guard = CustomContentFilter(
        blocked_keywords=["password", "secret", "confidential"]
    )
    
    @tracer.trace(guardrails=[custom_guard])
    def process_document(content: str) -> str:
        """Process document with custom content filtering."""
        return f"Document processed: {content}"
    
    try:
        result = process_document("This document contains secret information")
        print(f"🔍 Custom filter applied: {result}")
    except Exception as e:
        print(f"🚫 Custom filter blocked: {e}")
    
    # Example 4: Conditional Guardrails
    print("\n📋 Example 4: Conditional Guardrails")
    
    def create_context_aware_function(user_role: str):
        """Create function with role-based guardrails."""
        if user_role == "admin":
            # Admins get lenient guardrails
            guards = [PIIGuardrail(
                name="Admin PII",
                confidence_threshold=0.9,  # Higher threshold
                block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE
            )]
        else:
            # Regular users get strict guardrails
            guards = [PIIGuardrail(
                name="User PII", 
                confidence_threshold=0.6,  # Lower threshold
                block_strategy=BlockStrategy.RAISE_EXCEPTION
            )]
        
        @tracer.trace(guardrails=guards)
        def handle_request(request_data: str) -> str:
            return f"[{user_role}] Processed: {request_data}"
        
        return handle_request
    
    # Test with different roles
    admin_handler = create_context_aware_function("admin")
    user_handler = create_context_aware_function("user")
    
    test_data = "User email: user@example.com"
    
    try:
        admin_result = admin_handler(test_data)
        print(f"👑 Admin result: {admin_result}")
    except Exception as e:
        print(f"👑 Admin blocked: {e}")
    
    try:
        user_result = user_handler(test_data)
        print(f"👤 User result: {user_result}")
    except Exception as e:
        print(f"👤 User blocked: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 @trace() Decorator Examples Complete!")
    print("\n📊 Key Features Demonstrated:")
    print("  • Function-level PII protection")
    print("  • Multiple guardrails with different strategies")
    print("  • Custom guardrail implementations")
    print("  • Role-based conditional protection")
    print("  • Rich metadata for monitoring and analysis")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 To run with real PII detection:")
        print("   pip install presidio-analyzer presidio-anonymizer")
        print("\n✅ This example demonstrates the API structure.")
