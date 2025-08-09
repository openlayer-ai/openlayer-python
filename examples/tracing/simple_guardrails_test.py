"""
Simple test for guardrails functionality without external dependencies.

This test demonstrates the guardrails system using a mock guardrail
that doesn't require presidio or other external libraries.
"""

import os
from typing import Any, Dict
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails.base import BaseGuardrail, GuardrailAction, GuardrailResult, GuardrailBlockedException

# Set environment variables (these can be dummy values for testing)
os.environ["OPENLAYER_API_KEY"] = "test_api_key"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "test_pipeline_id"
os.environ["OPENLAYER_DISABLE_PUBLISH"] = "true"  # Disable actual publishing for testing


class MockPIIGuardrail(BaseGuardrail):
    """Mock PII guardrail for testing without external dependencies."""
    
    def __init__(self, name: str = "Mock PII Guardrail", enabled: bool = True, **config):
        super().__init__(name=name, enabled=enabled, **config)
        self.blocked_patterns = config.get("blocked_patterns", ["SSN:", "Credit Card:"])
        self.redacted_patterns = config.get("redacted_patterns", ["Phone:", "Email:"])
    
    def check_input(self, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check inputs for mock PII patterns."""
        return self._check_data(inputs)
    
    def check_output(self, output: Any, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check output for mock PII patterns.""" 
        return self._check_data(output)
    
    def _check_data(self, data: Any) -> GuardrailResult:
        """Check data for mock PII patterns."""
        if not self.enabled:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                metadata={"action": "allow", "reason": "disabled"}
            )
        
        # Convert data to string for pattern matching
        text_data = str(data)
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.lower() in text_data.lower():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    metadata={"action": "blocked", "pattern": pattern},
                    reason=f"Detected blocked pattern: {pattern}"
                )
        
        # Check for redacted patterns
        modified_text = text_data
        redacted_patterns = []
        for pattern in self.redacted_patterns:
            if pattern.lower() in text_data.lower():
                redacted_patterns.append(pattern)
                modified_text = modified_text.replace(pattern, "[REDACTED]")
        
        if redacted_patterns:
            # For dict inputs, try to preserve structure
            if isinstance(data, dict):
                modified_data = {}
                for key, value in data.items():
                    modified_value = str(value)
                    for pattern in redacted_patterns:
                        modified_value = modified_value.replace(pattern, "[REDACTED]")
                    modified_data[key] = modified_value
            else:
                modified_data = modified_text
                
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                modified_data=modified_data,
                metadata={"action": "redacted", "patterns": redacted_patterns},
                reason=f"Redacted patterns: {', '.join(redacted_patterns)}"
            )
        
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            metadata={"action": "allow", "reason": "no_patterns_detected"}
        )


def test_1_no_pii():
    """Test 1: Normal input with no PII."""
    print("=== Test 1: No PII ===")
    
    guardrail = MockPIIGuardrail()
    
    @tracer.trace(guardrails=[guardrail])
    def process_text(text: str) -> str:
        return f"Processed: {text}"
    
    try:
        result = process_text("tell me about turtles")
        print(f"✅ Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_2_blocked_ssn():
    """Test 2: Input with SSN should be blocked."""
    print("\n=== Test 2: Blocked SSN ===")
    
    guardrail = MockPIIGuardrail()
    
    @tracer.trace(guardrails=[guardrail])
    def process_text(text: str) -> str:
        return f"Processed: {text}"
    
    try:
        result = process_text("here is my SSN: 123-45-6789")
        print(f"❌ Should have been blocked but got: {result}")
        return False
    except GuardrailBlockedException as e:
        print(f"✅ Correctly blocked: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_3_redacted_phone():
    """Test 3: Input with phone should be redacted."""
    print("\n=== Test 3: Redacted Phone ===")
    
    guardrail = MockPIIGuardrail()
    
    @tracer.trace(guardrails=[guardrail])
    def process_text(text: str) -> str:
        return f"Processed: {text}"
    
    try:
        result = process_text("here is my Phone: 555-1234")
        print(f"✅ Result: {result}")
        if "[REDACTED]" in result:
            print("✅ Phone number correctly redacted")
            return True
        else:
            print("❌ Phone number was not redacted")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_4_output_redaction():
    """Test 4: Output redaction."""
    print("\n=== Test 4: Output Redaction ===")
    
    guardrail = MockPIIGuardrail()
    
    @tracer.trace(guardrails=[guardrail])
    def generate_response(query: str) -> str:
        # Function that returns PII in output
        return "Contact us at Phone: 555-HELP"
    
    try:
        result = generate_response("how to contact support")
        print(f"✅ Result: {result}")
        if "[REDACTED]" in result:
            print("✅ Output correctly redacted")
            return True
        else:
            print("❌ Output was not redacted")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_5_disabled_guardrail():
    """Test 5: Disabled guardrail should allow everything."""
    print("\n=== Test 5: Disabled Guardrail ===")
    
    guardrail = MockPIIGuardrail(enabled=False)
    
    @tracer.trace(guardrails=[guardrail])
    def process_text(text: str) -> str:
        return f"Processed: {text}"
    
    try:
        result = process_text("SSN: 123-45-6789")  # Would normally be blocked
        print(f"✅ Result: {result}")
        if "SSN:" in result:
            print("✅ Disabled guardrail correctly allowed request")
            return True
        else:
            print("❌ Request was modified when it shouldn't have been")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_6_multiple_guardrails():
    """Test 6: Multiple guardrails."""
    print("\n=== Test 6: Multiple Guardrails ===")
    
    guardrail1 = MockPIIGuardrail(
        name="Guardrail 1",
        blocked_patterns=["SSN:"],
        redacted_patterns=["Phone:"]
    )
    guardrail2 = MockPIIGuardrail(
        name="Guardrail 2", 
        blocked_patterns=[],
        redacted_patterns=["Email:"]
    )
    
    @tracer.trace(guardrails=[guardrail1, guardrail2])
    def process_text(text: str) -> str:
        return f"Processed: {text}"
    
    try:
        result = process_text("Contact: Phone: 555-1234, Email: test@example.com")
        print(f"✅ Result: {result}")
        if "[REDACTED]" in result:
            print("✅ Multiple guardrails applied successfully")
            return True
        else:
            print("❌ Guardrails did not modify the text")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("Simple Guardrails Test")
    print("=" * 50)
    
    tests = [
        test_1_no_pii,
        test_2_blocked_ssn,
        test_3_redacted_phone,
        test_4_output_redaction,
        test_5_disabled_guardrail,
        test_6_multiple_guardrails,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Guardrails system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
