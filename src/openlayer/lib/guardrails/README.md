# Openlayer Guardrails System

The Openlayer Guardrails system provides a flexible framework for protecting against security risks, PII leakage, and other concerns in traced functions. Guardrails can intercept function inputs and outputs, taking actions like allowing, blocking, or modifying data based on configurable rules.

## Overview

Guardrails integrate seamlessly with Openlayer's tracing system, automatically adding metadata about their actions to trace steps. This provides visibility into when and how guardrails are protecting your applications.

### Key Features

- **Flexible Actions**: Allow, block, or modify data based on detection results
- **Input & Output Protection**: Guardrails can protect both function inputs and outputs
- **Extensible Architecture**: Easy to add new guardrail types and detection methods
- **Trace Integration**: Automatic metadata logging to Openlayer traces
- **Multiple Guardrails**: Support for applying multiple guardrails to a single function
- **Configurable Thresholds**: Adjustable confidence levels and detection rules

## Quick Start

### Basic Usage

```python
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails import PIIGuardrail

# Create a PII guardrail
pii_guardrail = PIIGuardrail(
    name="PII Protection",
    block_entities={"US_SSN", "CREDIT_CARD"},  # Block high-risk PII
    redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS"}  # Redact medium-risk PII
)

# Apply to traced functions
@tracer.trace(guardrails=[pii_guardrail])
def process_user_input(user_query: str) -> str:
    return f"Processing: {user_query}"

# Usage examples:
process_user_input("tell me about turtles")  # ✅ Allowed
process_user_input("my SSN is 123-45-6789")  # 🚫 Blocked
process_user_input("call me at 555-1234")    # ✏️ Phone number redacted
```

### Installation Requirements

The PII guardrail requires Microsoft Presidio:

```bash
pip install presidio-analyzer presidio-anonymizer
```

## Guardrail Actions

Guardrails can take three types of actions:

### 1. ALLOW
- **When**: No sensitive data detected or data is considered safe
- **Result**: Function executes normally with original data
- **Metadata**: Records that no action was taken

### 2. BLOCK
- **When**: High-risk sensitive data is detected (e.g., SSN, credit cards)
- **Result**: Raises `GuardrailBlockedException`, preventing function execution
- **Metadata**: Records what was blocked and why

### 3. MODIFY
- **When**: Medium-risk sensitive data is detected (e.g., phone numbers, emails)
- **Result**: Function executes with redacted/modified data
- **Metadata**: Records what was modified and how

## Built-in Guardrails

### PIIGuardrail

Protects against Personally Identifiable Information using Microsoft Presidio.

```python
from openlayer.lib.guardrails import PIIGuardrail

pii_guardrail = PIIGuardrail(
    name="PII Protection",
    block_entities={"US_SSN", "CREDIT_CARD", "US_PASSPORT"},
    redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION"},
    confidence_threshold=0.8,  # Minimum confidence to trigger (0.0-1.0)
    language="en"  # Language for analysis
)
```

**Supported Entity Types:**
- **High-risk (typically blocked)**: `US_SSN`, `CREDIT_CARD`, `CRYPTO`, `IBAN_CODE`, `US_BANK_NUMBER`, `US_DRIVER_LICENSE`, `US_PASSPORT`
- **Medium-risk (typically redacted)**: `PHONE_NUMBER`, `EMAIL_ADDRESS`, `PERSON`, `LOCATION`, `DATE_TIME`, `NRP`, `MEDICAL_LICENSE`, `URL`

## Creating Custom Guardrails

### Basic Custom Guardrail

```python
from openlayer.lib.guardrails.base import BaseGuardrail, GuardrailAction, GuardrailResult

class ToxicityGuardrail(BaseGuardrail):
    def __init__(self, name: str = "Toxicity Filter", **config):
        super().__init__(name=name, **config)
        self.toxic_words = config.get("toxic_words", ["badword1", "badword2"])
    
    def check_input(self, inputs: Dict[str, Any]) -> GuardrailResult:
        # Check inputs for toxic content
        text_content = str(inputs)
        for word in self.toxic_words:
            if word.lower() in text_content.lower():
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason=f"Toxic content detected: {word}"
                )
        return GuardrailResult(action=GuardrailAction.ALLOW)
    
    def check_output(self, output: Any, inputs: Dict[str, Any]) -> GuardrailResult:
        # Similar logic for outputs
        return GuardrailResult(action=GuardrailAction.ALLOW)

# Register and use
from openlayer.lib.guardrails.base import register_guardrail
register_guardrail("toxicity", ToxicityGuardrail)

# Create instance
toxicity_guard = ToxicityGuardrail(toxic_words=["spam", "scam"])
```

## Advanced Usage

### Multiple Guardrails

```python
# Apply multiple guardrails in sequence
@tracer.trace(guardrails=[pii_guardrail, toxicity_guardrail, custom_guardrail])
def secure_function(user_input: str) -> str:
    return process_input(user_input)
```

### Configuration Options

```python
# Highly customized PII guardrail
strict_pii = PIIGuardrail(
    name="Strict PII Filter",
    enabled=True,
    block_entities={"US_SSN", "CREDIT_CARD", "US_BANK_NUMBER"},
    redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"},
    confidence_threshold=0.9,  # Very strict
    language="en"
)

# Lenient PII guardrail for development
dev_pii = PIIGuardrail(
    name="Development PII Filter", 
    enabled=False,  # Disabled for development
    confidence_threshold=0.5  # Lower threshold
)
```

### Conditional Guardrails

```python
# Enable/disable based on environment
import os
production_mode = os.getenv("ENVIRONMENT") == "production"

pii_guardrail = PIIGuardrail(
    name="Production PII Filter",
    enabled=production_mode,
    confidence_threshold=0.9 if production_mode else 0.5
)
```

## Metadata and Observability

Guardrails automatically add metadata to trace steps, providing visibility into their actions:

```json
{
  "guardrails": {
    "input_pii_protection": {
      "action": "redacted",
      "reason": "Redacted PII entities: PHONE_NUMBER",
      "metadata": {
        "detected_entities": ["PHONE_NUMBER"],
        "redacted_entities": ["PHONE_NUMBER"],
        "confidence_threshold": 0.7
      }
    },
    "output_pii_protection": {
      "action": "allow",
      "reason": "no_pii_detected",
      "metadata": {
        "detected_entities": [],
        "confidence_threshold": 0.7
      }
    }
  }
}
```

## Error Handling

### GuardrailBlockedException

When a guardrail blocks execution, it raises a `GuardrailBlockedException`:

```python
try:
    result = secure_function("my SSN is 123-45-6789")
except GuardrailBlockedException as e:
    print(f"Blocked by {e.guardrail_name}: {e.reason}")
    print(f"Metadata: {e.metadata}")
```

### Graceful Degradation

Guardrails are designed to fail gracefully:
- If a guardrail encounters an error, it logs the error but doesn't break the trace
- The error is recorded in the trace metadata
- Function execution continues normally

## Performance Considerations

- Guardrails add latency to function execution
- PII detection using Presidio can be CPU-intensive for large text
- Consider caching guardrail results for repeated content
- Use appropriate confidence thresholds to balance accuracy and performance
- Disable guardrails in development/testing environments if needed

## Integration with Other Systems

### LLM Guard Integration (Future)

The guardrails system is designed to support multiple detection backends:

```python
# Future: LLM Guard integration
from openlayer.lib.guardrails import LLMGuardGuardrail

llm_guard = LLMGuardGuardrail(
    scanners=["Toxicity", "BanSubstrings", "PromptInjection"]
)
```

### Custom Detection Engines

Implement the `BaseGuardrail` interface to integrate any detection system:

```python
class CustomDetectionGuardrail(BaseGuardrail):
    def __init__(self, **config):
        super().__init__(**config)
        # Initialize your detection engine
        self.detector = YourDetectionEngine(**config)
    
    def check_input(self, inputs):
        results = self.detector.analyze(inputs)
        # Convert to GuardrailResult
        return self._convert_results(results)
```

## Best Practices

1. **Layer Guardrails**: Use multiple guardrails for defense in depth
2. **Environment-Specific Config**: Different settings for dev/staging/production
3. **Monitor Performance**: Track guardrail latency and effectiveness
4. **Regular Updates**: Keep detection rules and models updated
5. **Test Thoroughly**: Verify guardrails work with your specific data patterns
6. **Document Policies**: Clear documentation of what gets blocked/modified
7. **Audit Logs**: Review guardrail actions regularly for tuning

## Troubleshooting

### Common Issues

1. **High False Positives**: Lower confidence threshold or adjust entity types
2. **Performance Issues**: Optimize text preprocessing, use caching
3. **Missing Detections**: Increase confidence threshold, add custom patterns
4. **Import Errors**: Ensure required dependencies (presidio) are installed

### Debugging

Enable debug logging to see guardrail decisions:

```python
import logging
logging.getLogger("openlayer.lib.guardrails").setLevel(logging.DEBUG)
```

## Examples

See the `examples/tracing/` directory for complete working examples:
- `guardrails_example.py` - Comprehensive examples with Presidio
- `simple_guardrails_test.py` - Basic functionality test without dependencies
