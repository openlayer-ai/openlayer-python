"""PII (Personally Identifiable Information) guardrail using Presidio."""

import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import BaseGuardrail, GuardrailAction, GuardrailResult, BlockStrategy, register_guardrail

if TYPE_CHECKING:
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerResult
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        # When presidio isn't available, we'll use string literals for type annotations
        pass

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    HAVE_PRESIDIO = True
except ImportError:
    HAVE_PRESIDIO = False
    AnalyzerEngine = None
    AnonymizerEngine = None
    RecognizerResult = None
    OperatorConfig = None

logger = logging.getLogger(__name__)


class PIIGuardrail(BaseGuardrail):
    """PII detection and protection guardrail using Microsoft Presidio."""
    
    # Default entity types that trigger blocking (high-risk PII)
    DEFAULT_BLOCK_ENTITIES = {
        'CREDIT_CARD', 'CRYPTO', 'IBAN_CODE', 'IP_ADDRESS', 'US_SSN',
        'US_BANK_NUMBER', 'US_DRIVER_LICENSE', 'US_PASSPORT'
    }
    
    # Default entity types that get redacted/modified (medium-risk PII)
    DEFAULT_REDACT_ENTITIES = {
        'PHONE_NUMBER', 'EMAIL_ADDRESS', 'PERSON', 'LOCATION',
        'DATE_TIME', 'NRP', 'MEDICAL_LICENSE', 'URL'
    }
    
    def __init__(
        self,
        name: str = "PII Guardrail",
        enabled: bool = True,
        block_entities: Optional[Set[str]] = None,
        redact_entities: Optional[Set[str]] = None,
        confidence_threshold: float = 0.7,
        language: str = "en",
        block_strategy: BlockStrategy = BlockStrategy.RETURN_ERROR_MESSAGE,
        block_message: str = "Request blocked due to sensitive information",
        **config
    ):
        """Initialize PII guardrail.
        
        Args:
            name: Human-readable name for this guardrail
            enabled: Whether this guardrail is active
            block_entities: Set of entity types that should block execution
            redact_entities: Set of entity types that should be redacted
            confidence_threshold: Minimum confidence score to trigger action (0.0-1.0)
            language: Language code for analysis (default: "en")
            block_strategy: How to handle blocked requests (graceful vs exception)
            block_message: Custom message for blocked requests
            **config: Additional configuration
        """
        if not HAVE_PRESIDIO:
            raise ImportError(
                "Presidio is required for PII guardrail. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )
        
        super().__init__(name=name, enabled=enabled, **config)
        
        self.block_entities = block_entities or self.DEFAULT_BLOCK_ENTITIES.copy()
        self.redact_entities = redact_entities or self.DEFAULT_REDACT_ENTITIES.copy()
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.block_strategy = block_strategy
        self.block_message = block_message
        
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        logger.debug(f"Initialized PII guardrail with block_entities={self.block_entities}, "
                    f"redact_entities={self.redact_entities}, threshold={confidence_threshold}")
    
    def check_input(self, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check function inputs for PII."""
        return self._check_data(inputs, data_type="input")
    
    def check_output(self, output: Any, inputs: Dict[str, Any]) -> GuardrailResult:
        """Check function output for PII."""
        return self._check_data(output, data_type="output")
    
    def _check_data(self, data: Any, data_type: str) -> GuardrailResult:
        """Check arbitrary data for PII.
        
        Args:
            data: Data to check (can be dict, string, or other types)
            data_type: Type of data being checked ("input" or "output")
            
        Returns:
            GuardrailResult with appropriate action
        """
        if not self.enabled:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                metadata={"guardrail": self.name, "action": "allow", "reason": "disabled"}
            )
        
        # Extract text content from data
        text_content = self._extract_text(data)
        if not text_content:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                metadata={"guardrail": self.name, "action": "allow", "reason": "no_text_content"}
            )
        
        # Analyze for PII
        analysis_results = []
        detected_entities = set()
        
        for text in text_content:
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=list(self.block_entities | self.redact_entities)
            )
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in results 
                if result.score >= self.confidence_threshold
            ]
            
            analysis_results.extend(filtered_results)
            detected_entities.update(result.entity_type for result in filtered_results)
        
        # Determine action based on detected entities
        blocked_entities = detected_entities & self.block_entities
        redacted_entities = detected_entities & self.redact_entities
        
        metadata = {
            "guardrail": self.name,
            "detected_entities": list(detected_entities),
            "blocked_entities": list(blocked_entities),
            "redacted_entities": list(redacted_entities),
            "confidence_threshold": self.confidence_threshold,
            "data_type": data_type
        }
        
        if blocked_entities:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                metadata={**metadata, "action": "blocked"},
                reason=f"Detected high-risk PII entities: {', '.join(blocked_entities)}",
                block_strategy=self.block_strategy,
                error_message=self.block_message
            )
        
        elif redacted_entities:
            # Redact the sensitive information
            modified_data = self._redact_data(data, analysis_results)
            return GuardrailResult(
                action=GuardrailAction.MODIFY,
                modified_data=modified_data,
                metadata={**metadata, "action": "redacted"},
                reason=f"Redacted PII entities: {', '.join(redacted_entities)}"
            )
        
        else:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                metadata={**metadata, "action": "allow", "reason": "no_pii_detected"}
            )
    
    def _extract_text(self, data: Any) -> List[str]:
        """Extract text content from various data types.
        
        Args:
            data: Data to extract text from
            
        Returns:
            List of text strings found in the data
        """
        texts = []
        
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                texts.extend(self._extract_text(value))
        elif isinstance(data, (list, tuple)):
            for item in data:
                texts.extend(self._extract_text(item))
        elif hasattr(data, '__str__') and not isinstance(data, (int, float, bool)):
            # Convert other types to string, but skip basic numeric/boolean types
            text_repr = str(data)
            if text_repr and text_repr not in ('True', 'False', 'None'):
                texts.append(text_repr)
        
        return [text for text in texts if text and len(text.strip()) > 0]
    
    def _redact_data(self, data: Any, analysis_results: List["RecognizerResult"]) -> Any:
        """Redact PII from data based on analysis results.
        
        Args:
            data: Original data
            analysis_results: List of PII detection results
            
        Returns:
            Data with PII redacted
        """
        if isinstance(data, str):
            return self._redact_text(data, analysis_results)
        elif isinstance(data, dict):
            return {key: self._redact_data(value, analysis_results) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._redact_data(item, analysis_results) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._redact_data(item, analysis_results) for item in data)
        else:
            # For other types, convert to string, redact, and return as string
            if hasattr(data, '__str__'):
                text_repr = str(data)
                if text_repr and text_repr not in ('True', 'False', 'None'):
                    return self._redact_text(text_repr, analysis_results)
            return data
    
    def _redact_text(self, text: str, analysis_results: List["RecognizerResult"]) -> str:
        """Redact PII from a text string.
        
        Args:
            text: Original text
            analysis_results: List of PII detection results for this text
            
        Returns:
            Text with PII redacted
        """
        # Filter results to only those that should be redacted (not blocked)
        relevant_results = [
            result for result in analysis_results
            if result.entity_type in self.redact_entities and result.score >= self.confidence_threshold
        ]
        
        if not relevant_results:
            return text
        
        # Use Presidio anonymizer to redact
        try:
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=relevant_results,
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE-REDACTED]"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL-REDACTED]"}),
                    "PERSON": OperatorConfig("replace", {"new_value": "[NAME-REDACTED]"}),
                    "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION-REDACTED]"}),
                }
            )
            return anonymized_result.text
        except Exception as e:
            logger.warning(f"Failed to anonymize text: {e}")
            # Fallback to simple replacement
            return "[REDACTED]"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this guardrail for trace logging."""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "block_entities": list(self.block_entities),
            "redact_entities": list(self.redact_entities),
            "confidence_threshold": self.confidence_threshold,
            "language": self.language,
            "presidio_available": HAVE_PRESIDIO
        })
        return base_metadata


# Register the PII guardrail
register_guardrail("pii", PIIGuardrail)
