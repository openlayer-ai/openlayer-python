"""
Example: Using trace_openai() with Guardrails

This example demonstrates how to use guardrails with OpenAI helper functions
for automatic LLM call protection and monitoring.
"""

import os
from typing import List, Dict, Any
from openlayer.lib.tracing import tracer
from openlayer.lib.guardrails import PIIGuardrail, BlockStrategy

# Set environment variables
os.environ["OPENLAYER_API_KEY"] = "your_api_key_here"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your_pipeline_id_here"

# Mock OpenAI for demonstration (replace with real openai import)
class MockOpenAI:
    """Mock OpenAI client for demonstration purposes."""
    
    class Chat:
        class Completions:
            def create(self, **kwargs):
                messages = kwargs.get("messages", [])
                last_message = messages[-1].get("content", "") if messages else ""
                
                # Simulate different responses based on input
                if "SSN" in last_message or "social security" in last_message.lower():
                    response_text = "I can't help with social security numbers for privacy reasons."
                elif "phone" in last_message.lower():
                    response_text = "I can help you with phone-related questions."
                elif "email" in last_message.lower():
                    response_text = "I can assist with email-related topics."
                else:
                    response_text = f"Here's information about: {last_message}"
                
                class MockResponse:
                    class Choice:
                        class Message:
                            content = response_text
                        message = Message()
                    choices = [Choice()]
                    
                    class Usage:
                        prompt_tokens = len(last_message.split()) if last_message else 0
                        completion_tokens = len(response_text.split())
                        total_tokens = len(last_message.split()) + len(response_text.split())
                    usage = Usage()
                    
                    model = kwargs.get("model", "gpt-3.5-turbo")
                
                return MockResponse()
        
        completions = Completions()
    
    chat = Chat()


def main():
    """Main example demonstrating trace_openai() with guardrails."""
    print("🤖 trace_openai() with Guardrails Examples")
    print("=" * 60)
    
    # Example 1: Global Guardrails Configuration
    print("\n📋 Example 1: Global Guardrails for All LLM Calls")
    
    # Configure global guardrails - applies to ALL helper functions
    global_pii_guard = PIIGuardrail(
        name="Global PII Protection",
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE,
        block_message="Content blocked by global PII protection",
        confidence_threshold=0.7
    )
    
    # Set global configuration
    tracer.configure(
        api_key="your_api_key_here",
        inference_pipeline_id="your_pipeline_id_here", 
        guardrails=[global_pii_guard]  # Global guardrails
    )
    
    print("✅ Global guardrails configured")
    print("   All trace_openai() calls will now use PII protection")
    
    # Create traced OpenAI client
    openai_client = MockOpenAI()
    traced_client = tracer.trace_openai(openai_client)
    
    @tracer.trace()
    def chat_with_ai(user_message: str) -> str:
        """Chat function using traced OpenAI client."""
        response = traced_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    # Test with safe content
    try:
        result = chat_with_ai("Explain quantum computing")
        print(f"✅ Safe query result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with PII content (should be protected by global guardrails)
    try:
        result = chat_with_ai("My SSN is 123-45-6789, can you store this?")
        print(f"🛡️  PII protected result: {result}")
    except Exception as e:
        print(f"🚫 PII blocked: {e}")
    
    # Example 2: RAG Pipeline with Guardrails
    print("\n📋 Example 2: RAG Pipeline with Comprehensive Protection")
    
    @tracer.trace()  # This function gets its own trace step
    def retrieve_context(query: str) -> str:
        """Simulate context retrieval."""
        return f"Retrieved context for: {query}"
    
    @tracer.trace()  # This function also gets its own trace step
    def generate_answer(query: str, context: str) -> str:
        """Generate answer using traced OpenAI (with global guardrails)."""
        full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # This OpenAI call automatically uses global guardrails
        response = traced_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content
    
    @tracer.trace()  # Main RAG function
    def rag_pipeline(user_query: str) -> str:
        """Complete RAG pipeline with automatic guardrail protection."""
        context = retrieve_context(user_query)
        answer = generate_answer(user_query, context)
        return answer
    
    # Test RAG pipeline
    try:
        result = rag_pipeline("What are the benefits of renewable energy?")
        print(f"🔍 RAG result: {result}")
    except Exception as e:
        print(f"❌ RAG error: {e}")
    
    # Test RAG with sensitive data
    try:
        result = rag_pipeline("Store this credit card: 4532-1234-5678-9012")
        print(f"🛡️  RAG protected: {result}")
    except Exception as e:
        print(f"🚫 RAG blocked: {e}")
    
    # Example 3: Per-Application Guardrail Configuration
    print("\n📋 Example 3: Application-Specific Guardrails")
    
    # Create specialized guardrails for different use cases
    customer_service_guard = PIIGuardrail(
        name="Customer Service PII",
        block_entities={"US_SSN", "CREDIT_CARD"},  # Block financial info
        redact_entities={"PHONE_NUMBER", "EMAIL_ADDRESS"},  # Redact contact info
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE,
        confidence_threshold=0.6
    )
    
    # Temporarily override global guardrails for customer service
    tracer.configure(guardrails=[customer_service_guard])
    
    @tracer.trace()
    def customer_service_chat(customer_message: str) -> str:
        """Customer service chat with specialized guardrails."""
        response = traced_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a customer service representative."},
                {"role": "user", "content": customer_message}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    # Test customer service scenarios
    test_cases = [
        "I need help with my account",
        "My phone number is 555-123-4567", 
        "My credit card 4532-1234-5678-9012 was charged incorrectly"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = customer_service_chat(test_case)
            print(f"🎧 Customer service {i}: {result}")
        except Exception as e:
            print(f"🚫 Customer service {i} blocked: {e}")
    
    # Example 4: Multi-Model Setup with Different Guardrails
    print("\n📋 Example 4: Multi-Model Configuration")
    
    # Reset to no global guardrails for this example
    tracer.configure(guardrails=None)
    
    # Create different clients with different protection levels
    class AIModelManager:
        def __init__(self):
            self.openai_client = tracer.trace_openai(MockOpenAI())
            
            # Configure different guardrails per model/use case
            self.strict_guard = PIIGuardrail(
                name="Strict Protection",
                confidence_threshold=0.5,
                block_strategy=BlockStrategy.RAISE_EXCEPTION
            )
            
            self.lenient_guard = PIIGuardrail(
                name="Lenient Protection", 
                confidence_threshold=0.8,
                block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE
            )
        
        @tracer.trace()
        def strict_chat(self, message: str) -> str:
            """High-security chat with strict guardrails."""
            # Manually apply strict guardrails to this specific call
            tracer.add_chat_completion_step_to_trace(
                guardrails=[self.strict_guard],
                inputs={"prompt": [{"role": "user", "content": message}]},
                output="Strict response generated",
                name="Strict OpenAI Chat",
                provider="OpenAI",
                model="gpt-4"
            )
            return "Strict response generated"
        
        @tracer.trace()
        def lenient_chat(self, message: str) -> str:
            """Standard chat with lenient guardrails."""
            tracer.add_chat_completion_step_to_trace(
                guardrails=[self.lenient_guard],
                inputs={"prompt": [{"role": "user", "content": message}]},
                output="Lenient response generated",
                name="Lenient OpenAI Chat",
                provider="OpenAI", 
                model="gpt-3.5-turbo"
            )
            return "Lenient response generated"
    
    ai_manager = AIModelManager()
    
    test_message = "My email is user@example.com and phone is 555-0123"
    
    try:
        strict_result = ai_manager.strict_chat(test_message)
        print(f"🔒 Strict model: {strict_result}")
    except Exception as e:
        print(f"🚫 Strict model blocked: {e}")
    
    try:
        lenient_result = ai_manager.lenient_chat(test_message)
        print(f"🔓 Lenient model: {lenient_result}")
    except Exception as e:
        print(f"🚫 Lenient model blocked: {e}")
    
    # Example 5: Monitoring and Analytics
    print("\n📋 Example 5: Guardrail Analytics")
    
    # Re-enable global guardrails for monitoring
    analytics_guard = PIIGuardrail(
        name="Analytics PII",
        block_strategy=BlockStrategy.RETURN_ERROR_MESSAGE
    )
    tracer.configure(guardrails=[analytics_guard])
    
    @tracer.trace()
    def monitored_chat_session(messages: List[str]) -> List[str]:
        """Chat session with comprehensive monitoring."""
        responses = []
        
        for i, message in enumerate(messages):
            try:
                response = traced_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": message}]
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"[BLOCKED] {str(e)}")
        
        return responses
    
    # Test session with mixed content
    test_session = [
        "Hello, how are you?",
        "My SSN is 123-45-6789",
        "What's the weather like?",
        "Call me at 555-987-6543",
        "Tell me a joke"
    ]
    
    try:
        session_results = monitored_chat_session(test_session)
        print("📊 Chat session results:")
        for i, (query, response) in enumerate(zip(test_session, session_results), 1):
            print(f"   {i}. Q: {query}")
            print(f"      A: {response}")
    except Exception as e:
        print(f"❌ Session error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 trace_openai() Examples Complete!")
    print("\n📊 Key Features Demonstrated:")
    print("  • Global guardrails for all LLM calls")
    print("  • RAG pipeline protection")
    print("  • Application-specific configurations")
    print("  • Multi-model setups")
    print("  • Comprehensive monitoring and analytics")
    print("  • Automatic metadata collection")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 To run with real OpenAI:")
        print("   pip install openai")
        print("   pip install presidio-analyzer presidio-anonymizer")
        print("\n✅ This example demonstrates the API integration.")
