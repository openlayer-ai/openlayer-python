#!/usr/bin/env python3
"""
Comprehensive example showing how to use Openlayer's trace metadata update functionality.

This example demonstrates how to set user_id, metadata, and other context information
dynamically during trace execution without having to pass them as function arguments.
"""

import os
import time
from typing import Dict, Any, List
from datetime import datetime

# Set up Openlayer configuration
os.environ["OPENLAYER_API_KEY"] = "your-api-key-here"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your-pipeline-id-here"

from openlayer.lib import trace, trace_async, update_current_trace, update_current_step


class UserSession:
    """Simulated user session with context information."""
    
    def __init__(self, user_id: str, session_id: str, preferences: Dict[str, Any]):
        self.user_id = user_id
        self.session_id = session_id
        self.preferences = preferences
        self.interaction_count = 0


class ChatApplication:
    """Example application that uses Openlayer tracing with dynamic metadata updates."""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
    
    @trace()
    def handle_user_request(self, request_text: str, session_token: str) -> str:
        """Main request handler that dynamically sets trace metadata."""
        
        # Get user session (this info isn't available as function arguments)
        user_session = self.get_user_session(session_token)
        
        # Set trace-level metadata with user context
        update_current_trace(
            name=f"chat_request_{user_session.user_id}",
            user_id=user_session.user_id,
            tags=["chat", "user_request", user_session.preferences.get("tier", "free")],
            metadata={
                "session_id": user_session.session_id,
                "user_tier": user_session.preferences.get("tier", "free"),
                "interaction_count": user_session.interaction_count,
                "timestamp": datetime.now().isoformat(),
                "request_length": len(request_text),
            },
            input={"user_request": request_text},
        )
        
        # Process the request through multiple steps
        processed_request = self.preprocess_request(request_text, user_session)
        response = self.generate_response(processed_request, user_session)
        final_response = self.postprocess_response(response, user_session)
        
        # Update trace with final output
        update_current_trace(
            output={"response": final_response, "processing_time": "0.5s"},
            metadata={
                "response_length": len(final_response),
                "processing_complete": True
            }
        )
        
        user_session.interaction_count += 1
        return final_response
    
    @trace()
    def preprocess_request(self, text: str, user_session: UserSession) -> str:
        """Preprocess user request with step-level metadata."""
        
        # Update current step with preprocessing context
        update_current_step(
            metadata={
                "preprocessing_type": "standard",
                "user_preferences_applied": True,
                "content_filter": user_session.preferences.get("content_filter", "moderate")
            },
            attributes={
                "step_category": "preprocessing",
                "user_tier": user_session.preferences.get("tier", "free")
            }
        )
        
        # Simulate preprocessing
        processed = text.strip().lower()
        if user_session.preferences.get("formal_language", False):
            processed = self.make_formal(processed)
            
        return processed
    
    @trace()
    def generate_response(self, processed_text: str, user_session: UserSession) -> str:
        """Generate AI response with model metadata."""
        
        # Set model-specific metadata
        model_version = "gpt-4" if user_session.preferences.get("tier") == "premium" else "gpt-3.5-turbo"
        
        update_current_step(
            metadata={
                "model_used": model_version,
                "temperature": 0.7,
                "max_tokens": 500,
                "response_type": "conversational"
            },
            attributes={
                "step_category": "llm_generation",
                "model_tier": user_session.preferences.get("tier", "free")
            }
        )
        
        # Simulate AI response generation
        time.sleep(0.1)  # Simulate processing time
        
        if "hello" in processed_text:
            response = f"Hello! How can I help you today, valued {user_session.preferences.get('tier', 'free')} user?"
        else:
            response = f"I understand you're asking about: {processed_text}. Let me help with that."
            
        return response
    
    @trace()
    def postprocess_response(self, response: str, user_session: UserSession) -> str:
        """Postprocess response with personalization metadata."""
        
        update_current_step(
            metadata={
                "personalization_applied": True,
                "content_filtering": user_session.preferences.get("content_filter", "moderate"),
                "user_language": user_session.preferences.get("language", "en")
            }
        )
        
        # Apply user preferences
        if user_session.preferences.get("include_emoji", False):
            response = f"😊 {response}"
            
        if user_session.preferences.get("formal_language", False):
            response = response.replace("you're", "you are").replace("can't", "cannot")
            
        return response
    
    def get_user_session(self, session_token: str) -> UserSession:
        """Get or create user session."""
        if session_token not in self.active_sessions:
            # Simulate session lookup
            self.active_sessions[session_token] = UserSession(
                user_id=f"user_{len(self.active_sessions) + 1}",
                session_id=session_token,
                preferences={
                    "tier": "premium" if session_token.startswith("premium_") else "free",
                    "content_filter": "strict",
                    "include_emoji": True,
                    "formal_language": False,
                    "language": "en"
                }
            )
        return self.active_sessions[session_token]
    
    def make_formal(self, text: str) -> str:
        """Convert text to more formal language."""
        return text.replace("can't", "cannot").replace("won't", "will not")


@trace()
def batch_processing_example():
    """Example showing batch processing with trace metadata updates."""
    
    # Set trace metadata for batch job
    update_current_trace(
        name="batch_user_requests",
        tags=["batch", "processing", "multiple_users"],
        metadata={
            "batch_size": 3,
            "processing_start": datetime.now().isoformat(),
        }
    )
    
    app = ChatApplication()
    results = []
    
    # Process multiple requests
    test_requests = [
        ("Hello there!", "premium_session_123"),
        ("What's the weather like?", "free_session_456"), 
        ("Help me with coding", "premium_session_789")
    ]
    
    for i, (request, session) in enumerate(test_requests):
        result = app.handle_user_request(request, session)
        results.append(result)
        
        # Update batch progress
        update_current_trace(
            metadata={
                "requests_processed": i + 1,
                "progress_percentage": ((i + 1) / len(test_requests)) * 100
            }
        )
    
    # Update final batch metadata
    update_current_trace(
        output={"batch_results": results, "total_processed": len(results)},
        metadata={
            "processing_complete": True,
            "processing_end": datetime.now().isoformat(),
            "success_rate": 100.0
        }
    )
    
    return results


@trace()
def error_handling_example():
    """Example showing error handling with trace metadata."""
    
    update_current_trace(
        name="error_handling_demo",
        metadata={"expected_behavior": "demonstrate error tracing"}
    )
    
    try:
        # Simulate some processing
        update_current_step(
            metadata={"processing_step": "initial_validation"}
        )
        
        # Simulate an error condition
        raise ValueError("Simulated processing error")
        
    except ValueError as e:
        # Update trace with error information
        update_current_trace(
            metadata={
                "error_occurred": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "recovery_attempted": True
            },
            output={"status": "error", "message": "Handled gracefully"}
        )
        
        return f"Error handled: {str(e)}"


@trace_async()
async def async_example():
    """Example showing async trace metadata updates."""
    
    update_current_trace(
        name="async_processing",
        metadata={"execution_mode": "async"},
        tags=["async", "demo"]
    )
    
    # Simulate async processing steps
    import asyncio
    
    update_current_step(
        metadata={"step": "async_sleep_simulation"}
    )
    await asyncio.sleep(0.1)
    
    update_current_trace(
        metadata={"async_complete": True},
        output="Async processing completed"
    )
    
    return "Async result"


def main():
    """Run all examples."""
    print("🚀 Running Openlayer Trace Metadata Update Examples")
    print("=" * 60)
    
    # Example 1: Basic chat application with user context
    print("\n1. Chat Application Example:")
    app = ChatApplication()
    
    response1 = app.handle_user_request("Hello there!", "premium_session_123")
    print(f"Response 1: {response1}")
    
    response2 = app.handle_user_request("What can you help with?", "free_session_456")
    print(f"Response 2: {response2}")
    
    # Example 2: Batch processing
    print("\n2. Batch Processing Example:")
    batch_results = batch_processing_example()
    print(f"Batch processed {len(batch_results)} requests")
    
    # Example 3: Error handling
    print("\n3. Error Handling Example:")
    error_result = error_handling_example()
    print(f"Error result: {error_result}")
    
    # Example 4: Async processing
    print("\n4. Async Processing Example:")
    import asyncio
    async_result = asyncio.run(async_example())
    print(f"Async result: {async_result}")
    
    print("\n✅ All examples completed!")
    print("\nCheck your Openlayer dashboard to see the traces with rich metadata including:")
    print("  • User IDs and session information")
    print("  • Dynamic tags and custom metadata")
    print("  • Processing steps with context")
    print("  • Error handling and recovery information")
    print("  • Async execution metadata")


if __name__ == "__main__":
    main()