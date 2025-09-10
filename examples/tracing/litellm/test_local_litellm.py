#!/usr/bin/env python3
"""
Test script for LiteLLM tracing with local LiteLLM server.

This script demonstrates how to test the LiteLLM integration with:
1. Local LiteLLM proxy server
2. Custom API base URLs
3. Various providers and models

Prerequisites:
- LiteLLM server running locally (e.g., litellm --port 4000)
- API keys configured in environment or LiteLLM config
"""

import os
import sys
import time
from typing import Dict, Any

# Add the src directory to the path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    import litellm
    from openlayer.lib import trace_litellm
    from openlayer.lib.tracing import tracer
    from openlayer.lib.tracing.tracer import configure
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to install required dependencies:")
    print("pip install litellm openlayer")
    sys.exit(1)


class LiteLLMTester:
    """Test LiteLLM tracing with various configurations."""
    
    def __init__(self, base_url: str = None, api_key: str = None, openlayer_base_url: str = None):
        """Initialize the tester with optional custom base URL and API key."""
        self.base_url = base_url or "http://localhost:4000"
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-1234")
        self.openlayer_base_url = openlayer_base_url or "http://localhost:8080/v1"
        
        # Configure OpenLayer base URL programmatically
        configure(base_url=self.openlayer_base_url)
        print(f"🔧 OpenLayer configured for: {self.openlayer_base_url}")
        
        # Configure LiteLLM for local testing
        if base_url:
            # Set custom API base for testing with local LiteLLM server
            os.environ["LITELLM_BASE_URL"] = self.base_url
            
        # Enable tracing
        trace_litellm()
        print(f"✅ LiteLLM tracing enabled")
        print(f"🔗 LiteLLM Base URL: {self.base_url}")
        print(f"🏠 OpenLayer Base URL: {self.openlayer_base_url}")
        
    def test_basic_completion(self, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Test basic completion with tracing."""
        print(f"\n📝 Testing basic completion with {model}")
        
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"}
                ],
                temperature=0.5,
                max_tokens=50,
                api_base=self.base_url,
                api_key=self.api_key,
                inference_id=f"test-basic-{int(time.time())}"
            )
            
            result = {
                "status": "success",
                "model": response.model,
                "content": response.choices[0].message.content,
                "usage": response.usage.model_dump() if response.usage else None,
                "provider": getattr(response, '_hidden_params', {}).get('custom_llm_provider', 'unknown')
            }
            
            print(f"✅ Success: {result['content'][:100]}...")
            print(f"📊 Usage: {result['usage']}")
            print(f"🏢 Provider: {result['provider']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_streaming_completion(self, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Test streaming completion with tracing."""
        print(f"\n🌊 Testing streaming completion with {model}")
        
        try:
            stream = litellm.completion(
                model=model,
                messages=[
                    {"role": "user", "content": "Count from 1 to 5, one number per line."}
                ],
                stream=True,
                temperature=0.3,
                max_tokens=50,
                api_base=self.base_url,
                api_key=self.api_key,
                inference_id=f"test-stream-{int(time.time())}"
            )
            
            collected_content = []
            chunk_count = 0
            
            for chunk in stream:
                chunk_count += 1
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    print(content, end="", flush=True)
            
            full_content = "".join(collected_content)
            
            result = {
                "status": "success",
                "model": model,
                "content": full_content,
                "chunks": chunk_count,
                "provider": "streamed"  # Provider detection in streaming is complex
            }
            
            print(f"\n✅ Streaming complete: {chunk_count} chunks")
            print(f"📝 Content: {full_content}")
            
            return result
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_multiple_providers(self, models: list = None) -> Dict[str, Any]:
        """Test multiple providers/models with tracing."""
        if models is None:
            models = [
                "gpt-3.5-turbo",
                "claude-3-haiku-20240307", 
                "gemini-pro",
                "llama-2-7b-chat"
            ]
        
        print(f"\n🔄 Testing multiple providers: {models}")
        
        results = {}
        prompt = "What is the capital of Japan?"
        
        with tracer.create_step(
            name="Multi-Provider Test",
            metadata={"test_type": "provider_comparison", "models": models}
        ) as step:
            
            for model in models:
                try:
                    print(f"\n🧪 Testing {model}...")
                    
                    response = litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=30,
                        api_base=self.base_url,
                        api_key=self.api_key,
                        inference_id=f"multi-test-{model.replace('/', '-')}-{int(time.time())}"
                    )
                    
                    results[model] = {
                        "status": "success",
                        "content": response.choices[0].message.content,
                        "usage": response.usage.model_dump() if response.usage else None,
                        "provider": getattr(response, '_hidden_params', {}).get('custom_llm_provider', 'unknown')
                    }
                    
                    print(f"✅ {model}: {results[model]['content'][:50]}...")
                    
                except Exception as e:
                    results[model] = {"status": "error", "error": str(e)}
                    print(f"❌ {model}: {e}")
            
            step.log(results=results)
        
        return results
    
    def test_function_calling(self, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Test function calling with tracing."""
        print(f"\n🔧 Testing function calling with {model}")
        
        functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "user", "content": "What's the weather like in Tokyo?"}
                ],
                functions=functions,
                function_call="auto",
                api_base=self.base_url,
                api_key=self.api_key,
                inference_id=f"test-func-{int(time.time())}"
            )
            
            message = response.choices[0].message
            
            if message.function_call:
                result = {
                    "status": "success",
                    "function_name": message.function_call.name,
                    "arguments": message.function_call.arguments,
                    "usage": response.usage.model_dump() if response.usage else None
                }
                print(f"✅ Function called: {result['function_name']}")
                print(f"📋 Arguments: {result['arguments']}")
            else:
                result = {
                    "status": "success",
                    "content": message.content,
                    "note": "No function call triggered",
                    "usage": response.usage.model_dump() if response.usage else None
                }
                print(f"✅ Regular response: {result['content']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Function calling error: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self):
        """Run all test scenarios."""
        print("🚀 Starting comprehensive LiteLLM tracing tests")
        print("=" * 60)
        
        results = {
            "basic": self.test_basic_completion(),
            "streaming": self.test_streaming_completion(),
            "multi_provider": self.test_multiple_providers(),
            "function_calling": self.test_function_calling(),
        }
        
        print("\n" + "=" * 60)
        print("📊 Test Summary:")
        
        for test_name, result in results.items():
            status = result.get("status", "unknown")
            emoji = "✅" if status == "success" else "❌"
            print(f"{emoji} {test_name}: {status}")
        
        return results


def main():
    """Main test function."""
    print("🧪 LiteLLM Tracing Test Suite")
    print("=" * 40)
    
    # Configuration
    base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
    api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
    openlayer_base_url = os.getenv("OPENLAYER_BASE_URL", "http://localhost:8080/v1")
    
    # You can also set OpenLayer configuration
    os.environ.setdefault("OPENLAYER_API_KEY", "sk-ol-vMcEc8O_Tw52HDIF8ihNsiIlzmHLnXxC")
    os.environ.setdefault("OPENLAYER_INFERENCE_PIPELINE_ID", "efefdd4f-12ab-4343-a164-7c10d2d48d61")
    
    print(f"🔗 LiteLLM Base URL: {base_url}")
    print(f"🏠 OpenLayer Base URL: {openlayer_base_url}")
    print(f"🔑 API Key: {api_key[:8]}...")
    
    # Initialize tester
    tester = LiteLLMTester(base_url=base_url, api_key=api_key, openlayer_base_url=openlayer_base_url)
    
    # Run tests
    try:
        results = tester.run_all_tests()
        
        print("\n🎯 All tests completed!")
        print("Check your OpenLayer dashboard for detailed traces.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
