#!/usr/bin/env python3
"""
Simple Oracle OCI Generative AI tracing example.

This script demonstrates basic usage of the OCI Generative AI tracer
with Openlayer integration.

Requirements:
- pip install oci openlayer
- OCI CLI configured or OCI config file set up
- Access to OCI Generative AI service

Usage:
    python simple_oci_example.py
"""

import os
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    GenericChatRequest,
    Message,
)

# Import the Openlayer tracer
from openlayer.lib.integrations import trace_oci_genai


def main():
    """Main function to demonstrate OCI Generative AI tracing."""
    
    # Configuration - Update these values for your environment
    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "your-compartment-ocid-here")
    ENDPOINT = os.getenv("OCI_GENAI_ENDPOINT", "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com")
    
    if COMPARTMENT_ID == "your-compartment-ocid-here":
        print("❌ Please set OCI_COMPARTMENT_ID environment variable or update the script")
        print("   export OCI_COMPARTMENT_ID='ocid1.compartment.oc1..your-actual-ocid'")
        return
    
    try:
        # Load OCI configuration
        print("🔧 Loading OCI configuration...")
        config = oci.config.from_file()
        
        # Create the OCI Generative AI client
        print("🌐 Creating OCI Generative AI client...")
        client = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=ENDPOINT
        )
        
        # Apply Openlayer tracing
        print("📊 Enabling Openlayer tracing...")
        traced_client = trace_oci_genai(client)
        
        # Example 1: Non-streaming request
        print("\n🚀 Example 1: Non-streaming chat completion")
        print("-" * 50)
        
        chat_request = GenericChatRequest(
            messages=[
                Message(
                    role="user",
                    content="What are the main benefits of Oracle Cloud Infrastructure?"
                )
            ],
            model_id="cohere.command-r-plus",
            max_tokens=150,
            temperature=0.7,
            is_stream=False
        )
        
        chat_details = ChatDetails(
            compartment_id=COMPARTMENT_ID,
            chat_request=chat_request
        )
        
        response = traced_client.chat(
            chat_details,
            inference_id="simple-example-non-streaming"
        )
        
        print("✅ Response received:")
        print(f"Model: {response.data.model_id}")
        print(f"Content: {response.data.choices[0].message.content}")
        print(f"Tokens: {response.data.usage.prompt_tokens} + {response.data.usage.completion_tokens} = {response.data.usage.total_tokens}")
        
        # Example 2: Streaming request
        print("\n🚀 Example 2: Streaming chat completion")
        print("-" * 50)
        
        streaming_request = GenericChatRequest(
            messages=[
                Message(
                    role="user",
                    content="Tell me a very short story about AI and cloud computing."
                )
            ],
            model_id="meta.llama-3.1-70b-instruct",
            max_tokens=100,
            temperature=0.8,
            is_stream=True
        )
        
        streaming_details = ChatDetails(
            compartment_id=COMPARTMENT_ID,
            chat_request=streaming_request
        )
        
        print("📡 Streaming response:")
        
        streaming_response = traced_client.chat(
            streaming_details,
            inference_id="simple-example-streaming"
        )
        
        content_parts = []
        for chunk in streaming_response:
            if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                if chunk.data.choices and hasattr(chunk.data.choices[0], 'delta'):
                    delta = chunk.data.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        print(delta.content, end='', flush=True)
                        content_parts.append(delta.content)
        
        print("\n" + "-" * 50)
        print("✅ Streaming completed!")
        print(f"📊 Generated {len(''.join(content_parts))} characters")
        
        print("\n🎉 All examples completed successfully!")
        print("📊 Check your Openlayer dashboard to view the traces.")
        
    except ImportError as e:
        if "oci" in str(e):
            print("❌ OCI SDK not installed. Install with: pip install oci")
        elif "openlayer" in str(e):
            print("❌ Openlayer not installed. Install with: pip install openlayer")
        else:
            print(f"❌ Import error: {e}")
    except oci.exceptions.ConfigFileNotFound:
        print("❌ OCI config file not found. Please run 'oci setup config' or check ~/.oci/config")
    except oci.exceptions.InvalidConfig as e:
        print(f"❌ Invalid OCI configuration: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()