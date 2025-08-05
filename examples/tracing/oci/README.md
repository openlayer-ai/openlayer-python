# Oracle OCI Generative AI Tracing with Openlayer

This directory contains examples for integrating Oracle Cloud Infrastructure (OCI) Generative AI with Openlayer tracing.

## Overview

Oracle OCI Generative AI is a fully managed service that provides state-of-the-art, customizable large language models (LLMs) through a single API. The Openlayer integration allows you to automatically trace and monitor all interactions with OCI Generative AI models.

## Prerequisites

1. **OCI Account**: Access to Oracle Cloud Infrastructure with Generative AI service enabled
2. **OCI Configuration**: Properly configured OCI CLI or config file
3. **Python Packages**:
   ```bash
   pip install oci openlayer
   ```

## Files

### `oci_genai_tracing.ipynb`
Comprehensive Jupyter notebook demonstrating:
- Basic non-streaming chat completions
- Streaming chat completions
- Advanced parameter configuration
- Error handling
- Multi-turn conversations

### `simple_oci_example.py`
Simple Python script for quick testing:
```bash
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..your-actual-ocid"
python simple_oci_example.py
```

## Quick Start

### 1. Configure OCI

Set up your OCI configuration using one of these methods:

**Option A: OCI CLI Setup**
```bash
oci setup config
```

**Option B: Environment Variables**
```bash
export OCI_CONFIG_FILE="~/.oci/config"
export OCI_CONFIG_PROFILE="DEFAULT"
```

**Option C: Instance Principal** (when running on OCI compute)
```python
from oci.auth.signers import InstancePrincipalsSecurityTokenSigner
config = {}
signer = InstancePrincipalsSecurityTokenSigner()
```

### 2. Basic Usage

```python
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import ChatDetails, GenericChatRequest, Message
from openlayer.lib.integrations import trace_oci_genai

# Configure OCI client
config = oci.config.from_file()
client = GenerativeAiInferenceClient(
    config=config,
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
)

# Apply Openlayer tracing
traced_client = trace_oci_genai(client)

# Make a request
chat_request = GenericChatRequest(
    messages=[Message(role="user", content="Hello, AI!")],
    model_id="cohere.command-r-plus",
    max_tokens=100,
    temperature=0.7
)

chat_details = ChatDetails(
    compartment_id="your-compartment-ocid",
    chat_request=chat_request
)

response = traced_client.chat(chat_details, inference_id="my-custom-id")
```

## Supported Models

The integration supports all OCI Generative AI models including:

### Cohere Models
- `cohere.command-r-16k` - 16K context window
- `cohere.command-r-plus` - Enhanced capabilities

### Meta Llama Models  
- `meta.llama-3.1-70b-instruct` - 70B parameters, 128K context
- `meta.llama-3.1-405b-instruct` - 405B parameters, largest available

## Features Traced

The Openlayer integration automatically captures:

- ✅ **Request Details**: Model ID, parameters, messages
- ✅ **Response Data**: Generated content, token usage
- ✅ **Performance Metrics**: Latency, time to first token (streaming)
- ✅ **Error Information**: When requests fail
- ✅ **Custom Inference IDs**: For request tracking
- ✅ **Model Parameters**: Temperature, top_p, max_tokens, etc.

## Streaming Support

Both streaming and non-streaming requests are fully supported:

```python
# Non-streaming
chat_request = GenericChatRequest(..., is_stream=False)
response = traced_client.chat(chat_details)

# Streaming  
chat_request = GenericChatRequest(..., is_stream=True)
for chunk in traced_client.chat(chat_details):
    print(chunk.data.choices[0].delta.content, end='')
```

## Configuration Options

### OCI Endpoints by Region
- **US East (Ashburn)**: `https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com`
- **US West (Phoenix)**: `https://inference.generativeai.us-phoenix-1.oci.oraclecloud.com`  
- **UK South (London)**: `https://inference.generativeai.uk-london-1.oci.oraclecloud.com`
- **Germany Central (Frankfurt)**: `https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com`

### Model Parameters
```python
GenericChatRequest(
    messages=[...],
    model_id="cohere.command-r-plus",
    max_tokens=500,           # Maximum tokens to generate
    temperature=0.7,          # Creativity (0.0-1.0)
    top_p=0.8,               # Nucleus sampling
    top_k=40,                # Top-k sampling  
    frequency_penalty=0.2,    # Reduce repetition
    presence_penalty=0.1,     # Encourage new topics
    stop=["\n\n"],           # Stop sequences
    is_stream=True           # Enable streaming
)
```

## Error Handling

The integration gracefully handles errors and traces them:

```python
try:
    response = traced_client.chat(chat_details)
except oci.exceptions.ServiceError as e:
    print(f"OCI Service Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
# All errors are automatically traced by Openlayer
```

## Best Practices

1. **Use Custom Inference IDs**: For better tracking and debugging
2. **Set Appropriate Timeouts**: For long-running requests
3. **Monitor Token Usage**: To manage costs
4. **Handle Rate Limits**: Implement retry logic
5. **Secure Credentials**: Use IAM roles and policies

## Troubleshooting

### Common Issues

**Config File Not Found**
```bash
oci setup config
```

**Authentication Errors**
```bash
oci iam user get --user-id $(oci iam user list --query 'data[0].id' --raw-output)
```

**Service Unavailable**
- Check if Generative AI is available in your region
- Verify compartment OCID is correct
- Ensure proper IAM permissions

**Import Errors**
```bash
pip install --upgrade oci openlayer
```

## Support

- **OCI Generative AI Documentation**: [docs.oracle.com](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- **Openlayer Documentation**: [openlayer.com/docs](https://openlayer.com/docs)
- **OCI Python SDK**: [github.com/oracle/oci-python-sdk](https://github.com/oracle/oci-python-sdk)

## License

This integration follows the same license as the main Openlayer project.