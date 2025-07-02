# Fixed Async Streaming Example

## The Problem
Using both `@trace()` decorator and `trace_async_openai()` creates duplicate traces that break tests.

## The Solution
Use **ONLY** `trace_async_openai()` - remove all decorators:

```python
import asyncio
from openai import AsyncOpenAI
from openlayer.lib import trace_async_openai

class say_hi:
    def __init__(self):
        self.openai_client = trace_async_openai(AsyncOpenAI())

    # ❌ Remove @trace() or @trace_async() decorators
    async def hi(self, cur_str: str):
        messages = [
            {
                "role": "system",
                "content": "say hi !",
            },
            {"role": "user", "content": cur_str}
        ]
        temperature = 0
        
        # This single call will be properly traced by trace_async_openai
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=temperature,
            max_tokens=100,
            stream=True,
        )
        
        complete_answer = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                chunk_content = delta.content
                complete_answer += chunk_content
                yield chunk_content

# Usage remains the same
obj_ = say_hi()

print("Streaming response:")
async for chunk in obj_.hi("hi you are an async assistant"):
    print(chunk, end="")
print("\nStreaming finished.")
```

## What This Fixes
- ✅ **Single trace only** - no more duplicate requests
- ✅ **Tests work properly** - only one request to test against
- ✅ **Complete tracing info** - input, output, tokens, cost, timing all captured
- ✅ **Proper async streaming** - chunks yielded correctly

## Why This Works
The `trace_async_openai()` wrapper is specifically designed for async OpenAI calls and:
- Automatically captures function input (cur_str parameter)
- Traces the complete streaming response 
- Includes OpenAI-specific metrics (tokens, cost, model)
- Maintains proper async context
- **Generates only ONE trace entry**

## Key Insight
Your sync version works because you're not double-tracing. Apply the same principle to async: **use only one tracing method, not both together**.