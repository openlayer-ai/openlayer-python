# Openlayer Duplicate Traces Analysis & Solutions

## Problem Summary

Your code is generating duplicate traces in Openlayer because you're using **both** `@trace()` decorator **and** `trace_async_openai()` simultaneously. This creates two separate traces:

1. **Function-level trace** from `@trace()`: Captures the async generator object as output (not useful)
2. **OpenAI-level trace** from `trace_async_openai()`: Captures only the OpenAI response without function context

## Root Cause Analysis

### Issue 1: Async Generator Handling
The `@trace()` and `trace_async()` decorators don't properly handle async generators. They capture the generator object itself as the output, not the actual streamed content.

```python
# Current behavior in tracer.py
output = await func(*func_args, **func_kwargs)  # This returns <async_generator>
step.log(output=output)  # Logs the generator object, not content
```

### Issue 2: Double Tracing
- `@trace()` creates a user-level trace for your `hi()` function
- `trace_async_openai()` creates an OpenAI-specific trace for the API call
- Both traces are independent and don't coordinate

## Solutions

### Solution 1: Use Only Client-Level Tracing (Recommended)

Remove the `@trace()` decorator and rely solely on `trace_async_openai()`:

```python
import asyncio
from openai import AsyncOpenAI
from openlayer.lib import trace_async_openai

class say_hi:
    def __init__(self):
        self.openai_client = trace_async_openai(AsyncOpenAI())

    # Remove @trace() decorator
    async def hi(self, cur_str: str):
        messages = [
            {"role": "system", "content": "say hi !"},
            {"role": "user", "content": cur_str}
        ]
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
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
```

### Solution 2: Use Only Function-Level Tracing

Remove `trace_async_openai()` and use only `@trace_async()` with a non-streaming approach:

```python
import asyncio
from openai import AsyncOpenAI
from openlayer.lib.tracing.tracer import trace_async

class say_hi:
    def __init__(self):
        self.openai_client = AsyncOpenAI()  # No tracing wrapper

    @trace_async()
    async def hi(self, cur_str: str) -> str:  # Return string, not generator
        messages = [
            {"role": "system", "content": "say hi !"},
            {"role": "user", "content": cur_str}
        ]
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
            max_tokens=100,
            stream=True,  # Still stream internally
        )
        complete_answer = ""
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                complete_answer += delta.content
        return complete_answer  # Return complete response
```

### Solution 3: Custom Async Streaming Decorator (Advanced)

Create a specialized decorator that properly handles async generators:

```python
import asyncio
import inspect
import time
from functools import wraps
from typing import AsyncGenerator, Any
from openlayer.lib.tracing.tracer import create_step

def trace_async_streaming(
    *step_args,
    inference_pipeline_id: str = None,
    **step_kwargs,
):
    """Decorator specifically for async streaming functions."""
    
    def decorator(func):
        func_signature = inspect.signature(func)

        @wraps(func)
        async def wrapper(*func_args, **func_kwargs):
            if step_kwargs.get("name") is None:
                step_kwargs["name"] = func.__name__
                
            with create_step(
                *step_args, 
                inference_pipeline_id=inference_pipeline_id, 
                **step_kwargs
            ) as step:
                # Bind arguments
                bound = func_signature.bind(*func_args, **func_kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)
                inputs.pop("self", None)
                inputs.pop("cls", None)

                # Execute the async generator
                async_gen = func(*func_args, **func_kwargs)
                collected_output = []
                
                async def traced_generator():
                    try:
                        async for chunk in async_gen:
                            collected_output.append(str(chunk))
                            yield chunk
                    except Exception as exc:
                        step.log(metadata={"Exceptions": str(exc)})
                        raise
                    finally:
                        # Log the complete output
                        end_time = time.time()
                        latency = (end_time - step.start_time) * 1000
                        complete_output = "".join(collected_output)
                        
                        step.log(
                            inputs=inputs,
                            output=complete_output,
                            end_time=end_time,
                            latency=latency,
                        )

                return traced_generator()
        return wrapper
    return decorator

# Usage:
class say_hi:
    def __init__(self):
        self.openai_client = AsyncOpenAI()  # No trace_async_openai

    @trace_async_streaming()
    async def hi(self, cur_str: str):
        messages = [
            {"role": "system", "content": "say hi !"},
            {"role": "user", "content": cur_str}
        ]
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
            max_tokens=100,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content
```

## Recommended Approach

**Use Solution 1** (client-level tracing only) because:

1. **Simplest**: Just remove the `@trace()` decorator
2. **Most reliable**: `trace_async_openai()` is specifically designed for streaming
3. **Complete data**: Captures all OpenAI-specific metrics (tokens, cost, etc.)
4. **Less error-prone**: Avoids the complexity of handling async generators

## Why `trace_async()` Doesn't Work Well

The current `trace_async()` implementation has these limitations:

1. **Generator object capture**: It captures `<async_generator>` as output, not the actual content
2. **Timing issues**: It completes before the generator is fully consumed
3. **No streaming awareness**: It doesn't understand that the function yields values over time

## Testing Your Fix

After implementing Solution 1, you should see:
- **Single trace** per function call
- **Complete output** showing the full generated response
- **Proper timing** and token counts
- **No duplicate entries** in Openlayer

## Future Improvements

Consider contributing to the Openlayer project by:
1. Improving async generator handling in the decorators
2. Adding detection for double-tracing scenarios
3. Creating specialized decorators for streaming functions