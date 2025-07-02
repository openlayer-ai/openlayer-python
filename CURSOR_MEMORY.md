# Openlayer Python SDK - Cursor Memory

## Key Lessons Learned

### Duplicate Trace Issue with Async Streaming

**Problem**: When using both `@trace()` decorator and `trace_async_openai()` together, duplicate traces are generated:
1. One trace from `@trace()` decorator with function input parameters
2. Another trace from `trace_async_openai()` with the OpenAI chat completion request
3. **CRITICAL**: This breaks tests because tests are executed over both separate requests instead of one unified trace

**Root Cause**: 
- **CRITICAL**: `@trace_async()` is fundamentally broken for async generators
  - Cannot `await` an async generator function (returns generator object, not values)
  - Logs the generator object as output instead of actual streamed content
  - Records timing before generator is consumed (incorrect latency)
  - Never captures the actual yielded values
- `trace_async_openai()` creates separate traces for OpenAI calls  
- This creates conflicting/duplicate trace data that confuses test execution
- Tests expect single request but get two separate ones to validate

**Key Files**:
- `src/openlayer/lib/tracing/tracer.py` - Contains trace() and trace_async() decorators
- `src/openlayer/lib/integrations/async_openai_tracer.py` - Contains trace_async_openai()

**Solution Strategy**:
1. **RECOMMENDED**: Remove all decorators and use ONLY `trace_async_openai()` for async streaming
2. Alternative: Use ONLY `@trace_async()` decorator (but lose OpenAI-specific metrics)
3. **NEVER**: Mix decorators with client tracing - this always causes duplicates

**Confirmed Working Solutions**:

**Option 1 - No Function Tracing** (simplest):
```python
class say_hi:
    def __init__(self):
        self.openai_client = trace_async_openai(AsyncOpenAI())
    
    # ❌ Remove @trace() decorator
    async def hi(self, cur_str: str):
        # trace_async_openai handles all tracing automatically
        response = await self.openai_client.chat.completions.create(...)
        # ... rest of streaming logic
```

**Option 2 - With Function Tracing** (recommended):
```python
class say_hi:
    def __init__(self):
        self.openai_client = trace_async_openai(AsyncOpenAI())
    
    @trace_async()  # ✅ Works when function returns string
    async def hi(self, cur_str: str) -> str:  # Return complete response
        response = await self.openai_client.chat.completions.create(...)
        complete_answer = ""
        async for chunk in response:
            complete_answer += chunk.choices[0].delta.content or ""
        return complete_answer  # ✅ Return instead of yield
```

## Project Structure Insights

### Tracing Architecture
- Context variables are used to maintain trace state across async calls
- Each trace consists of steps that can be nested
- Root steps trigger data upload to Openlayer
- Streaming responses are handled differently from regular responses

### Integration Patterns
- LLM integrations wrap client methods rather than using decorators
- Each provider (OpenAI, Anthropic, etc.) has its own tracer module
- All tracers follow similar patterns but handle provider-specific details

## Best Practices Discovered

1. **Don't double-trace**: Avoid using both decorators and client tracing simultaneously
2. **Async generators need special handling**: Regular trace decorators don't work well with streaming responses  
3. **Context preservation**: Async tracing requires proper context variable management
4. **Provider-specific tracing**: Use provider-specific tracers for better integration

## Technology Stack Notes

- Uses `contextvars` for maintaining trace context across async boundaries
- Integrates with multiple LLM providers through wrapper functions
- Supports both sync and async operations
- Uses step-based tracing model with nested structure