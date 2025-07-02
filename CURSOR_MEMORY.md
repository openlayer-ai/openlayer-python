# Openlayer Python SDK - Cursor Memory

## Key Lessons Learned

### Duplicate Trace Issue with Async Streaming

**Problem**: When using both `@trace()` decorator and `trace_async_openai()` together, duplicate traces are generated:
1. One trace from `@trace()` decorator showing async_generator as output (incomplete)
2. Another trace from `trace_async_openai()` showing only the OpenAI response (missing function context)

**Root Cause**: 
- The `@trace()` and `trace_async()` decorators don't handle async generators properly
- They capture the generator object itself as output, not the streamed content
- `trace_async_openai()` creates separate traces for OpenAI calls
- This creates conflicting/duplicate trace data

**Key Files**:
- `src/openlayer/lib/tracing/tracer.py` - Contains trace() and trace_async() decorators
- `src/openlayer/lib/integrations/async_openai_tracer.py` - Contains trace_async_openai()

**Solution Strategy**:
1. Either use ONLY `@trace_async()` decorator OR ONLY `trace_async_openai()`, not both
2. Modify decorators to properly handle async generators by consuming them
3. Create a specialized decorator for async streaming functions

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