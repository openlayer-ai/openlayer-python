"""
Example: Promoting inputs and outputs to top-level trace columns.

The `promote` parameter on @trace() lets you surface function inputs *and*
output fields as top-level columns in the trace data, so you can create
Openlayer tests against them (e.g. "agent_tool_call_count < 10").

Keys are resolved from **inputs first**, then from the **output** (dict,
Pydantic model, or dataclass).  Use a list to keep original names, or a dict
to alias them and avoid collisions between parent and child steps.
"""

import dataclasses
import os
from typing import Any, Dict, List

from pydantic import BaseModel

os.environ["OPENLAYER_API_KEY"] = "your-api-key-here"
os.environ["OPENLAYER_INFERENCE_PIPELINE_ID"] = "your-pipeline-id-here"

from openlayer.lib import trace
from openlayer.lib.tracing import tracer


# ---------------------------------------------------------------------------
# 1. Promote from a Pydantic model output
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    answer: str
    tool_call_count: int
    tool_names: List[str]


@trace(promote={
    "user_query": "agent_input_query",       # from input
    "tool_call_count": "agent_tool_calls",   # from output
    "tool_names": "agent_tools",             # from output
})
def run_agent(user_query: str) -> AgentResult:
    """Simulates an agent that uses tools to answer a question.

    The trace data will include three top-level columns:
      - agent_input_query  (from the `user_query` input)
      - agent_tool_calls   (from the Pydantic output's `tool_call_count`)
      - agent_tools        (from the Pydantic output's `tool_names`)
    """
    # ... agent logic would go here ...
    return AgentResult(
        answer="Paris is the capital of France.",
        tool_call_count=2,
        tool_names=["web_search", "summarize"],
    )


# ---------------------------------------------------------------------------
# 2. Promote from a dict output (list form -- no aliasing)
# ---------------------------------------------------------------------------

@trace(promote=["score", "confidence"])
def evaluate(text: str) -> Dict[str, Any]:
    """Evaluates text quality. `score` and `confidence` become top-level columns."""
    return {"score": 0.95, "confidence": 0.87, "explanation": "Well-structured."}


# ---------------------------------------------------------------------------
# 3. Promote from a dataclass output
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RetrievalResult:
    documents: List[str]
    doc_count: int
    avg_relevance: float


@trace(promote={"doc_count": "retrieval_doc_count", "avg_relevance": "retrieval_relevance"})
def retrieve(query: str) -> RetrievalResult:
    """Retrieves relevant documents. Promotes doc_count and avg_relevance."""
    return RetrievalResult(
        documents=["doc_a", "doc_b", "doc_c"],
        doc_count=3,
        avg_relevance=0.82,
    )


# ---------------------------------------------------------------------------
# 4. Nested traces -- child steps promote to the same top-level row
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    tool_call_count: int
    tool_names: List[str]
    result: str


@trace(promote={"tool_call_count": "child_tool_calls", "tool_names": "child_tools"})
def inner_agent_step(task: str) -> ToolResult:
    """A child step whose output fields are promoted to the parent trace.

    Even though this is a nested step, `promote` writes to the shared Trace
    object, so `child_tool_calls` and `child_tools` become top-level columns.
    """
    return ToolResult(
        tool_call_count=5,
        tool_names=["search", "calculator", "code_exec", "summarize", "translate"],
        result=f"Completed: {task}",
    )


@trace(promote={"user_query": "input_query"})
def orchestrator(user_query: str) -> str:
    """Parent function that delegates to a child step.

    After execution the trace will have top-level columns from *both* levels:
      - input_query       (parent input)
      - child_tool_calls  (child output)
      - child_tools       (child output)
    """
    step1 = inner_agent_step("look up facts")
    step2 = inner_agent_step("summarize findings")
    return f"{step1.result} | {step2.result}"


# ---------------------------------------------------------------------------

def main():
    print("=== 1. Promote from Pydantic output ===")
    result = run_agent("What is the capital of France?")
    print(f"  Answer: {result.answer}")
    print(f"  Tool calls: {result.tool_call_count}")
    print()

    print("=== 2. Promote from dict output (list form) ===")
    scores = evaluate("The quick brown fox.")
    print(f"  Score: {scores['score']}, Confidence: {scores['confidence']}")
    print()

    print("=== 3. Promote from dataclass output ===")
    docs = retrieve("machine learning basics")
    print(f"  Retrieved {docs.doc_count} docs, avg relevance: {docs.avg_relevance}")
    print()

    print("=== 4. Nested traces -- child promote to parent row ===")
    captured_trace = None

    @trace(promote={"user_query": "input_query"})
    def traced_orchestrator(user_query: str) -> str:
        nonlocal captured_trace
        step1 = inner_agent_step("look up facts")
        step2 = inner_agent_step("summarize findings")
        captured_trace = tracer.get_current_trace()
        return f"{step1.result} | {step2.result}"

    out = traced_orchestrator("hello world")
    print(f"  Result: {out}")
    print(f"  Trace metadata: {captured_trace.metadata}")
    assert captured_trace.metadata["input_query"] == "hello world", "parent input promoted"
    assert captured_trace.metadata["child_tool_calls"] == 5, "child output promoted"
    assert "search" in captured_trace.metadata["child_tools"], "child list output promoted"
    print("  All promoted columns verified!")
    print()

    print("Check your Openlayer dashboard -- promoted fields appear as top-level")
    print("columns you can write tests against (e.g. child_tool_calls < 10).")


if __name__ == "__main__":
    main()
