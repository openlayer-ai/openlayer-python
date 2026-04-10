"""Module with methods used to trace Azure Content Understanding."""

import logging
import mimetypes
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from urllib.parse import urlparse
from pathlib import Path

try:
    from azure.ai.contentunderstanding import ContentUnderstandingClient

    HAVE_AZURE_CONTENT_UNDERSTANDING = True
except ImportError:
    HAVE_AZURE_CONTENT_UNDERSTANDING = False

if TYPE_CHECKING:
    from azure.ai.contentunderstanding import ContentUnderstandingClient
    from azure.ai.contentunderstanding.models import AnalysisInput, AnalysisResult

from ..tracing import tracer
from ..tracing.attachments import Attachment
from ..tracing.content import AudioContent, FileContent, ImageContent

logger = logging.getLogger(__name__)


def trace_azure_content_understanding(
    client: "ContentUnderstandingClient",
) -> "ContentUnderstandingClient":
    """Patch the Azure Content Understanding client to trace analyses.

    The following information is collected for each analysis:
    - start_time: The time when the analysis was requested.
    - end_time: The time when the analysis result was retrieved.
    - latency: The total time from request to result (including polling).
    - model: The analyzer ID used to perform the analysis.
    - model_parameters: The API version used.
    - raw_output: The raw analysis result dict.
    - inputs: The inputs provided to the analysis (file URLs or binary indicators).
    - metadata: Additional metadata (analyzer_id, api_version, created_at).

    Parameters
    ----------
    client : ContentUnderstandingClient
        The Azure Content Understanding client to patch.

    Returns
    -------
    ContentUnderstandingClient
        The patched client.
    """
    if not HAVE_AZURE_CONTENT_UNDERSTANDING:
        raise ImportError(
            "azure-ai-contentunderstanding library is not installed. "
            "Please install it with: pip install azure-ai-contentunderstanding"
        )

    begin_analyze_func = client.begin_analyze

    @wraps(begin_analyze_func)
    def traced_begin_analyze(*args, **kwargs):
        inference_id = kwargs.pop("inference_id", None)
        start_time = time.time()

        poller = begin_analyze_func(*args, **kwargs)

        original_result = poller.result

        def traced_result(*result_args, **result_kwargs):
            result = original_result(*result_args, **result_kwargs)
            end_time = time.time()

            try:
                analyzer_id = kwargs.get("analyzer_id", args[0] if args else "unknown")
                analysis_inputs = kwargs.get("inputs", args[1] if len(args) > 1 else [])

                usage = _extract_usage_from_poller(poller)
                usage_info = _parse_usage(usage)

                output_data = parse_output(result)
                trace_args = create_trace_args(
                    end_time=end_time,
                    inputs=parse_inputs(analyzer_id, analysis_inputs),
                    output=output_data,
                    latency=(end_time - start_time) * 1000,
                    model=usage_info["model"] or analyzer_id,
                    model_parameters=get_model_parameters(result),
                    raw_output=result.as_dict(),
                    id=inference_id,
                    metadata=get_metadata(result, usage),
                    tokens=usage_info["total_tokens"],
                    prompt_tokens=usage_info["prompt_tokens"],
                    completion_tokens=usage_info["completion_tokens"],
                )
                add_to_trace(**trace_args)

            # pylint: disable=broad-except
            except Exception as e:
                logger.error(
                    "Failed to trace the Azure Content Understanding analysis with Openlayer. %s",
                    e,
                )

            return result

        poller.result = traced_result
        return poller

    client.begin_analyze = traced_begin_analyze
    return client


def _extract_usage_from_poller(poller: Any) -> Dict[str, Any]:
    """Extract UsageDetails from the LRO poller's final pipeline response.

    After poller.result() returns, the underlying polling method stores the last
    HTTP response in _pipeline_response. The full ContentAnalyzerAnalyzeOperationStatus
    JSON contains a ``usage`` field that is discarded by the SDK's deserialization
    (which only extracts the nested ``result`` field). We read it here directly.
    """
    try:
        full_json = poller.polling_method()._pipeline_response.http_response.json()
        usage = full_json.get("usage") or {}
        logger.debug("Azure Content Understanding usage data: %s", usage)
        return usage
    # pylint: disable=broad-except
    except Exception as e:
        logger.debug("Could not extract usage from poller: %s", e)
        return {}


def _parse_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a UsageDetails dict into model name and token counts.

    The ``tokens`` field is a dict keyed as ``"<model_name>/<token_type>"``
    (e.g., ``"gpt-4.1/input"``, ``"gpt-4.1/output"``).

    Returns a dict with keys: model, prompt_tokens, completion_tokens,
    contextualization_tokens, total_tokens.
    """
    tokens_by_type: Dict[str, int] = usage.get("tokens") or {}
    contextualization_tokens: int = usage.get("contextualizationTokens") or 0

    prompt_tokens = 0
    completion_tokens = 0
    completion_model = None

    for key, count in tokens_by_type.items():
        if key.endswith("-input"):
            prompt_tokens += count
        elif key.endswith("-output"):
            completion_tokens += count
            if completion_model is None and count > 0:
                completion_model = key[: -len("-output")]

    return {
        "model": completion_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "contextualization_tokens": contextualization_tokens,
        "total_tokens": prompt_tokens + completion_tokens + contextualization_tokens,
    }


def parse_inputs(
    analyzer_id: str,
    analysis_inputs: List[Any],
) -> Dict[str, Any]:
    """Parse the inputs provided to begin_analyze into a traceable dict.

    Each AnalysisInput is converted into a typed content object (FileContent,
    ImageContent, or AudioContent) backed by an Attachment, so that Openlayer
    can display and store the source file alongside the trace step.

    Parameters
    ----------
    analyzer_id : str
        The ID of the analyzer used.
    analysis_inputs : list
        The list of AnalysisInput objects.

    Returns
    -------
    Dict[str, Any]
        A dict with analyzer_id and a list of content objects.
    """
    files = []
    for item in analysis_inputs:
        content = _analysis_input_to_content(item)
        if content is not None:
            files.append(content)

    return {
        "analyzer_id": analyzer_id,
        "files": files,
    }


def _analysis_input_to_content(
    item: "AnalysisInput",
) -> Optional[Union[FileContent, ImageContent, AudioContent]]:
    """Convert a single AnalysisInput into a typed content object with an Attachment.

    Uses the input's own mime_type when available; otherwise guesses from the URL.
    """
    mime_type: Optional[str] = getattr(item, "mime_type", None)
    display_name: Optional[str] = getattr(item, "name", None)

    url = getattr(item, "url", None)
    data = getattr(item, "data", None)

    if url:
        if not mime_type:
            mime_type = mimetypes.guess_type(url)[0] or "application/octet-stream"
        if not display_name:
            display_name = Path(urlparse(url).path).name or "file"
        attachment = Attachment.from_url(url, name=display_name, media_type=mime_type)
    elif data:
        if not mime_type:
            mime_type = "application/octet-stream"
        display_name = display_name or "file"
        attachment = Attachment.from_bytes(data, name=display_name, media_type=mime_type)
    else:
        return None

    if mime_type.startswith("image/"):
        return ImageContent(attachment=attachment)
    if mime_type.startswith("audio/") or mime_type.startswith("video/"):
        return AudioContent(attachment=attachment)
    return FileContent(attachment=attachment)


def parse_output(
    result: "AnalysisResult",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse the AnalysisResult into a human-readable output dict.

    Parameters
    ----------
    result : AnalysisResult
        The analysis result returned by poller.result().

    Returns
    -------
    Union[Dict[str, Any], List[Dict[str, Any]]]
        A dict (or list of dicts) with the markdown and extracted fields
        for each content item in the result.
    """
    contents = getattr(result, "contents", None) or []
    parsed = []
    for content in contents:
        item = {}
        if hasattr(content, "markdown") and content.markdown:
            item["markdown"] = content.markdown
        if hasattr(content, "fields") and content.fields:
            item["fields"] = _simplify_fields(content.fields)
        parsed.append(item)

    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _simplify_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested field structure into a simpler key→value mapping."""
    simplified = {}
    for key, field in fields.items():
        if hasattr(field, "value_string") and field.value_string is not None:
            simplified[key] = field.value_string
        elif hasattr(field, "value_number") and field.value_number is not None:
            simplified[key] = field.value_number
        elif hasattr(field, "value_object") and field.value_object is not None:
            simplified[key] = _simplify_fields(field.value_object)
        elif hasattr(field, "value_array") and field.value_array is not None:
            simplified[key] = [
                _simplify_fields(item) if hasattr(item, "value_object") else item
                for item in field.value_array
            ]
        elif hasattr(field, "as_dict"):
            simplified[key] = field.as_dict()
        else:
            simplified[key] = str(field)
    return simplified


def get_model_parameters(result: "AnalysisResult") -> Dict[str, Any]:
    """Extract model parameters from the analysis result."""
    return {
        "api_version": getattr(result, "api_version", None),
    }


def get_metadata(result: "AnalysisResult", usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract metadata from the analysis result and usage details."""
    metadata = {}
    if hasattr(result, "analyzer_id") and result.analyzer_id:
        metadata["analyzer_id"] = result.analyzer_id
    if hasattr(result, "api_version") and result.api_version:
        metadata["api_version"] = result.api_version
    if hasattr(result, "created_at") and result.created_at:
        metadata["created_at"] = str(result.created_at)
    if hasattr(result, "warnings") and result.warnings:
        metadata["warnings"] = [str(w) for w in result.warnings]
    if usage:
        contextualization_tokens = usage.get("contextualizationTokens")
        if contextualization_tokens is not None:
            metadata["contextualization_tokens"] = contextualization_tokens
        for key in ("documentPagesMinimal", "documentPagesBasic", "documentPagesStandard",
                    "audioHours", "videoHours"):
            if usage.get(key) is not None:
                metadata[key] = usage[key]
    return metadata


def create_trace_args(
    end_time: float,
    inputs: Dict,
    output: Union[str, Dict, List],
    latency: float,
    model: str,
    tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_parameters: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    raw_output: Optional[Dict] = None,
    id: Optional[str] = None,
) -> Dict:
    """Returns a dictionary with the trace arguments."""
    trace_args = {
        "end_time": end_time,
        "inputs": inputs,
        "output": output,
        "latency": latency,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model": model,
        "model_parameters": model_parameters,
        "raw_output": raw_output,
        "metadata": metadata if metadata else {},
    }
    if id:
        trace_args["id"] = id
    return trace_args


def add_to_trace(**kwargs) -> None:
    """Add an Azure Content Understanding analysis step to the trace."""
    tracer.add_chat_completion_step_to_trace(
        **kwargs,
        name="Azure Content Understanding Analysis",
        provider="Azure OpenAI",
    )
