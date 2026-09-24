"""Module with methods used to trace the Azure AI Speech SDK.

Traces speech-to-text (``SpeechRecognizer``), speech translation
(``TranslationRecognizer``) and text-to-speech (``SpeechSynthesizer``) calls made
with ``azure-cognitiveservices-speech``.

The wrapper runs wherever the Speech SDK runs (the customer's backend), so the
Azure credential never leaves it: only an allowlist of non-secret configuration
(region, language, voice, custom endpoint ID, output format) is recorded. Audio
is attached only when ``attachment_upload_enabled`` is on in the tracer
configuration, and it is uploaded separately rather than inlined into the trace.
"""

import contextvars
import logging
import mimetypes
import time
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

try:
    import azure.cognitiveservices.speech as speechsdk

    HAVE_AZURE_SPEECH = True
except ImportError:
    HAVE_AZURE_SPEECH = False

if TYPE_CHECKING:
    import azure.cognitiveservices.speech as speechsdk

from ..tracing import tracer
from ..tracing.attachments import Attachment
from ..tracing.content import AudioContent

logger = logging.getLogger(__name__)

# No space: cost lookup matches `provider` against an llm-costs slug exactly.
PROVIDER = "Azure_Speech"

# Speech SDK offsets and durations are expressed in 100-nanosecond ticks.
_TICKS_PER_MS = 10_000

_RECOGNIZE_METHODS = ("recognize_once", "recognize_once_async")
_SYNTHESIZE_METHODS = {
    "speak_text": "text",
    "speak_text_async": "text",
    "speak_ssml": "ssml",
    "speak_ssml_async": "ssml",
}

_warned_audio_dropped = False

# True while a traced sync method runs; see ``_wrap_method``.
_in_traced_sync_call: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "openlayer_azure_speech_in_traced_sync_call", default=False
)


def trace_azure_speech(client: Any) -> Any:
    """Patch an Azure Speech recognizer or synthesizer to trace its calls.

    Supported clients and methods:

    - ``SpeechRecognizer`` / ``TranslationRecognizer``: ``recognize_once`` and
      ``recognize_once_async``.
    - ``SpeechSynthesizer``: ``speak_text``, ``speak_ssml`` and their ``_async``
      variants.

    The following information is collected for each call:

    - start_time, end_time and latency (for ``_async`` methods, until
      ``future.get()`` returns).
    - inputs: the recognition language (and translation target languages), or
      the text/SSML that was synthesized.
    - output: the recognized text (plus translations), or the synthesized audio's
      duration and size.
    - model: the custom endpoint ID or ``speech-to-text`` for recognition; the
      voice name or ``text-to-speech`` for synthesis.
    - model_parameters: region, language, voice, endpoint ID and output format.
      The subscription key and authorization token are never recorded.
    - metadata: result ID, result reason, offset/duration, and no-match or
      cancellation details (the Speech SDK reports failures as canceled results
      rather than raising).

    Two extra keyword arguments are accepted by every traced method and are never
    forwarded to the Speech SDK:

    - ``inference_id``: sets the ID of the traced step.
    - ``openlayer_audio`` (recognition only): the audio being recognized, as a
      file path, raw bytes, or an ``Attachment``. The Speech SDK does not expose
      the audio behind an ``AudioConfig``, so input audio is captured only when
      passed explicitly.

    Audio (``openlayer_audio`` and synthesized output) is attached only when
    ``attachment_upload_enabled=True`` is set via ``openlayer.lib.init()`` or
    ``configure()``; otherwise it is dropped and never leaves the process.

    Parameters
    ----------
    client : SpeechRecognizer | TranslationRecognizer | SpeechSynthesizer
        The Azure Speech client to patch.

    Returns
    -------
    SpeechRecognizer | TranslationRecognizer | SpeechSynthesizer
        The patched client.
    """
    if not HAVE_AZURE_SPEECH:
        raise ImportError(
            "azure-cognitiveservices-speech library is not installed. "
            "Please install it with: pip install azure-cognitiveservices-speech"
        )

    if getattr(client, "_openlayer_patched", False) is True:
        return client

    if isinstance(client, (speechsdk.SpeechRecognizer, speechsdk.translation.TranslationRecognizer)):
        for method in _RECOGNIZE_METHODS:
            _wrap_method(client, method, _recognition_tracer(client))
    elif isinstance(client, speechsdk.SpeechSynthesizer):
        for method, input_key in _SYNTHESIZE_METHODS.items():
            _wrap_method(client, method, _synthesis_tracer(client, input_key))
    else:
        raise ValueError(
            "Invalid client. Please provide a SpeechRecognizer, TranslationRecognizer "
            "or SpeechSynthesizer from azure.cognitiveservices.speech."
        )

    client._openlayer_patched = True
    return client


def _patch_azure_speech() -> None:
    """Patch the Speech SDK client classes' ``__init__`` so every newly-constructed
    recognizer and synthesizer is auto-traced. Idempotent."""
    if not HAVE_AZURE_SPEECH:
        return
    # pylint: disable=import-outside-toplevel
    from ._auto import _patch_class_init

    for cls in _client_classes():
        _patch_class_init(cls, trace_azure_speech)


def _unpatch_azure_speech() -> None:
    if not HAVE_AZURE_SPEECH:
        return
    # pylint: disable=import-outside-toplevel
    from ._auto import _unpatch_class_init

    for cls in _client_classes():
        _unpatch_class_init(cls)


def _client_classes() -> List[type]:
    return [
        speechsdk.SpeechRecognizer,
        speechsdk.translation.TranslationRecognizer,
        speechsdk.SpeechSynthesizer,
    ]


# ----------------------------- Wrapping ----------------------------- #


def _wrap_method(
    client: Any,
    method_name: str,
    make_step: Callable[[Any, tuple, Dict[str, Any], Any, float, float, Optional[str]], None],
) -> None:
    """Wrap ``client.<method_name>`` so a step is traced when its result is ready.

    ``_async`` methods return a ``ResultFuture``; the step is traced when
    ``future.get()`` returns, the same way the Content Understanding tracer hooks
    ``poller.result``.

    The SDK implements each sync method as ``self.<method>_async(...).get()``, so
    a traced sync call would also run through the traced async method. The sync
    wrapper sets ``_in_traced_sync_call`` so the inner async call passes through
    and the call is traced once.
    """
    original = getattr(client, method_name)
    is_async = method_name.endswith("_async")

    @wraps(original)
    def traced(*args: Any, **kwargs: Any) -> Any:
        if is_async and _in_traced_sync_call.get():
            return original(*args, **kwargs)

        inference_id = kwargs.pop("inference_id", None)
        audio = kwargs.pop("openlayer_audio", None)
        start_time = time.time()

        if not is_async:
            token = _in_traced_sync_call.set(True)
            try:
                result = original(*args, **kwargs)
            finally:
                _in_traced_sync_call.reset(token)
            _safe_trace(make_step, result, args, kwargs, audio, start_time, inference_id)
            return result

        future = original(*args, **kwargs)
        original_get = future.get

        @wraps(original_get)
        def traced_get() -> Any:
            result = original_get()
            _safe_trace(make_step, result, args, kwargs, audio, start_time, inference_id)
            return result

        future.get = traced_get
        return future

    setattr(client, method_name, traced)


def _safe_trace(
    make_step: Callable[..., None],
    result: Any,
    args: tuple,
    kwargs: Dict[str, Any],
    audio: Any,
    start_time: float,
    inference_id: Optional[str],
) -> None:
    try:
        make_step(result, args, kwargs, audio, start_time, time.time(), inference_id)
    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Failed to trace the Azure Speech call with Openlayer. %s", e)


def _recognition_tracer(client: Any) -> Callable[..., None]:
    is_translation = isinstance(client, speechsdk.translation.TranslationRecognizer)

    def make_step(
        result: Any,
        _args: tuple,
        _kwargs: Dict[str, Any],
        audio: Any,
        start_time: float,
        end_time: float,
        inference_id: Optional[str],
    ) -> None:
        config = get_model_parameters(client)
        inputs: Dict[str, Any] = {"language": config.get("language")}
        if is_translation:
            inputs["targetLanguages"] = list(getattr(client, "target_languages", None) or [])
        audio_content = _audio_input_content(audio)
        if audio_content is not None:
            inputs["audio"] = audio_content

        text = getattr(result, "text", None)
        translations = getattr(result, "translations", None)
        output: Union[str, Dict[str, Any], None] = text
        if is_translation:
            output = {"text": text, "translations": dict(translations or {})}

        add_to_trace(
            **create_trace_args(
                name="Azure Speech Translation" if is_translation else "Azure Speech Recognition",
                start_time=start_time,
                end_time=end_time,
                inputs=inputs,
                output=output,
                model=config.get("endpoint_id") or "speech-to-text",
                model_parameters=config,
                raw_output=getattr(result, "json", None) or None,
                metadata=get_result_metadata(result),
                id=inference_id,
            )
        )

    return make_step


def _synthesis_tracer(client: Any, input_key: str) -> Callable[..., None]:
    def make_step(
        result: Any,
        args: tuple,
        kwargs: Dict[str, Any],
        _audio: Any,
        start_time: float,
        end_time: float,
        inference_id: Optional[str],
    ) -> None:
        config = get_model_parameters(client)
        text = args[0] if args else kwargs.get(input_key)

        audio_data: bytes = getattr(result, "audio_data", None) or b""
        output: Dict[str, Any] = {
            "audioDurationMs": _timedelta_ms(getattr(result, "audio_duration", None)),
            "audioSizeBytes": len(audio_data),
        }
        if audio_data and _audio_upload_enabled():
            attachment = Attachment.from_bytes(
                audio_data,
                name=f"synthesis.{_extension(config.get('output_format'))}",
                media_type=_synthesis_media_type(config.get("output_format")),
            )
            output["audio"] = AudioContent(attachment=attachment)

        add_to_trace(
            **create_trace_args(
                name="Azure Speech Synthesis",
                start_time=start_time,
                end_time=end_time,
                inputs={input_key: text},
                output=output,
                model=config.get("voice") or "text-to-speech",
                model_parameters=config,
                # The result carries the audio bytes; never copy it into the trace.
                raw_output=None,
                metadata=get_result_metadata(result),
                id=inference_id,
            )
        )

    return make_step


# ----------------------------- Parsing ----------------------------- #


def get_model_parameters(client: Any) -> Dict[str, Any]:
    """Read an allowlist of non-secret settings from the client's properties.

    The property collection also holds the subscription key and authorization
    token, so it must never be dumped wholesale.
    """
    property_ids = {
        "region": speechsdk.PropertyId.SpeechServiceConnection_Region,
        "endpoint_id": speechsdk.PropertyId.SpeechServiceConnection_EndpointId,
        "language": speechsdk.PropertyId.SpeechServiceConnection_RecoLanguage,
        "recognition_mode": speechsdk.PropertyId.SpeechServiceConnection_RecoMode,
        "voice": speechsdk.PropertyId.SpeechServiceConnection_SynthVoice,
        "synthesis_language": speechsdk.PropertyId.SpeechServiceConnection_SynthLanguage,
        "output_format": speechsdk.PropertyId.SpeechServiceConnection_SynthOutputFormat,
    }
    properties = getattr(client, "properties", None)
    if properties is None:
        return {}

    parameters: Dict[str, Any] = {}
    for name, property_id in property_ids.items():
        try:
            value = properties.get_property(property_id)
        # pylint: disable=broad-except
        except Exception:
            continue
        if value:
            parameters[name] = value
    return parameters


def get_result_metadata(result: Any) -> Dict[str, Any]:
    """Extract the result ID, reason, timings and failure details."""
    metadata: Dict[str, Any] = {}

    result_id = getattr(result, "result_id", None)
    if result_id:
        metadata["resultId"] = result_id

    reason = getattr(result, "reason", None)
    if reason is not None:
        metadata["reason"] = _enum_name(reason)

    offset = getattr(result, "offset", None)
    if isinstance(offset, int):
        metadata["offsetMs"] = offset / _TICKS_PER_MS
    duration = getattr(result, "duration", None)
    if isinstance(duration, int):
        metadata["durationMs"] = duration / _TICKS_PER_MS

    if reason is not None and _enum_name(reason) == "NoMatch":
        no_match = getattr(result, "no_match_details", None)
        if no_match is not None and getattr(no_match, "reason", None) is not None:
            metadata["noMatchReason"] = _enum_name(no_match.reason)

    if reason is not None and _enum_name(reason) == "Canceled":
        cancellation = getattr(result, "cancellation_details", None)
        if cancellation is not None:
            # Recognition results expose ``code``; synthesis results ``error_code``.
            code = getattr(cancellation, "error_code", None) or getattr(cancellation, "code", None)
            metadata["cancellation"] = {
                "reason": _enum_name(getattr(cancellation, "reason", None)),
                "errorCode": _enum_name(code),
                "errorDetails": getattr(cancellation, "error_details", None),
            }

    return metadata


def _enum_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def _timedelta_ms(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value.total_seconds()) * 1000
    except AttributeError:
        return None


# ----------------------------- Audio ----------------------------- #


def _audio_upload_enabled() -> bool:
    return bool(tracer._resolve("attachment_upload_enabled"))  # pylint: disable=protected-access


def _audio_input_content(audio: Any) -> Optional[AudioContent]:
    """Build an ``AudioContent`` for explicitly-passed input audio.

    Returns None (and warns once) when attachment uploads are disabled, so audio
    never reaches Openlayer unless the tracer is configured to upload it.
    """
    global _warned_audio_dropped  # pylint: disable=global-statement

    if audio is None:
        return None
    if not _audio_upload_enabled():
        if not _warned_audio_dropped:
            logger.warning(
                "Openlayer: `openlayer_audio` was passed but attachment uploads are "
                "disabled, so the audio is not attached. Enable them with "
                "openlayer.lib.init(attachment_upload_enabled=True)."
            )
            _warned_audio_dropped = True
        return None

    if isinstance(audio, AudioContent):
        return audio
    if isinstance(audio, Attachment):
        return AudioContent(attachment=audio)
    if isinstance(audio, (bytes, bytearray)):
        return AudioContent(attachment=Attachment.from_bytes(bytes(audio), name="audio.wav", media_type="audio/wav"))
    if isinstance(audio, (str, Path)):
        # Read the bytes instead of using Attachment.from_file(), which would
        # record the absolute local path in the trace.
        path = Path(audio).expanduser()
        media_type = mimetypes.guess_type(str(path))[0] or "audio/wav"
        return AudioContent(attachment=Attachment.from_bytes(path.read_bytes(), name=path.name, media_type=media_type))

    logger.warning("Openlayer: unsupported `openlayer_audio` type %s; audio not attached.", type(audio).__name__)
    return None


def _synthesis_media_type(output_format: Optional[str]) -> str:
    """Map a Speech output format (e.g. ``audio-24khz-48kbitrate-mono-mp3``) to a
    MIME type. The default format is RIFF (WAV)."""
    fmt = (output_format or "").lower()
    if "mp3" in fmt:
        return "audio/mpeg"
    if "ogg" in fmt:
        return "audio/ogg"
    if "webm" in fmt:
        return "audio/webm"
    if fmt.startswith("raw"):
        return "audio/pcm"
    return "audio/wav"


def _extension(output_format: Optional[str]) -> str:
    return {
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "audio/pcm": "pcm",
    }.get(_synthesis_media_type(output_format), "wav")


# ----------------------------- Trace ----------------------------- #


def create_trace_args(
    name: str,
    start_time: float,
    end_time: float,
    inputs: Dict[str, Any],
    output: Any,
    model: str,
    model_parameters: Optional[Dict[str, Any]] = None,
    raw_output: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    """Returns a dictionary with the trace arguments."""
    trace_args = {
        "name": name,
        "start_time": start_time,
        "end_time": end_time,
        "latency": (end_time - start_time) * 1000,
        "inputs": inputs,
        "output": output,
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model": model,
        "model_parameters": model_parameters,
        "raw_output": raw_output,
        "metadata": metadata if metadata else {},
    }
    if id:
        trace_args["id"] = id
    return trace_args


def add_to_trace(**kwargs: Any) -> None:
    """Add an Azure Speech step to the trace."""
    tracer.add_chat_completion_step_to_trace(**kwargs, provider=PROVIDER)
