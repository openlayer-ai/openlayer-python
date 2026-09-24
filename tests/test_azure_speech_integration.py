"""Tests for the Azure AI Speech (``azure-cognitiveservices-speech``) tracer.

No network calls. Real ``SpeechRecognizer`` / ``SpeechSynthesizer`` objects are
constructed with a fake key (the constructors don't connect), and their
recognize/speak methods are replaced with stubs that return fake results *before*
tracing, so the tracer wraps the stub. Steps are asserted by patching the tracer
module's ``add_to_trace``, matching tests/test_google_genai_integration.py; the
credential-leak test runs the real step path and serializes the whole trace.
"""

# The Speech SDK isn't installed in the lint env, and pytest fixtures hide autouse
# functions from static analysis.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedFunction=false
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportCallIssue=false

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from datetime import timedelta
from urllib.parse import quote
from unittest.mock import patch

import pytest

speechsdk = pytest.importorskip("azure.cognitiveservices.speech")

from openlayer.lib.integrations import azure_speech_tracer as ast
from openlayer.lib.tracing.attachments import Attachment

FAKE_KEY = "FAKE-AZURE-SPEECH-KEY-0123456789"
# A minimal RIFF/WAVE header, as the SDK's default output format produces.
RIFF_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-audio"


# ------------------------------- fixtures ------------------------------- #
@pytest.fixture(autouse=True)
def _disable_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every tracer publish path off."""
    monkeypatch.setenv("OPENLAYER_DISABLE_PUBLISH", "true")
    monkeypatch.setenv("OPENLAYER_API_KEY", "fake")

    from openlayer.lib.tracing import tracer as _tracer

    monkeypatch.setattr(_tracer, "_publish", False, raising=False)


@pytest.fixture(autouse=True)
def _reset_class_patches():
    """Undo any class-level ``__init__`` patch so the idempotency marker doesn't
    leak between tests."""
    yield
    ast._unpatch_azure_speech()


# ------------------------------- helpers ------------------------------- #
def _speech_config(**overrides: Any) -> Any:
    config = speechsdk.SpeechConfig(subscription=FAKE_KEY, region="eastus")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _push_stream_audio() -> Any:
    return speechsdk.audio.AudioConfig(stream=speechsdk.audio.PushAudioInputStream())


def _make_recognizer(language: str = "en-US") -> Any:
    return speechsdk.SpeechRecognizer(
        speech_config=_speech_config(), audio_config=_push_stream_audio(), language=language
    )


def _make_synthesizer(voice: str = "en-US-JennyNeural") -> Any:
    return speechsdk.SpeechSynthesizer(
        speech_config=_speech_config(speech_synthesis_voice_name=voice), audio_config=None
    )


def _recognition_result(
    text: str = "Hello world.",
    reason: Any = None,
    offset: int = 5_000_000,
    duration: int = 12_300_000,
    no_match_details: Any = None,
    cancellation_details: Any = None,
    translations: Optional[Dict[str, str]] = None,
) -> SimpleNamespace:
    result = SimpleNamespace(
        text=text,
        reason=reason if reason is not None else speechsdk.ResultReason.RecognizedSpeech,
        result_id="res-123",
        offset=offset,
        duration=duration,
        json=json.dumps({"DisplayText": text, "RecognitionStatus": "Success"}),
        no_match_details=no_match_details,
        cancellation_details=cancellation_details,
    )
    if translations is not None:
        result.translations = translations
    return result


def _synthesis_result(
    audio: bytes = b"RIFF....WAVEfmt fake-audio",
    reason: Any = None,
    cancellation_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        audio_data=audio,
        audio_duration=timedelta(milliseconds=1500),
        reason=reason if reason is not None else speechsdk.ResultReason.SynthesizingAudioCompleted,
        result_id="syn-456",
        cancellation_details=cancellation_details,
    )


class _Future:
    """Stand-in for ``speechsdk.ResultFuture``."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def get(self) -> Any:
        return self._result


def _stub(client: Any, method: str, result: Any) -> List[Dict[str, Any]]:
    """Replace ``client.<method>`` with a stub; returns the recorded calls."""
    calls: List[Dict[str, Any]] = []

    def _impl(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return result

    setattr(client, method, _impl)
    return calls


# ------------------------------- dependency handling ------------------------------- #
class TestDependency:
    def test_module_exposes_availability_flag(self) -> None:
        assert ast.HAVE_AZURE_SPEECH is True

    def test_raises_helpful_import_error_without_sdk(self) -> None:
        with patch.object(ast, "HAVE_AZURE_SPEECH", False):
            with pytest.raises(ImportError) as exc_info:
                ast.trace_azure_speech(object())
        assert "pip install azure-cognitiveservices-speech" in str(exc_info.value)

    def test_rejects_unsupported_objects(self) -> None:
        with pytest.raises(ValueError):
            ast.trace_azure_speech(object())


# ------------------------------- recognition ------------------------------- #
class TestRecognition:
    def test_recognize_once_emits_step(self) -> None:
        recognizer = _make_recognizer(language="pt-BR")
        _stub(recognizer, "recognize_once", _recognition_result("Olá mundo."))
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            result = recognizer.recognize_once()

        assert result.text == "Olá mundo."
        mock_add.assert_called_once()
        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "Azure Speech Recognition"
        assert kwargs["output"] == "Olá mundo."
        assert kwargs["inputs"] == {"language": "pt-BR"}
        assert kwargs["model"] == "speech-to-text"
        assert kwargs["model_parameters"]["region"] == "eastus"
        assert kwargs["model_parameters"]["language"] == "pt-BR"
        assert kwargs["latency"] >= 0
        assert kwargs["start_time"] <= kwargs["end_time"]
        metadata = kwargs["metadata"]
        assert metadata["reason"] == "RecognizedSpeech"
        assert metadata["resultId"] == "res-123"
        # 100-ns ticks -> ms
        assert metadata["offsetMs"] == 500.0
        assert metadata["durationMs"] == 1230.0
        assert "RecognitionStatus" in kwargs["raw_output"]

    def test_custom_endpoint_id_is_the_model(self) -> None:
        config = _speech_config()
        config.endpoint_id = "custom-model-abc"
        recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=_push_stream_audio())
        _stub(recognizer, "recognize_once", _recognition_result())
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()

        assert mock_add.call_args.kwargs["model"] == "custom-model-abc"

    def test_no_match_is_recorded(self) -> None:
        recognizer = _make_recognizer()
        no_match = SimpleNamespace(reason=speechsdk.NoMatchReason.InitialSilenceTimeout)
        _stub(
            recognizer,
            "recognize_once",
            _recognition_result(text="", reason=speechsdk.ResultReason.NoMatch, no_match_details=no_match),
        )
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()

        metadata = mock_add.call_args.kwargs["metadata"]
        assert metadata["reason"] == "NoMatch"
        assert metadata["noMatchReason"] == "InitialSilenceTimeout"

    def test_cancellation_is_recorded_as_error(self) -> None:
        recognizer = _make_recognizer()
        cancellation = SimpleNamespace(
            reason=speechsdk.CancellationReason.Error,
            code=speechsdk.CancellationErrorCode.AuthenticationFailure,
            error_details="WebSocket upgrade failed: Authentication error (401).",
        )
        _stub(
            recognizer,
            "recognize_once",
            _recognition_result(text="", reason=speechsdk.ResultReason.Canceled, cancellation_details=cancellation),
        )
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()

        metadata = mock_add.call_args.kwargs["metadata"]
        assert metadata["reason"] == "Canceled"
        assert metadata["cancellation"] == {
            "reason": "Error",
            "errorCode": "AuthenticationFailure",
            "errorDetails": "WebSocket upgrade failed: Authentication error (401).",
        }

    def test_recognize_once_async_traces_on_get(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once_async", _Future(_recognition_result("async text")))
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            future = recognizer.recognize_once_async()
            mock_add.assert_not_called()
            result = future.get()

        assert result.text == "async text"
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["output"] == "async text"

    def test_translation_output_includes_translations(self) -> None:
        config = speechsdk.translation.SpeechTranslationConfig(subscription=FAKE_KEY, region="eastus")
        config.speech_recognition_language = "en-US"
        config.add_target_language("de")
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=config, audio_config=_push_stream_audio()
        )
        _stub(
            recognizer,
            "recognize_once",
            _recognition_result(
                "Hello.", reason=speechsdk.ResultReason.TranslatedSpeech, translations={"de": "Hallo."}
            ),
        )
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()

        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "Azure Speech Translation"
        assert kwargs["output"] == {"text": "Hello.", "translations": {"de": "Hallo."}}
        assert kwargs["inputs"]["language"] == "en-US"
        assert kwargs["inputs"]["targetLanguages"] == ["de"]

    def test_openlayer_kwargs_never_reach_the_sdk(self) -> None:
        recognizer = _make_recognizer()
        calls = _stub(recognizer, "recognize_once", _recognition_result())
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once(inference_id="abc-123", openlayer_audio=b"raw-audio")

        assert calls[0]["kwargs"] == {}
        assert mock_add.call_args.kwargs["id"] == "abc-123"

    def test_provider_has_no_spaces(self) -> None:
        """Cost lookup matches provider against a slug exactly; spaces never match."""
        assert " " not in ast.PROVIDER

        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())
        ast.trace_azure_speech(recognizer)

        from openlayer.lib.tracing import tracer as _tracer

        with patch.object(_tracer, "add_chat_completion_step_to_trace") as mock_step:
            recognizer.recognize_once()

        assert mock_step.call_args.kwargs["provider"] == "Azure_Speech"


# ------------------------------- synthesis ------------------------------- #
class TestSynthesis:
    @pytest.mark.parametrize("method,input_key", [("speak_text", "text"), ("speak_ssml", "ssml")])
    def test_speak_emits_step(self, method: str, input_key: str) -> None:
        synthesizer = _make_synthesizer()
        _stub(synthesizer, method, _synthesis_result())
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            result = getattr(synthesizer, method)("Hi there")

        assert result.result_id == "syn-456"
        kwargs = mock_add.call_args.kwargs
        assert kwargs["name"] == "Azure Speech Synthesis"
        assert kwargs["inputs"] == {input_key: "Hi there"}
        assert kwargs["model"] == "en-US-JennyNeural"
        assert kwargs["model_parameters"]["voice"] == "en-US-JennyNeural"
        assert kwargs["output"] == {"audioDurationMs": 1500.0, "audioSizeBytes": 26}
        assert kwargs["metadata"]["reason"] == "SynthesizingAudioCompleted"
        # audio bytes must never land in raw_output
        assert kwargs["raw_output"] is None

    def test_speak_text_accepts_keyword_text(self) -> None:
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_text", _synthesis_result())
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            synthesizer.speak_text(text="keyword text")

        assert mock_add.call_args.kwargs["inputs"] == {"text": "keyword text"}

    def test_speak_text_async_traces_on_get(self) -> None:
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_text_async", _Future(_synthesis_result()))
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            future = synthesizer.speak_text_async("Hi")
            mock_add.assert_not_called()
            future.get()

        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["inputs"] == {"text": "Hi"}

    def test_synthesis_cancellation_is_recorded(self) -> None:
        synthesizer = _make_synthesizer()
        cancellation = SimpleNamespace(
            reason=speechsdk.CancellationReason.Error,
            error_code=speechsdk.CancellationErrorCode.ConnectionFailure,
            error_details="Connection failed.",
        )
        _stub(
            synthesizer,
            "speak_text",
            _synthesis_result(audio=b"", reason=speechsdk.ResultReason.Canceled, cancellation_details=cancellation),
        )
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            synthesizer.speak_text("Hi")

        assert mock_add.call_args.kwargs["metadata"]["cancellation"] == {
            "reason": "Error",
            "errorCode": "ConnectionFailure",
            "errorDetails": "Connection failed.",
        }


# ------------------------------- audio capture ------------------------------- #
class TestAudioCapture:
    def test_no_audio_attached_when_uploads_disabled(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_text", _synthesis_result())
        ast.trace_azure_speech(recognizer)
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "_audio_upload_enabled", return_value=False), patch.object(
            ast, "add_to_trace"
        ) as mock_add:
            recognizer.recognize_once(openlayer_audio=b"raw-audio")
            synthesizer.speak_text("Hi")

        recognition, synthesis = (call.kwargs for call in mock_add.call_args_list)
        assert "audio" not in recognition["inputs"]
        assert "audio" not in synthesis["output"]

    def test_input_audio_attached_when_uploads_enabled(self, tmp_path: Any) -> None:
        wav = tmp_path / "caller.wav"
        wav.write_bytes(b"RIFF fake wav")
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "_audio_upload_enabled", return_value=True), patch.object(
            ast, "add_to_trace"
        ) as mock_add:
            recognizer.recognize_once(openlayer_audio=str(wav))

        audio = mock_add.call_args.kwargs["inputs"]["audio"]
        assert isinstance(audio, Attachment)
        assert audio.name == "caller.wav"
        assert audio.media_type in ("audio/wav", "audio/x-wav")
        # Read as bytes: the local path must not be recorded in the trace.
        assert audio.file_path is None
        assert audio.get_bytes() == b"RIFF fake wav"

    def test_output_audio_attached_when_uploads_enabled(self) -> None:
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_text", _synthesis_result(audio=RIFF_WAV))
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "_audio_upload_enabled", return_value=True), patch.object(
            ast, "add_to_trace"
        ) as mock_add:
            synthesizer.speak_text("Hi")

        audio = mock_add.call_args.kwargs["output"]["audio"]
        assert isinstance(audio, Attachment)
        assert audio.media_type == "audio/wav"
        assert audio.get_bytes() == RIFF_WAV
        # Never inlined into the trace JSON; the uploader sends it separately.
        assert audio.data_base64 is None

    def test_audio_shape_is_renderable_by_the_frontend(self) -> None:
        """Mirror of the frontend's ``parseMultimodalContent`` (lib/util/multimodal.ts):
        it renders a value that IS an attachment ({storageUri, mediaType}) or an
        object whose direct values are attachments. A typed ``{type, attachment}``
        item nested inside a dict is NOT found, so audio must be a bare attachment
        directly under the input/output value."""
        from openlayer.lib.tracing import tracer as _tracer
        from openlayer.lib.tracing.attachment_uploader import find_attachments

        def is_attachment_like(value: Any) -> bool:
            return isinstance(value, dict) and isinstance(value.get("storageUri"), str)

        def renders_audio(value: Any) -> bool:
            if is_attachment_like(value):
                return True
            return isinstance(value, dict) and any(
                is_attachment_like(v) and str(v.get("mediaType", "")).startswith("audio/") for v in value.values()
            )

        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_text", _synthesis_result())
        ast.trace_azure_speech(recognizer)
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "_audio_upload_enabled", return_value=True):
            with _tracer.create_step(name="voice turn") as root:
                recognizer.recognize_once(openlayer_audio=b"RIFF in")
                synthesizer.speak_text("Hi")

        recognition, synthesis = root.steps
        # The uploader must still find both attachments ...
        for step in (recognition, synthesis):
            found = find_attachments(step.inputs) + find_attachments(step.output)
            assert len(found) == 1
            found[0].storage_uri = "s3://bucket/attachments/x.wav"  # simulate upload

        # ... and the serialized shape must be what the frontend parses.
        recognition_dict, synthesis_dict = recognition.to_dict(), synthesis.to_dict()
        assert renders_audio(recognition_dict["inputs"]["audio"])  # inputs render per key
        assert renders_audio(synthesis_dict["output"])  # output renders as one value

    def test_audio_upload_setting_is_read_from_tracer_config(self) -> None:
        from openlayer.lib.tracing import tracer as _tracer

        with patch.object(_tracer, "_resolve", return_value=True) as mock_resolve:
            assert ast._audio_upload_enabled() is True
        mock_resolve.assert_called_with("attachment_upload_enabled")


# ------------------------------- credential safety ------------------------------- #
class TestAudioEncoding:
    """Synthesized audio is labeled by its actual encoding (review finding)."""

    @pytest.mark.parametrize(
        "output_format,data,media_type,extension",
        [
            ("", RIFF_WAV, "audio/wav", "wav"),
            ("audio-24khz-48kbitrate-mono-mp3", b"\xff\xf3\x44\xc4\x00", "audio/mpeg", "mp3"),
            ("audio-24khz-48kbitrate-mono-mp3", b"ID3\x04rest", "audio/mpeg", "mp3"),
            ("ogg-24khz-16bit-mono-opus", b"OggS\x00\x02", "audio/ogg", "ogg"),
            ("webm-24khz-16bit-mono-opus", b"\x1a\x45\xdf\xa3\x01", "audio/webm", "webm"),
            ("amr-wb-16000hz", b"#!AMR-WB\n\x04", "audio/amr-wb", "amr"),
            ("amr-wb-16000hz", RIFF_WAV, "audio/wav", "wav"),  # the service actually returns RIFF here
            ("raw-24khz-16bit-mono-truesilk", b"\x0e\x00\xa7\x54", "application/octet-stream", "bin"),
            ("audio-24khz-16bit-48kbps-mono-opus", b"\x68\x0b\xeb", "application/octet-stream", "bin"),
            ("g722-16khz-64kbps", b"\x7a\xde", "application/octet-stream", "bin"),
            ("", b"\x01\x02\x03", "application/octet-stream", "bin"),  # unknown: not guessed as WAV
        ],
    )
    def test_labels_by_actual_encoding(self, output_format: str, data: bytes, media_type: str, extension: str) -> None:
        _, got_type, got_ext, _ = ast._describe_synthesis_audio(data, output_format)
        assert (got_type, got_ext) == (media_type, extension)

    @pytest.mark.parametrize(
        "output_format,format_tag,sample_rate,bits,encoding",
        [
            ("raw-8khz-8bit-mono-mulaw", 7, 8000, 8, "mulaw"),
            ("raw-8khz-8bit-mono-alaw", 6, 8000, 8, "alaw"),
            ("raw-22050hz-16bit-mono-pcm", 1, 22050, 16, "pcm"),
            ("raw-48khz-16bit-mono-pcm", 1, 48000, 16, "pcm"),
        ],
    )
    def test_wraps_headerless_audio_in_wav(
        self, output_format: str, format_tag: int, sample_rate: int, bits: int, encoding: str
    ) -> None:
        import struct

        samples = b"\x01\x02\x03\x04"
        data, media_type, _, metadata = ast._describe_synthesis_audio(samples, output_format)
        assert media_type == "audio/wav"
        assert data[:4] == b"RIFF" and data[8:16] == b"WAVEfmt " and data[36:40] == b"data"
        assert struct.unpack("<H", data[20:22])[0] == format_tag
        assert struct.unpack("<I", data[24:28])[0] == sample_rate
        assert struct.unpack("<H", data[34:36])[0] == bits
        assert data[44:] == samples
        assert metadata == {
            "outputFormat": output_format,
            "encoding": encoding,
            "sampleRateHz": sample_rate,
            "wrappedInWav": True,
        }

    def test_wrapped_pcm_is_readable_by_wave(self, tmp_path: Any) -> None:
        import wave

        data, _, _, _ = ast._describe_synthesis_audio(b"\x00\x01" * 1600, "raw-16khz-16bit-mono-pcm")
        path = tmp_path / "out.wav"
        path.write_bytes(data)
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels(), wav.getnframes()) == (16000, 2, 1, 1600)

    def test_raw_pcm_that_starts_like_an_mpeg_frame_is_not_mp3(self) -> None:
        # Real raw PCM from the service starts with bytes like ff ff fe ff.
        _, media_type, _, metadata = ast._describe_synthesis_audio(b"\xff\xff\xfe\xff", "raw-16khz-16bit-mono-pcm")
        assert media_type == "audio/wav"
        assert metadata["wrappedInWav"] is True

    def test_synthesis_step_carries_the_wrapped_audio(self) -> None:
        config = _speech_config()
        config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        _stub(synthesizer, "speak_text", _synthesis_result(audio=b"\xff\x7f\x80\x00"))
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "_audio_upload_enabled", return_value=True), patch.object(
            ast, "add_to_trace"
        ) as mock_add:
            synthesizer.speak_text("Hi")

        audio = mock_add.call_args.kwargs["output"]["audio"]
        assert (audio.name, audio.media_type) == ("synthesis.wav", "audio/wav")
        assert audio.get_bytes()[:4] == b"RIFF"
        assert audio.metadata["encoding"] == "mulaw"


class TestErrorRedaction:
    """The Speech SDK embeds the endpoint URL (with its query string) in connection
    errors, so credentials in a custom endpoint must be redacted (review finding)."""

    ENDPOINT_TOKEN = "SECRET-ENDPOINT-TOKEN"

    def test_real_sdk_connection_failure_is_redacted(self) -> None:
        """Offline: the SDK fails to connect to a refused local port and reports the full URL."""
        from openlayer.lib.tracing import tracer as _tracer

        config = speechsdk.SpeechConfig(
            subscription=FAKE_KEY,
            endpoint=f"ws://127.0.0.1:1/speech/recognition/v1?token={self.ENDPOINT_TOKEN}&sig=SIGSECRET99",
        )
        stream = speechsdk.audio.PushAudioInputStream()
        stream.write(b"\x00" * 3200)
        stream.close()
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=speechsdk.audio.AudioConfig(stream=stream), language="en-US"
        )
        ast.trace_azure_speech(recognizer)

        with _tracer.create_step(name="turn") as root:
            result = recognizer.recognize_once()

        # The raw SDK text really does contain the token (the caller still gets it).
        assert self.ENDPOINT_TOKEN in result.cancellation_details.error_details
        serialized = json.dumps(root.to_dict())
        assert self.ENDPOINT_TOKEN not in serialized
        assert "SIGSECRET99" not in serialized
        assert FAKE_KEY not in serialized
        details = root.steps[0].metadata["cancellation"]["errorDetails"]
        assert "token=[REDACTED]" in details
        assert "Connection failed" in details

    def test_js_style_unable_to_contact_server_message(self) -> None:
        text = "Unable to contact server. StatusCode: 1006, wss://h.example.com/v1?token=abc123def&x=1 Reason: 401"
        assert ast._redact_secrets(text) == (
            "Unable to contact server. StatusCode: 1006, wss://h.example.com/v1?token=[REDACTED]&x=[REDACTED] Reason: 401"
        )

    def test_bearer_pairs_userinfo_and_known_secrets(self) -> None:
        assert ast._redact_secrets("Authorization: Bearer eyJ.a.b") == "Authorization: Bearer [REDACTED]"
        assert ast._redact_secrets("failed (subscription-key=abcdef123, sig: ZZZZZZ)") == (
            "failed (subscription-key=[REDACTED], sig: [REDACTED])"
        )
        assert (
            ast._redact_secrets("wss://user:pa55word@h.example.com/p") == "wss://[REDACTED]:[REDACTED]@h.example.com/p"
        )
        secret = "k3y/with+special=chars"
        assert ast._redact_secrets(f"{secret} and {quote(secret, safe='')}", [secret]) == "[REDACTED] and [REDACTED]"

    def test_ordinary_error_text_is_untouched(self) -> None:
        text = "Unsupported voice xx-XX-NoSuchVoiceNeural. websocket error code: 1007"
        assert ast._redact_secrets(text, ["short"]) == text

    def test_collect_secrets_reads_key_and_endpoint_query(self) -> None:
        config = speechsdk.SpeechConfig(
            subscription=FAKE_KEY, endpoint="wss://h.example.com/p?token=endpoint-token-1&x=1"
        )
        recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=_push_stream_audio())
        secrets = ast._collect_secrets(recognizer)
        assert FAKE_KEY in secrets and "endpoint-token-1" in secrets
        assert "1" not in secrets


class TestCredentialSafety:
    def test_key_never_appears_in_serialized_trace(self) -> None:
        """Runs the real step path (no add_to_trace mock) and serializes the trace."""
        from openlayer.lib.tracing import tracer as _tracer

        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())
        synthesizer = _make_synthesizer()
        _stub(synthesizer, "speak_ssml", _synthesis_result())
        ast.trace_azure_speech(recognizer)
        ast.trace_azure_speech(synthesizer)

        with _tracer.create_step(name="voice turn") as root:
            recognizer.recognize_once()
            synthesizer.speak_ssml("<speak>Hi</speak>")

        serialized = json.dumps(root.to_dict())
        assert len(root.steps) == 2
        assert FAKE_KEY not in serialized


# ------------------------------- sync -> async delegation ------------------------------- #
class TestSyncDelegatesToAsync:
    """The SDK implements ``recognize_once()`` as ``self.recognize_once_async().get()``
    (and ``speak_*`` likewise), so a sync call runs through BOTH wrapped methods.
    Only the async method is stubbed here; the real sync method delegates to it."""

    def test_recognize_once_emits_a_single_step(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once_async", _Future(_recognition_result("once")))
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            result = recognizer.recognize_once(inference_id="abc-123")

        assert result.text == "once"
        assert mock_add.call_count == 1, "sync call must not also trace the inner async call"
        assert mock_add.call_args.kwargs["id"] == "abc-123"

    @pytest.mark.parametrize("method,input_key", [("speak_text", "text"), ("speak_ssml", "ssml")])
    def test_speak_emits_a_single_step(self, method: str, input_key: str) -> None:
        synthesizer = _make_synthesizer()
        _stub(synthesizer, f"{method}_async", _Future(_synthesis_result()))
        ast.trace_azure_speech(synthesizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            getattr(synthesizer, method)("Hi")

        assert mock_add.call_count == 1
        assert mock_add.call_args.kwargs["inputs"] == {input_key: "Hi"}

    def test_async_call_after_sync_call_is_still_traced(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once_async", _Future(_recognition_result()))
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()
            recognizer.recognize_once_async().get()

        assert mock_add.call_count == 2


# ------------------------------- robustness ------------------------------- #
class TestRobustness:
    def test_tracing_failure_does_not_break_the_call(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result("still works"))
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace", side_effect=RuntimeError("boom")):
            result = recognizer.recognize_once()

        assert result.text == "still works"

    def test_sdk_exception_propagates(self) -> None:
        recognizer = _make_recognizer()

        def _raise() -> Any:
            raise RuntimeError("sdk failure")

        recognizer.recognize_once = _raise
        ast.trace_azure_speech(recognizer)

        with patch.object(ast, "add_to_trace") as mock_add, pytest.raises(RuntimeError, match="sdk failure"):
            recognizer.recognize_once()
        mock_add.assert_not_called()


# ------------------------------- idempotency & auto-instrument ------------------------------- #
class TestIdempotency:
    def test_double_patch_wraps_once(self) -> None:
        recognizer = _make_recognizer()
        _stub(recognizer, "recognize_once", _recognition_result())

        ast.trace_azure_speech(recognizer)
        first = recognizer.recognize_once
        assert ast.trace_azure_speech(recognizer) is recognizer
        assert recognizer.recognize_once is first

        with patch.object(ast, "add_to_trace") as mock_add:
            recognizer.recognize_once()
        assert mock_add.call_count == 1

    def test_auto_instrument_patches_new_clients(self) -> None:
        ast._patch_azure_speech()
        recognizer = _make_recognizer()
        synthesizer = _make_synthesizer()

        assert getattr(recognizer, "_openlayer_patched", False) is True
        assert getattr(synthesizer, "_openlayer_patched", False) is True

    def test_unpatch_restores_constructors(self) -> None:
        ast._patch_azure_speech()
        ast._unpatch_azure_speech()
        recognizer = _make_recognizer()
        assert getattr(recognizer, "_openlayer_patched", False) is False

    def test_registered_for_auto_instrument(self) -> None:
        from openlayer.lib.integrations._auto import _REGISTRY_BY_NAME

        spec = _REGISTRY_BY_NAME["azure_speech"]
        assert spec.probe == "azure.cognitiveservices.speech"

    def test_public_alias(self) -> None:
        from openlayer.lib import trace_azure_speech

        recognizer = _make_recognizer()
        assert trace_azure_speech(recognizer) is recognizer
        assert getattr(recognizer, "_openlayer_patched", False) is True
