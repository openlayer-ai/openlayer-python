"""Attachment abstraction for unstructured data in traces.

This module provides the Attachment class for representing binary/media content
(images, audio, video, PDFs, etc.) that is associated with trace steps but
stored separately from structured trace data.
"""

import base64
import hashlib
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """Unstructured data attached to a trace step.

    Attachments represent binary/media content that needs special handling
    for storage and display. The attachment holds references to where the
    data is stored, not the raw data itself (except for small inline data).

    Examples of attachments:
        - Audio files (input to STT, output from TTS)
        - Images (input to vision models, output from image generation)
        - PDFs and documents
        - Debug artifacts and screenshots

    Attributes:
        id: Unique identifier for this attachment.
        name: Human-readable name (e.g., "input_audio.wav").
        media_type: MIME type (e.g., "audio/wav", "image/png").
        storage_uri: Openlayer storage reference (set after upload).
        url: External URL reference.
        file_path: Local file path (for development/debugging).
        data_base64: Inline base64 data (for small attachments).
        size_bytes: Size of the attachment in bytes.
        checksum_md5: MD5 checksum for integrity/deduplication.
        metadata: Extensible metadata dict (duration, dimensions, etc.).
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""

    # Content type
    media_type: str = "application/octet-stream"

    # Storage references (priority: storage_uri > url > file_path)
    storage_uri: Optional[str] = None  # Openlayer managed storage
    url: Optional[str] = None  # External URL
    file_path: Optional[str] = None  # Local file path

    # Inline data (for small attachments when configured)
    data_base64: Optional[str] = None

    # Size and integrity
    size_bytes: Optional[int] = None
    checksum_md5: Optional[str] = None

    # Extensible metadata
    # For audio: duration_seconds, sample_rate, channels
    # For images: width, height
    # For documents: page_count
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal: holds pending bytes for upload (not serialized)
    _pending_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize attachment for JSON transport.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result = {
            "id": self.id,
            "name": self.name,
            "mediaType": self.media_type,
        }

        # Add references (only non-None)
        if self.storage_uri:
            result["storageUri"] = self.storage_uri
        if self.url:
            result["url"] = self.url
        if self.file_path:
            result["filePath"] = self.file_path
        if self.data_base64:
            result["dataBase64"] = self.data_base64

        # Add optional fields
        if self.size_bytes is not None:
            result["sizeBytes"] = self.size_bytes
        if self.checksum_md5:
            result["checksumMd5"] = self.checksum_md5
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attachment":
        """Deserialize attachment from JSON.

        Args:
            data: Dictionary representation of an attachment.

        Returns:
            Attachment instance.
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            media_type=data.get("mediaType", "application/octet-stream"),
            storage_uri=data.get("storageUri"),
            url=data.get("url"),
            file_path=data.get("filePath"),
            data_base64=data.get("dataBase64"),
            size_bytes=data.get("sizeBytes"),
            checksum_md5=data.get("checksumMd5"),
            metadata=data.get("metadata", {}),
        )

    def is_uploaded(self) -> bool:
        """Check if attachment has been uploaded to Openlayer storage.

        Returns:
            True if storage_uri is set.
        """
        return self.storage_uri is not None

    def has_data(self) -> bool:
        """Check if attachment has data available for upload.

        Returns:
            True if data is available via file_path, data_base64, or _pending_bytes.
        """
        if self._pending_bytes is not None:
            return True
        if self.data_base64 is not None:
            return True
        if self.file_path:
            return Path(self.file_path).exists()
        return False

    def is_valid(self) -> bool:
        """Check if attachment is valid and should be included in trace data.

        An attachment is valid if it has been uploaded, has an external URL,
        or has data available for upload.

        Returns:
            True if the attachment should be serialized.
        """
        return self.is_uploaded() or self.url is not None or self.has_data()

    def get_reference(self) -> Optional[str]:
        """Get the best available reference URI.

        Returns:
            The most reliable reference to this attachment's data.
        """
        return self.storage_uri or self.url or self.file_path

    # ----- Factory methods -----

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        media_type: Optional[str] = None,
        extract_metadata: bool = True,
    ) -> "Attachment":
        """Create an attachment from a local file.

        Args:
            file_path: Path to the file.
            name: Display name (defaults to filename).
            media_type: MIME type (auto-detected if not provided).
            extract_metadata: Whether to extract metadata like duration/dimensions.

        Returns:
            Attachment instance with file reference.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            logger.warning(
                "Attachment file does not exist: %s. "
                "The attachment will be ignored.",
                path,
            )

        if media_type is None:
            media_type = (
                mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            )

        attachment = cls(
            name=name or path.name,
            media_type=media_type,
            file_path=str(path.absolute()),
        )

        if path.exists():
            stat = path.stat()
            attachment.size_bytes = stat.st_size

            # Compute checksum for deduplication
            try:
                with open(path, "rb") as f:
                    attachment.checksum_md5 = hashlib.md5(f.read()).hexdigest()
            except Exception as e:
                logger.debug("Could not compute checksum for %s: %s", path, e)

            if extract_metadata:
                attachment._extract_metadata_from_file(path)

        return attachment

    @classmethod
    def from_url(
        cls,
        url: str,
        name: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> "Attachment":
        """Create an attachment from an external URL.

        Args:
            url: The URL pointing to the media.
            name: Display name (extracted from URL if not provided).
            media_type: MIME type (guessed from URL if not provided).

        Returns:
            Attachment instance with URL reference.
        """
        if media_type is None:
            media_type = mimetypes.guess_type(url)[0] or "application/octet-stream"

        # Extract filename from URL if name not provided
        if name is None:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            name = Path(parsed.path).name or "attachment"

        return cls(
            name=name,
            media_type=media_type,
            url=url,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        name: str,
        media_type: str,
        inline: bool = False,
    ) -> "Attachment":
        """Create an attachment from raw bytes.

        Args:
            data: The binary data.
            name: Display name for the attachment.
            media_type: MIME type of the data.
            inline: If True, store data as base64 inline (for small attachments).

        Returns:
            Attachment instance with data ready for upload.
        """
        attachment = cls(
            name=name,
            media_type=media_type,
            size_bytes=len(data),
            checksum_md5=hashlib.md5(data).hexdigest(),
        )

        if inline:
            attachment.data_base64 = base64.b64encode(data).decode("utf-8")
        else:
            # Store bytes for later upload
            attachment._pending_bytes = data

        return attachment

    @classmethod
    def from_base64(
        cls,
        data_base64: str,
        name: str,
        media_type: str,
    ) -> "Attachment":
        """Create an attachment from base64-encoded data.

        Args:
            data_base64: Base64-encoded binary data.
            name: Display name for the attachment.
            media_type: MIME type of the data.

        Returns:
            Attachment instance with inline data.
        """
        # Decode to get size and checksum
        try:
            decoded = base64.b64decode(data_base64)
            size_bytes = len(decoded)
            checksum_md5 = hashlib.md5(decoded).hexdigest()
        except Exception:
            size_bytes = None
            checksum_md5 = None

        return cls(
            name=name,
            media_type=media_type,
            data_base64=data_base64,
            size_bytes=size_bytes,
            checksum_md5=checksum_md5,
        )

    def get_bytes(self) -> Optional[bytes]:
        """Get the binary data for this attachment.

        Returns:
            The attachment data as bytes, or None if not available locally.
        """
        # Check for pending bytes first
        if self._pending_bytes is not None:
            return self._pending_bytes

        # Check for inline data
        if self.data_base64:
            try:
                return base64.b64decode(self.data_base64)
            except Exception as e:
                logger.error("Failed to decode base64 data: %s", e)
                return None

        # Read from file
        if self.file_path:
            path = Path(self.file_path)
            if path.exists():
                try:
                    return path.read_bytes()
                except Exception as e:
                    logger.error("Failed to read file %s: %s", path, e)
                    return None

        return None

    def _extract_metadata_from_file(self, path: Path) -> None:
        """Extract metadata from file based on media type.

        Args:
            path: Path to the file.
        """
        # TODO: Implement metadata extraction from file
        pass

    def __repr__(self) -> str:
        """String representation of the attachment."""
        ref = self.storage_uri or self.url or self.file_path or "(no reference)"
        return (
            f"Attachment(name={self.name!r}, "
            f"media_type={self.media_type!r}, ref={ref!r})"
        )
