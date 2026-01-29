"""Attachment upload handling for traces.

This module provides functionality to upload attachments to Openlayer storage
using the existing upload infrastructure from openlayer.lib.data._upload.
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..._client import Openlayer
    from .steps import Step
    from .traces import Trace

from ..data._upload import STORAGE, StorageType, upload_bytes
from .attachments import Attachment

logger = logging.getLogger(__name__)


def find_attachments(data: Any) -> List[Attachment]:
    """Recursively find all Attachment objects in a data structure.

    This function traverses dicts, lists, tuples, and objects with
    'attachment' attributes to find any Attachment objects embedded within.

    Args:
        data: Any data structure that may contain Attachment objects.

    Returns:
        A list of all Attachment objects found.
    """
    if isinstance(data, Attachment):
        return [data]
    elif isinstance(data, dict):
        result = []
        for value in data.values():
            result.extend(find_attachments(value))
        return result
    elif isinstance(data, (list, tuple)):
        result = []
        for item in data:
            result.extend(find_attachments(item))
        return result
    elif hasattr(data, "attachment"):
        # Handle ContentItem objects (ImageContent, AudioContent, etc.)
        attachment = getattr(data, "attachment")
        if isinstance(attachment, Attachment):
            return [attachment]
    return []


class AttachmentUploader:
    """Handles uploading attachments to Openlayer storage.

    This class manages the upload of attachment data to Openlayer's storage
    backend using the same infrastructure as other data uploads. It supports
    S3, GCS, Azure, and local storage backends.
    """

    def __init__(self, client: "Openlayer", storage: StorageType = STORAGE):
        """Initialize the attachment uploader.

        Args:
            client: The Openlayer client instance.
            storage: Storage type override. Defaults to the global STORAGE setting.
        """
        self._client = client
        self._storage = storage
        self._storage_uri_cache: Dict[str, str] = {}  # checksum -> storage_uri

    def upload_attachment(self, attachment: "Attachment") -> "Attachment":
        """Upload a single attachment if needed.

        If the attachment already has a storage_uri or external URL,
        it is returned as-is. Otherwise, the attachment data is uploaded
        to Openlayer storage and the storage_uri is set.

        Args:
            attachment: The attachment to upload.

        Returns:
            The attachment with storage_uri populated (if upload was needed).
        """
        # Skip if already uploaded
        if attachment.is_uploaded():
            logger.debug("Attachment %s already uploaded", attachment.name)
            return attachment

        # Skip if it has an external URL (no upload needed)
        if attachment.url:
            logger.debug(
                "Attachment %s has external URL, skipping upload", attachment.name
            )
            return attachment

        # Check if we have data to upload
        if not attachment.has_data():
            logger.warning(
                "Attachment %s has no data available for upload",
                attachment.name,
            )
            return attachment

        # Check cache by checksum for deduplication
        if (
            attachment.checksum_md5
            and attachment.checksum_md5 in self._storage_uri_cache
        ):
            attachment.storage_uri = self._storage_uri_cache[attachment.checksum_md5]
            logger.debug(
                "Using cached storage_uri for attachment %s (checksum: %s)",
                attachment.name,
                attachment.checksum_md5,
            )
            return attachment

        try:
            # Generate a unique object name for storage
            object_name = self._generate_object_name(attachment)

            # Get presigned URL from Openlayer API
            presigned_response = self._client.storage.presigned_url.create(
                object_name=object_name,
            )

            # Get the bytes to upload
            file_bytes = attachment.get_bytes()
            if file_bytes is None:
                raise ValueError(f"No data available for attachment {attachment.name}")

            # Upload using the shared upload function
            upload_bytes(
                storage=self._storage,
                url=presigned_response.url,
                data=file_bytes,
                object_name=object_name,
                content_type=attachment.media_type,
                fields=(
                    dict(presigned_response.fields)
                    if presigned_response.fields
                    else None
                ),
            )

            # Set the storage URI on the attachment
            attachment.storage_uri = presigned_response.storage_uri

            # Cache for deduplication
            if attachment.checksum_md5:
                self._storage_uri_cache[attachment.checksum_md5] = (
                    attachment.storage_uri
                )

            # Clear data after upload (no longer needed, avoid duplicating in JSON)
            attachment._pending_bytes = None
            attachment.data_base64 = None

            logger.debug(
                "Uploaded attachment %s to %s",
                attachment.name,
                attachment.storage_uri,
            )

        except Exception as e:
            logger.error(
                "Failed to upload attachment %s: %s",
                attachment.name,
                e,
            )

        return attachment

    def _generate_object_name(self, attachment: "Attachment") -> str:
        """Generate a unique object name for storage.

        Args:
            attachment: The attachment to generate a name for.

        Returns:
            A unique object name for storage.
        """
        # Use checksum if available for deduplication, otherwise UUID
        unique_id = attachment.checksum_md5 or str(uuid.uuid4())

        # Extract extension from name or media type
        extension = ""
        if "." in attachment.name:
            extension = attachment.name.rsplit(".", 1)[-1]
        elif "/" in attachment.media_type:
            # Try to get extension from media type (e.g., "image/png" -> "png")
            subtype = attachment.media_type.split("/")[-1]
            # Handle special cases
            extension_map = {
                "mpeg": "mp3",
                "jpeg": "jpg",
                "x-wav": "wav",
                "x-m4a": "m4a",
            }
            extension = extension_map.get(subtype, subtype)

        if extension:
            return f"attachments/{unique_id}.{extension}"
        return f"attachments/{unique_id}"

    def upload_trace_attachments(self, trace: "Trace") -> int:
        """Upload all attachments in a trace.

        Recursively processes all steps in the trace and uploads any
        attachments that have data available. This includes attachments
        in the step's attachments list, as well as any Attachment objects
        embedded in the step's inputs or outputs.

        Args:
            trace: The trace containing steps with attachments.

        Returns:
            The number of attachments uploaded.
        """
        seen_ids: set = set()

        def process_step(step: "Step") -> int:
            """Process a step and return the number of attachments uploaded."""
            step_upload_count = 0

            # Collect attachments from all sources
            all_attachments: List[Attachment] = list(step.attachments)
            all_attachments.extend(find_attachments(step.inputs))
            all_attachments.extend(find_attachments(step.output))

            # Process each attachment (deduplicate by ID)
            for attachment in all_attachments:
                if attachment.id in seen_ids:
                    continue
                seen_ids.add(attachment.id)

                if not attachment.is_uploaded() and attachment.has_data():
                    self.upload_attachment(attachment)
                    if attachment.is_uploaded():
                        step_upload_count += 1

            # Process nested steps recursively
            for nested_step in step.steps:
                step_upload_count += process_step(nested_step)

            return step_upload_count

        upload_count = sum(process_step(step) for step in trace.steps)

        if upload_count > 0:
            logger.info("Uploaded %d attachment(s) for trace", upload_count)

        return upload_count


# Module-level uploader instance (lazy initialized)
_uploader: Optional[AttachmentUploader] = None


def get_uploader() -> Optional[AttachmentUploader]:
    """Get or create the attachment uploader.

    Returns:
        The AttachmentUploader instance if uploads are enabled, None otherwise.
    """
    global _uploader
    from . import tracer

    if not tracer._configured_attachment_upload_enabled:
        return None

    if _uploader is None:
        client = tracer._get_client()
        if client:
            _uploader = AttachmentUploader(client)

    return _uploader


def reset_uploader() -> None:
    """Reset the uploader instance.

    This is called when tracer.configure() is called to ensure
    the uploader is recreated with new settings.
    """
    global _uploader
    _uploader = None


def upload_trace_attachments(trace: "Trace") -> int:
    """Upload all attachments in a trace.

    This is a convenience function that gets the uploader and
    uploads all attachments in the trace.

    Args:
        trace: The trace to upload attachments for.

    Returns:
        The number of attachments uploaded, or 0 if uploads are disabled.
    """
    uploader = get_uploader()
    if uploader is None:
        return 0
    return uploader.upload_trace_attachments(trace)
