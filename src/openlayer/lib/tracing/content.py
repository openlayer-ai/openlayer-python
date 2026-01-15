"""Module with the different Content classes that can be used in the input/output
of a chat completion step (multimodal)."""

from dataclasses import dataclass, field
from typing import Any, Dict, Union

from .attachments import Attachment
from .enums import ContentType


@dataclass
class TextContent:
    """Text content item."""

    text: str
    type: ContentType = field(default=ContentType.TEXT, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "text": self.text}


@dataclass
class ImageContent:
    """Image content item."""

    attachment: Attachment
    type: ContentType = field(default=ContentType.IMAGE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "attachment": self.attachment.to_dict()}


@dataclass
class AudioContent:
    """Audio content item."""

    attachment: Attachment
    type: ContentType = field(default=ContentType.AUDIO, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "attachment": self.attachment.to_dict()}


@dataclass
class FileContent:
    """File content item (PDFs, documents, etc.)."""

    attachment: Attachment
    type: ContentType = field(default=ContentType.FILE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "attachment": self.attachment.to_dict()}


# Union type for type hints
ContentItem = Union[TextContent, ImageContent, AudioContent, FileContent]
