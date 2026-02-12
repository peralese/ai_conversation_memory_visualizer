from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    CHATGPT = "CHATGPT"
    CLAUDE = "CLAUDE"
    GEMINI = "GEMINI"
    UNKNOWN = "UNKNOWN"


class SpeakerRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Conversation:
    id: str
    source: SourceType
    title: str
    created_at: datetime
    updated_at: datetime
    participants: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    id: str
    conversation_id: str
    timestamp: str
    speaker_role: SpeakerRole
    text: str
    parent_message_id: Optional[str] = None
    token_count: Optional[int] = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRecord:
    item_id: str
    vector: list[float]
    model_name: str
    created_at: datetime


@dataclass
class Cluster:
    cluster_id: int
    label: str
    member_ids: list[str]
    centroid: Optional[list[float]]
    created_at: datetime


@dataclass
class TopicEvent:
    cluster_id: int
    timestamp: str
    conversation_id: str
    message_id: str


@dataclass
class CanonicalConversationBundle:
    conversation: Conversation
    messages: list[Message]

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "conversation": _convert(asdict(self.conversation)),
            "messages": [_convert(asdict(m)) for m in self.messages],
        }


def _convert(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value
