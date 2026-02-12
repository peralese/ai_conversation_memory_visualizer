from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dt_parser

from src.models import CanonicalConversationBundle, Conversation, Message, SourceType, SpeakerRole
from src.parsers.chatgpt_parser import flatten_content_parts


def normalize(raw_conversation: dict[str, Any], source: SourceType) -> CanonicalConversationBundle:
    if source == SourceType.CHATGPT:
        return _normalize_chatgpt(raw_conversation)
    raise ValueError(f"Normalizer is not implemented for source {source}")


def _normalize_chatgpt(raw: dict[str, Any]) -> CanonicalConversationBundle:
    conv_id = str(raw.get("id") or raw.get("conversation_id") or _fallback_id(raw))
    created = _normalize_timestamp(raw.get("create_time") or raw.get("created_at"))
    updated = _normalize_timestamp(raw.get("update_time") or raw.get("updated_at") or created)

    conversation = Conversation(
        id=conv_id,
        source=SourceType.CHATGPT,
        title=(raw.get("title") or "Untitled").strip() or "Untitled",
        created_at=created,
        updated_at=updated,
        participants=["user", "assistant"],
        tags=[],
        raw_metadata={
            "conversation_template_id": raw.get("conversation_template_id"),
            "is_archived": raw.get("is_archived"),
        },
    )

    mapping = raw.get("mapping") or {}
    messages: list[Message] = []

    for node_id, node in mapping.items():
        if not isinstance(node, dict):
            continue

        message = node.get("message")
        if not isinstance(message, dict):
            continue

        author_role = ((message.get("author") or {}).get("role") or "user").lower()
        role = _normalize_role(author_role)

        content = message.get("content") or {}
        text = flatten_content_parts(content)
        if not text:
            continue

        msg_time = _normalize_timestamp(message.get("create_time") or node.get("create_time") or created)
        token_count = _extract_token_count(message)

        messages.append(
            Message(
                id=str(message.get("id") or node_id),
                conversation_id=conv_id,
                timestamp=msg_time.isoformat(),
                speaker_role=role,
                text=text,
                parent_message_id=node.get("parent"),
                token_count=token_count,
                raw_metadata={
                    "status": message.get("status"),
                    "recipient": message.get("recipient"),
                    "end_turn": message.get("end_turn"),
                    "weight": message.get("weight"),
                },
            )
        )

    messages.sort(key=lambda m: m.timestamp)
    return CanonicalConversationBundle(conversation=conversation, messages=messages)


def _normalize_role(role: str) -> SpeakerRole:
    if role in ("assistant", "user", "system", "tool"):
        return SpeakerRole(role)
    if role in ("critic", "developer"):
        return SpeakerRole.SYSTEM
    return SpeakerRole.USER


def _normalize_timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, (int, float)):
        # ChatGPT exports often use unix seconds.
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        dt = dt_parser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    return datetime.now(timezone.utc)


def _extract_token_count(message: dict[str, Any]) -> int | None:
    metadata = message.get("metadata") or {}
    for key in ("token_count", "completion_tokens", "prompt_tokens"):
        val = metadata.get(key)
        if isinstance(val, int):
            return val
    return None


def _fallback_id(raw: dict[str, Any]) -> str:
    title = raw.get("title") or "conversation"
    created = str(raw.get("create_time") or "0")
    return f"{title}-{created}"
