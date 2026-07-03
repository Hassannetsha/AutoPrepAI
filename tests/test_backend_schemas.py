import json
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from presentation.api.schemas import (
    ChatResponse,
    ChatResponsePayload,
    ConversationMessageOut,
    ConversationOut,
    ConversationRenameRequest,
    FeedbackRequest,
)


class TestChatResponsePayload:
    def test_defaults(self):
        payload = ChatResponsePayload()
        assert payload.shape is None
        assert payload.logs == []
        assert payload.metadata == {}
        assert payload.output_file is None
        assert payload.download_url is None

    def test_with_values(self):
        payload = ChatResponsePayload(
            shape=(100, 5),
            logs=["step 1 done", "step 2 done"],
            metadata={"intents": ["missing_values"]},
            data_preview_before=[{"col1": 1, "col2": "a"}],
            output_file="processed/abc.csv",
            download_url="https://example.com/download",
        )
        assert payload.shape == (100, 5)
        assert len(payload.logs) == 2
        assert payload.metadata["intents"] == ["missing_values"]

    def test_json_serializable(self):
        payload = ChatResponsePayload(shape=(10, 3))
        data = json.loads(payload.model_dump_json())
        assert data["shape"] == [10, 3]


class TestFeedbackRequest:
    def test_valid(self):
        req = FeedbackRequest(conversation_id=str(uuid4()), accept=True)
        assert req.accept is True

    def test_valid_false(self):
        req = FeedbackRequest(conversation_id=str(uuid4()), accept=False)
        assert req.accept is False


class TestChatResponse:
    def test_valid(self):
        cid = uuid4()
        resp = ChatResponse(
            conversation_id=cid,
            assistant_message="Hello",
            result=ChatResponsePayload(),
            finished=False,
        )
        assert resp.conversation_id == cid
        assert resp.assistant_message == "Hello"
        assert resp.finished is False

    def test_default_finished(self):
        resp = ChatResponse(
            conversation_id=uuid4(),
            assistant_message="Done",
            result=ChatResponsePayload(),
        )
        assert resp.finished is False


class TestConversationMessageOut:
    def test_valid(self):
        now = datetime.now()
        msg = ConversationMessageOut(
            id=1,
            role="user",
            content="hello",
            payload={"key": "val"},
            created_at=now,
        )
        assert msg.id == 1
        assert msg.role == "user"
        assert msg.payload == {"key": "val"}

    def test_from_attributes_config(self):
        assert hasattr(ConversationMessageOut, "model_config")
        assert ConversationMessageOut.model_config.get("from_attributes") is True


class TestConversationOut:
    def test_valid(self):
        now = datetime.now()
        conv = ConversationOut(
            id=uuid4(),
            title="Test Chat",
            created_at=now,
            messages=[],
        )
        assert len(conv.messages) == 0
        assert conv.id is not None

    def test_with_messages(self):
        now = datetime.now()
        msg = ConversationMessageOut(id=1, role="assistant", content="hi", payload=None, created_at=now)
        conv = ConversationOut(id=uuid4(), title="Test Chat", created_at=now, messages=[msg])
        assert len(conv.messages) == 1

    def test_from_attributes_config(self):
        assert ConversationOut.model_config.get("from_attributes") is True


class TestConversationRenameRequest:
    def test_valid(self):
        req = ConversationRenameRequest(title="New Title")
        assert req.title == "New Title"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            ConversationRenameRequest(title="")

    def test_whitespace_title_raises(self):
        with pytest.raises(ValidationError):
            ConversationRenameRequest(title="   ")

    def test_too_long_title_raises(self):
        with pytest.raises(ValidationError):
            ConversationRenameRequest(title="x" * 256)
