import uuid
from datetime import datetime

import pytest

from backend.models import Conversation, ConversationMessage, User


class TestUserModel:
    def test_user_creation(self):
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            hashed_password="hashed_pwd",
            first_name="John",
            last_name="Doe",
        )
        assert user.email == "test@example.com"
        assert user.is_verified is False
        assert user.last_verification_sent is None
        assert user.phone_number is None

    def test_user_with_optional_fields(self):
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            hashed_password="hashed_pwd",
            first_name="John",
            last_name="Doe",
            phone_number="01234567890",
            is_verified=True,
        )
        assert user.phone_number == "01234567890"
        assert user.is_verified is True

    def test_conversations_relationship_default(self):
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            hashed_password="hashed_pwd",
            first_name="John",
            last_name="Doe",
        )
        assert hasattr(user, "conversations")
        assert list(user.conversations) == []


class TestConversationModel:
    def test_conversation_creation(self):
        conv = Conversation(
            id=uuid.uuid4(),
            title="New Chat",
            user_id=uuid.uuid4(),
        )
        assert conv.title == "New Chat"
        assert conv.created_at is not None
        assert conv.updated_at is not None


class TestConversationMessageModel:
    def test_message_creation(self):
        msg = ConversationMessage(
            conversation_id=uuid.uuid4(),
            role="user",
            content="Hello",
        )
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.payload is None

    def test_message_with_payload(self):
        msg = ConversationMessage(
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="Processed",
            payload={"shape": [100, 5], "logs": ["done"]},
        )
        assert msg.payload["shape"] == [100, 5]

    def test_message_relationship(self):
        msg = ConversationMessage(
            conversation_id=uuid.uuid4(),
            role="user",
            content="Hello",
        )
        assert msg.conversation is None  # not loaded without session
