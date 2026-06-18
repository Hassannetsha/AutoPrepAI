"""Tests for the conversations CRUD routes."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.Routes.conversations import router
from backend.models import Conversation, User


class TestListConversations:
    @patch("backend.Routes.conversations.get_db")
    @patch("backend.Routes.conversations.get_current_user")
    def test_lists_conversations(self, mock_get_user, mock_get_db):
        mock_user = MagicMock(spec=User)
        mock_user.id = uuid.uuid4()
        mock_get_user.return_value = mock_user

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []

        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.get(
            "/conversations/",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json() == []


class TestGetConversation:
    @patch("backend.Routes.conversations.get_db")
    @patch("backend.Routes.conversations.get_current_user")
    def test_get_own_conversation(self, mock_get_user, mock_get_db):
        user_id = uuid.uuid4()
        conv_id = uuid.uuid4()

        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_get_user.return_value = mock_user

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = user_id
        conv.title = "Test Chat"
        conv.messages = []
        mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.get(
            f"/conversations/{conv_id}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Chat"

    def test_invalid_uuid_returns_400(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.get(
            "/conversations/not-a-uuid",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "Invalid conversation_id" in response.text


class TestDeleteConversation:
    @patch("backend.Routes.conversations.get_db")
    @patch("backend.Routes.conversations.get_current_user")
    def test_delete_own_conversation(self, mock_get_user, mock_get_db):
        user_id = uuid.uuid4()
        conv_id = uuid.uuid4()

        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_get_user.return_value = mock_user

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = user_id
        mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Conversation deleted successfully"
        mock_db.delete.assert_called_once_with(conv)

    def test_delete_invalid_id_returns_400(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.delete(
            "/conversations/bad-id",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400


class TestRenameConversation:
    @patch("backend.Routes.conversations.get_db")
    @patch("backend.Routes.conversations.get_current_user")
    def test_rename_own_conversation(self, mock_get_user, mock_get_db):
        user_id = uuid.uuid4()
        conv_id = uuid.uuid4()

        mock_user = MagicMock(spec=User)
        mock_user.id = user_id
        mock_get_user.return_value = mock_user

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = user_id
        conv.title = "Old Title"
        mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        response = client.patch(
            f"/conversations/{conv_id}/rename",
            json={"title": "New Title"},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert conv.title == "New Title"
