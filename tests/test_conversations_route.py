"""Tests for the conversations CRUD routes."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from presentation.api.routes.conversations import router
from data_access.database.models import Conversation, User


class TestListConversations:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        from presentation.api.main import app
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_lists_conversations(self):
        self._mock_db.query.return_value.filter.return_value.all.return_value = []

        from fastapi.testclient import TestClient
        from presentation.api.main import app
        client = TestClient(app)

        response = client.get(
            "/conversations/",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json() == []


class TestGetConversation:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        from presentation.api.main import app
        self._current_user_id = mock_auth_user.id
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_get_own_conversation(self):
        conv_id = uuid.uuid4()

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = self._current_user_id
        conv.title = "Test Chat"
        conv.messages = []
        self._mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from presentation.api.main import app
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
        from presentation.api.main import app
        client = TestClient(app)

        response = client.get(
            "/conversations/not-a-uuid",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "Invalid conversation_id" in response.text


class TestDeleteConversation:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        from presentation.api.main import app
        self._current_user_id = mock_auth_user.id
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_delete_own_conversation(self):
        conv_id = uuid.uuid4()

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = self._current_user_id
        self._mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from presentation.api.main import app
        client = TestClient(app)

        response = client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Conversation deleted successfully"
        self._mock_db.delete.assert_called_once_with(conv)

    def test_delete_invalid_id_returns_400(self):
        from fastapi.testclient import TestClient
        from presentation.api.main import app
        client = TestClient(app)

        response = client.delete(
            "/conversations/bad-id",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400


class TestRenameConversation:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        from presentation.api.main import app
        self._current_user_id = mock_auth_user.id
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_rename_own_conversation(self):
        conv_id = uuid.uuid4()

        conv = MagicMock(spec=Conversation)
        conv.id = conv_id
        conv.user_id = self._current_user_id
        conv.title = "Old Title"
        self._mock_db.get.return_value = conv

        from fastapi.testclient import TestClient
        from presentation.api.main import app
        client = TestClient(app)

        response = client.patch(
            f"/conversations/{conv_id}/rename",
            json={"title": "New Title"},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        assert conv.title == "New Title"
