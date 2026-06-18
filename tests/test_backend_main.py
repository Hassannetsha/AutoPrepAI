"""Integration tests for the FastAPI backend endpoints."""

import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.ml_service import MLPipelineService
from backend.models import Conversation, User


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_auth_user():
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.is_verified = True
    return user


@pytest.fixture
def valid_csv_bytes():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    return df.to_csv(index=False).encode("utf-8")


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatEndpoint:
    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_chat_requires_dataset(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user
        mock_get_db.return_value = MagicMock()

        response = client.post(
            "/chat",
            data={"message": "hello", "mode": "chat"},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "Dataset is required" in response.text

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_chat_with_empty_dataset(self, mock_get_db, mock_get_user, client, mock_auth_user, valid_csv_bytes):
        mock_get_user.return_value = mock_auth_user
        mock_get_db.return_value = MagicMock()

        response = client.post(
            "/chat",
            data={"message": "hello", "mode": "chat"},
            files={"dataset": ("test.csv", io.BytesIO(b""), "text/csv")},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_upload_only_message(
        self, mock_upload, mock_service, mock_get_db, mock_get_user, client, mock_auth_user, valid_csv_bytes
    ):
        mock_get_user.return_value = mock_auth_user
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.get.return_value = None

        mock_service.dataframe_from_upload.return_value = pd.DataFrame(
            {"col1": [1, 2], "col2": ["a", "b"]}
        )

        response = client.post(
            "/chat",
            data={"message": "I've uploaded a dataset: test.csv", "mode": "chat"},
            files={"dataset": ("test.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "Dataset loaded successfully" in data["assistant_message"]

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_invalid_conversation_id(
        self, mock_upload, mock_service, mock_get_db, mock_get_user, client, mock_auth_user, valid_csv_bytes
    ):
        mock_get_user.return_value = mock_auth_user
        mock_service.dataframe_from_upload.return_value = pd.DataFrame({"a": [1]})

        response = client.post(
            "/chat",
            data={
                "message": "clean data",
                "mode": "chat",
                "conversation_id": "not-a-uuid",
            },
            files={"dataset": ("test.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_conversation_not_found(
        self, mock_upload, mock_service, mock_get_db, mock_get_user, client, mock_auth_user, valid_csv_bytes
    ):
        mock_get_user.return_value = mock_auth_user
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.get.return_value = None
        mock_service.dataframe_from_upload.return_value = pd.DataFrame({"a": [1]})

        cid = uuid.uuid4()
        response = client.post(
            "/chat",
            data={
                "message": "clean data",
                "mode": "chat",
                "conversation_id": str(cid),
            },
            files={"dataset": ("test.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 404


class TestChatFeedback:
    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_feedback_no_session(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user

        response = client.post(
            "/chat/feedback",
            json={"conversation_id": str(uuid.uuid4()), "accept": True},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "No active session" in response.text

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_feedback_invalid_id(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user

        response = client.post(
            "/chat/feedback",
            json={"conversation_id": "bad-id", "accept": True},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400


class TestDownloadEndpoint:
    @patch("backend.main.get_current_user")
    @patch("backend.main.generate_download_url")
    def test_download(self, mock_gen_url, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user
        mock_gen_url.return_value = "https://example.com/file.csv"

        response = client.get(
            "/download/some/path/file.csv",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 307  # Redirect


class TestConversationsEndpoint:
    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_get_conversation_not_found(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.get.return_value = None

        cid = uuid.uuid4()
        response = client.get(
            f"/conversations/{cid}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 404

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_get_conversation_invalid_id(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user

        response = client.get(
            "/conversations/bad-id",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400

    @patch("backend.main.get_current_user")
    @patch("backend.main.get_db")
    def test_get_conversation_unauthorized(self, mock_get_db, mock_get_user, client, mock_auth_user):
        mock_get_user.return_value = mock_auth_user
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        conv = MagicMock(spec=Conversation)
        conv.user_id = uuid.uuid4()  # different from mock_auth_user.id
        mock_db.get.return_value = conv

        cid = uuid.uuid4()
        response = client.get(
            f"/conversations/{cid}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 403
