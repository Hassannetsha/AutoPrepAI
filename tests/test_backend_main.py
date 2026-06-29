"""Integration tests for the FastAPI backend endpoints."""

import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from data_access.main import app
from business_logic.cleaning_coordinator.ml_service import MLPipelineService
from data_access.database.models import Conversation


@pytest.fixture
def client():
    return TestClient(app)


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
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_chat_requires_dataset(self, client, mock_auth_user, valid_csv_bytes):
        response = client.post(
            "/chat",
            data={"message": "hello", "mode": "chat"},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "Dataset is required" in response.text

    def test_chat_with_empty_dataset(self, client, mock_auth_user, valid_csv_bytes):
        response = client.post(
            "/chat",
            data={"message": "hello", "mode": "chat"},
            files={"dataset": ("test.csv", io.BytesIO(b""), "text/csv")},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400

    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_upload_only_message(
        self, mock_upload, mock_service, client, mock_auth_user, valid_csv_bytes
    ):
        def _add_and_set_id(conv):
            conv.id = uuid.uuid4()
        self._mock_db.add.side_effect = _add_and_set_id
        self._mock_db.get.return_value = None

        mock_service.dataframe_from_upload.return_value = pd.DataFrame(
            {"col1": [1, 2], "col2": ["a", "b"]}
        )
        mock_service._to_jsonable.side_effect = lambda x: x

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

    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_invalid_conversation_id(
        self, mock_upload, mock_service, client, mock_auth_user, valid_csv_bytes
    ):
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

    @patch("backend.main.MLPipelineService")
    @patch("backend.main.upload_file_to_b2")
    def test_chat_conversation_not_found(
        self, mock_upload, mock_service, client, mock_auth_user, valid_csv_bytes
    ):
        self._mock_db.get.return_value = None
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
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_feedback_no_session(self, client, mock_auth_user):
        response = client.post(
            "/chat/feedback",
            json={"conversation_id": str(uuid.uuid4()), "accept": True},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400
        assert "No active session" in response.text

    def test_feedback_invalid_id(self, client, mock_auth_user):
        response = client.post(
            "/chat/feedback",
            json={"conversation_id": "bad-id", "accept": True},
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400


class TestDownloadEndpoint:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        yield
        app.dependency_overrides.clear()

    @patch("backend.main.generate_download_url")
    def test_download(self, mock_gen_url, client, mock_auth_user):
        mock_gen_url.return_value = "https://example.com/file.csv"

        response = client.get(
            "/download/some/path/file.csv",
            headers={"Authorization": "Bearer test_token"},
            follow_redirects=False,
        )
        assert response.status_code == 307  # Redirect
        assert response.headers["location"] == "https://example.com/file.csv"


class TestConversationsEndpoint:
    @pytest.fixture(autouse=True)
    def _auth_override(self, mock_auth_user):
        from business_logic.auth.dependencies import get_current_user
        from data_access.database.connection import get_db
        self._mock_db = MagicMock()
        app.dependency_overrides[get_current_user] = lambda: mock_auth_user
        app.dependency_overrides[get_db] = lambda: self._mock_db
        yield
        app.dependency_overrides.clear()

    def test_get_conversation_not_found(self, client, mock_auth_user):
        self._mock_db.get.return_value = None

        cid = uuid.uuid4()
        response = client.get(
            f"/conversations/{cid}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 404

    def test_get_conversation_invalid_id(self, client, mock_auth_user):
        response = client.get(
            "/conversations/bad-id",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 400

    def test_get_conversation_unauthorized(self, client, mock_auth_user):
        conv = MagicMock(spec=Conversation)
        conv.user_id = uuid.uuid4()  # different from mock_auth_user.id
        self._mock_db.get.return_value = conv

        cid = uuid.uuid4()
        response = client.get(
            f"/conversations/{cid}",
            headers={"Authorization": "Bearer test_token"},
        )
        assert response.status_code == 403
