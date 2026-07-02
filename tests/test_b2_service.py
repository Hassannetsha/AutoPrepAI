"""Tests for the B2 (Backblaze) storage service."""

from unittest.mock import MagicMock, patch

import pytest

from data_access.storage.b2_service import (
    delete_file_from_b2,
    generate_download_url,
    get_b2_client,
    upload_file_to_b2,
)


class TestGetB2Client:
    @patch("data_access.storage.b2_service.boto3.client")
    def test_creates_client_with_s3v4(self, mock_boto3_client):
        get_b2_client()
        mock_boto3_client.assert_called_once()
        _, kwargs = mock_boto3_client.call_args
        assert kwargs["config"].signature_version == "s3v4"


class TestUploadFileToB2:
    @patch("data_access.storage.b2_service.get_b2_client")
    def test_upload_calls_put_object(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = upload_file_to_b2(b"data", key="test/file.csv", content_type="text/csv")
        assert result == "test/file.csv"
        mock_client.put_object.assert_called_once_with(
            Bucket="autoprepai-datasets",
            Key="test/file.csv",
            Body=b"data",
            ContentType="text/csv",
        )


class TestGenerateDownloadUrl:
    @patch("data_access.storage.b2_service.get_b2_client")
    def test_generates_presigned_url(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://example.com/presigned"
        mock_get_client.return_value = mock_client

        url = generate_download_url("test/file.csv", expires_in=3600)
        assert url == "https://example.com/presigned"
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "autoprepai-datasets", "Key": "test/file.csv"},
            ExpiresIn=3600,
        )


class TestDeleteFileFromB2:
    @patch("data_access.storage.b2_service.get_b2_client")
    def test_deletes_all_versions(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_object_versions.return_value = {
            "Versions": [
                {"Key": "test/file.csv", "VersionId": "v1"},
                {"Key": "test/file.csv", "VersionId": "v2"},
            ],
            "DeleteMarkers": [],
        }
        mock_get_client.return_value = mock_client

        delete_file_from_b2("test/file.csv")
        assert mock_client.delete_object.call_count == 2

    @patch("data_access.storage.b2_service.get_b2_client")
    def test_handles_delete_markers(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_object_versions.return_value = {
            "Versions": [],
            "DeleteMarkers": [
                {"Key": "test/file.csv", "VersionId": "m1"},
            ],
        }
        mock_get_client.return_value = mock_client

        delete_file_from_b2("test/file.csv")
        mock_client.delete_object.assert_called_once_with(
            Bucket="autoprepai-datasets",
            Key="test/file.csv",
            VersionId="m1",
        )

    @patch("data_access.storage.b2_service.get_b2_client")
    def test_no_files_to_delete(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_object_versions.return_value = {
            "Versions": [],
            "DeleteMarkers": [],
        }
        mock_get_client.return_value = mock_client

        delete_file_from_b2("test/file.csv")
        mock_client.delete_object.assert_not_called()
