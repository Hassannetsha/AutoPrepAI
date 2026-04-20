import boto3
from botocore.config import Config
from backend.settings import B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL


def get_b2_client():
    return boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
        config=Config(signature_version="s3v4")
    )

def upload_file_to_b2(file_bytes: bytes, key: str, content_type= "text/csv") -> str:
    """Upload bytes to B2 and return the object key."""
    client = get_b2_client()
    client.put_object(
        Bucket=B2_BUCKET_NAME,
        Key=key,
        Body=file_bytes,
        ContentType=content_type
    )
    return key

def generate_download_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL valid for 1 hour by default."""
    client = get_b2_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": B2_BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in
    )
    return url

def delete_file_from_b2(key: str) -> None:
    """Delete all versions of a file from B2."""
    client = get_b2_client()

    # List all versions of the file
    versions = client.list_object_versions(Bucket=B2_BUCKET_NAME, Prefix=key)

    # Delete each version
    for version in versions.get("Versions", []):
        client.delete_object(
            Bucket=B2_BUCKET_NAME,
            Key=version["Key"],
            VersionId=version["VersionId"]
        )

    # Delete any hide markers too
    for marker in versions.get("DeleteMarkers", []):
        client.delete_object(
            Bucket=B2_BUCKET_NAME,
            Key=marker["Key"],
            VersionId=marker["VersionId"]
        )