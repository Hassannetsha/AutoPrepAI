# backend/routes/conversations.py

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.b2_service import get_b2_client, delete_file_from_b2
from backend.database import get_db
from backend.models import Conversation, User
from auth.dependencies import get_current_user
from backend.settings import B2_BUCKET_NAME, B2_KEY_ID

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 1️⃣ Validate UUID
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")

    # 2️⃣ Get conversation
    conversation = db.get(Conversation, conversation_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 3️⃣ Security check (VERY IMPORTANT)
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # 4️⃣ Delete all B2 files related to this conversation
    try:
        client = get_b2_client()
        for prefix in [f"inputs/{conversation_uuid}/", f"processed/{conversation_uuid}/"]:
            response = client.list_objects_v2(Bucket=B2_BUCKET_NAME, Prefix=prefix)
            files = response.get("Contents", [])
            print(f"Found {len(files)} files to delete under {prefix}")
            for obj in files:
                delete_file_from_b2(obj["Key"])
                print(f"Deleted: {obj['Key']}")
    except Exception as exc:
        print(f"B2 cleanup error: {exc}")


    # 5️⃣ Delete conversation (messages auto-delete via cascade)
    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}