# backend/routes/conversations.py

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.b2_service import get_b2_client, delete_file_from_b2
from backend.database import get_db
from backend.models import Conversation, User
from auth.dependencies import get_current_user
from backend.settings import B2_BUCKET_NAME, B2_KEY_ID
from backend.schemas import ConversationRenameRequest

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversations for the current user"""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).all()
    return [
        {
            "id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "message_count": len(conv.messages) if conv.messages else 0,
        }
        for conv in conversations
    ]


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific conversation with all its messages"""
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")

    conversation = db.get(Conversation, conversation_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": [
            {
                "id": str(msg.id),
                "content": msg.content,
                "sender": msg.role,
                "created_at": msg.created_at,
                "payload": msg.payload,
            }
            for msg in (conversation.messages or [])
        ],
    }


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


@router.patch("/{conversation_id}/rename")
def rename_conversation(
    conversation_id: str,
    body: ConversationRenameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate UUID
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")

    # Fetch conversation
    conversation = db.get(Conversation, conversation_uuid)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Ownership check
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Rename
    conversation.title = body.title.strip()
    db.commit()
    db.refresh(conversation)

    return {"id": conversation.id, "title": conversation.title}