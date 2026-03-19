# backend/routes/conversations.py

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Conversation, User
from auth.dependencies import get_current_user

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

    # 4️⃣ Delete conversation (messages auto-delete via cascade)
    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}