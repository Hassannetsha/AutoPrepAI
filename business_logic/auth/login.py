import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from data_access.database.connection import get_db, SessionLocal
from data_access.database.models import User, Conversation
from business_logic.utils.utils import verify_password, create_access_token, create_reset_password_token, decode_reset_password_token, hash_password
from business_logic.utils.email_utils import send_password_reset_email
from business_logic.auth.schemas import ForgotPasswordRequest, ResetPasswordRequest, UserLogin
from business_logic.auth import dependencies

router = APIRouter()
PASSWORD_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&.#_-])[A-Za-z\d@$!%*?&.#_-]{8,}$"
@router.post("/logout")
def logout(current_user=Depends(dependencies.get_current_user)):
    from business_logic.services import session_store as utilities
    db = SessionLocal()
    try:
        user_conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).all()
        for conv in user_conversations:
            utilities.sessions.pop(str(conv.id), None)
    finally:
        db.close()
    return {"message": "Logged out successfully"}

@router.post("/login")
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    # Access fields from the body
    email = user_data.email
    password = user_data.password

    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Please verify your email address before logging in")

    token = create_access_token({"user_id": user.id})

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    
    if user:
        token = create_reset_password_token(user.email)
        send_password_reset_email(
            to_email=user.email, 
            first_name=user.first_name, 
            token=token
        )
    return {"message": "If that email is in our system, we have sent a reset link."}

@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):

    email = decode_reset_password_token(request.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if verify_password(request.new_password, user.hashed_password):
        raise HTTPException(
            status_code=400, 
            detail="New password cannot be the same as your current password."
        )
    if not re.match(PASSWORD_REGEX, request.new_password):
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be at least 8 characters long, "
                "contain at least one uppercase letter, "
                "one lowercase letter, "
                "one number, "
                "and one special character (@$!%*?&.#_-)."
            )
        )
    if request.new_password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    user.hashed_password = hash_password(request.new_password)
    db.commit()
    
    return {"message": "Password has been reset successfully. You can now log in."}