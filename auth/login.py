from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
from auth.utils import verify_password, create_access_token, create_reset_password_token, decode_reset_password_token, hash_password
from auth.email_utils import send_password_reset_email
from auth.schemas import ForgotPasswordRequest, ResetPasswordRequest, UserLogin

router = APIRouter()

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
    
    user.hashed_password = hash_password(request.new_password)
    db.commit()
    
    return {"message": "Password has been reset successfully. You can now log in."}