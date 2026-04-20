from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
from auth.utils import create_verification_token, decode_verification_token, hash_password
from auth.schemas import UserSignup, ResendVerificationRequest
from auth.email_utils import send_verification_email, send_welcome_email

router = APIRouter()

@router.post("/signup")
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    email = user_data.email
    password = user_data.password
    first_name = user_data.first_name
    last_name = user_data.last_name
    phone_number=user_data.phone_number
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=email,
        hashed_password=hash_password(password),
        first_name=first_name,
        last_name=last_name,
        phone_number=phone_number,
        is_verified=False
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_verification_token(email=user.email)
    send_verification_email(
        to_email=user.email,
        first_name=user.first_name,
        token=token
    )

    return {"message": "User created. Please check your email to verify your account."}

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    email = decode_verification_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
        
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if user.is_verified:
        return {"message": "Email is already verified"}
        
    user.is_verified = True
    db.commit()
    
    # Redirect to your frontend login or confirmation page
    # e.g., return RedirectResponse(url="http://localhost:3000/login?verified=true")
    return {"message": "Email verified successfully! You can now log in."}

@router.post("/resend-verification")
def resend_verification(request: ResendVerificationRequest, db: Session = Depends(get_db)):
    email = request.email
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return {"message": "If an account exists, a verification link has been sent."}
    
    if user.is_verified:
        return {"message": "Account is already verified. You can log in."}
    
    now = datetime.now(timezone.utc)
    
    if user.last_verification_sent:
        time_since_last = now - user.last_verification_sent
        if time_since_last < timedelta(minutes=1):
            # Calculate exactly how many seconds are left
            seconds_left = 60 - int(time_since_last.total_seconds())
            raise HTTPException(
                status_code=429, # 429 means "Too Many Requests"
                detail=f"Please wait {seconds_left} seconds before requesting another email."
            )
    
    new_token = create_verification_token(email=user.email)
    send_verification_email(
        to_email=user.email,
        first_name=user.first_name,
        token=new_token
    )

    user.last_verification_sent = now
    db.commit()

    return {"message": "A new verification link has been sent to your email."}