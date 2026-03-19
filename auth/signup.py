from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
from auth.utils import hash_password
from auth.schemas import UserSignup
from auth.email_utils import send_welcome_email

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
        phone_number=phone_number
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    
    # send email (non-blocking recommended later)
    send_welcome_email(
        to_email=user.email,
        first_name=user.first_name,
        last_name=user.last_name
    )

    return {"message": "User created"}