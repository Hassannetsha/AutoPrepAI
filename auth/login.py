from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
from auth.utils import verify_password, create_access_token
from auth.schemas import UserLogin

router = APIRouter()

@router.post("/login")
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    # Access fields from the body
    email = user_data.email
    password = user_data.password

    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": user.id})

    return {
        "access_token": token,
        "token_type": "bearer"
    }