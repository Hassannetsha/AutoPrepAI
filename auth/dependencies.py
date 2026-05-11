from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt
from backend.database import get_db
from backend.models import User
from sqlalchemy.orm import Session

security = HTTPBearer()
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

def get_current_user(token=Depends(security), db: Session = Depends(get_db)):

    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user