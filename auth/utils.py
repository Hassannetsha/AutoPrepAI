from uuid import UUID

from fastapi import HTTPException
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Use argon2 instead of bcrypt
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # convert UUIDs to string before encoding
    for k, v in to_encode.items():
        if isinstance(v, UUID):
            to_encode[k] = str(v)
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

#email verification token
VERIFICATION_TOKEN_EXPIRE_HOURS = 1

def create_verification_token(email: str):
    expire = datetime.utcnow() + timedelta(hours=VERIFICATION_TOKEN_EXPIRE_HOURS)
    to_encode = {"sub": email, "exp": expire, "type": "email_verification"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_verification_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "email_verification":
            return None
        return payload.get("sub") # Returns the email
    except jwt.JWTError:
        return None
    
#password reset token

def create_reset_password_token(email: str):
    expire = datetime.utcnow() + timedelta(minutes=15)
    # Notice we label the type as "password_reset" so it can't be mixed up with verification tokens
    to_encode = {"sub": email, "exp": expire, "type": "password_reset"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_reset_password_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "password_reset":
            return None
        return payload.get("sub")
    except jwt.JWTError:
        return None