from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
from auth.schemas import AdminDeleteUserRequest

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

@router.delete("/delete_user")
def delete_user(
    request: AdminDeleteUserRequest = Body(...),
    db: Session = Depends(get_db)
):
    if request.admin_username != ADMIN_USERNAME or request.admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid admin credentials")

    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": f"User with email {request.email} has been deleted"}