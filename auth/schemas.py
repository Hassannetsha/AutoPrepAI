from pydantic import BaseModel, EmailStr, Field

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str
    first_name: str = Field(min_length=2, max_length=50)
    last_name: str = Field(min_length=2, max_length=50)

    phone_number: str | None = Field(
        default=None,
        pattern=r"^01[0-9]{9}$"
    )

class DeleteUserRequest(BaseModel):
    email: EmailStr

class AdminDeleteUserRequest(BaseModel):
    admin_username: str
    admin_password: str
    email: EmailStr

class ResendVerificationRequest(BaseModel):
    email: EmailStr

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str