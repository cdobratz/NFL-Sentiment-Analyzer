from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class UserPreferences(BaseModel):
    favorite_teams: List[str] = []
    notification_settings: dict = {}
    theme: str = "light"


class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    role: UserRole = UserRole.USER


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    preferences: Optional[UserPreferences] = None


class UserInDB(UserBase):
    id: str = Field(alias="_id")
    hashed_password: str
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    model_config = ConfigDict(populate_by_name=True)


class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: UserPreferences
