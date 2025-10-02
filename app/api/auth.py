from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import (
    OAuth2PasswordRequestForm,
    HTTPBearer,
    HTTPAuthorizationCredentials,
)
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from typing import Optional
import logging

from ..core.config import settings
from ..core.database import get_database, get_redis
from ..core.dependencies import get_current_user
from ..models.user import UserCreate, UserResponse, UserInDB, UserRole

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


async def authenticate_user(email: str, password: str, db):
    """Authenticate user with email and password"""
    user = await db.users.find_one({"email": email})
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db=Depends(get_database)):
    """Register a new user"""
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    existing_username = await db.users.find_one({"username": user_data.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user_dict = user_data.dict(exclude={"password"})
    user_dict.update(
        {
            "hashed_password": hashed_password,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {},
        }
    )

    result = await db.users.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)

    return UserResponse(**user_dict)


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_database)
):
    """Login user and return access token"""
    user = await authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    await db.users.update_one(
        {"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}}
    )

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user["_id"])}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return UserResponse(
        id=str(current_user["_id"]),
        email=current_user["email"],
        username=current_user["username"],
        role=current_user["role"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"],
        last_login=current_user.get("last_login"),
        preferences=current_user.get("preferences", {}),
    )


@router.post("/refresh")
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh access token"""
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(current_user["_id"])}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


@router.delete("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user),
    redis=Depends(get_redis),
):
    """Logout user and blacklist token"""
    token = credentials.credentials

    # Add token to blacklist in Redis if available
    if redis:
        try:
            # Decode token to get expiration time
            payload = jwt.decode(
                token, settings.secret_key, algorithms=[settings.algorithm]
            )
            exp = payload.get("exp")

            if exp:
                # Calculate TTL for blacklist entry
                ttl = exp - datetime.utcnow().timestamp()
                if ttl > 0:
                    await redis.setex(f"blacklist:{token}", int(ttl), "1")
        except Exception as e:
            logger.warning(f"Failed to blacklist token: {e}")

    return {"message": "Successfully logged out"}


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    profile_data: dict,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Update user profile"""
    # Validate allowed fields
    allowed_fields = {"username", "email"}
    update_data = {k: v for k, v in profile_data.items() if k in allowed_fields}

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No valid fields to update"
        )

    # Check if email is already taken by another user
    if "email" in update_data:
        existing_user = await db.users.find_one(
            {"email": update_data["email"], "_id": {"$ne": current_user["_id"]}}
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

    # Check if username is already taken by another user
    if "username" in update_data:
        existing_user = await db.users.find_one(
            {"username": update_data["username"], "_id": {"$ne": current_user["_id"]}}
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
            )

    # Update user
    await db.users.update_one({"_id": current_user["_id"]}, {"$set": update_data})

    # Get updated user
    updated_user = await db.users.find_one({"_id": current_user["_id"]})

    return UserResponse(
        id=str(updated_user["_id"]),
        email=updated_user["email"],
        username=updated_user["username"],
        role=updated_user["role"],
        is_active=updated_user["is_active"],
        created_at=updated_user["created_at"],
        last_login=updated_user.get("last_login"),
        preferences=updated_user.get("preferences", {}),
    )


@router.put("/change-password")
async def change_password(
    password_data: dict,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Change user password"""
    current_password = password_data.get("currentPassword")
    new_password = password_data.get("newPassword")

    if not current_password or not new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password and new password are required",
        )

    # Verify current password
    if not verify_password(current_password, current_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Hash new password
    hashed_new_password = get_password_hash(new_password)

    # Update password
    await db.users.update_one(
        {"_id": current_user["_id"]}, {"$set": {"hashed_password": hashed_new_password}}
    )

    return {"message": "Password changed successfully"}
