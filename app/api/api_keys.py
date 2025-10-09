"""
API key management endpoints for administrators.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.dependencies import get_current_admin_user
from ..core.api_keys import api_key_manager, APIKeyScope, APIKey
from ..core.exceptions import ValidationError, NotFoundError

router = APIRouter(prefix="/admin/api-keys", tags=["Admin", "API Keys"])


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating API keys"""

    name: str
    scopes: List[APIKeyScope]
    expires_in_days: Optional[int] = None
    rate_limit: int = 1000
    metadata: dict = Field(default_factory=dict)


class APIKeyResponse(BaseModel):
    """Response model for API key information (without the actual key)"""

    id: str
    name: str
    scopes: List[APIKeyScope]
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: int
    created_by: str
    metadata: dict


class CreateAPIKeyResponse(BaseModel):
    """Response model for API key creation (includes the actual key)"""

    api_key: str
    key_info: APIKeyResponse
    warning: str = "Store this API key securely. It will not be shown again."


@router.post("/", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest, current_user: dict = Depends(get_current_admin_user)
):
    """
    Create a new API key with the provided name, scopes, expiration, rate limit, and metadata.

    Parameters:
        request (CreateAPIKeyRequest): Payload containing name, scopes, optional expires_in_days, rate_limit, and metadata.

    Returns:
        CreateAPIKeyResponse: Contains the newly created API key value (shown only once) and associated key metadata (APIKeyResponse).

    Raises:
        ValidationError: If one or more requested scopes are invalid.
        HTTPException: On unexpected server-side failures (status 500).
    """
    try:
        # Validate scopes
        valid_scopes = [scope for scope in APIKeyScope]
        for scope in request.scopes:
            if scope not in valid_scopes:
                raise ValidationError(f"Invalid scope: {scope}")

        # Create the API key
        key, api_key_obj = await api_key_manager.create_api_key(
            name=request.name,
            scopes=request.scopes,
            created_by=str(current_user["_id"]),
            expires_in_days=request.expires_in_days,
            rate_limit=request.rate_limit,
            metadata=request.metadata,
        )

        # Return response with actual key (only time it's shown)
        return CreateAPIKeyResponse(
            api_key=key,
            key_info=APIKeyResponse(
                id=api_key_obj.id,
                name=api_key_obj.name,
                scopes=api_key_obj.scopes,
                status=api_key_obj.status.value,
                created_at=api_key_obj.created_at,
                expires_at=api_key_obj.expires_at,
                last_used_at=api_key_obj.last_used_at,
                usage_count=api_key_obj.usage_count,
                rate_limit=api_key_obj.rate_limit,
                created_by=api_key_obj.created_by,
                metadata=api_key_obj.metadata,
            ),
        )

    except ValidationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(current_user: dict = Depends(get_current_admin_user)):
    """
    Retrieve all API keys and their associated metadata.

    Returns:
        List[APIKeyResponse]: A list of APIKeyResponse objects representing each API key and its usage/metadata. The actual secret key values are not included.
    """
    try:
        api_keys = await api_key_manager.list_api_keys()

        return [
            APIKeyResponse(
                id=key.id,
                name=key.name,
                scopes=key.scopes,
                status=key.status.value,
                created_at=key.created_at,
                expires_at=key.expires_at,
                last_used_at=key.last_used_at,
                usage_count=key.usage_count,
                rate_limit=key.rate_limit,
                created_by=key.created_by,
                metadata=key.metadata,
            )
            for key in api_keys
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}",
        )


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str, current_user: dict = Depends(get_current_admin_user)
):
    """
    Retrieve metadata for the API key identified by `key_id`.

    Parameters:
        key_id (str): The unique identifier of the API key to retrieve.

    Returns:
        APIKeyResponse: The API key's metadata (id, name, scopes, status, timestamps, usage, rate limit, creator, and metadata) excluding the secret key.

    Raises:
        NotFoundError: If no API key with `key_id` exists.
    """
    try:
        api_key = await api_key_manager.get_api_key_by_id(key_id)

        if not api_key:
            raise NotFoundError("API key", key_id)

        return APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            scopes=api_key.scopes,
            status=api_key.status.value,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
            last_used_at=api_key.last_used_at,
            usage_count=api_key.usage_count,
            rate_limit=api_key.rate_limit,
            created_by=api_key.created_by,
            metadata=api_key.metadata,
        )

    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API key: {str(e)}",
        )


@router.get("/{key_id}/usage")
async def get_api_key_usage(
    key_id: str, current_user: dict = Depends(get_current_admin_user)
):
    """
    Retrieve usage statistics for the API key identified by `key_id`.

    Returns detailed usage metrics such as total requests, average daily usage, last usage timestamp, and rate limit information.

    Parameters:
        key_id (str): Identifier of the API key to query.

    Returns:
        dict: Usage statistics including keys like `total_requests`, `average_daily_usage`, `last_used_at`, and `rate_limit`.

    Raises:
        NotFoundError: If no API key with `key_id` exists.
        HTTPException: If an internal error occurs while retrieving usage.
    """
    try:
        usage_stats = await api_key_manager.get_api_key_usage(key_id)

        if not usage_stats:
            raise NotFoundError("API key", key_id)

        return usage_stats

    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API key usage: {str(e)}",
        )


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str, current_user: dict = Depends(get_current_admin_user)
):
    """
    Revoke the API key identified by `key_id`, disabling it permanently.

    Attempts to revoke the key and returns a confirmation payload on success. This operation is irreversible and requires an admin user.

    Parameters:
        key_id (str): Identifier of the API key to revoke.

    Returns:
        dict: Confirmation containing `message`, `key_id`, `revoked_by` (admin user id), and `revoked_at` (ISO 8601 UTC timestamp).

    Raises:
        NotFoundError: If no API key with `key_id` exists.
        HTTPException: For unexpected failures during revocation (results in a 500 response).
    """
    try:
        success = await api_key_manager.revoke_api_key(key_id, str(current_user["_id"]))

        if not success:
            raise NotFoundError("API key", key_id)

        return {
            "message": "API key revoked successfully",
            "key_id": key_id,
            "revoked_by": str(current_user["_id"]),
            "revoked_at": datetime.utcnow().isoformat(),
        }

    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}",
        )
