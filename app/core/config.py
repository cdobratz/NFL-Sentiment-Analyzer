from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, ValidationInfo
from typing import Optional, List
import os
import warnings


class Settings(BaseSettings):
    # Application settings
    app_name: str = "NFL Sentiment Analyzer"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"  # Environment: development, staging, production

    # Database settings
    mongodb_url: str
    database_name: str = "nfl_sentiment"

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_key_prefix: str = "nfl_sentiment"  # Prefix for all Redis keys in this app

    # Authentication settings
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # External API settings
    twitter_bearer_token: Optional[str] = None
    espn_api_key: Optional[str] = None
    draftkings_api_key: Optional[str] = None
    mgm_api_key: Optional[str] = None

    # MLOps settings
    huggingface_token: Optional[str] = None
    hopsworks_api_key: Optional[str] = None
    hopsworks_project: str = "nfl-sentiment-analyzer"
    hopsworks_host: str = "https://c.app.hopsworks.ai"
    wandb_api_key: Optional[str] = None
    wandb_project: str = "nfl-sentiment-analyzer"
    wandb_entity: Optional[str] = None

    # Model storage settings
    models_dir: str = "./models"
    deployments_dir: str = "./deployments"

    # CORS settings - safe development default, override with ALLOWED_ORIGINS env var for production
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Monitoring
    sentry_dsn: Optional[str] = None

    @field_validator("mongodb_url")
    @classmethod
    def validate_mongodb_url(cls, v):
        """Validate MongoDB connection string format"""
        if not v:
            raise ValueError("mongodb_url is required")
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                "mongodb_url must start with 'mongodb://' or 'mongodb+srv://'"
            )
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key meets minimum security requirements"""
        if not v:
            raise ValueError("secret_key is required")
        if len(v) < 32:
            raise ValueError(
                "secret_key must be at least 32 characters for security. "
                "Generate one using: openssl rand -hex 32"
            )
        # Check for common placeholder values
        placeholder_values = [
            "your-super-secret-key-change-this-in-production",
            "change-me",
            "secret",
            "your-secret-key",
            "changeme",
            "your_secret_key",
        ]
        if v.lower() in placeholder_values:
            raise ValueError(
                "secret_key appears to be a placeholder value. "
                "Generate a secure random key using: openssl rand -hex 32"
            )
        return v

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v, info: ValidationInfo):
        """Parse CORS origins from environment variable or return default"""
        if isinstance(v, str):
            # Parse comma-separated string from environment variable
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            # Prevent wildcard in non-development environments
            if "*" in origins:
                # Get environment from the settings being validated
                env = info.data.get("environment", "production").lower()
                if env in ["production", "prod", "staging"]:
                    raise ValueError(
                        "CORS wildcard '*' is not allowed in production or staging. "
                        "Use explicit origins instead."
                    )
                warnings.warn(
                    "CORS wildcard '*' detected - acceptable for development only",
                    UserWarning,
                )
            return origins
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Provide the global Settings instance used by the application.

    Returns:
        Settings: The application's Settings instance populated from environment variables and defaults.
    """
    return settings
