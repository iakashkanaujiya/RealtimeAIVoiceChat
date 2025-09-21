from pathlib import Path
from dotenv import load_dotenv

from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings

load_dotenv()  # Load environment variables from .env file


class Settings(BaseSettings):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    debug: bool = False
    log_level: str = "INFO"
    allowed_origins: str = "*"
    # OpenAI Api key
    openai_api_key: SecretStr
    # Google Api key
    google_api_key: SecretStr
    # Grow Api Key
    groq_api_key: SecretStr


settings = Settings()
