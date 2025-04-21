from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_KEY: str
    API_KEY_NAME: str = "X-API-Key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings() 