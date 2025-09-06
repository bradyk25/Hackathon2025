"""
Configuration management for the Ghostwriter data pipeline.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    dbtwin_api_url: str = "https://api.dbtwin.com"
    dbtwin_api_key: Optional[str] = None
    listening_post_api_url: str = "https://api.listeningpost.com"
    listening_post_api_key: Optional[str] = None
    
    # Privacy Settings
    pii_detection_threshold: float = 0.8
    phi_detection_threshold: float = 0.9
    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Data Generation Settings
    default_synthetic_multiplier: int = 10
    random_seed: int = 42
    enable_reproducibility: bool = True
    
    # File Storage
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"
    max_file_size: str = "100MB"
    
    # Logging
    log_level: str = "info"
    log_file: str = "./logs/pipeline.log"
    
    # Development
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_max_file_size_bytes() -> int:
    """Convert max file size string to bytes"""
    settings = get_settings()
    size_str = settings.max_file_size.upper()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)
