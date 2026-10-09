# config.py
import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    DEBUG = False
    TESTING = False

    # Model configuration
    MODEL_PATH = str((BASE_DIR / 'models' / 'random_forest_model.pkl').resolve())
    SCALER_PATH = str((BASE_DIR / 'models' / 'scaler.pkl').resolve())
    ENCODER_PATH = str((BASE_DIR / 'models' / 'le_category.pkl').resolve())
    FEATURES_PATH = str((BASE_DIR / 'models' / 'feature_columns.pkl').resolve())

    # Dataset
    DATASET_PATH = str((PROJECT_ROOT / 'Data' / 'data' / 'processed_dataset.csv').resolve())

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
