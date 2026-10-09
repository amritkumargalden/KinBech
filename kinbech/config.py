# config.py
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Model configuration
    MODEL_PATH = 'models/random_forest_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    ENCODER_PATH = 'models/le_category.pkl'
    FEATURES_PATH = 'models/feature_columns.pkl'
    
    # Dataset
    DATASET_PATH = 'data/processed_dataset.csv'
    
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
    'default': DevelopmentConfig
}
