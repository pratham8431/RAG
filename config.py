import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION')
    AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
    AZURE_DEPLOYMENT_EMBEDDING = os.environ.get('AZURE_DEPLOYMENT_EMBEDDING')