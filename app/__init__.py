from flask import Flask
from app.routes.rag_routes import rag_routes

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('../config.py')
    
    # Register blueprints
    app.register_blueprint(rag_routes)
    
    return app