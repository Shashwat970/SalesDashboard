# config.py
class Config:
    """Configuration settings for the Flask application."""
    # Flask secret key for session management
    SECRET_KEY = '8c7d5c709e6f3b0a1d2e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c'
    # MySQL database connection details
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = 'shash972' # ⚠️ IMPORTANT: Replace with your MySQL password!
    MYSQL_DB = 'sales_forecast_db'

    # Model storage directory (relative to app.py)
    MODEL_SAVE_DIR = 'models'

    # Uploads directory (for data files)
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
