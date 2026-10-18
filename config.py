import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "human_anomaly_db")
    MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "1500"))
    MONGO_PING_ON_STARTUP = True
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", str(BASE_DIR / "processed"))
    SAMPLE_EVERY_SECONDS = float(os.getenv("SAMPLE_EVERY_SECONDS", "1.0"))
    MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "900"))
    POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "")
    PROCESSING_WORKERS = int(os.getenv("PROCESSING_WORKERS", "1"))
    PROCESSING_QUEUE_SIZE = int(os.getenv("PROCESSING_QUEUE_SIZE", "3"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
