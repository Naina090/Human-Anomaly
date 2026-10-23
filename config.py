import os
import secrets
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from dotenv import load_dotenv

# Resolve .env next to this file so VS Code's current working directory does
# not affect which configuration the website reads.
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "250")) * 1024 * 1024
    MONGO_URI = os.getenv("MONGO_URI", "").strip()
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "human_anomaly_db")
    MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "1500"))
    MONGO_PING_ON_STARTUP = True
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", str(BASE_DIR / "processed"))
    SAMPLE_EVERY_SECONDS = float(os.getenv("SAMPLE_EVERY_SECONDS", "1.0"))
    MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "900"))
    POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "")
    FALL_TRANSITION_WINDOW_SECONDS = float(os.getenv("FALL_TRANSITION_WINDOW_SECONDS", "1.5"))
    FALL_MIN_VERTICAL_DROP = float(os.getenv("FALL_MIN_VERTICAL_DROP", "0.12"))
    FALL_MIN_HORIZONTAL_PERSISTENCE = float(os.getenv("FALL_MIN_HORIZONTAL_PERSISTENCE", "0.6"))
    FALL_CANDIDATE_SCORE_THRESHOLD = float(os.getenv("FALL_CANDIDATE_SCORE_THRESHOLD", "0.55"))
    # Object detection is opt-in and only loads an existing local weight file.
    OBJECT_MODEL_PATH = os.getenv("OBJECT_MODEL_PATH", "").strip()
    OBJECT_MODEL_CONFIDENCE = float(os.getenv("OBJECT_MODEL_CONFIDENCE", "0.25"))
    OBJECT_TRACKER = os.getenv("OBJECT_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
    # Empty means use the frontal-face cascade bundled with opencv-python.
    FACE_CASCADE_PATH = os.getenv("FACE_CASCADE_PATH", "").strip()
    PROCESSING_WORKERS = int(os.getenv("PROCESSING_WORKERS", "1"))
    PROCESSING_QUEUE_SIZE = int(os.getenv("PROCESSING_QUEUE_SIZE", "3"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
