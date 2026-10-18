"""Application-scoped MongoDB connection and index setup."""
import logging
from flask import current_app
from pymongo import MongoClient
from pymongo.errors import PyMongoError

log = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    """Raised when a database operation cannot be completed."""


def init_database(app):
    try:
        client = app.config.get("MONGO_CLIENT")
        if client is None:
            client = MongoClient(
                app.config["MONGO_URI"],
                serverSelectionTimeoutMS=app.config["MONGO_TIMEOUT_MS"],
                connectTimeoutMS=app.config["MONGO_TIMEOUT_MS"],
                appname="human-anomaly-web",
            )
        db = client[app.config["MONGO_DB_NAME"]]
    except (PyMongoError, ValueError) as exc:
        app.extensions["mongo_client"] = None
        app.extensions["mongo_db"] = None
        app.extensions["mongo_available"] = False
        log.warning("MongoDB configuration is unavailable (%s).", type(exc).__name__)
        return
    app.extensions["mongo_client"] = client
    app.extensions["mongo_db"] = db
    app.extensions["mongo_available"] = False
    try:
        if app.config.get("MONGO_PING_ON_STARTUP", True):
            client.admin.command("ping")
        db.videos.create_index("upload_date", name="videos_upload_date")
        db.videos.create_index("processing_status", name="videos_status")
        db.videos.create_index([("original_filename", 1), ("upload_date", -1)], name="videos_filename_date")
        db.detection_events.create_index("video_id", name="events_video_id")
        db.detection_events.create_index("created_at", name="events_created_at")
        db.detection_events.create_index("event_type", name="events_event_type")
        db.processing_jobs.create_index("video_id", name="jobs_video_id")
        db.processing_jobs.create_index("status", name="jobs_status")
        app.extensions["mongo_available"] = True
    except PyMongoError as exc:
        # The exception text can include the configured host; never log a URI.
        log.warning("MongoDB is unavailable during startup (%s). The website will show a connection notice.", type(exc).__name__)


def get_database():
    client = current_app.extensions["mongo_client"]
    if client is None:
        raise DatabaseUnavailable("MongoDB is not configured. Set MONGO_URI and MONGO_DB_NAME in your .env file.")
    if not current_app.extensions.get("mongo_available"):
        try:
            client.admin.command("ping")
            current_app.extensions["mongo_available"] = True
        except PyMongoError as exc:
            raise DatabaseUnavailable("MongoDB is unavailable. Check MONGO_URI, network access, and database credentials.") from exc
    return current_app.extensions["mongo_db"]
