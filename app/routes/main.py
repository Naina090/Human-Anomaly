from flask import Blueprint, render_template, current_app
from app.services.repository import dashboard_stats
from app.services.fall_detector import detector_status

bp = Blueprint("main", __name__)


@bp.get("/")
def home():
    return render_template("index.html")


@bp.get("/dashboard")
def dashboard():
    stats = dashboard_stats()
    for video in stats["recent_videos"]:
        video["id"] = str(video["_id"])
        video["uploaded_display"] = video.get("upload_date").strftime("%Y-%m-%d %H:%M UTC")
    total_events = sum(item["count"] for item in stats["categories"])
    for item in stats["categories"]:
        item["percent"] = round(100 * item["count"] / total_events) if total_events else 0
    stats["mongo_connected"] = bool(current_app.extensions.get("mongo_available"))
    stats["detector"] = detector_status(current_app.config.get("POSE_MODEL_PATH", ""))
    for event in stats["recent"]:
        event["id"] = str(event["_id"])
        event["timestamp"] = event.get("timestamp_seconds", 0)
        event["category"] = event.get("event_type", "Uncategorized")
        event["method"] = event.get("detection_method", "Unknown")
    return render_template("dashboard.html", **stats)


@bp.get("/about")
def about():
    return render_template("about.html")
