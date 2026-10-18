import os
import uuid
import logging
import time
import os.path
from datetime import datetime, timezone
from bson import ObjectId
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, send_from_directory, abort, jsonify
from werkzeug.utils import secure_filename
from app.services.repository import create_video, get_video, get_events, update_video, create_job, update_job, get_job_for_video
from app.services.storage import safe_video_path, safe_processed_path
from app.services.video_processor import analyze_video, VideoError
from app.services.fall_detector import YoloPoseFallDetector, DetectorUnavailable, detector_status

bp = Blueprint("videos", __name__)
log = logging.getLogger(__name__)
ALLOWED = {"mp4", "avi", "mov", "mkv"}


@bp.get("/upload")
def upload_page():
    return render_template("upload.html", max_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
                           detector=detector_status(current_app.config.get("POSE_MODEL_PATH", "")))


@bp.post("/upload")
def upload():
    file = request.files.get("video")
    if not file or not file.filename:
        flash("Choose a video file first.", "error")
        return redirect(url_for("videos.upload_page"))
    clean = secure_filename(file.filename)
    ext = clean.rsplit(".", 1)[-1].lower() if "." in clean else ""
    if ext not in ALLOWED:
        flash("Supported extensions: MP4, AVI, MOV and MKV (decoder support varies by system).", "error")
        return redirect(url_for("videos.upload_page"))
    video_id = uuid.uuid4().hex
    stored = video_id + "." + ext
    path = os.path.join(current_app.config["UPLOAD_DIR"], stored)
    file.save(path)
    document = {
        "original_filename": clean,
        "stored_filename": stored,
        "file_path": stored,
        "upload_date": datetime.now(timezone.utc),
        "processing_status": "queued",
        "duration": None, "frame_rate": None, "width": None, "height": None,
        "processing_duration": None, "error_message": None,
    }
    try:
        record_id = create_video(document)
    except Exception:
        if os.path.isfile(path):
            os.remove(path)
        raise
    video_id = str(record_id)
    app = current_app._get_current_object()
    if not app.extensions["processing_slots"].acquire(blocking=False):
        update_video(video_id, {"processing_status": "failed", "error_message": "The processing queue is full. Try again shortly."})
        flash("The processing queue is full. Try again shortly.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    now = datetime.now(timezone.utc)
    try:
        job_id = create_job({"video_id": ObjectId(video_id), "status": "queued", "progress_percent": 0,
                             "started_at": None, "created_at": now, "completed_at": None, "error_message": None})
    except Exception:
        app.extensions["processing_slots"].release()
        raise
    app.extensions["processing_executor"].submit(_run_analysis, app, video_id, str(job_id), path)
    return redirect(url_for("videos.results", video_id=video_id))


def _run_analysis(app, video_id, job_id, path):
    started = time.monotonic()
    output_path = os.path.join(app.config["PROCESSED_DIR"], video_id + "_annotated.mp4")
    try:
        with app.app_context():
            update_job(job_id, {"status": "running", "started_at": datetime.now(timezone.utc)})
            update_video(video_id, {"processing_status": "processing"})
            detector = None
            model_path = app.config.get("POSE_MODEL_PATH", "")
            if model_path:
                detector = YoloPoseFallDetector(model_path)
            output_name = video_id + "_annotated.mp4"
            output_path = output_path if detector else None
            result = analyze_video(path, app.config["SAMPLE_EVERY_SECONDS"], app.config["MAX_DURATION_SECONDS"], detector,
                                   lambda pct: update_job(job_id, {"progress_percent": pct}), output_path)
            if not result.get("annotated_output") and output_path and os.path.isfile(output_path):
                os.remove(output_path)
            from app.db import get_database
            if result["events"]:
                for event in result["events"]:
                    event.update({"video_id": ObjectId(video_id), "created_at": datetime.now(timezone.utc)})
                get_database().detection_events.insert_many(result["events"])
            update_video(video_id, {"processing_status": "complete", "duration": result.get("duration"),
                "frame_rate": result.get("fps"), "width": result.get("width"), "height": result.get("height"),
                "processing_duration": time.monotonic()-started, "error_message": None,
                "analysis_mode": result.get("analysis_mode"), "detection_method": result.get("method"),
                "event_count": len(result["events"]),
                "processed_filename": output_name if result.get("annotated_output") else None})
            update_job(job_id, {"status": "complete", "progress_percent": 100,
                                "completed_at": datetime.now(timezone.utc)})
    except (VideoError, DetectorUnavailable) as exc:
        _fail_analysis(app, video_id, job_id, started, str(exc))
    except Exception as exc:
        log.warning("Video processing failed (%s).", type(exc).__name__)
        _fail_analysis(app, video_id, job_id, started, "The video could not be processed. Check the video codec and model configuration.")
    finally:
        if os.path.isfile(os.path.join(app.config["PROCESSED_DIR"], video_id + "_annotated.mp4")):
            with app.app_context():
                row = get_video(video_id)
                if not row or not row.get("processed_filename"):
                    try:
                        os.remove(os.path.join(app.config["PROCESSED_DIR"], video_id + "_annotated.mp4"))
                    except OSError:
                        log.warning("Could not remove incomplete processed video for record %s.", video_id[:8])
        app.extensions["processing_slots"].release()


def _fail_analysis(app, video_id, job_id, started, message):
    with app.app_context():
        now = datetime.now(timezone.utc)
        update_video(video_id, {"processing_status": "failed", "processing_duration": time.monotonic()-started,
                                "error_message": message})
        update_job(job_id, {"status": "failed", "error_message": message, "completed_at": now})


@bp.get("/results/<video_id>")
def results(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id)
    video["id"] = str(video["_id"])
    video["original_name"] = video.get("original_filename")
    video["uploaded_at"] = video.get("upload_date").strftime("%Y-%m-%d %H:%M UTC")
    video["status"] = video.get("processing_status")
    video["fps"] = video.get("frame_rate")
    video["processing_seconds"] = video.get("processing_duration")
    video["error"] = video.get("error_message")
    video["analysis_mode"] = video.get("analysis_mode", "pending")
    video["has_processed_video"] = bool(safe_processed_path(video, current_app.config["PROCESSED_DIR"]))
    for event in events:
        event["category"] = event.get("event_type", "Uncategorized")
        event["timestamp"] = event.get("timestamp_seconds", 0)
        event["score"] = event.get("anomaly_score", event.get("confidence"))
        event["method"] = event.get("detection_method", "Unknown")
    return render_template("results.html", video=video, events=events)


@bp.get("/api/videos/<video_id>/status")
def status(video_id):
    if not get_video(video_id):
        abort(404)
    job = get_job_for_video(video_id) or {}
    return jsonify({"status": job.get("status", "queued"), "progress": job.get("progress_percent", 0),
                    "error": job.get("error_message"), "analysis_mode": get_video(video_id).get("analysis_mode")})


@bp.get("/media/<video_id>")
def media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_DIR"], os.path.basename(target), as_attachment=False)


@bp.get("/processed/<video_id>")
def processed_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_processed_path(video, current_app.config["PROCESSED_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["PROCESSED_DIR"], os.path.basename(target), as_attachment=False)
