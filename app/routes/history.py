import os
from datetime import datetime, timezone
from bson import ObjectId
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, abort
from app.services.repository import (list_videos, delete_video, object_id, video_is_processing,
    get_video, queue_video_retry, create_job, update_video)
from app.services.storage import safe_video_path, safe_processed_path

bp = Blueprint("history", __name__)


def _format_upload_date(value):
    """Render current and legacy MongoDB upload_date values safely."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return "Unknown"
    else:
        return "Unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


@bp.get("/history")
def history():
    q = request.args.get("q", "").strip()[:100]
    status = request.args.get("status", "")
    try:
        page = max(1, min(100000, int(request.args.get("page", "1"))))
    except ValueError:
        page = 1
    videos, total = list_videos(q, status, page)
    for video in videos:
        video["id"] = str(video["_id"])
        video["original_name"] = video.get("original_filename")
        video["uploaded_at"] = _format_upload_date(video.get("upload_date"))
        video["status"] = video.get("processing_status")
        video["file_size"] = video.get("file_size", 0)
        video["face_analysis"] = video.get("face_analysis") or {"status": "not_run", "face_observations": 0}
        video["pose_analysis"] = video.get("pose_analysis") or {"status": "not_configured"}
        video["object_analysis"] = video.get("object_analysis") or {"status": "not_configured"}
    return render_template("history.html", videos=videos, q=q, status=status, page=page, total=total, per_page=25)


@bp.post("/history/<video_id>/retry")
def retry(video_id):
    if object_id(video_id) is None:
        abort(404)
    video = get_video(video_id)
    if not video:
        abort(404)
    if video.get("processing_status") not in {"failed", "interrupted"}:
        flash("Only failed or interrupted videos can be retried.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        flash("The original video file is missing or unavailable. This job cannot be retried.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    app = current_app._get_current_object()
    if not app.extensions["processing_slots"].acquire(blocking=False):
        flash("The processing queue is full. Try again shortly.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    try:
        claimed = queue_video_retry(video_id)
    except Exception:
        app.extensions["processing_slots"].release()
        raise
    if not claimed:
        app.extensions["processing_slots"].release()
        flash("This video was already claimed for retry or is no longer retryable.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    now = datetime.now(timezone.utc)
    try:
        job_id = create_job({"video_id": ObjectId(video_id), "status": "queued", "progress_percent": 0,
                             "started_at": None, "created_at": now, "completed_at": None,
                             "error_message": None, "retry": True})
        update_video(video_id, {"active_job_id": job_id})
    except Exception:
        try:
            update_video(video_id, {"processing_status": "failed",
                                    "error_message": "The retry job could not be queued. Try again after checking MongoDB.",
                                    "face_analysis": {"status": "not_run", "error_message": "The retry job was not started."}})
        finally:
            app.extensions["processing_slots"].release()
        raise
    from app.routes.videos import _run_analysis
    try:
        app.extensions["processing_executor"].submit(_run_analysis, app, video_id, str(job_id), target)
    except RuntimeError:
        app.extensions["processing_slots"].release()
        message = "The local worker is shutting down. Restart the website and retry again."
        update_video(video_id, {"processing_status": "failed", "error_message": message})
        update_job(str(job_id), {"status": "failed", "error_message": message,
                                 "completed_at": datetime.now(timezone.utc)})
        flash(message, "error")
        return redirect(url_for("videos.results", video_id=video_id))
    flash("Video analysis was queued again.", "success")
    return redirect(url_for("videos.results", video_id=video_id))


@bp.post("/history/<video_id>/delete")
def delete(video_id):
    if object_id(video_id) is None:
        abort(404)
    if video_is_processing(video_id):
        flash("This video is still being processed. Delete it after the job finishes.", "error")
        return redirect(url_for("history.history"))
    row = delete_video(video_id)
    if not row:
        abort(404)
    target = safe_video_path(row, current_app.config["UPLOAD_DIR"])
    if target:
        os.remove(target)
    processed = safe_processed_path(row, current_app.config["PROCESSED_DIR"])
    if processed:
        os.remove(processed)
    flash("Video and related events deleted.", "success")
    return redirect(url_for("history.history"))
