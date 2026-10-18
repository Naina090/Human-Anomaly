import os
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, abort
from app.services.repository import list_videos, delete_video, object_id, video_is_processing
from app.services.storage import safe_video_path, safe_processed_path

bp = Blueprint("history", __name__)


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
        video["uploaded_at"] = video.get("upload_date").strftime("%Y-%m-%d %H:%M")
        video["status"] = video.get("processing_status")
    return render_template("history.html", videos=videos, q=q, status=status, page=page, total=total, per_page=25)


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
