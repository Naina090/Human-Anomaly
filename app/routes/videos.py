import os
import uuid
import logging
import time
import os.path
from datetime import datetime, timezone
from bson import ObjectId
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, send_from_directory, abort, jsonify, Response
from werkzeug.utils import secure_filename
from app.services.repository import (create_video, get_video, get_events, update_video,
    create_job, update_job, get_job_for_video, get_face_frames, save_face_frames,
    get_pose_frames, save_pose_frames)
from app.db import DatabaseUnavailable
from app.services.storage import safe_video_path, safe_processed_path
from app.services.video_processor import analyze_video, VideoError
from app.services.fall_detector import YoloPoseFallDetector, DetectorUnavailable, detector_status, METHOD as POSE_METHOD
from app.services.face_detector import OpenCVFaceDetector, FaceDetectorUnavailable, detector_status as face_detector_status
from app.services.object_detector import (ObjectDetectorUnavailable, YoloObjectTracker,
    detector_status as object_detector_status)
from app.services.report_service import build_analysis_summary, video_report_csv
from app.services.video_validator import validate_video_file, VideoValidationError
from app.logging_utils import sanitized_traceback

bp = Blueprint("videos", __name__)
log = logging.getLogger(__name__)
ALLOWED = {"mp4", "avi", "mov", "mkv"}


@bp.get("/upload")
def upload_page():
    return render_template("upload.html", max_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
                           detector=detector_status(current_app.config.get("POSE_MODEL_PATH", "")),
                           face_detector=face_detector_status(current_app.config.get("FACE_CASCADE_PATH", "")),
                           object_detector=object_detector_status(
                               current_app.config.get("OBJECT_MODEL_PATH", ""),
                               current_app.config.get("OBJECT_TRACKER", "bytetrack.yaml")))


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
    try:
        file.save(path)
        file_size = os.path.getsize(path) if os.path.isfile(path) else 0
        if file_size <= 0:
            raise VideoValidationError("The selected video file is empty. Choose a non-empty video.")
        # Verify actual decodable video content before creating MongoDB records
        # or occupying a background-processing slot. This requires no AI model.
        video_metadata = validate_video_file(path)
        maximum_duration = current_app.config["MAX_DURATION_SECONDS"]
        if video_metadata["duration"] is not None and video_metadata["duration"] > maximum_duration:
            raise VideoValidationError(
                f"This video is longer than the configured {maximum_duration}-second limit. Choose a shorter video."
            )
    except VideoValidationError as exc:
        if os.path.isfile(path):
            os.remove(path)
        flash(str(exc), "error")
        return redirect(url_for("videos.upload_page"))
    except OSError as exc:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                log.warning("Could not remove incomplete upload after a filesystem error.")
        log.error("Video upload filesystem failure (%s); traceback follows:\n%s",
                  type(exc).__name__, sanitized_traceback(exc, current_app.config))
        raise
    document = {
        "original_filename": clean,
        "stored_filename": stored,
        "file_path": stored,
        "upload_date": datetime.now(timezone.utc),
        "processing_status": "queued",
        **video_metadata,
        "validation_status": "valid",
        "validation_message": "OpenCV opened the file and decoded a video frame.",
        "processing_duration": None, "error_message": None,
        "file_size": file_size,
        "face_analysis": {"status": "queued", "method": OpenCVFaceDetector.METHOD,
                          "frames_sampled": 0, "frames_with_faces": 0,
                          "face_observations": 0, "max_faces_per_frame": 0},
        "pose_analysis": {"status": "queued" if current_app.config.get("POSE_MODEL_PATH") else "not_configured",
                          "method": POSE_METHOD, "frames_sampled": 0,
                          "frames_with_people": 0},
        "object_analysis": {"status": "queued" if current_app.config.get("OBJECT_MODEL_PATH") else "not_configured",
                            "method": YoloObjectTracker.METHOD, "frames_sampled": 0,
                            "observations": 0, "counts_by_class": {},
                            "person_track_count_estimate": 0},
    }
    try:
        record_id = create_video(document)
    except DatabaseUnavailable as exc:
        if os.path.isfile(path):
            os.remove(path)
        return render_template(
            "error.html",
            error_label="UPLOAD NOT QUEUED",
            error_title="MongoDB is temporarily unavailable.",
            message=("OpenCV validated the video, but MongoDB was unavailable, so no video record or analysis job was saved. "
                     "The temporary upload was removed. " + str(exc)),
        ), 503
    except Exception:
        if os.path.isfile(path):
            os.remove(path)
        raise
    video_id = str(record_id)
    app = current_app._get_current_object()
    if not app.extensions["processing_slots"].acquire(blocking=False):
        message = "The processing queue is full. Try again shortly."
        update_video(video_id, {"processing_status": "failed", "error_message": message,
                                "face_analysis": {"status": "not_run", "method": OpenCVFaceDetector.METHOD,
                                                  "error_message": message}})
        flash("The processing queue is full. Try again shortly.", "error")
        return redirect(url_for("videos.results", video_id=video_id))
    now = datetime.now(timezone.utc)
    try:
        job_id = create_job({"video_id": ObjectId(video_id), "status": "queued", "progress_percent": 0,
                             "started_at": None, "created_at": now, "completed_at": None, "error_message": None})
    except Exception:
        app.extensions["processing_slots"].release()
        update_video(video_id, {"processing_status": "failed",
                                "error_message": "The processing job could not be saved to MongoDB.",
                                "face_analysis": {"status": "not_run", "method": OpenCVFaceDetector.METHOD}})
        raise
    try:
        app.extensions["processing_executor"].submit(_run_analysis, app, video_id, str(job_id), path)
    except RuntimeError:
        app.extensions["processing_slots"].release()
        message = "The local processing worker is shutting down. Retry this video after restarting the website."
        update_video(video_id, {"processing_status": "failed", "error_message": message})
        update_job(str(job_id), {"status": "failed", "error_message": message,
                                 "completed_at": datetime.now(timezone.utc)})
    return redirect(url_for("videos.results", video_id=video_id))


def _run_analysis(app, video_id, job_id, path):
    started = time.monotonic()
    output_path = os.path.join(app.config["PROCESSED_DIR"], video_id + "_annotated.mp4")
    try:
        with app.app_context():
            update_job(job_id, {"status": "running", "started_at": datetime.now(timezone.utc)})
            update_video(video_id, {"processing_status": "processing"})
            detector = None
            face_detector = None
            object_detector = None
            face_init_error = None
            object_init_error = None
            pose_init_error = None
            try:
                face_detector = OpenCVFaceDetector(app.config.get("FACE_CASCADE_PATH", ""))
            except FaceDetectorUnavailable as exc:
                face_init_error = str(exc)
            except Exception as exc:
                log.error("Face detector initialization failed (%s); traceback follows:\n%s",
                          type(exc).__name__, sanitized_traceback(exc, app.config))
                face_init_error = "The OpenCV face detector failed to initialize."
            update_video(video_id, {"face_analysis": {
                "status": "processing" if face_detector else "unavailable",
                "method": OpenCVFaceDetector.METHOD,
                "error_message": face_init_error,
            }})
            object_model_path = app.config.get("OBJECT_MODEL_PATH", "")
            if object_model_path:
                try:
                    object_detector = YoloObjectTracker(
                        object_model_path,
                        app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25),
                        app.config.get("OBJECT_TRACKER", "bytetrack.yaml"),
                    )
                except ObjectDetectorUnavailable as exc:
                    object_init_error = str(exc)
                except Exception as exc:
                    log.error("Object detector initialization failed (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, app.config))
                    object_init_error = "The configured object detector could not be initialized."
            object_status = "processing" if object_detector else (
                "unavailable" if object_model_path else "not_configured")
            update_video(video_id, {"object_analysis": {
                "status": object_status, "method": YoloObjectTracker.METHOD,
                "model_name": os.path.basename(object_model_path) if object_model_path else None,
                "error_message": object_init_error,
            }})
            model_path = app.config.get("POSE_MODEL_PATH", "")
            if model_path:
                update_video(video_id, {"pose_analysis": {"status": "processing",
                    "method": POSE_METHOD,
                    "model_name": os.path.basename(model_path)}})
                try:
                    detector = YoloPoseFallDetector(model_path, {
                        "transition_window_seconds": app.config.get("FALL_TRANSITION_WINDOW_SECONDS", 1.5),
                        "minimum_vertical_drop": app.config.get("FALL_MIN_VERTICAL_DROP", .12),
                        "minimum_horizontal_persistence": app.config.get("FALL_MIN_HORIZONTAL_PERSISTENCE", .6),
                        "candidate_score_threshold": app.config.get("FALL_CANDIDATE_SCORE_THRESHOLD", .55),
                    })
                except DetectorUnavailable as exc:
                    pose_init_error = str(exc)
                except Exception as exc:
                    log.error("Pose detector initialization failed (%s); traceback follows:\n%s",
                              type(exc).__name__, sanitized_traceback(exc, app.config))
                    pose_init_error = "The configured pose detector could not be initialized."
            else:
                update_video(video_id, {"pose_analysis": {"status": "not_configured",
                    "method": POSE_METHOD, "model_name": None}})
            output_name = video_id + "_annotated.mp4"
            output_path = output_path if (detector or face_detector or object_detector) else None
            result = analyze_video(path, app.config["SAMPLE_EVERY_SECONDS"], app.config["MAX_DURATION_SECONDS"], detector,
                                   lambda pct: update_job(job_id, {"progress_percent": pct}), output_path,
                                   face_detector=face_detector, object_detector=object_detector)
            if not result.get("annotated_output") and output_path and os.path.isfile(output_path):
                os.remove(output_path)
            from app.db import get_database
            if result["events"]:
                for event in result["events"]:
                    event.update({"video_id": ObjectId(video_id), "created_at": datetime.now(timezone.utc)})
                get_database().detection_events.insert_many(result["events"])
            if result.get("face_frames"):
                save_face_frames(video_id, result["face_frames"])
            if result.get("pose_frames"):
                save_pose_frames(video_id, result["pose_frames"])
            face_result = result.get("face_analysis", {})
            if face_init_error:
                face_result.update({"status": "unavailable", "error_message": face_init_error})
            object_result = result.get("object_analysis", {})
            if object_init_error:
                object_result.update({"status": "unavailable", "error_message": object_init_error})
            elif not object_model_path:
                object_result.update({"status": "not_configured", "error_message": None})
            object_result["model_name"] = os.path.basename(object_model_path) if object_model_path else None
            object_result["method"] = YoloObjectTracker.METHOD
            object_result["package_version"] = getattr(object_detector, "package_version", None)
            object_result["confidence_threshold"] = (object_result.get("confidence_threshold")
                                                       if object_result.get("confidence_threshold") is not None
                                                       else (app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25)
                                                             if object_detector else None))
            result["object_analysis"] = object_result
            pose_result = result.get("pose_analysis", {})
            if pose_init_error:
                pose_result.update({"status": "unavailable", "error_message": pose_init_error,
                                    "model_name": os.path.basename(model_path) if model_path else None})
            elif not model_path:
                pose_result.update({"status": "not_configured", "error_message": None})
            result["pose_analysis"] = pose_result
            anomaly_events = [event for event in result["events"]
                              if event.get("event_type") != "object_detection"]
            object_events = [event for event in result["events"]
                             if event.get("event_type") == "object_detection"]
            update_video(video_id, {"processing_status": "complete", "duration": result.get("duration"),
                "frame_rate": result.get("fps"), "width": result.get("width"), "height": result.get("height"),
                "processing_duration": time.monotonic()-started, "error_message": None,
                "analysis_mode": result.get("analysis_mode"), "detection_method": result.get("method"),
                "event_count": len(anomaly_events), "object_detection_count": len(object_events),
                "face_analysis": face_result,
                "face_analysis_status": face_result.get("status", "unavailable"),
                "pose_analysis": pose_result,
                "object_analysis": object_result,
                "analysis_settings": {"sample_every_seconds": app.config["SAMPLE_EVERY_SECONDS"],
                                      "max_duration_seconds": app.config["MAX_DURATION_SECONDS"],
                                      "fall_transition_window_seconds": app.config.get("FALL_TRANSITION_WINDOW_SECONDS", 1.5),
                                      "fall_min_vertical_drop": app.config.get("FALL_MIN_VERTICAL_DROP", .12),
                                      "fall_min_horizontal_persistence": app.config.get("FALL_MIN_HORIZONTAL_PERSISTENCE", .6),
                                      "fall_candidate_score_threshold": app.config.get("FALL_CANDIDATE_SCORE_THRESHOLD", .55),
                                      "object_confidence": app.config.get("OBJECT_MODEL_CONFIDENCE", 0.25),
                                      "object_tracker": app.config.get("OBJECT_TRACKER", "bytetrack.yaml")},
                "analysis_summary": build_analysis_summary(result, app.config["SAMPLE_EVERY_SECONDS"]),
                "processed_filename": output_name if result.get("annotated_output") else None})
            update_job(job_id, {"status": "complete", "progress_percent": 100,
                                "completed_at": datetime.now(timezone.utc)})
    except (VideoError, DetectorUnavailable) as exc:
        _fail_analysis(app, video_id, job_id, started, str(exc))
    except Exception as exc:
        log.error("Video worker failed for record %s (%s); traceback follows:\n%s",
                  video_id[:8], type(exc).__name__, sanitized_traceback(exc, app.config))
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
                                "error_message": message,
                                "face_analysis": {"status": "failed", "method": OpenCVFaceDetector.METHOD,
                                                  "error_message": "Face analysis did not complete because the video job failed."}})
        update_job(job_id, {"status": "failed", "error_message": message, "completed_at": now})


@bp.get("/results/<video_id>")
def results(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id, limit=2000)
    video["id"] = str(video["_id"])
    video["original_name"] = video.get("original_filename")
    # Older MongoDB records may predate upload_date, or contain a serialized
    # timestamp. Keep their results page usable instead of calling strftime
    # on a missing/non-datetime value.
    uploaded_at = video.get("upload_date")
    if isinstance(uploaded_at, datetime):
        if uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)
        video["uploaded_at"] = uploaded_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elif isinstance(uploaded_at, str):
        try:
            parsed_upload_date = datetime.fromisoformat(uploaded_at.replace("Z", "+00:00"))
            if parsed_upload_date.tzinfo is None:
                parsed_upload_date = parsed_upload_date.replace(tzinfo=timezone.utc)
            video["uploaded_at"] = parsed_upload_date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            video["uploaded_at"] = "Unknown"
    else:
        video["uploaded_at"] = "Unknown"
    video["status"] = video.get("processing_status") or "unknown"
    video["fps"] = video.get("frame_rate")
    video["processing_seconds"] = video.get("processing_duration")
    video["error"] = video.get("error_message")
    video["analysis_mode"] = video.get("analysis_mode", "pending")
    video["has_processed_video"] = bool(safe_processed_path(video, current_app.config["PROCESSED_DIR"]))
    video["face_analysis"] = video.get("face_analysis") or {"status": "not_run", "face_observations": 0}
    video["pose_analysis"] = video.get("pose_analysis") or {"status": "not_configured", "frames_sampled": 0}
    video["object_analysis"] = video.get("object_analysis") or {"status": "not_configured", "counts_by_class": {}}
    face_frames = get_face_frames(video_id)
    pose_frames = get_pose_frames(video_id)
    for face_frame in face_frames:
        face_frame["timestamp"] = face_frame.get("timestamp_seconds", 0)
    for event in events:
        event["category"] = event.get("event_type", "Uncategorized")
        event["timestamp"] = event.get("timestamp_seconds", 0)
        event["score"] = event.get("anomaly_score", event.get("confidence"))
        event["method"] = event.get("detection_method", "Unknown")
        event["label"] = event.get("label", "")
        event["track"] = event.get("track_id")
        event["frame"] = event.get("frame_number")
        event["details"] = event.get("details") or event.get("metadata") or {}
    event_types = sorted({event.get("event_type", "unknown") for event in events})
    event_labels = sorted({event["label"] for event in events if event.get("label")})
    event_tracks = sorted({str(event["track_id"]) for event in events if event.get("track_id") is not None})
    return render_template("results.html", video=video, events=events, face_frames=face_frames,
                           pose_frames=pose_frames,
                           event_types=event_types, event_labels=event_labels, event_tracks=event_tracks,
                           event_limit_reached=len(events) >= 2000)


@bp.get("/results/<video_id>/report.csv")
def video_report(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    events = get_events(video_id, limit=50000)
    content = video_report_csv(video, events)
    return Response(content, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=analysis-{video_id[:8]}.csv"})


@bp.get("/api/videos/<video_id>/status")
def status(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    job = get_job_for_video(video_id) or {}
    status = job.get("status", video.get("processing_status", "queued"))
    if status == "running":
        status = "processing"
    return jsonify({"status": status, "progress": job.get("progress_percent", 0),
                    "error": job.get("error_message"), "analysis_mode": video.get("analysis_mode"),
                    "face_status": (video.get("face_analysis") or {}).get("status", "not_run"),
                    "face_error": (video.get("face_analysis") or {}).get("error_message"),
                    "pose_status": (video.get("pose_analysis") or {}).get("status", "not_configured"),
                    "pose_error": (video.get("pose_analysis") or {}).get("error_message"),
                    "object_status": (video.get("object_analysis") or {}).get("status", "not_configured"),
                    "object_error": (video.get("object_analysis") or {}).get("error_message")})


@bp.get("/media/<video_id>")
def media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_DIR"], os.path.basename(target), as_attachment=False)


@bp.get("/media/<video_id>/download")
def download_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_video_path(video, current_app.config["UPLOAD_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_DIR"], os.path.basename(target),
                               as_attachment=True, download_name=video.get("original_filename", "video"))


@bp.get("/processed/<video_id>")
def processed_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_processed_path(video, current_app.config["PROCESSED_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["PROCESSED_DIR"], os.path.basename(target), as_attachment=False)


@bp.get("/processed/<video_id>/download")
def download_processed_media(video_id):
    video = get_video(video_id)
    if not video:
        abort(404)
    target = safe_processed_path(video, current_app.config["PROCESSED_DIR"])
    if not target:
        abort(404)
    return send_from_directory(current_app.config["PROCESSED_DIR"], os.path.basename(target),
                               as_attachment=True, download_name="annotated-analysis.mp4")
