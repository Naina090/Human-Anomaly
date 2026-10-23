import time
import logging
from collections import Counter
import cv2
from flask import current_app, has_app_context
from app.logging_utils import sanitized_traceback

log = logging.getLogger(__name__)

class VideoError(Exception):
    pass


def analyze_video(path, sample_seconds, max_duration, detector=None, progress=None, output_path=None,
                 face_detector=None, object_detector=None):
    start = time.monotonic()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoError("OpenCV could not read this video. Try an MP4 encoded with H.264.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0:
        capture.release()
        raise VideoError("The file has no readable video frames.")
    if duration > max_duration:
        capture.release()
        raise VideoError(f"Video is {duration:.0f}s; maximum allowed is {max_duration}s.")
    # Read at a bounded interval without retaining a complete video in memory.
    interval = max(1, int(fps * sample_seconds))
    read = 0
    events = []
    pose_frames = []
    pose_error = None
    pose_frames_sampled = 0
    pose_frames_with_people = 0
    face_frames = []
    face_error = None
    object_error = None
    object_frames_sampled = 0
    object_counts = Counter()
    person_track_ids = set()
    object_observations = 0
    frame_index = 0
    writer = None
    annotated = False
    if (detector is not None or face_detector is not None or object_detector is not None) and output_path:
        try:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                writer.release()
                writer = None
        except cv2.error:
            writer = None
    try:
        while frame_index < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                read += 1
                source_frame = frame
                if detector is not None:
                    try:
                        timestamp = frame_index / fps
                        if hasattr(detector, "inspect_with_evidence"):
                            frame, event, pose_evidence = detector.inspect_with_evidence(frame, timestamp)
                            pose_evidence["frame_number"] = int(frame_index)
                            pose_evidence["detection_method"] = getattr(detector, "METHOD", "pose estimator")
                            pose_frames.append(pose_evidence)
                            pose_frames_sampled += 1
                            pose_frames_with_people += int(pose_evidence.get("person_count", 0) > 0)
                        else:
                            frame, event = detector.inspect(frame, timestamp)
                            pose_frames_sampled += 1
                        annotated = writer is not None
                        if event:
                            events.append(event)
                    except Exception as exc:
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional pose inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        pose_error = "Pose inference stopped after a model error. See server logs for the exception type."
                        detector = None
                if object_detector is not None:
                    try:
                        timestamp = frame_index / fps
                        observations = object_detector.detect(source_frame, timestamp, frame_index)
                        object_frames_sampled += 1
                        object_observations += len(observations)
                        for observation in observations:
                            events.append(observation)
                            object_counts[observation["label"]] += 1
                            if observation.get("label") == "person" and observation.get("track_id") is not None:
                                person_track_ids.add(observation["track_id"])
                        frame = object_detector.annotate(frame, observations)
                        annotated = annotated or writer is not None
                    except Exception as exc:
                        # Optional object inference must not invalidate decoded video,
                        # face analysis, or the existing fall heuristic.
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional object inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        object_error = "Object detection stopped after a model inference error."
                        object_detector = None
                if face_detector is not None:
                    try:
                        boxes = face_detector.detect(source_frame)
                        face_frames.append({"frame_index": int(frame_index),
                                            "timestamp_seconds": float(frame_index / fps),
                                            "face_count": len(boxes), "boxes": boxes})
                        frame = face_detector.annotate(frame, boxes)
                        annotated = writer is not None
                    except Exception as exc:
                        # A face-detector error must not stop the existing anomaly
                        # pipeline or invalidate successfully decoded video frames.
                        config = current_app.config if has_app_context() else {}
                        log.error("Optional face inference stopped (%s); traceback follows:\n%s",
                                  type(exc).__name__, sanitized_traceback(exc, config))
                        face_error = "Face detection stopped after an OpenCV processing error."
                        face_detector = None
                if progress:
                    progress(min(95, int(95 * frame_index / max(frame_count, 1))))
            if writer is not None:
                try:
                    writer.write(frame)
                except cv2.error:
                    writer.release()
                    writer = None
                    annotated = False
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if read == 0:
        raise VideoError("No readable frames were found in this file.")
    frames_with_faces = sum(1 for item in face_frames if item["face_count"] > 0)
    face_counts = [item["face_count"] for item in face_frames]
    return {"duration": duration, "fps": fps, "width": width, "height": height,
            "processing_seconds": time.monotonic() - start, "sampled_frames": read,
            "events": events, "method": getattr(detector, "METHOD", "YOLO pose COCO keypoints")
            if (detector or pose_frames_sampled or pose_error) else "validation_only",
            "analysis_mode": "pose_heuristic" if (detector or pose_frames_sampled or pose_error) else "validation_only",
            "annotated_output": bool(annotated), "pose_frames": pose_frames,
            "pose_analysis": {"status": "failed" if pose_error else ("complete" if pose_frames_sampled else "not_configured"),
                              "method": getattr(detector, "METHOD", "YOLO pose COCO keypoints") if detector else "YOLO pose COCO keypoints",
                              "model_name": getattr(detector, "model_name", None),
                              "model_version": getattr(detector, "model_version", None),
                              "frames_sampled": pose_frames_sampled,
                              "frames_with_people": pose_frames_with_people,
                              "error_message": pose_error},
            "face_frames": face_frames,
            "face_analysis": {"status": "failed" if face_error else ("complete" if face_counts else "unavailable"),
                              "method": face_detector.METHOD if face_detector else "OpenCV Haar cascade (haarcascade_frontalface_default)",
                              "frames_sampled": len(face_frames),
                              "frames_with_faces": frames_with_faces,
                              "face_observations": sum(face_counts),
                              "max_faces_per_frame": max(face_counts, default=0),
                              "error_message": face_error},
            "object_analysis": {"status": "failed" if object_error else ("complete" if object_frames_sampled else "not_run"),
                                "method": getattr(object_detector, "METHOD", "Ultralytics YOLO COCO detector + ByteTrack"),
                                "frames_sampled": object_frames_sampled,
                                "observations": object_observations,
                                "counts_by_class": dict(sorted(object_counts.items())),
                                "person_track_count_estimate": len(person_track_ids),
                                "person_track_count_note": "Approximate temporary tracker IDs, not unique identities.",
                                "confidence_threshold": getattr(object_detector, "confidence", None),
                                "error_message": object_error}}
