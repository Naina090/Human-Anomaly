"""Formatting helpers for evidence-based reports built from saved MongoDB data."""
import csv
import io

from app.services.repository import report_data


def build_analysis_summary(result, sample_seconds):
    """Summarize only the detector outputs and metadata present in ``result``."""
    duration = result.get("duration")
    sampled = int(result.get("sampled_frames") or 0)
    parts = []
    if duration is not None:
        parts.append(f"Video duration: {float(duration):.2f} seconds; {sampled} frames sampled at approximately {float(sample_seconds):g}-second intervals.")

    objects = result.get("object_analysis") or {}
    if objects.get("status") == "complete":
        counts = objects.get("counts_by_class") or {}
        if counts:
            count_text = ", ".join(f"{label}: {count} frame observations" for label, count in sorted(counts.items()))
            parts.append(f"Supported object observations: {count_text}.")
        else:
            parts.append("No supported objects were observed in the sampled frames; this does not rule out objects between samples.")
        tracks = int(objects.get("person_track_count_estimate") or 0)
        parts.append(f"Approximate person tracks: {tracks}; tracker IDs are temporary associations, not unique identities.")
    elif objects.get("status") in {"unavailable", "failed"}:
        parts.append("Object detection was unavailable: " + (objects.get("error_message") or "the configured detector could not run."))
    else:
        parts.append("No object detector was configured; no object predictions were produced.")

    pose = result.get("pose_analysis") or {}
    if pose.get("status") == "complete":
        parts.append(f"Pose estimation sampled {int(pose.get('frames_sampled') or 0)} frames; "
                     f"people were observed in {int(pose.get('frames_with_people') or 0)} sampled frames. "
                     "Keypoints are model estimates, not identity data.")
    elif pose.get("status") in {"unavailable", "failed"}:
        parts.append("Pose estimation was unavailable: " +
                     (pose.get("error_message") or "the configured model could not run."))
    else:
        parts.append("No pose model was configured; no pose or fall estimate was produced.")

    events = result.get("events") or []
    fall_events = [event for event in events if event.get("event_type") == "potential_fall"]
    if pose.get("status") in {"unavailable", "failed"}:
        pass
    elif pose.get("status") == "complete" or result.get("analysis_mode") == "pose_heuristic":
        if fall_events:
            times = ", ".join(f"{float(event.get('timestamp_seconds', 0)):.2f}s (score {event.get('anomaly_score')})" for event in fall_events)
            parts.append("The unvalidated pose transition heuristic produced candidate review signals at " + times + ". Scores are review signals, not proof of an incident.")
        else:
            parts.append("The unvalidated pose transition heuristic produced no candidate event in the sampled frames; this is not evidence that no fall occurred.")
    elif pose.get("status") not in {"not_configured", None}:
        parts.append("No pose-based anomaly model was configured, so no fall/anomaly prediction was produced.")
    return " ".join(parts)


def video_report_csv(video, events):
    """Return a flat report with one metadata row and one row per stored event."""
    output = io.StringIO(newline="")
    columns = ["record_type", "video_id", "filename", "status", "duration_seconds",
               "sample_interval_seconds", "pose_status", "pose_model", "pose_model_version",
               "pose_frames_sampled", "object_status", "object_model", "object_counts_by_class",
               "person_track_count_estimate", "summary", "event_type", "label",
               "timestamp_seconds", "start_timestamp_seconds", "end_timestamp_seconds",
               "frame_number", "confidence", "anomaly_score", "track_id", "detection_method", "details"]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    video_id = str(video.get("_id", ""))
    writer.writerow({"record_type": "video", "video_id": video_id,
        "filename": video.get("original_filename", ""),
        "status": video.get("processing_status", ""),
        "duration_seconds": video.get("duration", ""),
        "sample_interval_seconds": (video.get("analysis_settings") or {}).get("sample_every_seconds", ""),
        "pose_status": (video.get("pose_analysis") or {}).get("status", ""),
        "pose_model": (video.get("pose_analysis") or {}).get("model_name", ""),
        "pose_model_version": (video.get("pose_analysis") or {}).get("model_version", ""),
        "pose_frames_sampled": (video.get("pose_analysis") or {}).get("frames_sampled", ""),
        "object_status": (video.get("object_analysis") or {}).get("status", ""),
        "object_model": (video.get("object_analysis") or {}).get("model_name", ""),
        "object_counts_by_class": repr((video.get("object_analysis") or {}).get("counts_by_class", {})),
        "person_track_count_estimate": (video.get("object_analysis") or {}).get("person_track_count_estimate", ""),
        "summary": video.get("analysis_summary", "")})
    for event in events:
        writer.writerow({"record_type": "event", "video_id": video_id,
            "filename": video.get("original_filename", ""),
            "status": video.get("processing_status", ""),
            "event_type": event.get("event_type", ""), "label": event.get("label", ""),
            "timestamp_seconds": event.get("timestamp_seconds", ""),
            "start_timestamp_seconds": event.get("start_timestamp_seconds", ""),
            "end_timestamp_seconds": event.get("end_timestamp_seconds", ""),
            "frame_number": event.get("frame_number", ""),
            "confidence": event.get("confidence", ""),
            "anomaly_score": event.get("anomaly_score", ""),
            "track_id": event.get("track_id", ""),
            "detection_method": event.get("detection_method", ""),
            "details": repr(event.get("details", event.get("metadata", "")))})
    return output.getvalue()


__all__ = ["report_data", "video_report_csv"]
