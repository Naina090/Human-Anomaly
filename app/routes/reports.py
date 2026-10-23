import csv
import io
from datetime import datetime
from flask import Blueprint, render_template, request, Response, abort
from app.services.repository import report_data

bp = Blueprint("reports", __name__)


def validated_date(value):
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        abort(400)


@bp.get("/reports")
def reports():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, events = report_data(start, end)
    for video in videos:
        video["id"] = str(video["_id"])
        video["status"] = video.get("processing_status")
    for event in events:
        event["video_id"] = str(event["video_id"])
        event["category"] = event.get("event_type", "Uncategorized")
        event["timestamp"] = event.get("timestamp_seconds")
        event["score"] = event.get("anomaly_score", event.get("confidence"))
        event["method"] = event.get("detection_method", "Unknown")
    return render_template("reports.html", videos=videos, events=events, start=start, end=end)


@bp.get("/reports.csv")
def csv_export():
    start, end = validated_date(request.args.get("start", "")), validated_date(request.args.get("end", ""))
    if start and end and start > end:
        abort(400)
    videos, events = report_data(start, end)
    out = io.StringIO(); writer = csv.writer(out)
    writer.writerow(["video_id", "filename", "uploaded_at", "status", "event_category", "label",
                     "timestamp_seconds", "start_timestamp_seconds", "end_timestamp_seconds",
                     "frame_number", "confidence", "anomaly_score", "track_id", "method", "details"])
    names = {str(v["_id"]): (v.get("original_filename", ""), v.get("upload_date", ""), v.get("processing_status", "")) for v in videos}
    for e in events:
        event_video_id = str(e["video_id"])
        filename, uploaded, status = names.get(event_video_id, (e.get("original_filename", ""), "", ""))
        writer.writerow([event_video_id, filename, uploaded, status, e.get("event_type", ""),
                         e.get("label", ""), e.get("timestamp_seconds", ""),
                         e.get("start_timestamp_seconds", ""), e.get("end_timestamp_seconds", ""),
                         e.get("frame_number", ""),
                         e.get("confidence", ""), e.get("anomaly_score", ""), e.get("track_id", ""),
                         e.get("detection_method", ""), repr(e.get("details", e.get("metadata", "")))])
    return Response(out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=anomaly-report.csv"})
