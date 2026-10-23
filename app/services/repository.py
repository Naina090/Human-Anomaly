"""MongoDB persistence operations kept separate from Flask route handlers."""
import logging
import re
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING
from pymongo.errors import PyMongoError
from flask import current_app
from app.db import DatabaseUnavailable, _friendly_connection_error, get_database

log = logging.getLogger(__name__)


def object_id(value):
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _run(fn):
    try:
        return fn(get_database())
    except DatabaseUnavailable:
        raise
    except PyMongoError as exc:
        current_app.extensions["mongo_available"] = False
        log.warning("MongoDB persistence operation failed (%s, code=%s): %s",
                    type(exc).__name__, getattr(exc, "code", None), _friendly_connection_error(exc))
        raise DatabaseUnavailable("MongoDB could not complete the request. Check that the database is reachable.") from exc


def create_video(document):
    return _run(lambda db: db.videos.insert_one(document).inserted_id)


def create_job(document):
    return _run(lambda db: db.processing_jobs.insert_one(document).inserted_id)


def update_job(job_id, changes):
    oid = object_id(job_id)
    return False if oid is None else _run(lambda db: current_job_update(db, oid, changes))


def current_job_update(db, oid, changes):
    return db.processing_jobs.update_one({"_id": oid}, {"$set": changes}).matched_count == 1


def get_job_for_video(video_id):
    oid = object_id(video_id)
    return None if oid is None else _run(lambda db: db.processing_jobs.find_one({"video_id": oid}, sort=[("created_at", -1)]))


def video_is_processing(video_id):
    oid = object_id(video_id)
    return False if oid is None else _run(lambda db: db.processing_jobs.count_documents(
        {"video_id": oid, "status": {"$in": ["queued", "running"]}}) > 0)


def update_video(video_id, changes):
    oid = object_id(video_id)
    if oid is None:
        return False
    changes["updated_at"] = datetime.now(timezone.utc)
    return _run(lambda db: db.videos.update_one({"_id": oid}, {"$set": changes}).matched_count == 1)


def get_video(video_id):
    oid = object_id(video_id)
    if oid is None:
        return None
    return _run(lambda db: db.videos.find_one({"_id": oid}))


def get_events(video_id, limit=2000):
    oid = object_id(video_id)
    if oid is None:
        return []
    limit = max(1, min(50000, int(limit)))
    return _run(lambda db: list(db.detection_events.find(
        {"video_id": oid, "superseded": {"$ne": True}}).sort("timestamp_seconds", 1).limit(limit)))


def get_face_frames(video_id, limit=1200):
    oid = object_id(video_id)
    if oid is None:
        return []
    return _run(lambda db: list(db.face_detection_frames.find({"video_id": oid, "superseded": {"$ne": True}},
        {"_id": 0}).sort("timestamp_seconds", 1).limit(max(1, min(2000, int(limit))))))


def save_face_frames(video_id, frames):
    oid = object_id(video_id)
    if oid is None or not frames:
        return 0
    now = datetime.now(timezone.utc)
    documents = [{**frame, "video_id": oid, "created_at": now} for frame in frames]
    return _run(lambda db: len(db.face_detection_frames.insert_many(documents).inserted_ids))


def get_pose_frames(video_id, limit=1200):
    oid = object_id(video_id)
    if oid is None:
        return []
    return _run(lambda db: list(db.pose_detection_frames.find(
        {"video_id": oid, "superseded": {"$ne": True}}, {"_id": 0}
    ).sort("timestamp_seconds", 1).limit(max(1, min(2000, int(limit))))))


def save_pose_frames(video_id, frames):
    oid = object_id(video_id)
    if oid is None or not frames:
        return 0
    now = datetime.now(timezone.utc)
    documents = [{**frame, "video_id": oid, "created_at": now} for frame in frames]
    return _run(lambda db: len(db.pose_detection_frames.insert_many(documents).inserted_ids))


def queue_video_retry(video_id):
    """Atomically claim a failed/interrupted video for one retry worker."""
    oid = object_id(video_id)
    if oid is None:
        return False
    def claim(db):
        result = db.videos.update_one(
            {"_id": oid, "processing_status": {"$in": ["failed", "interrupted"]}},
            {"$set": {"processing_status": "queued", "error_message": None,
                       "face_analysis": {"status": "queued", "method": "OpenCV Haar cascade (haarcascade_frontalface_default)"},
                       "pose_analysis": {"status": "queued"}, "object_analysis": {"status": "queued"},
                       "event_count": 0, "analysis_mode": "pending", "detection_method": None,
                       "processing_duration": None, "processed_filename": None,
                       "updated_at": datetime.now(timezone.utc)}})
        if result.modified_count != 1:
            return False
        # Preserve prior actual results but mark them as superseded so a retry's
        # result page and analytics do not combine multiple attempts.
        db.detection_events.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        db.face_detection_frames.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        db.pose_detection_frames.update_many({"video_id": oid, "superseded": {"$ne": True}},
            {"$set": {"superseded": True, "superseded_at": datetime.now(timezone.utc)}})
        return True
    return _run(claim)


def mark_incomplete_jobs_interrupted():
    """Mark work left by a prior local Flask process; no worker survives restart."""
    now = datetime.now(timezone.utc)
    def recover(db):
        active_ids = [row["_id"] for row in db.videos.find(
            {"processing_status": {"$in": ["queued", "processing"]}}, {"_id": 1})]
        if not active_ids:
            return 0, 0
        jobs = db.processing_jobs.update_many(
            {"video_id": {"$in": active_ids}, "status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "interrupted", "error_message": "The Flask process stopped before this job completed.",
                       "completed_at": now}})
        videos = db.videos.update_many(
            {"_id": {"$in": active_ids}, "processing_status": {"$in": ["queued", "processing"]}},
            {"$set": {"processing_status": "interrupted",
                       "error_message": "The Flask process stopped before this job completed.",
                       "pose_analysis.status": "interrupted",
                       "face_analysis.status": "interrupted",
                       "object_analysis.status": "interrupted",
                       "updated_at": now}})
        return jobs.modified_count, videos.modified_count
    return _run(recover)


def dashboard_stats():
    def query(db):
        videos = db.videos
        events = db.detection_events
        total = videos.count_documents({})
        completed = videos.count_documents({"processing_status": "complete"})
        failed = videos.count_documents({"processing_status": "failed"})
        active = videos.count_documents({"processing_status": {"$in": ["queued", "processing"]}})
        # Dashboard anomaly metrics intentionally exclude ordinary object
        # observations. Those are shown separately on per-video results.
        active_events = {"superseded": {"$ne": True}, "event_type": {"$ne": "object_detection"}}
        event_count = events.count_documents(active_events)
        object_query = {"superseded": {"$ne": True}, "event_type": "object_detection"}
        object_observations = events.count_documents(object_query)
        object_categories = list(events.aggregate([
            {"$match": object_query},
            {"$group": {"_id": "$label", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]))
        track_groups = list(events.aggregate([
            {"$match": {**object_query, "label": "person", "track_id": {"$ne": None}}},
            {"$group": {"_id": {"video_id": "$video_id", "track_id": "$track_id"}}},
            {"$limit": 10001},
        ]))
        face_summary = list(videos.aggregate([
            {"$group": {"_id": None,
                        "completed": {"$sum": {"$cond": [{"$eq": ["$face_analysis.status", "complete"]}, 1, 0]}},
                        "observations": {"$sum": {"$ifNull": ["$face_analysis.face_observations", 0]}}}}
        ]))
        face_summary = face_summary[0] if face_summary else {"completed": 0, "observations": 0}
        categories = list(events.aggregate([{"$match": active_events},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]))
        recent = list(events.aggregate([
            {"$match": active_events}, {"$sort": {"created_at": -1}}, {"$limit": 8},
            {"$lookup": {"from": "videos", "localField": "video_id", "foreignField": "_id", "as": "video"}},
            {"$unwind": {"path": "$video", "preserveNullAndEmptyArrays": True}},
        ]))
        recent_videos = list(videos.find({}, {"original_filename": 1, "upload_date": 1,
            "processing_status": 1, "duration": 1, "processing_duration": 1}).sort("upload_date", DESCENDING).limit(6))
        return {"videos": total, "done": completed, "failed": failed, "active": active, "events": event_count,
                "queued": videos.count_documents({"processing_status": "queued"}),
                "processing": videos.count_documents({"processing_status": "processing"}),
                "object_observations": object_observations,
                "object_categories": [{"label": row["_id"] or "Unknown", "count": row["count"]}
                                      for row in object_categories],
                "person_track_estimate": len(track_groups),
                "person_track_estimate_capped": len(track_groups) > 10000,
                "face_videos": face_summary.get("completed", 0),
                "face_observations": face_summary.get("observations", 0),
                "categories": [{"category": row["_id"] or "Uncategorized", "count": row["count"]} for row in categories],
                "recent_videos": recent_videos,
                "recent": [{**row, "original_filename": (row.get("video") or {}).get("original_filename", "Deleted video")} for row in recent]}
    return _run(query)


def list_videos(search="", status="", page=1, per_page=25):
    query = {}
    if search:
        query["original_filename"] = {"$regex": re.escape(search[:100]), "$options": "i"}
    if status in {"complete", "failed", "processing", "queued", "interrupted"}:
        query["processing_status"] = status
    page = max(1, int(page))
    def find(db):
        coll = db.videos
        total = coll.count_documents(query)
        docs = list(coll.find(query).sort("upload_date", DESCENDING).skip((page - 1) * per_page).limit(per_page))
        event_counts = {row["_id"]: row["count"] for row in db.detection_events.aggregate([
            {"$match": {"video_id": {"$in": [d["_id"] for d in docs]},
                        "superseded": {"$ne": True}, "event_type": {"$ne": "object_detection"}}},
            {"$group": {"_id": "$video_id", "count": {"$sum": 1}}},
        ])} if docs else {}
        for doc in docs:
            doc["event_count"] = event_counts.get(doc["_id"], 0)
        return docs, total
    return _run(find)


def delete_video(video_id):
    oid = object_id(video_id)
    if oid is None:
        return None
    def delete(db):
        video = db.videos.find_one({"_id": oid})
        if video is None:
            return None
        db.detection_events.delete_many({"video_id": oid})
        db.face_detection_frames.delete_many({"video_id": oid})
        db.pose_detection_frames.delete_many({"video_id": oid})
        db.processing_jobs.delete_many({"video_id": oid})
        db.videos.delete_one({"_id": oid})
        return video
    return _run(delete)


def report_data(start=None, end=None):
    query = {}
    date_filter = {}
    if start:
        date_filter["$gte"] = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end:
        date_filter["$lt"] = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
    if date_filter:
        query["upload_date"] = date_filter
    def find(db):
        videos = list(db.videos.find(query).sort("upload_date", DESCENDING).limit(5000))
        ids = [v["_id"] for v in videos]
        events = list(db.detection_events.find({"video_id": {"$in": ids}, "superseded": {"$ne": True}}).sort("timestamp_seconds", 1)) if ids else []
        names = {v["_id"]: v.get("original_filename", "") for v in videos}
        for event in events:
            event["original_filename"] = names.get(event["video_id"], "")
        return videos, events
    return _run(find)
