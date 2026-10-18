"""MongoDB persistence operations kept separate from Flask route handlers."""
import re
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING
from pymongo.errors import PyMongoError
from flask import current_app
from app.db import DatabaseUnavailable, get_database


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
    return None if oid is None else _run(lambda db: db.processing_jobs.find_one({"video_id": oid}, sort=[("started_at", -1)]))


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


def get_events(video_id):
    oid = object_id(video_id)
    if oid is None:
        return []
    return _run(lambda db: list(db.detection_events.find({"video_id": oid}).sort("timestamp_seconds", 1)))


def dashboard_stats():
    def query(db):
        videos = db.videos
        events = db.detection_events
        total = videos.count_documents({})
        completed = videos.count_documents({"processing_status": "complete"})
        failed = videos.count_documents({"processing_status": "failed"})
        active = videos.count_documents({"processing_status": {"$in": ["queued", "processing"]}})
        event_count = events.count_documents({})
        categories = list(events.aggregate([{"$group": {"_id": "$event_type", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]))
        recent = list(events.aggregate([
            {"$sort": {"created_at": -1}}, {"$limit": 8},
            {"$lookup": {"from": "videos", "localField": "video_id", "foreignField": "_id", "as": "video"}},
            {"$unwind": {"path": "$video", "preserveNullAndEmptyArrays": True}},
        ]))
        recent_videos = list(videos.find({}, {"original_filename": 1, "upload_date": 1,
            "processing_status": 1, "duration": 1, "processing_duration": 1}).sort("upload_date", DESCENDING).limit(6))
        return {"videos": total, "done": completed, "failed": failed, "active": active, "events": event_count,
                "categories": [{"category": row["_id"] or "Uncategorized", "count": row["count"]} for row in categories],
                "recent_videos": recent_videos,
                "recent": [{**row, "original_filename": (row.get("video") or {}).get("original_filename", "Deleted video")} for row in recent]}
    return _run(query)


def list_videos(search="", status="", page=1, per_page=25):
    query = {}
    if search:
        query["original_filename"] = {"$regex": re.escape(search[:100]), "$options": "i"}
    if status in {"complete", "failed", "processing", "queued"}:
        query["processing_status"] = status
    page = max(1, int(page))
    def find(db):
        coll = db.videos
        total = coll.count_documents(query)
        docs = list(coll.find(query).sort("upload_date", DESCENDING).skip((page - 1) * per_page).limit(per_page))
        event_counts = {row["_id"]: row["count"] for row in db.detection_events.aggregate([
            {"$match": {"video_id": {"$in": [d["_id"] for d in docs]}}},
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
        events = list(db.detection_events.find({"video_id": {"$in": ids}}).sort("timestamp_seconds", 1)) if ids else []
        names = {v["_id"]: v.get("original_filename", "") for v in videos}
        for event in events:
            event["original_filename"] = names.get(event["video_id"], "")
        return videos, events
    return _run(find)
