import io
import os
import time
from datetime import datetime, timezone

import cv2
import mongomock
import numpy as np
import pytest
from bson import ObjectId

from app import create_app
from app.db import DatabaseUnavailable
from app.services.repository import (
    create_video, dashboard_stats, delete_video, get_video, list_videos,
    object_id, report_data, update_video,
)
from app.services.fall_detector import YoloPoseFallDetector
from app.services.storage import safe_video_path, safe_processed_path


@pytest.fixture()
def app(tmp_path):
    client = mongomock.MongoClient()
    application = create_app({
        "TESTING": True, "SECRET_KEY": "test-secret", "MONGO_CLIENT": client,
        "MONGO_DB_NAME": "test_human_anomaly", "MONGO_PING_ON_STARTUP": False,
        "UPLOAD_DIR": str(tmp_path / "uploads"), "PROCESSED_DIR": str(tmp_path / "processed"),
        "MAX_CONTENT_LENGTH": 2_000_000, "MAX_DURATION_SECONDS": 30,
        "SAMPLE_EVERY_SECONDS": 0.1,
    })
    yield application
    application.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    client.close()


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client):
    client.get("/upload")
    with client.session_transaction() as session:
        return session["csrf_token"]


def video_doc(name="clip.mp4", status="processing"):
    return {
        "original_filename": name, "stored_filename": "internal.mp4", "file_path": "internal.mp4",
        "upload_date": datetime.now(timezone.utc), "processing_status": status,
        "duration": None, "frame_rate": None, "width": None, "height": None,
        "processing_duration": None, "error_message": None,
    }


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def wait_for_status(app, video_id, statuses=("complete", "failed")):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = app.extensions["mongo_db"].videos.find_one({"_id": ObjectId(video_id)})
        if row and row["processing_status"] in statuses:
            return row
        time.sleep(.02)
    raise AssertionError("Background video job did not finish in time")


def test_app_startup_and_pages_render(client):
    for url in ("/", "/dashboard", "/about", "/history", "/reports", "/upload"):
        assert client.get(url).status_code == 200


def test_dashboard_shows_database_and_detector_readiness_without_fake_counts(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"MongoDB" in response.data and b"Connected" in response.data
    assert b"Pose analysis" in response.data and b"Validation only" in response.data
    assert b">0</strong>" in response.data
    assert b"No videos yet" in response.data and b"No events recorded" in response.data


def test_mongodb_connection_failure_is_friendly(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "MONGO_URI": "mongodb://127.0.0.1:1/",
                      "MONGO_TIMEOUT_MS": 20, "MONGO_DB_NAME": "connection_test",
                      "UPLOAD_DIR": str(tmp_path / "u"), "PROCESSED_DIR": str(tmp_path / "p")})
    response = app.test_client().get("/dashboard")
    assert response.status_code == 503
    assert b"MongoDB is unavailable" in response.data
    assert app.test_client().get("/").status_code == 200
    app.extensions["mongo_client"].close()


def test_repository_create_read_update_and_dashboard(app):
    with app.app_context():
        record_id = create_video(video_doc())
        assert isinstance(record_id, ObjectId)
        assert get_video(str(record_id))["original_filename"] == "clip.mp4"
        assert update_video(str(record_id), {"processing_status": "complete", "duration": 12.5})
        stats = dashboard_stats()
        assert (stats["videos"], stats["done"], stats["failed"], stats["events"]) == (1, 1, 0, 0)


def test_safe_objectid_validation():
    assert object_id("not-an-object-id") is None
    assert object_id("507f1f77bcf86cd799439011") == ObjectId("507f1f77bcf86cd799439011")


def test_fall_heuristic_requires_a_transition_and_persistence():
    class Tensor:
        def __init__(self, values): self.values = np.asarray(values)
        def cpu(self): return self
        def numpy(self): return self.values
        def __len__(self): return len(self.values)
    class Keypoints:
        def __init__(self, points): self.xy = Tensor([points]); self.conf = Tensor([[.9] * 17])
    class Result:
        def __init__(self, points): self.keypoints = Keypoints(points)
        def plot(self): return np.zeros((100, 100, 3), dtype=np.uint8)
    class PoseModel:
        def __init__(self, frames): self.frames = iter(frames)
        def predict(self, frame, verbose=False): return [Result(next(self.frames))]
    def pose(center, horizontal=False):
        pts = np.zeros((17, 2), dtype=float)
        if horizontal:
            pts[5], pts[6], pts[11], pts[12] = [35, center], [65, center], [42, center+3], [72, center+3]
        else:
            pts[5], pts[6], pts[11], pts[12] = [45, center-20], [55, center-20], [45, center+20], [55, center+20]
        return pts.tolist()
    detector = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    detector.model = PoseModel([pose(40), pose(78, True), pose(78, True)])
    detector.previous = None; detector.track_center = None; detector.upright_seen = False; detector.horizontal_since = None
    detector.drop_seen_at = None; detector.drop_fraction = None; detector.emitted = False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.inspect(frame, 0)[1] is None
    assert detector.inspect(frame, 1)[1] is None
    event = detector.inspect(frame, 2)[1]
    assert event["event_type"] == "potential_fall"
    assert event["confidence"] is None and 0 <= event["anomaly_score"] <= 1

    static = YoloPoseFallDetector.__new__(YoloPoseFallDetector)
    static.model = PoseModel([pose(80, True), pose(80, True), pose(80, True)])
    static.previous = None; static.track_center = None; static.upright_seen = False; static.horizontal_since = None
    static.drop_seen_at = None; static.drop_fraction = None; static.emitted = False
    assert all(static.inspect(frame, t)[1] is None for t in (0, 1, 2))


def test_background_worker_persists_only_detector_emitted_events(client, app, tmp_path, monkeypatch):
    import app.routes.videos as video_routes
    model_marker = tmp_path / "configured-model.pt"
    model_marker.write_bytes(b"test marker")
    app.config["POSE_MODEL_PATH"] = str(model_marker)
    class FixtureDetector:
        METHOD = "synthetic detector fixture"
        def inspect(self, frame, timestamp):
            return frame, {"event_type": "potential_fall", "timestamp_seconds": timestamp,
                           "confidence": None, "anomaly_score": .8, "detection_method": self.METHOD,
                           "metadata": {"source": "synthetic test fixture"}}
    monkeypatch.setattr(video_routes, "YoloPoseFallDetector", lambda _path: FixtureDetector())
    payload = make_video(tmp_path / "pose-test.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "pose-test.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["analysis_mode"] == "pose_heuristic"
    event = app.extensions["mongo_db"].detection_events.find_one({"video_id": ObjectId(video_id)})
    assert event["event_type"] == "potential_fall" and event["confidence"] is None
    assert event["anomaly_score"] == .8 and event["metadata"]["source"] == "synthetic test fixture"
    results = client.get(f"/results/{video_id}")
    assert b"synthetic detector fixture" in results.data
    if row.get("processed_filename"):
        assert b"Annotated pose output" in results.data


def test_processed_path_is_restricted_to_generated_name(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    name = "507f1f77bcf86cd799439011_annotated.mp4"
    (root / name).write_bytes(b"video")
    assert safe_processed_path({"processed_filename": name}, str(root)) == str((root / name).resolve())
    outside = tmp_path / "private.txt"
    outside.write_text("keep")
    assert safe_processed_path({"processed_filename": str(outside)}, str(root)) is None


def test_missing_configured_model_marks_job_failed(client, app, tmp_path):
    app.config["POSE_MODEL_PATH"] = str(tmp_path / "missing-weights.pt")
    payload = make_video(tmp_path / "missing-model.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "missing-model.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    row = wait_for_status(app, video_id)
    assert row["processing_status"] == "failed"
    assert "weights" in row["error_message"].lower()


def test_safe_video_path_requires_generated_internal_name(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    good_name = "a" * 32 + ".mp4"
    good = folder / good_name
    good.write_bytes(b"video")
    assert safe_video_path({"stored_filename": good_name, "file_path": good_name}, str(folder)) == str(good.resolve())
    outside = tmp_path / "important.txt"
    outside.write_text("keep")
    assert safe_video_path({"stored_filename": str(outside), "file_path": str(outside)}, str(folder)) is None
    assert outside.read_text() == "keep"


def test_history_search_filter_event_and_csv(client, app):
    with app.app_context():
        one = create_video(video_doc("fall-study.mp4", "complete"))
        create_video(video_doc("scene-two.avi", "failed"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": one, "event_type": "prototype_test_event",
            "timestamp_seconds": 3.0, "confidence": None, "anomaly_score": None,
            "detection_method": "test-fixture", "created_at": datetime.now(timezone.utc)})
        stats = dashboard_stats()
        assert stats["events"] == 1 and stats["categories"][0]["category"] == "prototype_test_event"
        videos, events = report_data()
        assert len(videos) == 2 and len(events) == 1
        assert report_data("2099-01-01", "2099-01-02") == ([], [])
        assert list_videos("scene", "failed")[1] == 1
    response = client.get("/history?q=fall&status=complete")
    assert response.status_code == 200 and b"fall-study.mp4" in response.data and b"scene-two.avi" not in response.data
    csv_response = client.get("/reports.csv")
    assert csv_response.status_code == 200 and b"prototype_test_event" in csv_response.data


def test_delete_removes_only_video_and_its_events(app):
    with app.app_context():
        target = create_video(video_doc("remove.mp4"))
        keep = create_video(video_doc("keep.mp4"))
        db = app.extensions["mongo_db"]
        db.detection_events.insert_one({"video_id": target, "event_type": "x"})
        db.detection_events.insert_one({"video_id": keep, "event_type": "y"})
        deleted = delete_video(str(target))
        assert deleted["_id"] == target
        assert get_video(str(target)) is None
        assert db.detection_events.count_documents({"video_id": target}) == 0
        assert get_video(str(keep)) is not None
        assert db.detection_events.count_documents({"video_id": keep}) == 1


def test_delete_route_removes_owned_file(client, app, tmp_path):
    payload = make_video(tmp_path / "delete-me.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "delete-me.avi")}, content_type="multipart/form-data")
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    record = app.extensions["mongo_db"].videos.find_one({})
    stored_path = os.path.join(app.config["UPLOAD_DIR"], record["stored_filename"])
    assert os.path.isfile(stored_path)
    deleted = client.post(f"/history/{video_id}/delete", data={"csrf_token": csrf(client)})
    assert deleted.status_code == 302
    assert not os.path.exists(stored_path)
    assert app.extensions["mongo_db"].videos.count_documents({}) == 0


def test_csrf_and_unsupported_upload(client):
    assert client.post("/history/no/delete").status_code == 400
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "bad.exe")}, content_type="multipart/form-data", follow_redirects=True)
    assert b"Supported extensions" in response.data


def test_upload_saves_processing_result_and_database_error_cleanup(client, app, tmp_path, monkeypatch):
    payload = make_video(tmp_path / "small.avi")
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "sample.avi")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    result_page = client.get(response.headers["Location"])
    assert result_page.status_code == 200 and b"Video validation complete" in result_page.data
    assert app.extensions["mongo_db"].videos.count_documents({"processing_status": "complete"}) == 1
    status = client.get(f"/api/videos/{video_id}/status")
    assert status.status_code == 200 and status.json["status"] == "complete"
    job = app.extensions["mongo_db"].processing_jobs.find_one({"video_id": ObjectId(video_id)})
    assert job["progress_percent"] == 100 and job["completed_at"] is not None

    import app.routes.videos as video_routes
    monkeypatch.setattr(video_routes, "create_video", lambda _doc: (_ for _ in ()).throw(DatabaseUnavailable("offline")))
    before = set(os.listdir(app.config["UPLOAD_DIR"]))
    failed = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(payload), "orphan.avi")}, content_type="multipart/form-data")
    assert failed.status_code == 503
    app.extensions["processing_executor"].shutdown(wait=True, cancel_futures=True)
    assert set(os.listdir(app.config["UPLOAD_DIR"])) == before


def test_invalid_video_is_persisted_as_failed(client, app):
    response = client.post("/upload", data={"csrf_token": csrf(client),
        "video": (io.BytesIO(b"not a video"), "broken.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 302
    video_id = response.headers["Location"].rsplit("/", 1)[-1]
    wait_for_status(app, video_id)
    result = client.get(response.headers["Location"])
    assert result.status_code == 200 and b"failed" in result.data
    assert app.extensions["mongo_db"].videos.count_documents({"processing_status": "failed"}) == 1
